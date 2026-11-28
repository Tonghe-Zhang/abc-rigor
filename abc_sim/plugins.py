"""External task plugins: add sim tasks without editing abc_sim.

A plugin registers one :class:`ExternalTask` per scene task.  It brings its own task specs (prompts / aliases),
evaluator classes, randomizer and assets, and picks ONE way to make its scene:

* ``scene_xml``      a static MJCF path (or a zero-argument callable returning one, resolved lazily so assets are only
                     fetched when the scene is built) -- the standard ``make_env`` path, like the built-in tasks;
* ``scene_builder``  ``builder(seed, options) -> MJCF string | mujoco.MjSpec | mujoco.MjModel``: a new scene per seed
                     (procedural layouts, MjSpec-built flex objects).  ``make_env`` builds the seed-``0`` scene (or
                     ``make_env(seed=...)``) and every randomized ``reset(seed=s)`` rebuilds for ``s``; the seed and
                     options are recorded in the randomization metadata, so ``randomizer.apply(state)`` replays it;
* ``env_factory``    ``factory(task=..., **make_env_kwargs) -> env``: a fully custom environment that follows the env
                     surface (:class:`SimEnvSurface`).  Use it only when the YAM ``MuJoCoYAMEnv`` cannot host the task.

Discovery (lazy, on the first task lookup):

* Python entry points in group ``abc_sim.tasks`` -- each entry point is a module (imported) or a callable (called with
  no arguments) that calls :func:`register_task`;
* the ``ABC_SIM_PLUGINS`` environment variable: comma-separated module names or ``module:function`` targets.

Large assets are described by :class:`abc_sim.asset_source.AssetPack` and fetched from the location configured at run
time (``ABC_SIM_ASSET_SOURCE``); nothing about where they live is committed here.  See docs/external-tasks.md.

Example::

    from abc_sim.plugins import ExternalTask, register_task
    from abc_sim.task_specs import SimTaskSpec

    register_task(ExternalTask(
        env_task="my_pick",
        specs=(SimTaskSpec(name="my_pick", env_task="my_pick", prompt="pick up the red cube",
                           evaluator_name="my_pick"),),
        scene_builder=build_scene,            # (seed, options) -> MjSpec
        evaluators={"my_pick": MyPickEvaluator},
    ))
"""

from __future__ import annotations

import importlib
import logging
import os
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

import mujoco
import numpy as np

from abc_sim.asset_source import AssetPack
from abc_sim.task_specs import SimTaskSpec, _normalize_key

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "abc_sim.tasks"
PLUGINS_ENV = "ABC_SIM_PLUGINS"

SceneBuilder = Callable[[int, Mapping[str, Any]], "str | mujoco.MjSpec | mujoco.MjModel"]


@runtime_checkable
class SimEnvSurface(Protocol):
    """What callers (deploy-anything, one-model RL, eval scripts) use of an abc_sim env.

    ``MuJoCoYAMEnv`` implements it; an ``env_factory`` plugin must implement at least this.
    """

    prompt: str | None
    cur_step: int
    model: mujoco.MjModel
    data: mujoco.MjData

    def reset(self, seed: int | None = None, options: dict | None = None, randomize: bool = True): ...

    def step(self, action, render_obs: bool = True): ...

    def apply_action(self, action, physics_steps: int = 1): ...

    def evaluate_task(self): ...

    def get_state(self): ...

    def forget_arm_state(self) -> None: ...

    def close(self) -> None: ...


@dataclass
class ExternalTask:
    """One externally provided scene task (see the module docstring)."""

    env_task: str
    specs: tuple[SimTaskSpec, ...] = ()
    scene_xml: Path | str | Callable[[], Path | str] | None = None
    scene_builder: SceneBuilder | None = None
    builder_options: Mapping[str, Any] = field(default_factory=dict)
    env_factory: Callable[..., Any] | None = None
    evaluators: Mapping[str, type] = field(default_factory=dict)
    randomizer: Callable[[], Any] | None = None   # zero-arg factory of a SceneRandomizer (static-XML tasks)
    physics_defaults: Mapping[str, Any] = field(default_factory=dict)
    assets: tuple[AssetPack, ...] = ()
    description: str = ""

    def __post_init__(self) -> None:
        modes = [m for m in ("scene_xml", "scene_builder", "env_factory") if getattr(self, m) is not None]
        if len(modes) != 1:
            raise ValueError(f"external task {self.env_task!r}: give exactly one of scene_xml / scene_builder / "
                             f"env_factory (got {modes or 'none'})")
        if self.randomizer is not None and self.scene_xml is None:
            raise ValueError(f"external task {self.env_task!r}: randomizer is for scene_xml tasks (a scene_builder "
                             "is its own randomizer)")
        self.specs = tuple(self.specs)
        for spec in self.specs:
            if spec.env_task != self.env_task:
                raise ValueError(f"spec {spec.name!r} has env_task {spec.env_task!r}, expected {self.env_task!r}")

    @property
    def mode(self) -> str:
        return "xml" if self.scene_xml is not None else "builder" if self.scene_builder is not None else "factory"

    def resolve_scene_xml(self) -> Path:
        """The static scene file (fetching assets on first use)."""
        if self.scene_xml is None:
            raise ValueError(f"external task {self.env_task!r} has no static scene_xml")
        xml = self.scene_xml() if callable(self.scene_xml) else self.scene_xml
        path = Path(xml)
        if not path.exists():
            raise FileNotFoundError(f"external task {self.env_task!r}: scene {path} does not exist")
        return path

    def asset_paths(self) -> dict[str, Path]:
        """Fetch every declared pack; ``{pack name: local root}``."""
        return {pack.name: pack.path() for pack in self.assets}


_TASKS: dict[str, ExternalTask] = {}
_SPECS: dict[str, SimTaskSpec] = {}
_EVALUATORS: dict[str, type] = {}
_LOADED = False
_LOCK = threading.RLock()


def register_task(task: ExternalTask, *, replace: bool = False) -> ExternalTask:
    """Register an external task.  Names may not shadow built-in tasks; re-registering needs ``replace=True``."""
    from abc_sim import task_specs as _ts
    from abc_sim.task_eval import registry as _ev
    from abc_sim.task_registry import _SCENE_TASK_LOOKUP

    with _LOCK:
        if task.env_task in _SCENE_TASK_LOOKUP:
            raise ValueError(f"{task.env_task!r} is a built-in scene task")
        if task.env_task in _TASKS and not replace:
            raise ValueError(f"external task {task.env_task!r} is already registered (pass replace=True)")
        keys = {}
        for spec in task.specs:
            for alias in {spec.name, spec.prompt, *spec.aliases}:
                key = _normalize_key(alias)
                if key in _ts._LOOKUP:
                    raise ValueError(f"task name / alias {alias!r} of {spec.name!r} collides with a built-in task")
                other = _SPECS.get(key)
                if other is not None and other.env_task != task.env_task:
                    raise ValueError(f"task name / alias {alias!r} is already used by {other.name!r}")
                keys[key] = spec
        for name in task.evaluators:
            if name in _ev._TASK_EVALUATORS:
                raise ValueError(f"evaluator {name!r} collides with a built-in evaluator")
        if task.env_task in _TASKS:
            unregister_task(task.env_task)
        _TASKS[task.env_task] = task
        _SPECS.update(keys)
        _EVALUATORS.update(task.evaluators)
    return task


def unregister_task(env_task: str) -> None:
    """Remove an external task (tests, hot reload)."""
    with _LOCK:
        task = _TASKS.pop(env_task, None)
        if task is None:
            return
        for key in [k for k, s in _SPECS.items() if s.env_task == env_task]:
            del _SPECS[key]
        for name in task.evaluators:
            _EVALUATORS.pop(name, None)


def load_plugins(*, force: bool = False) -> None:
    """Import every plugin once (entry points + ``ABC_SIM_PLUGINS``).  A failing plugin is logged, not fatal."""
    global _LOADED
    with _LOCK:
        if _LOADED and not force:
            return
        _LOADED = True
        targets: list[tuple[str, Callable[[], Any]]] = []
        try:
            from importlib.metadata import entry_points

            for ep in entry_points(group=ENTRY_POINT_GROUP):
                targets.append((f"entry point {ep.name}", ep.load))
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - broken metadata
            logger.warning("abc_sim plugins: cannot read entry points: %s", exc)
        for item in filter(None, (s.strip() for s in os.environ.get(PLUGINS_ENV, "").split(","))):
            targets.append((item, lambda item=item: _import_target(item)))
        for label, load in targets:
            try:
                obj = load()
                if callable(obj) and not isinstance(obj, type(os)):
                    obj()
            except Exception:
                logger.exception("abc_sim plugins: loading %s failed", label)


def _import_target(target: str) -> Any:
    module, _, attr = target.partition(":")
    mod = importlib.import_module(module)
    return getattr(mod, attr) if attr else mod


# ---------------------------------------------------------------------------------------------------- lookups
def list_external_tasks() -> tuple[ExternalTask, ...]:
    load_plugins()
    return tuple(_TASKS.values())


def list_external_specs() -> tuple[SimTaskSpec, ...]:
    load_plugins()
    return tuple(spec for task in _TASKS.values() for spec in task.specs)


def maybe_get_external_spec(name: str | None) -> SimTaskSpec | None:
    if not name:
        return None
    load_plugins()
    return _SPECS.get(_normalize_key(name))


def maybe_get_external_task(task: str | SimTaskSpec | None) -> ExternalTask | None:
    """The external task behind a scene-task name, task name, alias, prompt or spec."""
    if task is None:
        return None
    load_plugins()
    if isinstance(task, SimTaskSpec):
        return _TASKS.get(task.env_task)
    hit = _TASKS.get(task)
    if hit is not None:
        return hit
    spec = _SPECS.get(_normalize_key(task))
    return None if spec is None else _TASKS.get(spec.env_task)


def get_external_evaluator(name: str) -> type | None:
    load_plugins()
    return _EVALUATORS.get(name)


# ---------------------------------------------------------------------------------------------------- builders
def compile_built_scene(env: Any, built: Any) -> mujoco.MjModel | str:
    """Normalise a builder result: MJCF strings are returned as-is (compiled by the env), specs are compiled."""
    if isinstance(built, str):
        return built
    if isinstance(built, mujoco.MjSpec):
        return built.compile()
    if isinstance(built, mujoco.MjModel):
        return built
    raise TypeError(f"scene_builder returned {type(built).__name__}; expected str, MjSpec or MjModel")


def load_built_scene(env: Any, built: Any) -> None:
    """Swap the env's scene to a builder result."""
    scene = compile_built_scene(env, built)
    if isinstance(scene, str):
        env.reload_from_xml(scene)
    else:
        env.reload_from_model(scene)


def _make_builder_randomizer_cls():
    from abc_sim.randomization.core import RandomizationState, SceneRandomizer

    class BuilderRandomizer(SceneRandomizer):
        """Rebuilds the scene of an external ``scene_builder`` task for every randomized reset."""

        perturbations: ClassVar[list] = []

        def __init__(self, task: ExternalTask | None = None, options: Mapping[str, Any] | None = None) -> None:
            super().__init__()
            self.task = task
            self.options = dict(options or {})
            self.current_seed: int | None = None

        def clone(self) -> BuilderRandomizer:
            return type(self)(self.task, self.options)

        def bind_env(self, env: Any) -> None:   # the scene has no XML file: nothing to read
            self._env_ref = env
            self._scene_xml_transform_options = None

        def build(self, seed: int) -> None:
            env = self._env_ref
            if env is None:
                raise RuntimeError("BuilderRandomizer is not bound to an env")
            if self.current_seed == seed and getattr(env, "model", None) is not None:
                return
            robot_qpos = env._get_reset_robot_qpos() if getattr(env, "model", None) is not None else None
            load_built_scene(env, self.task.scene_builder(int(seed), dict(self.options)))
            mujoco.mj_resetData(env.model, env.data)
            if robot_qpos is not None:
                env._set_robot_qpos(robot_qpos)
            mujoco.mj_forward(env.model, env.data)
            self.current_seed = int(seed)

        def randomize(self, model: Any, data: Any, seed: int | None = None, request: Any | None = None):
            if seed is None:
                seed = int(np.random.default_rng().integers(2**31 - 1))
            self.build(int(seed))
            return RandomizationState(
                seed=int(seed), object_states={}, scale_states={},
                metadata={"scene_builder": {"task": self.task.env_task, "seed": int(seed),
                                            "options": dict(self.options)}},
            )

        def _restore_scene(self, model: Any, data: Any, state: RandomizationState) -> RandomizationState:
            meta = state.metadata.get("scene_builder") or {}
            if meta.get("task", self.task.env_task) != self.task.env_task:
                raise ValueError(f"replay state is for {meta.get('task')!r}, not {self.task.env_task!r}")
            if meta.get("options") is not None:
                self.options = dict(meta["options"])
            self.build(int(meta.get("seed", state.seed)))
            return state

    return BuilderRandomizer


_BUILDER_CLS = None


def builder_randomizer(task: ExternalTask, options: Mapping[str, Any] | None = None):
    global _BUILDER_CLS
    if _BUILDER_CLS is None:
        _BUILDER_CLS = _make_builder_randomizer_cls()
    return _BUILDER_CLS(task, {**task.builder_options, **(options or {})})


__all__ = [
    "ENTRY_POINT_GROUP",
    "PLUGINS_ENV",
    "AssetPack",
    "ExternalTask",
    "SimEnvSurface",
    "builder_randomizer",
    "get_external_evaluator",
    "list_external_specs",
    "list_external_tasks",
    "load_built_scene",
    "load_plugins",
    "maybe_get_external_spec",
    "maybe_get_external_task",
    "register_task",
    "unregister_task",
]
