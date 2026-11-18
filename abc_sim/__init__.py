"""abc-sim: Standalone MuJoCo simulation for the YAM bimanual robot."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

__version__ = "0.1.0"

if TYPE_CHECKING:
    from abc_sim.env import MuJoCoYAMEnv
    from abc_sim.physics import PhysicsProfile
    from abc_sim.task_specs import SimTaskSpec


def __getattr__(name: str):
    if name == "MuJoCoYAMEnv":
        from abc_sim.env import MuJoCoYAMEnv

        return MuJoCoYAMEnv

    if name in {"get_i2rt_sim_config", "get_i2rt_config", "RobotSystemConfig"}:
        from abc_sim.config import RobotSystemConfig, get_i2rt_config, get_i2rt_sim_config

        return {
            "get_i2rt_sim_config": get_i2rt_sim_config,
            "get_i2rt_config": get_i2rt_config,
            "RobotSystemConfig": RobotSystemConfig,
        }[name]

    if name in {"RandomizationSamplingError", "RandomizationState", "TASK_RANDOMIZERS"}:
        from abc_sim.randomization import (
            RandomizationSamplingError,
            RandomizationState,
            TASK_RANDOMIZERS,
        )

        return {
            "RandomizationSamplingError": RandomizationSamplingError,
            "RandomizationState": RandomizationState,
            "TASK_RANDOMIZERS": TASK_RANDOMIZERS,
        }[name]

    if name in {"SimTaskSpec", "get_task_spec", "list_task_specs", "maybe_get_task_spec"}:
        from abc_sim.task_specs import (
            SimTaskSpec,
            get_task_spec,
            list_task_specs,
            maybe_get_task_spec,
        )

        return {
            "SimTaskSpec": SimTaskSpec,
            "get_task_spec": get_task_spec,
            "list_task_specs": list_task_specs,
            "maybe_get_task_spec": maybe_get_task_spec,
        }[name]

    if name in {"TaskEvalResult", "TaskEvaluator", "make_task_evaluator"}:
        from abc_sim.task_eval import TaskEvalResult, TaskEvaluator, make_task_evaluator

        return {
            "TaskEvalResult": TaskEvalResult,
            "TaskEvaluator": TaskEvaluator,
            "make_task_evaluator": make_task_evaluator,
        }[name]

    if name in {"ConveyorPickRuntime", "TaskRuntime", "make_task_runtime"}:
        from abc_sim.task_runtime import ConveyorPickRuntime, TaskRuntime, make_task_runtime

        return {
            "ConveyorPickRuntime": ConveyorPickRuntime,
            "TaskRuntime": TaskRuntime,
            "make_task_runtime": make_task_runtime,
        }[name]

    if name in {
        "DEFAULT_SCENE_XML",
        "ResolvedTask",
        "SCENE_XMLS",
        "SceneTaskSpec",
        "get_scene_task_spec",
        "get_task_physics_defaults",
        "get_task_randomizer",
        "get_task_scene_xml",
        "list_scene_task_names",
        "list_scene_tasks",
        "maybe_get_scene_task_spec",
        "resolve_env_task_name",
        "resolve_task",
    }:
        from abc_sim.task_registry import (
            DEFAULT_SCENE_XML,
            ResolvedTask,
            SCENE_XMLS,
            SceneTaskSpec,
            get_scene_task_spec,
            get_task_physics_defaults,
            get_task_randomizer,
            get_task_scene_xml,
            list_scene_task_names,
            list_scene_tasks,
            maybe_get_scene_task_spec,
            resolve_env_task_name,
            resolve_task,
        )

        return {
            "DEFAULT_SCENE_XML": DEFAULT_SCENE_XML,
            "ResolvedTask": ResolvedTask,
            "SCENE_XMLS": SCENE_XMLS,
            "SceneTaskSpec": SceneTaskSpec,
            "get_scene_task_spec": get_scene_task_spec,
            "get_task_physics_defaults": get_task_physics_defaults,
            "get_task_randomizer": get_task_randomizer,
            "get_task_scene_xml": get_task_scene_xml,
            "list_scene_task_names": list_scene_task_names,
            "list_scene_tasks": list_scene_tasks,
            "maybe_get_scene_task_spec": maybe_get_scene_task_spec,
            "resolve_env_task_name": resolve_env_task_name,
            "resolve_task": resolve_task,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def background_config(
    *,
    randomize_background: bool = False,
    background_seed: int | None = None,
    background_id: int | None = None,
    background_variation_asset_dir: str | Path | None = None,
    background_resample: str = "episode",
    background_slot_scale: float = 1.0,
):
    """``BackgroundConfig`` for the ``make_env`` background options, or None when disabled."""
    if not randomize_background:
        if background_id is not None or background_seed is not None:
            raise ValueError("background_id / background_seed need randomize_background=True")
        return None
    if not background_variation_asset_dir:
        raise ValueError(
            "randomize_background=True needs background_variation_asset_dir "
            "(a background package directory or s3:// URI)"
        )
    from abc_sim.background import BackgroundConfig

    return BackgroundConfig(
        asset_dir=str(background_variation_asset_dir),
        seed=background_seed,
        background_id=background_id,
        resample=background_resample,
        slot_scale=background_slot_scale,
    )


def make_env(
    task: str | SimTaskSpec = "bottles",
    render_cameras: bool = True,
    camera_backend: str = "mujoco",
    camera_gpu_id: int | None = None,
    prompt: str | None = None,
    chunk_dim: int = 30,
    wrist_fov: float = 58.0,
    scene_xml: str | Path | None = None,
    scene_xml_string: str | None = None,
    scene_xml_transform_options=None,
    enable_task_randomizer: bool = True,
    max_episode_steps: int | None = None,
    terminate_on_success: bool = False,
    physics_profile: str | PhysicsProfile | None = None,
    randomize_background: bool = False,
    background_seed: int | None = None,
    background_id: int | None = None,
    background_variation_asset_dir: str | Path | None = None,
    background_resample: str = "episode",
    background_slot_scale: float = 1.0,
    **kwargs,
) -> "MuJoCoYAMEnv":
    """Create a MuJoCo YAM bimanual environment.

    Args:
        task: Task scene, catalogue name, or a custom ``SimTaskSpec``.
        render_cameras: Whether to render camera images in observations.
        prompt: Task prompt string included in observations. Defaults to the
            resolved task prompt when available.
        chunk_dim: Number of timesteps per action chunk.
        wrist_fov: Vertical field-of-view in degrees for the wrist cameras
            ("left" and "right"). Default 58 matches the real D405 mounting.
        scene_xml: Optional explicit XML file path for standard scene tasks.
        scene_xml_string: Optional in-memory XML override for standard scene
            tasks. Mutually exclusive with ``scene_xml``.
        scene_xml_transform_options: Optional runtime XML transform options for
            model-swapping tasks such as ``inhand_transfer``.
        enable_task_randomizer: Whether to attach and prepare the task
            randomizer during construction. Disable this for fully assembled
            replay/export scenes that need no initial sampler preparation.
            An explicit randomized reset can subsequently initialize sampling.
        max_episode_steps: Optional single-action step horizon. When reached,
            ``step`` returns ``truncated=True``.
        terminate_on_success: Whether ``step`` should return ``terminated=True``
            when the task evaluator reports success.
        physics_profile: Optional named or explicit contact profile, applied to
            every compiled model before simulation and rendering are initialized.
        randomize_background: Opt-in (default off). Replace the walls' and table's
            appearance with real wallpaper x table surface textures from the package in
            ``background_variation_asset_dir``. Visual only: collision geometry, objects
            and task physics are unchanged. See docs/background-variations.md.
        background_seed: Seed of the background stream, independent of the task /
            episode ``seed`` passed to ``reset``. ``None`` uses OS entropy. Give parallel
            workers distinct values (e.g. base + rank).
        background_id: Pin one combination in ``[0, n_combinations)``
            (``wallpaper_index * n_table_types + table_index``, sorted names).
        background_variation_asset_dir: Background package directory, local path or
            ``s3://`` URI (downloaded once per machine). Required with
            ``randomize_background``.
        background_resample: ``"episode"`` (default) draws a new background on every
            randomized reset; ``"env"`` keeps the first draw.
        background_slot_scale: background texture resolution relative to the package's
            (0.5 halves each side; ``make_batched_env`` defaults to 0.5).
        **kwargs: Additional kwargs passed to MuJoCoYAMEnv.

    Returns:
        Configured MuJoCoYAMEnv instance.
    """
    from abc_sim.config import get_i2rt_sim_config
    from abc_sim.env import MuJoCoYAMEnv
    from abc_sim.task_registry import (
        get_task_physics_defaults,
        get_task_randomizer,
        get_task_scene_xml,
        list_scene_task_names,
        resolve_task,
    )

    kwargs["physics_profile"] = physics_profile
    kwargs["background"] = background_config(
        randomize_background=randomize_background,
        background_seed=background_seed,
        background_id=background_id,
        background_variation_asset_dir=background_variation_asset_dir,
        background_resample=background_resample,
        background_slot_scale=background_slot_scale,
    )

    if scene_xml is not None and scene_xml_string is not None:
        raise ValueError("scene_xml and scene_xml_string are mutually exclusive")

    resolved_task = resolve_task(task)
    env_task = resolved_task.env_task or task

    if env_task == "inhand_transfer":
        import numpy as _np

        from abc_sim.randomization import (
            InHandTransferRandomizer,
            _INHAND_CATEGORIES,
            _OBJ_Z,
            _X_MAX,
            _X_MIN,
            _Y_LEFT_MAX,
            _Y_LEFT_MIN,
            _inhand_apply_scene_transforms,
            _inhand_build_xml,
            _inhand_get_variants,
        )

        if scene_xml is not None or scene_xml_string is not None:
            raise ValueError("scene_xml overrides are not supported for task='inhand_transfer'")

        config = get_i2rt_sim_config()
        seed = kwargs.pop("seed", None)
        rng = _np.random.default_rng(seed)

        categories = _INHAND_CATEGORIES
        category = categories[int(rng.integers(0, len(categories)))]
        variants = _inhand_get_variants(category)
        variant_dir = variants[int(rng.integers(0, len(variants)))]
        x = float(rng.uniform(_X_MIN, _X_MAX))
        y = float(rng.uniform(_Y_LEFT_MIN, _Y_LEFT_MAX))
        yaw = float(rng.uniform(-_np.pi, _np.pi))
        xml = _inhand_build_xml(category, variant_dir, x, y, _OBJ_Z, yaw)
        xml = _inhand_apply_scene_transforms(xml, scene_xml_transform_options)

        physics_kwargs = {
            **get_task_physics_defaults(env_task),
            **kwargs,
            "max_episode_steps": max_episode_steps,
            "terminate_on_success": terminate_on_success,
        }
        env = MuJoCoYAMEnv(
            config=config,
            render_cameras=render_cameras,
            camera_backend=camera_backend,
            camera_gpu_id=camera_gpu_id,
            prompt=prompt,
            chunk_dim=chunk_dim,
            wrist_fov=wrist_fov,
            scene_xml=None,
            **physics_kwargs,
        )
        env.reload_from_xml(xml)
        env.set_task(task)
        env._inhand_category = category
        env._inhand_variant = variant_dir.name

        # Bind a randomizer so XdofSimNode can call it on every reset.
        randomizer = InHandTransferRandomizer(
            scene_xml_transform_options=scene_xml_transform_options,
        )
        randomizer.bind_env(env)
        randomizer._rng = rng
        env._task_randomizer = randomizer
        return env

    scene_xml_path = Path(scene_xml) if scene_xml is not None else get_task_scene_xml(env_task)
    if scene_xml_path is None:
        raise ValueError(
            f"Unknown task '{task}'. Available: {list(list_scene_task_names())}"
        )

    tmp_scene_xml_path: Path | None = None
    if scene_xml_string is not None:
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".xml",
            prefix=f".{env_task}_",
            dir=scene_xml_path.parent,
            delete=False,
        ) as handle:
            handle.write(scene_xml_string)
            tmp_scene_xml_path = Path(handle.name)
        scene_xml_path = tmp_scene_xml_path

    config = get_i2rt_sim_config()
    physics_kwargs = {
        **get_task_physics_defaults(env_task),
        **kwargs,
        "max_episode_steps": max_episode_steps,
        "terminate_on_success": terminate_on_success,
    }
    try:
        env = MuJoCoYAMEnv(
            config=config,
            render_cameras=render_cameras,
            camera_backend=camera_backend,
            camera_gpu_id=camera_gpu_id,
            prompt=prompt,
            chunk_dim=chunk_dim,
            wrist_fov=wrist_fov,
            scene_xml=scene_xml_path,
            **physics_kwargs,
        )
    finally:
        if tmp_scene_xml_path is not None:
            tmp_scene_xml_path.unlink(missing_ok=True)
    env.set_task(task)
    env._scene_xml_transform_options = scene_xml_transform_options
    if scene_xml_string is not None:
        env._scene_xml_string = scene_xml_string

    base_randomizer = get_task_randomizer(env_task) if enable_task_randomizer else None
    if base_randomizer is not None:
        randomizer = base_randomizer.clone()
        randomizer.bind_env(env)
        env._task_randomizer = randomizer
        prepare_env = getattr(randomizer, "prepare_env", None)
        if callable(prepare_env):
            prepare_env()

    return env


def make_batched_env(
    task: str,
    *,
    num_worlds: int,
    prompt: str | None = None,
    camera_height: int = 480,
    camera_width: int = 640,
    camera_gpu_id: int | None = None,
    background_pool_size: int = 64,
    background_slot_scale: float = 0.5,
    **kwargs,
):
    """Create ``num_worlds`` copies of a task scene stepped together in MJWarp.

    The CPU env built by :func:`make_env` still owns the task's randomizer and
    resets each world; physics and camera rendering run batched on the GPU.

    With ``randomize_background=True`` every world shows its own background, redrawn on
    every ``reset``: the render context holds ``min(num_worlds, background_pool_size)``
    texture sets (world ``w`` uses set ``w % n_sets``) at ``background_slot_scale``
    resolution (0.5: about 8 MB of GPU texture per set).
    """
    from abc_sim.batched_env import BatchedWarpYAMEnv

    if kwargs.get("randomize_background"):
        # The batched env draws all sets itself on reset; the base env's own per-reset draw is idle.
        kwargs["background_resample"] = "env"
        kwargs["background_slot_scale"] = background_slot_scale
    base_env = make_env(task=task, prompt=prompt, render_cameras=False, **kwargs)
    return BatchedWarpYAMEnv(
        base_env,
        num_worlds=num_worlds,
        camera_height=camera_height,
        camera_width=camera_width,
        gpu_id=camera_gpu_id,
        background_pool_size=background_pool_size,
    )


__all__ = [
    "MuJoCoYAMEnv",
    "make_env",
    "make_batched_env",
    "background_config",
    "get_i2rt_sim_config",
    "get_i2rt_config",
    "RobotSystemConfig",
    "RandomizationSamplingError",
    "RandomizationState",
    "TASK_RANDOMIZERS",
    "SimTaskSpec",
    "get_task_spec",
    "list_task_specs",
    "maybe_get_task_spec",
    "TaskEvalResult",
    "TaskEvaluator",
    "make_task_evaluator",
    "ConveyorPickRuntime",
    "TaskRuntime",
    "make_task_runtime",
    "DEFAULT_SCENE_XML",
    "ResolvedTask",
    "SCENE_XMLS",
    "SceneTaskSpec",
    "get_scene_task_spec",
    "get_task_physics_defaults",
    "get_task_randomizer",
    "get_task_scene_xml",
    "list_scene_task_names",
    "list_scene_tasks",
    "maybe_get_scene_task_spec",
    "resolve_env_task_name",
    "resolve_task",
]
