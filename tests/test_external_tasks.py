"""External task plugins (abc_sim.plugins) and download-on-demand asset packs (abc_sim.asset_source)."""

from __future__ import annotations

import hashlib
import sys
import tarfile
import types
from pathlib import Path

import mujoco
import numpy as np
import pytest

from abc_sim import asset_source, make_env
from abc_sim.asset_source import AssetPack
from abc_sim.plugins import (
    ExternalTask,
    load_plugins,
    maybe_get_external_task,
    register_task,
    unregister_task,
)
from abc_sim.task_eval.base import TaskEvalResult
from abc_sim.task_registry import (
    DEFAULT_SCENE_XML,
    get_task_randomizer,
    list_scene_task_names,
)
from abc_sim.task_specs import SimTaskSpec, get_task_spec, list_task_specs

TOY = "toy_external_box"


def _box_x(seed: int) -> float:
    return 0.55 + 0.01 * (int(seed) % 10)


def build_toy_scene(seed, options):
    spec = mujoco.MjSpec.from_file(str(DEFAULT_SCENE_XML))
    body = spec.worldbody.add_body(name="toy_box", pos=[_box_x(seed), 0.0, float(options.get("z", 0.9))])
    body.add_freejoint()
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.02, 0.02], mass=0.05)
    return spec


class ToyEvaluator:
    """reward = box x; success when the box is above ``threshold``."""

    def __init__(self, *, model, spec, threshold: float = 0.0):
        self.spec = spec
        self.threshold = float(threshold)
        self._adr = int(model.jnt_qposadr[model.body("toy_box").jntadr[0]])

    def reset(self, *, nworld: int = 1) -> None:
        pass

    def evaluate_qpos_batch(self, qpos_batch):
        q = np.atleast_2d(np.asarray(qpos_batch))
        x = q[:, self._adr]
        z = q[:, self._adr + 2]
        return TaskEvalResult(reward=x, success=z > self.threshold, metrics={"box_z": z})


def _spec(env_task: str, name: str, prompt: str, **kw) -> SimTaskSpec:
    return SimTaskSpec(name=name, env_task=env_task, prompt=prompt, evaluator_name=f"{name}_eval", **kw)


@pytest.fixture
def toy_builder_task():
    task = register_task(ExternalTask(
        env_task=TOY,
        specs=(_spec(TOY, "toy_pick_box", "pick up the toy box", aliases=("grab the toy box",)),),
        scene_builder=build_toy_scene,
        builder_options={"z": 0.9},
        evaluators={"toy_pick_box_eval": ToyEvaluator},
    ))
    yield task
    unregister_task(TOY)


def test_registry_lists_and_resolves_external_task(toy_builder_task):
    assert TOY in list_scene_task_names()
    assert any(s.name == "toy_pick_box" for s in list_task_specs())
    assert get_task_spec("grab the toy box").name == "toy_pick_box"
    assert maybe_get_external_task("toy_pick_box") is toy_builder_task
    assert maybe_get_external_task("put_plastic_bottles_in_bin") is None
    assert type(get_task_randomizer(TOY)).__name__ == "BuilderRandomizer"


def test_builder_task_rebuilds_per_seed_and_replays(toy_builder_task):
    env = make_env("toy_pick_box", render_cameras=False, seed=3)
    try:
        assert env.prompt == "pick up the toy box"
        assert env.model.body("toy_box").pos[0] == pytest.approx(_box_x(3))
        for seed in (5, 7):
            env.forget_arm_state()
            env.reset(seed=seed, randomize=True)
            assert env.model.body("toy_box").pos[0] == pytest.approx(_box_x(seed))
            result = env.evaluate_task()
            assert result is not None and result.num_worlds == 1
            assert result.scalar_reward() == pytest.approx(_box_x(seed), abs=1e-3)
            assert result.scalar_success()
        rnd = env._task_randomizer
        state5 = rnd.randomize(env.model, env.data, seed=5)
        assert state5.metadata["scene_builder"] == {"task": TOY, "seed": 5, "options": {"z": 0.9}}
        rnd.randomize(env.model, env.data, seed=8)
        rnd._restore_scene(env.model, env.data, state5)
        assert env.model.body("toy_box").pos[0] == pytest.approx(_box_x(5))
    finally:
        env.close()


def test_builder_task_evaluator_options_and_scene_xml_rejected(toy_builder_task):
    env = make_env("toy_pick_box", render_cameras=False, evaluator_options={"threshold": 5.0})
    try:
        env.forget_arm_state()
        env.reset(seed=1, randomize=True)
        assert not env.evaluate_task().scalar_success()
    finally:
        env.close()
    with pytest.raises(ValueError):
        make_env("toy_pick_box", render_cameras=False, scene_xml=str(DEFAULT_SCENE_XML))


class ConstEvaluator:
    def __init__(self, *, model, spec, value: float = 1.0):
        self.spec, self.value = spec, float(value)

    def reset(self, *, nworld: int = 1) -> None:
        pass

    def evaluate_qpos_batch(self, qpos_batch):
        n = np.atleast_2d(np.asarray(qpos_batch)).shape[0]
        return TaskEvalResult(reward=np.full(n, self.value), success=np.ones(n, bool), metrics={})


def test_xml_task_with_local_asset_pack(tmp_path, monkeypatch):
    pack_dir = tmp_path / "packs" / "toy_scene_v1"
    pack_dir.mkdir(parents=True)
    monkeypatch.setenv("ABC_SIM_ASSET_SOURCE", str(tmp_path / "packs"))
    pack = AssetPack(name="toy_scene", archive="toy_scene_v1")
    seen = {}

    def scene():   # resolved lazily: assets are fetched only when the scene is built
        seen["root"] = pack.path()
        return DEFAULT_SCENE_XML

    register_task(ExternalTask(
        env_task="toy_external_xml",
        specs=(_spec("toy_external_xml", "toy_xml_task", "touch the table"),),
        scene_xml=scene,
        assets=(pack,),
        evaluators={"toy_xml_task_eval": ConstEvaluator},
    ))
    try:
        assert "root" not in seen
        env = make_env("toy_xml_task", render_cameras=False, evaluator_options={"value": 0.25})
        try:
            assert seen["root"] == pack_dir
            env.reset(seed=0)
            assert env.prompt == "touch the table"
            assert env.evaluate_task().scalar_reward() == pytest.approx(0.25)
        finally:
            env.close()
    finally:
        unregister_task("toy_external_xml")


def test_factory_task_delegates():
    calls = {}

    def factory(*, task, **kwargs):
        calls.update(task=task, **kwargs)
        return "custom-env"

    register_task(ExternalTask(
        env_task="toy_external_factory",
        specs=(_spec("toy_external_factory", "toy_factory_task", "do the custom thing"),),
        env_factory=factory,
    ))
    try:
        out = make_env("toy_factory_task", render_cameras=False, builder_options={"k": 1})
        assert out == "custom-env"
        assert calls["task"].name == "toy_factory_task"
        assert calls["builder_options"] == {"k": 1}
    finally:
        unregister_task("toy_external_factory")


def test_registration_rejects_collisions(toy_builder_task):
    with pytest.raises(ValueError):
        ExternalTask(env_task="x", scene_xml="a.xml", scene_builder=build_toy_scene)
    with pytest.raises(ValueError):
        register_task(ExternalTask(env_task="bottles", scene_builder=build_toy_scene))
    with pytest.raises(ValueError):
        register_task(ExternalTask(env_task="toy_other", scene_builder=build_toy_scene,
                                   specs=(_spec("toy_other", "put_plastic_bottles_in_bin", "x"),)))
    with pytest.raises(ValueError):
        register_task(ExternalTask(env_task="toy_other", scene_builder=build_toy_scene,
                                   specs=(_spec("toy_other", "toy_other_task", "pick up the toy box"),)))
    with pytest.raises(ValueError):
        register_task(ExternalTask(env_task=TOY, scene_builder=build_toy_scene))
    register_task(ExternalTask(env_task=TOY, scene_builder=build_toy_scene), replace=True)
    assert maybe_get_external_task("toy_pick_box") is None


def test_plugins_env_var_loading(monkeypatch):
    mod = types.ModuleType("toy_abc_plugin")

    def register():
        register_task(ExternalTask(env_task="toy_env_plugin", scene_builder=build_toy_scene,
                                   specs=(_spec("toy_env_plugin", "toy_env_plugin_task", "plugin prompt"),)))

    mod.register = register
    monkeypatch.setitem(sys.modules, "toy_abc_plugin", mod)
    monkeypatch.setenv("ABC_SIM_PLUGINS", "toy_abc_plugin:register, missing_module_xyz")
    try:
        load_plugins(force=True)   # the missing module is logged, not raised
        assert get_task_spec("plugin prompt").env_task == "toy_env_plugin"
    finally:
        unregister_task("toy_env_plugin")


# ------------------------------------------------------------------------------------------- asset_source
def _make_tar(tmp_path: Path) -> tuple[Path, str]:
    src = tmp_path / "src" / "pack"
    src.mkdir(parents=True)
    (src / "scene.xml").write_text("<mujoco/>")
    arc = tmp_path / "store" / "pack_v1.tar.gz"
    arc.parent.mkdir()
    with tarfile.open(arc, "w:gz") as tf:
        tf.add(src, arcname="pack")
    return arc, hashlib.sha256(arc.read_bytes()).hexdigest()


def test_asset_pack_requires_a_source(monkeypatch):
    monkeypatch.delenv("ABC_SIM_ASSET_SOURCE", raising=False)
    monkeypatch.delenv("ABC_SIM_ASSET_SOURCE_NOPE", raising=False)
    with pytest.raises(RuntimeError, match="ABC_SIM_ASSET_SOURCE"):
        AssetPack(name="nope", archive="nope.tar.gz").location()


def test_asset_pack_local_dir_in_place(tmp_path, monkeypatch):
    (tmp_path / "d" / "inner").mkdir(parents=True)
    monkeypatch.setenv("ABC_SIM_ASSET_SOURCE", str(tmp_path))
    assert AssetPack(name="d", archive="d").path() == tmp_path / "d"
    assert AssetPack(name="d", archive="d", subdir="inner").path() == tmp_path / "d" / "inner"
    monkeypatch.setenv("ABC_SIM_ASSET_SOURCE_MY_PACK", f"file://{tmp_path / 'd'}")
    assert AssetPack(name="my-pack", archive="ignored").path() == tmp_path / "d"


def test_asset_pack_archive_extract_cache_and_checksum(tmp_path, monkeypatch):
    arc, sha = _make_tar(tmp_path)
    monkeypatch.setenv("ABC_SIM_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("ABC_SIM_ASSET_SOURCE", str(arc.parent))
    pack = AssetPack(name="pack", archive="pack_v1.tar.gz", sha256=sha, subdir="pack")
    root = pack.path()
    assert (root / "scene.xml").read_text() == "<mujoco/>"
    assert root.is_relative_to(tmp_path / "cache" / "assets")
    assert (root.parent / ".complete").exists()
    arc.unlink()                      # cached: the source is not read again
    assert pack.path() == root
    arc2, _ = _make_tar(tmp_path / "again")
    bad = AssetPack(name="pack2", archive=str(arc2), sha256="0" * 64)
    with pytest.raises(RuntimeError, match="sha256"):
        bad.path()
    assert not any((tmp_path / "cache" / "assets").glob("pack2-*/.complete"))


def test_asset_pack_unsupported_scheme(tmp_path, monkeypatch):
    monkeypatch.setenv("ABC_SIM_CACHE_DIR", str(tmp_path))
    with pytest.raises(ValueError):
        asset_source.resolve("ftp://example.invalid/pack.tar.gz", name="x")
