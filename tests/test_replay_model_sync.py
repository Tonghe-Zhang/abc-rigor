"""CPU regressions for persistent per-world texture assignment updates."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import mujoco
import numpy as np

from abc_sim.batched_env import BatchedWarpYAMEnv, model_signature
from abc_sim.rendering.replay.renderer import (
    PER_EPISODE_MODEL_FIELD_DTYPES,
    WarpReplayRuntime,
)

_SCENE = """
<mujoco>
  <asset>
    <texture name="red" type="2d" builtin="flat" width="2" height="2" rgb1="1 0 0"/>
    <texture name="blue" type="2d" builtin="flat" width="2" height="2" rgb1="0 0 1"/>
    <material name="object" texture="red"/>
  </asset>
  <worldbody>
    <geom type="sphere" size="0.02" material="object"/>
  </worldbody>
</mujoco>
"""
_RGB = mujoco.mjtTextureRole.mjTEXROLE_RGB


class _Array:
    """Host storage with Warp's array-copy interface for synchronization tests."""

    def __init__(self, values, dtype):
        self.values = np.array(values, dtype=dtype, copy=True)
        self.dtype = self.values.dtype
        self.shape = self.values.shape
        self.size = self.values.size


class _HostWarp:
    array = staticmethod(_Array)
    from_numpy = staticmethod(_Array)

    @staticmethod
    def copy(target, source):
        if target.shape != source.shape:
            raise ValueError("copy shape mismatch")
        np.copyto(target.values, source.values, casting="no")


def _runtime(model, nworld):
    runtime = WarpReplayRuntime.__new__(WarpReplayRuntime)
    runtime.mjm = model
    runtime.nworld = nworld
    runtime._wp = _HostWarp()
    runtime.m_warp = SimpleNamespace(
        **{
            name: _Array(np.repeat(getattr(model, name)[None], nworld, axis=0), dtype)
            for name, dtype in PER_EPISODE_MODEL_FIELD_DTYPES.items()
        }
    )
    return runtime


class TextureSyncTest(unittest.TestCase):
    def test_persistent_world_updates_texture_and_restores_untextured_material(self):
        model = mujoco.MjModel.from_xml_string(_SCENE)
        runtime = _runtime(model, 1)
        texture_field = runtime.m_warp.mat_texid
        for texture in (model.texture("blue").id, -1, model.texture("red").id):
            with self.subTest(texture=texture):
                model.mat_texid[0, _RGB] = texture
                runtime.sync_model_from_mujoco()
                self.assertIs(runtime.m_warp.mat_texid, texture_field)
                self.assertEqual(texture_field.dtype, np.int32)
                np.testing.assert_array_equal(
                    texture_field.values, model.mat_texid[None]
                )

    def test_world_snapshots_keep_distinct_ids_when_cpu_model_holds_last_world(self):
        model = mujoco.MjModel.from_xml_string(_SCENE)
        runtime = _runtime(model, 3)
        per_world = np.repeat(model.mat_texid[None], 3, axis=0)
        per_world[:, 0, _RGB] = [model.texture("red").id, -1, model.texture("blue").id]
        expected = per_world.copy()
        model.mat_texid[0, _RGB] = model.texture("blue").id

        runtime.sync_model_from_mujoco({"mat_texid": per_world})

        np.testing.assert_array_equal(runtime.m_warp.mat_texid.values, expected)
        np.testing.assert_array_equal(per_world, expected)
        per_world[:] = -1
        np.testing.assert_array_equal(runtime.m_warp.mat_texid.values, expected)

    def test_batched_reset_uploads_each_world_and_replaces_previous_episode(self):
        model = mujoco.MjModel.from_xml_string(_SCENE)
        data = mujoco.MjData(model)
        texture_choices = [model.texture("red").id, model.texture("blue").id, -1]
        base = SimpleNamespace(
            model=model,
            data=data,
            _task_reset_request=None,
            _last_randomization=None,
            prompt="",
        )

        def reset(*, seed, options, randomize):
            model.mat_texid[0, _RGB] = texture_choices[seed]
            base.prompt = f"texture {texture_choices[seed]}"

        base.reset = reset
        batch = BatchedWarpYAMEnv.__new__(BatchedWarpYAMEnv)
        batch.base_env = base
        batch.num_worlds = 3
        batch.seeds = [0] * 3
        batch.randomization = [None] * 3
        batch.prompts = [""] * 3
        batch._signature = model_signature(model)
        batch._bind_model = Mock()
        batch._runtime = _runtime(model, 3)
        batch._runtime.reset_from_mujoco = batch._runtime.sync_model_from_mujoco
        batch._runtime.load_state_batch = Mock()
        batch._runtime.forward = Mock()
        batch._background = batch._evaluator = None
        batch.get_obs = Mock(return_value={})
        for seeds in ([0, 1, 2], [2, 0, 1]):
            batch.reset(seeds)
            expected = [texture_choices[seed] for seed in seeds]
            np.testing.assert_array_equal(
                batch._runtime.m_warp.mat_texid.values[:, 0, _RGB], expected
            )
            self.assertEqual(
                batch.prompts, [f"texture {texture}" for texture in expected]
            )

    def test_background_sets_preserve_captured_task_textures_in_every_world(self):
        batch = BatchedWarpYAMEnv.__new__(BatchedWarpYAMEnv)
        batch.num_worlds = 3
        batch._background_sets = 2
        assignments = np.full(
            (3, 3, int(mujoco.mjtTextureRole.mjNTEXROLE)), -1, dtype=np.int32
        )
        assignments[:, 2, _RGB] = [2, 3, -1]
        original = assignments.copy()
        cpu_assignments = assignments[1].copy()
        original_cpu = cpu_assignments.copy()
        model = SimpleNamespace(ntex=4, mat_texid=cpu_assignments)
        batch._runtime = SimpleNamespace(mjm=model, set_mat_texid=Mock())
        batch._renderer = SimpleNamespace(update_textures=Mock())
        slots = ("wall", "table")
        draws = [
            (index, {slot: np.full((2, 2, 3), index, dtype=np.uint8) for slot in slots})
            for index in range(2)
        ]
        batch._background = SimpleNamespace(
            draw_sets=Mock(return_value=draws),
            mjwarp_gain=lambda slot: 1.0,
            info_of=lambda draw: {"set": draw},
        )
        with (
            patch(
                "abc_sim.batched_env.background_texture_ids",
                return_value={"wall": 0, "table": 1},
            ),
            patch(
                "abc_sim.batched_env.background_material_ids",
                return_value={"wall": 0, "table": 1},
            ),
        ):
            batch._apply_backgrounds(assignments, background_id=7)

        batch._background.draw_sets.assert_called_once_with(2, 7)
        batch._runtime.set_mat_texid.assert_called_once()
        actual = batch._runtime.set_mat_texid.call_args.args[0]
        expected = original.copy()
        expected[:, :2, _RGB] = [[0, 1], [4, 5], [0, 1]]
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(assignments, original)
        np.testing.assert_array_equal(model.mat_texid, original_cpu)
        self.assertEqual(actual.dtype, np.int32)
        self.assertEqual(batch.background_info, [{"set": 0}, {"set": 1}, {"set": 0}])
        self.assertEqual(
            set(batch._renderer.update_textures.call_args.args[0]), {0, 1, 4, 5}
        )


if __name__ == "__main__":
    unittest.main()
