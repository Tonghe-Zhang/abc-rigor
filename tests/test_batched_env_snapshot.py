"""CPU regressions for independent per-world simulator snapshots."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import mujoco
import numpy as np
from abc_sim.batched_env import BatchedWarpYAMEnv
from abc_sim.rendering.replay.renderer import PER_EPISODE_MODEL_FIELDS

_SCENE = """
<mujoco>
  <asset>
    <material name="object_color" rgba="1 0 0 1"/>
  </asset>
  <worldbody>
    <body name="region" pos="0.2 0.1 0.01">
      <geom type="box" size="0.1 0.1 0.001" contype="0" conaffinity="0"/>
    </body>
    <body name="object" pos="0 0 1">
      <freejoint name="object_joint"/>
      <geom type="box" size="0.02 0.02 0.02" mass="0.1" material="object_color"/>
    </body>
    <body name="actuated" pos="0.3 0 1">
      <joint name="hinge" type="hinge"/>
      <geom type="box" size="0.02 0.02 0.02" mass="0.1"/>
    </body>
    <body name="mocap_target" pos="0 0 1" mocap="true">
      <geom type="sphere" size="0.01" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <actuator>
    <general joint="hinge" dyntype="filter" dynprm="0.1"/>
  </actuator>
</mujoco>
"""
_DATA_ARRAY_FIELDS = ("qpos", "qvel", "ctrl", "act", "mocap_pos", "mocap_quat")


class SnapshotTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = mujoco.MjModel.from_xml_string(_SCENE)
        self.data = mujoco.MjData(self.model)
        self.env = BatchedWarpYAMEnv.__new__(BatchedWarpYAMEnv)
        self.env.base_env = SimpleNamespace(model=self.model, data=self.data)

    def test_later_world_reset_preserves_previous_snapshot(self) -> None:
        first = self.env._snapshot()
        expected = {name: value.copy() if isinstance(value, np.ndarray) else value for name, value in first.items()}

        self.model.mat_rgba[0] = [0, 0, 1, 1]
        self.model.body_pos[self.model.body("region").id] = [0.4, -0.2, 0.01]
        self.model.body_quat[self.model.body("region").id] = [0, 0, 0, 1]
        for name in _DATA_ARRAY_FIELDS:
            getattr(self.data, name)[:] += 0.25
        self.data.time = 1.0
        second = self.env._snapshot()

        for name in expected:
            with self.subTest(field=name):
                np.testing.assert_array_equal(first[name], expected[name])
        colors = np.stack([first["mat_rgba"], second["mat_rgba"]])
        np.testing.assert_array_equal(colors[:, 0], [[1, 0, 0, 1], [0, 0, 1, 1]])
        self.assertEqual(float(first["time"]), 0.0)
        self.assertEqual(float(second["time"]), 1.0)

    def test_snapshot_fields_have_independent_storage(self) -> None:
        snapshot = self.env._snapshot()
        for name in _DATA_ARRAY_FIELDS:
            with self.subTest(field=name):
                self.assertEqual(snapshot[name].dtype, np.float32)
                self.assertFalse(np.shares_memory(snapshot[name], getattr(self.data, name)))
        for name in PER_EPISODE_MODEL_FIELDS:
            with self.subTest(field=name):
                self.assertEqual(snapshot[name].dtype, np.float32)
                self.assertFalse(np.shares_memory(snapshot[name], getattr(self.model, name)))

    def test_parking_float32_state_does_not_mutate_cpu_buffers(self) -> None:
        # A float32 source must retain the same ownership contract as MjData.
        data = SimpleNamespace(
            **{name: np.array(getattr(self.data, name), dtype=np.float32) for name in _DATA_ARRAY_FIELDS},
            time=0.0,
        )
        self.env.base_env.data = data
        object_adr = self.model.joint("object_joint").qposadr[0]
        hinge_adr = self.model.joint("hinge").qposadr[0]
        data.qpos[object_adr + 2] = -1.0
        data.qpos[hinge_adr] = -0.75
        before = {name: getattr(data, name).copy() for name in _DATA_ARRAY_FIELDS}

        snapshot = self.env._snapshot()

        self.assertAlmostEqual(float(snapshot["qpos"][object_adr + 2]), 0.76)
        self.assertEqual(float(snapshot["qpos"][hinge_adr]), -0.75)
        for name in _DATA_ARRAY_FIELDS:
            with self.subTest(field=name):
                np.testing.assert_array_equal(getattr(data, name), before[name])
                snapshot[name][:] += 2
                np.testing.assert_array_equal(getattr(data, name), before[name])


if __name__ == "__main__":
    unittest.main()
