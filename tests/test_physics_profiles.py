"""Contact-level checks for opt-in friction conditions and model/replay lifecycle."""

import copy
import unittest
from unittest import mock

import mujoco
import numpy as np

from abc_sim.env import MuJoCoYAMEnv
from abc_sim.physics import (
    PhysicsProfile,
    ROUND2_LOW_FRICTION_V1,
    apply_physics_profile,
    resolve_physics_profile,
)
from abc_sim.randomization.core import RandomizationState, SceneRandomizer


def _xml(*, priority=0, pair=False, override=False):
    pair_xml = (
        '<contact><pair geom1="floor" geom2="ball" friction="7 .4 .02 .03 .04"/></contact>'
        if pair
        else ""
    )
    flag = '<flag override="enable"/>' if override else ""
    return f"""
      <mujoco>
        <option o_friction="8 .7 .05 .06 .07">{flag}</option>
        <worldbody>
          <geom name="floor" type="plane" size="1 1 .1" friction="5 .01 .001"/>
          <body pos="0 0 .08"><freejoint name="ball_joint"/>
            <geom name="ball" type="sphere" size=".1" friction=".6 .02 .002" priority="{priority}"/>
          </body>
        </worldbody>{pair_xml}
      </mujoco>
    """


class PhysicsProfileTest(unittest.TestCase):
    def test_effective_contacts_priority_pair_and_override(self):
        for priority, pair, override, expected in (
            (0, False, False, [1.5, 1.5]),
            (1, False, False, [0.6, 0.6]),
            (0, True, False, [1.5, 0.4]),
            (0, False, True, [1.5, 0.7]),
        ):
            with self.subTest(priority=priority, pair=pair, override=override):
                model = mujoco.MjModel.from_xml_string(
                    _xml(priority=priority, pair=pair, override=override)
                )
                original = copy.deepcopy(model)
                original_data = mujoco.MjData(original)
                mujoco.mj_forward(original, original_data)
                apply_physics_profile(model, "round2_low_friction_v1")
                data = mujoco.MjData(model)
                mujoco.mj_forward(model, data)
                self.assertGreater(data.ncon, 0)
                for contact in data.contact:
                    np.testing.assert_allclose(contact.friction[:2], expected)
                np.testing.assert_array_equal(
                    data.contact.friction[:, 2:], original_data.contact.friction[:, 2:]
                )
                np.testing.assert_array_equal(
                    model.geom_priority, original.geom_priority
                )
                np.testing.assert_array_equal(
                    model.geom_friction[:, 1:], original.geom_friction[:, 1:]
                )
                np.testing.assert_array_equal(
                    model.pair_friction[:, 2:], original.pair_friction[:, 2:]
                )
                once = model.geom_friction.copy()
                apply_physics_profile(model, ROUND2_LOW_FRICTION_V1)
                np.testing.assert_array_equal(model.geom_friction, once)

    def test_default_and_profile_metadata_roundtrip(self):
        model = mujoco.MjModel.from_xml_string(_xml())
        friction = model.geom_friction.copy()
        apply_physics_profile(model, None)
        np.testing.assert_array_equal(model.geom_friction, friction)
        self.assertEqual(
            PhysicsProfile.from_dict(ROUND2_LOW_FRICTION_V1.to_dict()),
            ROUND2_LOW_FRICTION_V1,
        )
        with self.assertRaises(ValueError):
            resolve_physics_profile("typo")
        for cap in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                PhysicsProfile("invalid", cap)

    def make_env(self):
        env = MuJoCoYAMEnv.__new__(MuJoCoYAMEnv)
        env.physics_profile = ROUND2_LOW_FRICTION_V1
        env._physics_dt = 0.002
        env._task_runtime = None
        env._task_spec = None
        env._scene_xml_string = _xml()
        env._control_decimation = 17
        env._render_cameras_flag = False
        env._build_index_maps = lambda: None
        env._bind_model(mujoco.MjModel.from_xml_string(_xml()))
        return env

    def test_profile_applies_before_renderer_and_survives_both_reload_paths(self):
        env = self.make_env()
        env._render_cameras_flag = True
        env._camera_backend = "mujoco"
        env._camera_height, env._camera_width = 8, 8

        def renderer(model, **kwargs):
            self.assertLessEqual(model.geom_friction[:, 0].max(), 1.5)
            return mock.Mock()

        with mock.patch("abc_sim.env.mujoco.Renderer", side_effect=renderer) as render:
            source = mujoco.MjModel.from_xml_string(_xml())
            env.reload_from_model(source)
            self.assertEqual(source.geom_friction[:, 0].max(), 5)
            env.reload_from_xml(_xml())
        self.assertEqual(render.call_count, 2)
        self.assertEqual(
            env.capture_state()["physics_profile"], ROUND2_LOW_FRICTION_V1.to_dict()
        )
        self.assertLessEqual(copy.deepcopy(env.model).geom_friction[:, 0].max(), 1.5)

    def test_replay_applies_recorded_profile_and_rejects_mismatched_environment(self):
        model = mujoco.MjModel.from_xml_string(_xml())
        data = mujoco.MjData(model)
        state = RandomizationState(
            seed=0,
            object_states={},
            metadata={"physics_profile": ROUND2_LOW_FRICTION_V1.to_dict()},
        )
        randomizer = SceneRandomizer()
        randomizer.apply(model, data, RandomizationState.from_dict(state.to_dict()))
        self.assertLessEqual(model.geom_friction[:, 0].max(), 1.5)
        env = self.make_env()
        randomizer.bind_env(env)
        randomizer.apply(env.model, env.data, state)
        env.physics_profile = None
        with self.assertRaisesRegex(ValueError, "Replay physics profile"):
            randomizer.apply(env.model, env.data, state)


if __name__ == "__main__":
    unittest.main()
