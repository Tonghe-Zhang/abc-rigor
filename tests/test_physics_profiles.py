"""Contact-level checks for opt-in friction conditions and model/replay lifecycle."""

import copy
import unittest
from unittest import mock

import mujoco
import numpy as np

from abc_sim.config import RobotSystemConfig
from abc_sim.env import MuJoCoYAMEnv
from abc_sim.physics import (
    PhysicsProfile,
    ROUND2_LOW_FRICTION_V1,
    apply_physics_profile,
    resolve_physics_profile,
)
from abc_sim.randomization.core import RandomizationState, SceneRandomizer
from abc_sim.randomization.tasks.ball_tray import BallTrayBalancingRandomizer
from abc_sim.randomization.tasks.blocks import BlocksRandomizer
from abc_sim.randomization.tasks.bottles import WaterBottleRandomizer
from abc_sim.randomization.tasks.chess import ChessRandomizer
from abc_sim.randomization.tasks.dishrack import DishRackRandomizer
from abc_sim.randomization.tasks.inhand_transfer import InHandTransferRandomizer
from abc_sim.randomization.tasks.mugs import MugFlipRandomizer, MugTreeRandomizer
from abc_sim.randomization.tasks.sweep import SweepRandomizer


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


def _friction_arrays(model):
    return {
        "geom": model.geom_friction.copy(),
        "flex": model.flex_friction.copy(),
        "pair": model.pair_friction.copy(),
        "override": model.opt.o_friction.copy(),
    }


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
                once = _friction_arrays(model)
                apply_physics_profile(model, ROUND2_LOW_FRICTION_V1)
                for name, values in _friction_arrays(model).items():
                    np.testing.assert_array_equal(values, once[name])

    def test_effective_flex_contacts_and_native_preservation(self):
        for priority, floor_friction, flex_friction, expected in (
            (0, 0.6, 5, 1.5),
            (1, 0.6, 5, 0.6),
            (0, 0.2, 0.4, 0.4),
        ):
            with self.subTest(floor_priority=priority, flex_friction=flex_friction):
                model = mujoco.MjModel.from_xml_string(f"""
                    <mujoco><worldbody>
                      <geom type="plane" size="1 1 .1" friction="{floor_friction} .02 .002" priority="{priority}"/>
                      <flexcomp name="cloth" type="grid" count="2 2 1"
                                spacing=".1 .1 .1" dim="2" pos="0 0 .005" radius=".01" mass=".1">
                        <contact friction="{flex_friction} .01 .001" condim="6"/>
                      </flexcomp>
                    </worldbody></mujoco>
                """)
                native = _friction_arrays(model)
                apply_physics_profile(model, None)
                for name, values in _friction_arrays(model).items():
                    np.testing.assert_array_equal(values, native[name])
                native_data = mujoco.MjData(model)
                mujoco.mj_forward(model, native_data)
                original_contacts = native_data.contact.friction.copy()
                apply_physics_profile(model, ROUND2_LOW_FRICTION_V1)
                data = mujoco.MjData(model)
                mujoco.mj_forward(model, data)
                self.assertEqual(data.ncon, 4)
                np.testing.assert_allclose(data.contact.friction[:, :2], expected)
                np.testing.assert_array_equal(
                    data.contact.friction[:, 2:], original_contacts[:, 2:]
                )
                np.testing.assert_array_equal(model.geom_friction, native["geom"])
                np.testing.assert_array_equal(
                    model.flex_friction[:, 1:], native["flex"][:, 1:]
                )
                self.assertAlmostEqual(
                    model.flex_friction[0, 0], min(flex_friction, 1.5)
                )
                once = _friction_arrays(model)
                apply_physics_profile(model, ROUND2_LOW_FRICTION_V1)
                for name, values in _friction_arrays(model).items():
                    np.testing.assert_array_equal(values, once[name])

    def test_default_and_profile_metadata_roundtrip(self):
        model = mujoco.MjModel.from_xml_string(_xml(pair=True, override=True))
        friction = _friction_arrays(model)
        apply_physics_profile(model, None)
        for name, values in _friction_arrays(model).items():
            np.testing.assert_array_equal(values, friction[name])
        self.assertEqual(
            PhysicsProfile.from_dict(ROUND2_LOW_FRICTION_V1.to_dict()),
            ROUND2_LOW_FRICTION_V1,
        )
        with self.assertRaises(ValueError):
            resolve_physics_profile("typo")
        for cap in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                PhysicsProfile("invalid", cap)

    def make_env(self, profile=ROUND2_LOW_FRICTION_V1):
        with mock.patch.object(
            MuJoCoYAMEnv,
            "setup_model",
            lambda env: env._bind_model(mujoco.MjModel.from_xml_string(_xml())),
        ):
            env = MuJoCoYAMEnv(
                RobotSystemConfig(), render_cameras=False, physics_profile=profile
            )
        env.get_init_q = lambda: np.zeros(0)
        env._scene_xml_string = _xml()
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

    def test_concrete_replays_reject_before_any_task_or_model_mutation(self):
        for cls in (
            WaterBottleRandomizer,
            MugTreeRandomizer,
            MugFlipRandomizer,
            DishRackRandomizer,
            ChessRandomizer,
            InHandTransferRandomizer,
            SweepRandomizer,
            BallTrayBalancingRandomizer,
            BlocksRandomizer,
        ):
            for recorded, bound in (
                (ROUND2_LOW_FRICTION_V1.to_dict(), None),
                (None, ROUND2_LOW_FRICTION_V1),
                ("missing", ROUND2_LOW_FRICTION_V1),
            ):
                with self.subTest(randomizer=cls.__name__, recorded=recorded):
                    env = self.make_env(bound)
                    randomizer = cls()
                    randomizer.bind_env(env)
                    state = RandomizationState(
                        seed=7,
                        object_states={
                            "ball_joint": {"pos": [0.2, 0, 0.3], "quat": [1, 0, 0, 0]}
                        },
                        scale_states={"ball_joint": 1.2},
                        metadata={}
                        if recorded == "missing"
                        else {"physics_profile": recorded},
                    )
                    qpos = env.data.qpos.copy()
                    body_pos = env.model.body_pos.copy()
                    randomizer_before = dict(randomizer.__dict__)
                    with mock.patch.object(
                        randomizer, "_restore_scene", wraps=randomizer._restore_scene
                    ) as restore:
                        with self.assertRaisesRegex(
                            ValueError, "Replay physics profile"
                        ):
                            randomizer.apply(env.model, env.data, state)
                        restore.assert_not_called()
                    np.testing.assert_array_equal(env.data.qpos, qpos)
                    np.testing.assert_array_equal(env.model.body_pos, body_pos)
                    self.assertEqual(randomizer.__dict__, randomizer_before)

    def test_standalone_bottle_replay_applies_recorded_profile(self):
        model = mujoco.MjModel.from_xml_string(_xml())
        data = mujoco.MjData(model)
        state = RandomizationState(
            seed=1,
            object_states={"ball_joint": {"pos": [0.2, 0, 0.3], "quat": [1, 0, 0, 0]}},
            metadata={
                "bottle_count": 2,
                "physics_profile": ROUND2_LOW_FRICTION_V1.to_dict(),
            },
        )
        WaterBottleRandomizer().apply(model, data, state)
        self.assertEqual(model.geom_friction[:, 0].max(), 1.5)
        np.testing.assert_allclose(data.qpos[:3], [0.2, 0, 0.3])

    def test_legacy_replay_requires_native_bound_physics(self):
        env = self.make_env(None)
        randomizer = SceneRandomizer()
        randomizer.bind_env(env)
        state = RandomizationState(seed=1, object_states={})
        before = copy.deepcopy(state.to_dict())
        randomizer.apply(env.model, env.data, state)
        self.assertEqual(state.to_dict(), before)
        self.assertEqual(env.model.geom_friction[:, 0].max(), 5)

    def test_replay_preflight_cannot_be_overridden(self):
        with self.assertRaisesRegex(TypeError, "must override _restore_scene"):
            type("UncheckedReplay", (SceneRandomizer,), {"apply": lambda *args: None})


if __name__ == "__main__":
    unittest.main()
