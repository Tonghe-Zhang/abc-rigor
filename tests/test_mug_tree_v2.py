"""Solved-contact tests for released and continuously stable mug hanging."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np
from abc_sim.randomization.assets.mugs import _build_mug_scene_xml
from abc_sim.task_eval.mug_tree_v2 import MugOnRackV2Evaluator
from abc_sim.task_specs import get_task_spec


class RectangularHandle:
    """Exact opening for the four-box physical ring used by these tests."""

    def threaded_by(self, endpoints, radius, **kwargs):
        endpoints = endpoints[np.argsort(endpoints[:, 0])]
        if endpoints[0, 0] > -0.009 or endpoints[1, 0] < 0.009:
            return False
        delta = endpoints[1] - endpoints[0]
        fractions = (np.array([-0.008, 0.008]) - endpoints[0, 0]) / delta[0]
        crossings = endpoints[0] + fractions[:, None] * delta
        return bool(np.all(np.abs(crossings[:, 1:]) + radius <= 0.0201))


def ring_body(index):
    return (
        f'<body name="mug_{index}" pos=".3 {0.2 * index} .029">'
        f'<freejoint name="mug_{index}_jnt"/>'
        '<geom name="top_ring_' + str(index) + '" type="box" pos="0 0 .024" size=".008 .028 .004" mass=".025"/>'
        '<geom type="box" pos="0 0 -.024" size=".008 .028 .004" mass=".025"/>'
        '<geom type="box" pos="0 -.024 0" size=".008 .004 .020" mass=".025"/>'
        '<geom type="box" pos="0 .024 0" size=".008 .004 .020" mass=".025"/>'
        "</body>"
    )


class MugHangingV2Tests(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string(
            '<mujoco><option timestep=".002"/>'
            '<default><geom friction=".6 .01 .001" solref=".005 1"/></default>'
            '<worldbody><geom type="plane" size="2 2 .1"/>'
            '<body name="mug_tree">'
            '<geom name="rack_base" type="box" pos="-.07 .1 .45" size=".02 .15 .10"/>'
            '<geom name="peg_1" type="capsule" fromto="-.05 0 .5 .05 0 .5" size=".005"/>'
            '<geom name="peg_2" type="capsule" fromto="-.05 .2 .5 .05 .2 .5" size=".005"/>'
            "</body>"
            '<body name="left_arm" mocap="true" pos="0 1 1">'
            '<body><joint type="slide" axis="1 0 0" stiffness="1000" damping="10"/>'
            '<geom name="left_contact" type="box" size=".006 .006 .006" mass=".1"/></body></body>'
            '<body name="right_arm" mocap="true" pos="0 -1 1">'
            '<body><joint type="slide" axis="1 0 0" stiffness="1000" damping="10"/>'
            '<geom name="right_contact" type="box" size=".006 .006 .006" mass=".1"/></body></body>'
            + ring_body(1)
            + ring_body(2)
            + "</worldbody><contact>"
            '<pair geom1="left_contact" geom2="rack_base"/>'
            '<pair geom1="right_contact" geom2="rack_base"/>'
            "</contact></mujoco>"
        )
        self.data = mujoco.MjData(self.model)
        pegs = [
            SimpleNamespace(
                geom_id=self.model.geom(f"peg_{index + 1}").id,
                endpoints=np.array([[-0.05, 0.2 * index, 0.5], [0.05, 0.2 * index, 0.5]]),
                radius=0.005,
                vertices=np.array([[-0.05, 0.2 * index, 0.5], [0.05, 0.2 * index, 0.5]]),
                edges=np.array([[0, 1]]),
            )
            for index in range(2)
        ]
        with (
            patch("abc_sim.task_eval.mug_tree_v2.measure_rack_pegs", return_value=pegs),
            patch(
                "abc_sim.task_eval.mug_tree_v2.measure_handle_aperture", side_effect=lambda *args: RectangularHandle()
            ),
        ):
            self.evaluator = MugOnRackV2Evaluator(model=self.model, spec=get_task_spec("hang_mug_on_mug_rack"))
        mujoco.mj_forward(self.model, self.data)
        self.evaluator.after_reset(self.model, self.data)

    def result(self):
        return self.evaluator.evaluate_model_data(self.model, self.data)

    def advance(self, ticks):
        for _ in range(ticks):
            mujoco.mj_step(self.model, self.data)
            self.evaluator.after_physics_step(self.model, self.data)
        return self.result()

    def move_mug(self, index, position):
        joint = self.model.joint(f"mug_{index}_jnt")
        address, dof = int(joint.qposadr[0]), int(joint.dofadr[0])
        self.data.qpos[address : address + 3] = position
        self.data.qpos[address + 3 : address + 7] = [1, 0, 0, 0]
        self.data.qvel[dof : dof + 6] = 0
        mujoco.mj_forward(self.model, self.data)

    def hang(self, index=1, peg=1):
        self.move_mug(index, [0, 0.2 * (peg - 1), 0.486])
        result = self.advance(600)
        self.assertTrue(result.metrics["mug_threaded"][0, index - 1])
        self.assertGreater(result.metrics["mug_peg_support_n"][0, index - 1, peg - 1], 0.9)
        return result

    def restart_observer(self):
        self.evaluator.reset()
        self.evaluator.after_reset(self.model, self.data)

    def test_reset_and_idle_start_at_zero(self):
        self.assertEqual(self.result().scalar_reward(), 0)
        self.assertEqual(self.advance(600).scalar_reward(), 0)
        self.assertFalse(self.result().metrics["mug_threaded"].any())

    def test_half_second_dwell_and_independent_partial_reward(self):
        self.assertEqual(self.hang().scalar_reward(), 0.5)
        self.restart_observer()
        self.assertEqual(self.result().scalar_reward(), 0)
        self.assertEqual(self.advance(249).scalar_reward(), 0)
        self.assertEqual(self.advance(3).scalar_reward(), 0.5)
        self.assertEqual(self.hang(2, 2).scalar_reward(), 1)
        self.assertTrue(self.result().scalar_success())
        self.move_mug(1, [0.3, 0.2, 0.029])
        result = self.advance(1)
        self.assertEqual(result.scalar_reward(), 0.5)
        self.assertFalse(result.scalar_success())
        self.assertTrue(result.metrics["ever_success"][0])

    def test_resting_on_tree_without_threading_earns_zero(self):
        self.move_mug(1, [-0.07, 0.1, 0.579])
        result = self.advance(600)
        self.assertEqual(result.scalar_reward(), 0)
        self.assertTrue(result.metrics["mug_stable"][0, 0])
        self.assertFalse(result.metrics["mug_threaded"][0, 0])

    def test_tip_not_through_far_side_does_not_qualify(self):
        self.move_mug(1, [0.051, 0, 0.486])
        result = self.advance(1)
        self.assertFalse(result.metrics["mug_threaded"][0, 0])
        self.assertEqual(result.scalar_reward(), 0)

    def test_threaded_airborne_mug_without_peg_support_earns_zero(self):
        self.move_mug(1, [0, 0, 0.5])
        body = self.model.body("mug_1").id
        self.data.xfrc_applied[body, 2] = self.model.body_subtreemass[body] * 9.81
        result = self.advance(600)
        self.assertTrue(result.metrics["mug_threaded"][0, 0])
        self.assertTrue(result.metrics["mug_stable"][0, 0])
        np.testing.assert_array_equal(result.metrics["mug_peg_support_n"][0, 0], 0)
        self.assertEqual(result.scalar_reward(), 0)

    def test_either_gripper_touching_mug_restarts_dwell(self):
        self.hang()
        self.hang(2, 2)
        for arm in range(2):
            with self.subTest(arm=arm):
                self.data.mocap_pos[arm] = [0, -0.0335, self.data.xpos[self.model.body("mug_1").id, 2]]
                result = self.advance(1)
                self.assertTrue(result.metrics["mug_touching_robot"][0, 0])
                self.assertEqual(result.scalar_reward(), 0.5)
                self.assertEqual(result.metrics["mug_settling_seconds"][0, 0], 0)
                self.data.mocap_pos[arm] = [0, 1 if arm == 0 else -1, 1]
                self.move_mug(1, [0, 0, 0.486])
                self.assertEqual(self.advance(200).scalar_reward(), 0.5)
                self.assertEqual(self.advance(400).scalar_reward(), 1)

    def test_either_gripper_touching_rack_removes_all_credit(self):
        self.hang()
        self.hang(2, 2)
        for arm in range(2):
            with self.subTest(arm=arm):
                self.data.mocap_pos[arm] = [-0.095, 0.1, 0.45]
                result = self.advance(1)
                self.assertTrue(result.metrics["rack_touching_robot"][0])
                self.assertEqual(result.scalar_reward(), 0)
                np.testing.assert_array_equal(result.metrics["mug_settling_seconds"], 0)
                self.data.mocap_pos[arm] = [0, 1 if arm == 0 else -1, 1]
                self.assertEqual(self.advance(249).scalar_reward(), 0)
                self.assertEqual(self.advance(4).scalar_reward(), 1)

    def test_support_must_come_from_the_threaded_peg(self):
        self.hang()
        self.evaluator._apertures[0].threaded_by = lambda endpoints, radius, **kwargs: abs(endpoints[0, 1] - 0.2) < 0.01
        result = self.advance(300)
        self.assertGreater(result.metrics["mug_peg_support_n"][0, 0, 0], 0.9)
        self.assertFalse(result.metrics["mug_threaded_pegs"][0, 0, 0])
        self.assertTrue(result.metrics["mug_threaded_pegs"][0, 0, 1])
        self.assertEqual(result.scalar_reward(), 0)

    def test_motion_and_deep_penetration_reset_settling(self):
        self.hang()
        dof = int(self.model.joint("mug_1_jnt").dofadr[0])
        self.data.qvel[dof] = 0.1
        result = self.advance(1)
        self.assertFalse(result.metrics["mug_stable"][0, 0])
        self.assertEqual(result.scalar_reward(), 0)
        self.move_mug(1, [0, 0, 0.480])
        result = self.advance(1)
        self.assertGreater(result.metrics["mug_rack_penetration_m"][0, 0], self.evaluator.max_penetration_m)
        self.assertEqual(result.scalar_reward(), 0)

    def test_switching_pegs_restarts_qualification(self):
        self.hang()
        self.restart_observer()
        self.advance(150)
        self.assertGreater(self.result().metrics["mug_settling_seconds"][0, 0], 0.29)
        self.move_mug(1, [0, 0.2, 0.486])
        self.assertEqual(self.advance(150).scalar_reward(), 0)
        self.assertEqual(self.advance(450).scalar_reward(), 0.5)

    def test_reads_do_not_advance_history_and_reset_clears_success(self):
        self.hang()
        result = self.result()
        result.reward[0] = -1
        result.metrics["mug_peg_settling_seconds"][:] = 100
        self.assertEqual(self.result().scalar_reward(), 0.5)
        self.assertLess(self.result().metrics["mug_settling_seconds"][0, 0], 2)
        self.restart_observer()
        self.assertEqual(self.result().scalar_reward(), 0)
        self.assertFalse(self.result().metrics["ever_success"][0])
        self.assertEqual(self.result().metrics["max_mugs_hanging_so_far"][0], 0)

    def test_reset_rack_translation_and_rotation_follow_live_body_frame(self):
        angle = 0.3
        quat = np.array([np.cos(angle / 2), 0, 0, np.sin(angle / 2)])
        tree = self.model.body("mug_tree").id
        self.model.body_pos[tree] = [0.2, 0.1, 0]
        self.model.body_quat[tree] = quat
        joint = self.model.joint("mug_1_jnt")
        address = int(joint.qposadr[0])
        self.move_mug(1, [0.2, 0.1, 0.486])
        self.data.qpos[address + 3 : address + 7] = quat
        mujoco.mj_forward(self.model, self.data)
        self.restart_observer()
        result = self.advance(600)
        self.assertEqual(result.scalar_reward(), 0.5)
        self.assertTrue(result.metrics["mug_threaded"][0, 0])

    def test_requires_native_history_and_exactly_one_observation_per_tick(self):
        with self.assertRaisesRegex(NotImplementedError, "native"):
            self.evaluator.reset(nworld=2)
        with self.assertRaisesRegex(NotImplementedError, "contacts"):
            self.evaluator.evaluate_qpos_batch(self.data.qpos[None])
        with self.assertRaisesRegex(ValueError, "every physics tick"):
            self.evaluator.after_physics_step(self.model, self.data)
        mujoco.mj_step(self.model, self.data, nstep=2)
        with self.assertRaisesRegex(ValueError, "every physics tick"):
            self.evaluator.after_physics_step(self.model, self.data)
        with self.assertRaisesRegex(RuntimeError, "Reset"):
            self.evaluator.evaluate_model_data(self.model, mujoco.MjData(self.model))


class MugHangingAssetTests(unittest.TestCase):
    def test_real_mug_settles_on_a_fully_threaded_branch_for_full_credit(self):
        xml = _build_mug_scene_xml(
            task_name="mug_tree",
            mug_variant="mug_0",
            mug_variants=["mug_0"],
            scale_states={},
            base_scene_xml=None,
            base_scene_dir=None,
        )
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        data.ctrl[:] = data.qpos[model.jnt_qposadr[model.actuator_trnid[:, 0]]]
        address = int(model.joint("mug_1_jnt").qposadr[0])
        # A released pose with the first rack branch inside mug_0's handle.
        data.qpos[address : address + 7] = [
            0.6969733592468916,
            -0.007138197059290641,
            1.0044640461289187,
            0.6986953026857661,
            -0.6990868890789936,
            -0.10619567560807149,
            -0.10874223652677817,
        ]
        mujoco.mj_forward(model, data)
        evaluator = MugOnRackV2Evaluator(model=model, spec=get_task_spec("hang_mug_on_mug_rack"))
        evaluator.after_reset(model, data)
        self.assertEqual(evaluator.evaluate_model_data(model, data).scalar_reward(), 0)
        for _ in range(round(3.0 / model.opt.timestep)):
            mujoco.mj_step(model, data)
            evaluator.after_physics_step(model, data)
            if data.time < 0.49:
                self.assertEqual(evaluator.evaluate_model_data(model, data).scalar_reward(), 0)
        result = evaluator.evaluate_model_data(model, data)
        self.assertEqual(result.scalar_reward(), 1)
        self.assertTrue(result.scalar_success())
        self.assertTrue(result.metrics["mug_threaded"][0, 0])
        self.assertTrue(result.metrics["mug_released"][0, 0])
        self.assertGreaterEqual(result.metrics["mug_settling_seconds"][0, 0], 0.5)
        weight = model.body_subtreemass[model.body("mug_1").id] * np.linalg.norm(model.opt.gravity)
        self.assertAlmostEqual(result.metrics["mug_peg_support_n"][0, 0].sum(), weight, delta=0.01)
        self.assertLess(result.metrics["mug_rack_penetration_m"][0, 0], 0.0001)


if __name__ == "__main__":
    unittest.main()
