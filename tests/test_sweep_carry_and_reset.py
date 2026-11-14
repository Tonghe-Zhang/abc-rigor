"""Sweep fails when a gripper carries a scrap; resets start the arms at init_q unless asked to keep them."""

from __future__ import annotations

import unittest

import mujoco
import numpy as np

from abc_sim import make_env
from abc_sim.task_eval.sweep import SweepAwayEvaluator


def _free_adr(model, joint):
    return int(model.jnt_qposadr[model.joint(joint).id])


class SweepCarryTest(unittest.TestCase):
    def setUp(self):
        self.env = make_env("sweep", render_cameras=False)
        self.env.reset(seed=1)
        self.ev = self.env._task_evaluator
        self.active = [i for i, on in enumerate(self.ev._active_trash_mask[0]) if on]
        self.ball = self.ev.trash_names[self.active[0]]
        self.adr = _free_adr(self.env.model, f"{self.ball}_jnt")

    def place(self, xyz):
        self.env.data.qpos[self.adr : self.adr + 3] = xyz
        self.env.data.qvel[:] = 0.0
        mujoco.mj_forward(self.env.model, self.env.data)
        return self.env.evaluate_task()

    def finger_point(self):
        geom = next(iter(sorted(self.ev._gripper_geom_ids)))
        return np.asarray(self.env.data.geom_xpos[geom]).copy()

    def bin_point(self):
        # The bin interior is a cylinder about the bin's local y axis; its centre is a safe target.
        adr = self.ev._bin_qpos_adr
        pos = self.env.data.qpos[adr : adr + 3].copy()
        rot = np.empty(9)
        mujoco.mju_quat2Mat(rot, self.env.data.qpos[adr + 3 : adr + 7])
        return pos + rot.reshape(3, 3) @ np.array(
            [0.0, 0.5 * (self.ev._bin_bottom_y + self.ev._bin_top_y), 0.0]
        )

    def test_a_scrap_carried_by_a_gripper_fails_the_episode(self):
        carried = self.place(
            self.finger_point()
        )  # touching a finger, in the air, on no support
        self.assertEqual(int(carried.metrics["num_scraps_carried_by_gripper"][0]), 1)
        self.assertTrue(bool(carried.metrics["ever_failed"][0]))
        self.assertEqual(float(carried.reward[0]), 0.0)
        # Dropping it in the bin afterwards does not undo the failure.
        later = self.place(self.bin_point())
        self.assertIn(self.ball, later.metrics["scraps_in_bin"][0])
        self.assertFalse(bool(later.success[0]))
        self.assertEqual(float(later.reward[0]), 0.0)

    def test_a_scrap_in_the_bin_without_gripper_contact_counts(self):
        result = self.place(self.bin_point())
        self.assertIn(self.ball, result.metrics["scraps_in_bin"][0])
        self.assertEqual(int(result.metrics["num_scraps_carried_by_gripper"][0]), 0)
        self.assertAlmostEqual(
            float(result.reward[0]), 1.0 / len(self.active), places=5
        )

    def test_lenient_flag_keeps_position_only_scoring(self):
        lenient = SweepAwayEvaluator(
            model=self.env.model, spec=self.env._task_spec, fail_on_gripper_carry=False
        )
        lenient.set_active_trash_joints(
            [f"{self.ev.trash_names[i]}_jnt" for i in self.active]
        )
        self.place(self.finger_point())
        result = lenient.evaluate_model_data(self.env.model, self.env.data)
        self.assertFalse(bool(result.metrics["ever_failed"][0]))


class SweepCarryContactRuleTest(unittest.TestCase):
    """Synthetic scene: a table, the bin, one scrap and a finger box, so each contact case is exact."""

    XML = """
    <mujoco><worldbody>
      <geom name="table_plane" type="plane" size="1 1 .1" pos="0 0 0"/>
      <body name="bin_container" pos="1 1 0.1"><freejoint name="bin_joint"/><geom type="box" size=".05 .05 .05"/></body>
      <body name="trash_1" pos="0 0 0.02"><freejoint name="trash_1_jnt"/><geom type="sphere" size="0.02" mass="0.01"/></body>
      <body name="left_link_left_finger" pos="0 0 0.06"><geom name="finger" type="box" size="0.03 0.03 0.0201"/></body>
    </worldbody></mujoco>
    """

    def score(self, ball_z):
        model = mujoco.MjModel.from_xml_string(self.XML)
        data = mujoco.MjData(model)
        evaluator = SweepAwayEvaluator(model=model, spec=None)
        evaluator.set_active_trash_joints(["trash_1_jnt"])
        data.qpos[_free_adr(model, "trash_1_jnt") + 2] = ball_z
        mujoco.mj_forward(model, data)
        contacts = {
            frozenset((model.geom(int(c.geom1)).name, model.geom(int(c.geom2)).name))
            for c in data.contact[: data.ncon]
        }
        return evaluator.evaluate_model_data(model, data), contacts

    def test_gripper_contact_while_resting_on_the_table_is_not_a_carry(self):
        result, contacts = self.score(
            0.02
        )  # touches the table plane and the finger's underside
        self.assertTrue(
            any("finger" in pair for pair in contacts)
            and any("table_plane" in pair for pair in contacts)
        )
        self.assertFalse(bool(result.metrics["ever_failed"][0]))

    def test_gripper_contact_in_the_air_is_a_carry(self):
        result, contacts = self.score(
            0.035
        )  # lifted off the table, still pressed against the finger
        self.assertTrue(any("finger" in pair for pair in contacts))
        self.assertFalse(any("table_plane" in pair for pair in contacts))
        self.assertTrue(bool(result.metrics["ever_failed"][0]))
        self.assertEqual(result.metrics["carried_scraps"][0], ["trash_1"])


class ResetArmPoseTest(unittest.TestCase):
    def test_reset_returns_the_arms_to_init_q_unless_asked_to_keep_them(self):
        env = make_env("conveyor_pick", render_cameras=False)
        env.reset(seed=0)
        init = env.get_state().copy()
        moved = init.copy()
        moved[1] += 0.4  # bend a shoulder joint
        env._set_qpos_from_state(moved)
        mujoco.mj_forward(env.model, env.data)
        env.reset(seed=1)
        np.testing.assert_allclose(env.get_state()[:6], init[:6], atol=1e-6)
        env._set_qpos_from_state(moved)
        mujoco.mj_forward(env.model, env.data)
        env.reset(seed=2, options={"keep_arm_state": True})
        np.testing.assert_allclose(env.get_state()[:6], moved[:6], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
