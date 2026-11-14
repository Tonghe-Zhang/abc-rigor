"""Sweep staged reward: tools by the handle, scraps brushed onto the dustpan, then dumped into the bin."""

from __future__ import annotations

import unittest

import mujoco
import numpy as np

from abc_sim.task_eval.sweep import SweepAwayEvaluator

R = 0.01  # scrap radius
PAN = np.array([0.3, 0.0, 0.0])  # dustpan body; its pan top is at z = 0.01
BIN_POINT = [1.0, 1.017, 0.1]  # centre of the bin interior (fallback bounds, identity pose)
AWAY = {
    "brush_jnt": [-0.5, 0.5, 0.2],
    "trash_1_jnt": [0.0, -0.3, R],
    "trash_2_jnt": [0.0, 0.3, R],
    "left_finger_jnt": [-0.6, -0.6, 0.3],
    "right_finger_jnt": [-0.6, -0.7, 0.3],
}

XML = """
<mujoco><worldbody>
  <geom name="table_plane" type="plane" size="1 1 .1"/>
  <body name="bin_container" pos="1 1 .1"><freejoint name="bin_joint"/><geom type="box" size=".05 .05 .05"/></body>
  <body name="dustpan" pos=".3 0 0"><freejoint name="dustpan_jnt"/>
    <geom name="pan" type="box" size=".05 .05 .005" pos="0 0 .005"/>
    <geom name="dustpan_handle_grip" type="box" size=".01 .01 .03" pos="0 -.1 .04"/></body>
  <body name="brush" pos="-.5 .5 .2"><freejoint name="brush_jnt"/>
    <geom name="brush_head" type="box" size=".02 .05 .02"/>
    <geom name="brush_handle_grip" type="box" size=".01 .03 .01" pos="0 .15 0"/></body>
  <body name="trash_1" pos="0 -.3 .01"><freejoint name="trash_1_jnt"/><geom type="sphere" size=".01" mass=".01"/></body>
  <body name="trash_2" pos="0 .3 .01"><freejoint name="trash_2_jnt"/><geom type="sphere" size=".01" mass=".01"/></body>
  <body name="left_link_left_finger" pos="-.6 -.6 .3"><freejoint name="left_finger_jnt"/>
    <geom type="box" size=".005 .005 .005"/></body>
  <body name="left_link_right_finger" pos="-.6 -.7 .3"><freejoint name="right_finger_jnt"/>
    <geom type="box" size=".005 .005 .005"/></body>
</worldbody></mujoco>
"""


def on_pan(dx=0.0):
    return [PAN[0] + dx, PAN[1], 0.01 + R - 0.001]


def brushing(scrap):
    """Brush pose whose head touches ``scrap`` from the -x side."""
    return [scrap[0] - R - 0.02 + 0.001, scrap[1], scrap[2]]


def pinching(handle):
    """Left and right finger poses squeezing a 2 cm wide handle from both sides."""
    return [handle[0] - 0.014, handle[1], handle[2]], [handle[0] + 0.014, handle[1], handle[2]]


class SweepStagedRewardTest(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)
        self.use_evaluator()

    def use_evaluator(self, **options):
        self.evaluator = SweepAwayEvaluator(model=self.model, spec=None, **options)
        self.evaluator.set_active_trash_joints(["trash_1_jnt", "trash_2_jnt"])

    def state(self, **poses):
        for joint, xyz in {**AWAY, **poses}.items():
            address = int(self.model.jnt_qposadr[self.model.joint(joint).id])
            self.data.qpos[address : address + 3] = xyz
        mujoco.mj_forward(self.model, self.data)
        return self.evaluator.evaluate_model_data(self.model, self.data)

    @staticmethod
    def score(result):
        return result.scalar_success(), round(float(result.reward[0]), 4)

    def grasp_tools(self):
        brush_handle = np.add(AWAY["brush_jnt"], [0, 0.15, 0])
        left, right = pinching(brush_handle)
        self.assertEqual(self.score(self.state(left_finger_jnt=left, right_finger_jnt=right)), (False, 0.1))
        dustpan_handle = PAN + [0, -0.1, 0.04]
        left, right = pinching(dustpan_handle)
        self.assertEqual(self.score(self.state(left_finger_jnt=left, right_finger_jnt=right)), (False, 0.2))

    def test_one_finger_on_a_handle_is_not_a_grasp(self):
        left, _ = pinching(np.add(AWAY["brush_jnt"], [0, 0.15, 0]))
        result = self.state(left_finger_jnt=left)
        self.assertEqual(self.score(result), (False, 0.0))
        self.assertFalse(result.metrics["tools_grasped"].any())

    def test_full_sweep_reaches_one(self):
        self.grasp_tools()
        for joint, dx in (("trash_1_jnt", -0.02), ("trash_2_jnt", 0.02)):
            scrap = on_pan(dx)
            self.state(**{joint: scrap, "brush_jnt": brushing(scrap)})
        on_dustpan = self.state(trash_1_jnt=on_pan(-0.02), trash_2_jnt=on_pan(0.02))
        self.assertEqual(int(on_dustpan.metrics["num_brushed_scraps_on_dustpan"][0]), 2)
        self.assertEqual(self.score(on_dustpan), (False, round(0.2 + 0.3, 4)))
        one_in = self.state(trash_1_jnt=BIN_POINT, trash_2_jnt=on_pan(0.02))
        self.assertEqual(self.score(one_in), (False, round(0.2 + (0.8 + 0.3) / 2, 4)))
        self.assertEqual(self.score(self.state(trash_1_jnt=BIN_POINT, trash_2_jnt=BIN_POINT)), (True, 1.0))

    def test_a_scrap_pushed_onto_the_dustpan_by_a_finger_earns_nothing(self):
        self.grasp_tools()
        scrap = on_pan()
        result = self.state(trash_1_jnt=scrap, left_finger_jnt=[scrap[0] - R - 0.004, 0, scrap[2]])
        self.assertFalse(bool(result.metrics["ever_failed"][0]))  # resting on the pan: not a carry
        self.assertEqual(self.score(result), (False, 0.2))
        dumped = self.state(trash_1_jnt=BIN_POINT)
        self.assertEqual(int(dumped.metrics["num_brushed_scraps_in_bin"][0]), 0)
        self.assertEqual(self.score(dumped), (False, 0.2))

    def test_a_gripper_touching_a_brushed_scrap_cancels_it_until_brushed_again(self):
        scrap = on_pan()
        self.assertEqual(self.score(self.state(trash_1_jnt=scrap, brush_jnt=brushing(scrap))), (False, 0.15))
        touched = self.state(trash_1_jnt=scrap, left_finger_jnt=[scrap[0], 0, scrap[2] + R + 0.004])
        self.assertEqual(self.score(touched), (False, 0.0))
        self.state(trash_1_jnt=[0.1, 0, R])  # off the dustpan
        self.assertEqual(self.score(self.state(trash_1_jnt=scrap, brush_jnt=brushing(scrap))), (False, 0.15))

    def test_a_scrap_pushed_by_a_brushed_scrap_counts_as_brushed(self):
        front = on_pan()
        back = [front[0] - 2 * R + 0.001, 0, front[2]]
        result = self.state(trash_1_jnt=back, trash_2_jnt=front, brush_jnt=brushing(back))
        self.assertEqual(int(result.metrics["num_brushed_scraps_on_dustpan"][0]), 2)

    def test_a_scrap_reaching_the_bin_without_the_dustpan_blocks_success(self):
        scrap = on_pan()
        self.state(trash_1_jnt=scrap, brush_jnt=brushing(scrap))
        result = self.state(trash_1_jnt=BIN_POINT, trash_2_jnt=BIN_POINT)
        self.assertEqual(int(result.metrics["num_scraps_in_bin"][0]), 2)
        self.assertEqual(self.score(result), (False, 0.4))

    def test_without_contacts_or_with_the_flag_off_the_reward_is_the_bin_fraction(self):
        self.state(trash_1_jnt=BIN_POINT)
        qpos_only = self.evaluator.evaluate_qpos_batch(self.data.qpos[None, :])
        self.assertFalse(qpos_only.metrics["staged_reward"])
        self.assertEqual(self.score(qpos_only), (False, 0.5))
        self.use_evaluator(staged_reward=False)
        self.assertEqual(self.score(self.state(trash_1_jnt=BIN_POINT, trash_2_jnt=BIN_POINT)), (True, 1.0))


if __name__ == "__main__":
    unittest.main()
