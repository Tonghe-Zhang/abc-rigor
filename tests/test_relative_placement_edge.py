"""Relative placement: nothing counts short of the reference's edge in the asked direction."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import mujoco
import numpy as np

from abc_sim.task_eval.put_relative import PutRelativeEvaluator
from abc_sim.task_specs import get_task_spec

XML = (
    "<mujoco><worldbody>"
    '<body name="mover" pos="0 0 .02"><freejoint name="mover_joint"/><geom type="sphere" size=".01" mass=".1"/></body>'
    # A 4 x 2 cm (half extents) reference box; the asked direction is +x.
    '<body name="reference" pos="0 0 .02"><freejoint name="reference_joint"/>'
    '<geom type="box" size=".04 .02 .02" mass=".1"/></body>'
    "</worldbody></mujoco>"
)


class RelativePlacementEdgeTest(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)
        self.strict = self.evaluator()
        self.lenient = self.evaluator(require_past_reference_edge=False, along_min_m=0.06)

    def evaluator(self, **options):
        evaluator = PutRelativeEvaluator(
            model=self.model, spec=get_task_spec("put_relative"), **options
        )
        evaluator.configure_from_randomization(
            self.model,
            SimpleNamespace(
                metadata={
                    "mover_object": "mover",
                    "reference_object": "reference",
                    "direction_offset_xy": [1, 0],
                },
                object_states={"mover_joint": {"pos": [0, 0, 0.02]}},
            ),
        )
        return evaluator

    def score(self, x, y, evaluator=None):
        address = int(self.model.jnt_qposadr[self.model.joint("mover_joint").id])
        self.data.qpos[address : address + 3] = [x, y, 0.02]
        result = (evaluator or self.strict).evaluate_qpos_batch(self.data.qpos[None, :])
        return result.scalar_success(), round(float(result.reward[0]), 6)

    def test_nothing_counts_short_of_the_reference_edge(self):
        for x, y in ((0.039, 0), (0.0, 0.07), (-0.02, 0), (-0.10, 0.03)):
            with self.subTest(x=x, y=y):
                self.assertEqual(self.score(x, y), (False, 0.0))

    def test_past_the_edge_within_the_cone_succeeds(self):
        for x, y in ((0.041, 0), (0.10, 0.08), (0.30, 0)):
            with self.subTest(x=x, y=y):
                self.assertEqual(self.score(x, y), (True, 1.0))
        self.assertFalse(
            self.score(0.05, 0.06)[0]
        )  # more sideways than out: outside the cone

    def test_partial_reward_past_the_edge_is_capped_and_linear(self):
        success, reward = self.score(0.10, 0.12)  # 4 cm too far sideways
        self.assertFalse(success)
        self.assertAlmostEqual(reward, 0.5 * (1 - 0.04 / 0.15), places=5)
        self.assertEqual(self.score(0.10, 0.30)[1], 0.0)

    def test_edge_follows_the_reference_rotation(self):
        address = int(self.model.jnt_qposadr[self.model.joint("reference_joint").id])
        self.data.qpos[address + 3 : address + 7] = [
            np.cos(np.pi / 4),
            0,
            0,
            np.sin(np.pi / 4),
        ]  # 90 deg yaw
        self.assertEqual(
            self.score(0.03, 0), (True, 1.0)
        )  # now only 2 cm deep along +x

    def test_lenient_keeps_the_fixed_minimum_and_distance_reward(self):
        self.assertFalse(self.score(0.05, 0, self.lenient)[0])  # below the 6 cm minimum
        self.assertTrue(self.score(0.06, 0, self.lenient)[0])
        self.assertTrue(self.score(0.10, 0.08, self.lenient)[0])
        self.assertAlmostEqual(
            self.score(-0.02, 0, self.lenient)[1], 1 - 0.08 / 0.25, places=5
        )


if __name__ == "__main__":
    unittest.main()
