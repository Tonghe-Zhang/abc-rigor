"""Conveyor and pour latch ``ever_failed`` once nothing left can raise the reward, so rollouts can stop."""

from __future__ import annotations

import unittest

import mujoco
import numpy as np

from abc_sim import make_env


def _adr(model, joint):
    return int(
        model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)]
    )


class ConveyorStopTest(unittest.TestCase):
    def setUp(self):
        self.env = make_env("conveyor_pick", render_cameras=False)
        self.addCleanup(self.env.close)
        self.env.reset(seed=3)

    def removed_and_failed_over(self, seconds, step_s=0.25):
        trace = []
        for _ in range(round(seconds / step_s)):
            self.env.advance_physics(round(step_s / self.env.model.opt.timestep))
            result = self.env.evaluate_task()
            trace.append(
                (
                    int(result.metrics["num_removed_objects"][0]),
                    bool(result.metrics["ever_failed"][0]),
                )
            )
        return trace

    def test_keeps_going_until_every_object_is_gone(self):
        trace = self.removed_and_failed_over(
            12.0
        )  # idle arms: all three ride off the belt
        self.assertTrue(any(removed in (1, 2) for removed, _ in trace))
        self.assertFalse(any(failed for removed, failed in trace if removed < 3))
        self.assertTrue(trace[-1] == (3, True))

    def test_stops_when_the_rest_is_in_the_bin(self):
        evaluator = self.env._task_evaluator
        adr = _adr(self.env.model, "conveyor_object_1_joint")
        x, y = np.mean(evaluator._bin_x_bounds), np.mean(evaluator._bin_y_bounds)
        self.env.data.qpos[adr : adr + 3] = (x, y, evaluator._bin_floor_z + 0.08)
        self.env.data.qvel[:] = 0.0
        mujoco.mj_forward(self.env.model, self.env.data)
        trace = self.removed_and_failed_over(12.0)
        self.assertEqual(
            trace[-1], (2, True)
        )  # one in the bin, two removed: nothing left to pick
        self.assertAlmostEqual(
            float(self.env.evaluate_task().reward[0]), 1.0 / 3.0, places=6
        )


class PourStopTest(unittest.TestCase):
    def setUp(self):
        self.env = make_env("pouring", render_cameras=False)
        self.addCleanup(self.env.close)
        self.env.reset(seed=0)
        self.ev = self.env._task_evaluator
        self.beads = [f"{name}_jnt" for name in self.ev.bead_names]
        self.start = self.env.data.qpos.copy()

    def score(self, **where):
        """Put beads in "target", "table" or "air" (others stay in the source cup) and score positions only."""
        q = self.start.copy()
        target = q[
            _adr(self.env.model, "cup_1_jnt") : _adr(self.env.model, "cup_1_jnt") + 3
        ]
        spots = {
            "target": lambda i: (
                target + [0, 0, self.ev._container_floor_z + 0.01 + 0.004 * i]
            ),
            "table": lambda i: [0.75, -0.45 + 0.03 * i, self.ev._table_z + 0.008],
            "air": lambda i: target + [0, 0.15, 0.15],
        }
        index = 0
        for place, count in where.items():
            for _ in range(count):
                a = _adr(self.env.model, self.beads[index])
                q[a : a + 3] = spots[place](index)
                index += 1
        result = self.ev.evaluate_qpos_batch(q[None, :])
        self.ev.reset()
        return (
            bool(result.success[0]),
            bool(result.metrics["ever_failed"][0]),
            int(result.metrics["num_spilled_beads"][0]),
        )

    def test_start_and_partial_states_keep_going(self):
        self.assertEqual(self.score(), (False, False, 0))  # all in the source cup
        self.assertEqual(
            self.score(target=5, table=2), (False, False, 2)
        )  # 3 still in the source cup
        self.assertEqual(
            self.score(target=8, air=2), (False, False, 0)
        )  # 2 still falling

    def test_everything_spilled_stops(self):
        self.assertEqual(self.score(table=10), (False, True, 10))

    def test_too_few_left_to_succeed_stops(self):
        self.assertEqual(
            self.score(target=8, table=2), (False, True, 2)
        )  # success needs 9

    def test_success_is_not_a_failure(self):
        self.assertEqual(self.score(target=9, table=1), (True, False, 1))


if __name__ == "__main__":
    unittest.main()
