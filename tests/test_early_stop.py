"""Conveyor latches ``ever_failed`` once nothing left can raise the reward, and the env ends the episode there."""

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


class EnvTerminationTest(unittest.TestCase):
    """RL episodes end on ``ever_failed`` by default; the flags restore the previous behavior."""

    def first_terminated(self, seconds=13.0, **env_kwargs):
        env = make_env("conveyor_pick", render_cameras=False, **env_kwargs)
        self.addCleanup(env.close)
        env.reset(seed=3)
        hold = env.get_state()
        for step in range(
            int(seconds / (env.model.opt.timestep * env._control_decimation))
        ):
            _, _, terminated, _, info = env.step(hold, render_obs=False)
            if terminated:
                return step, info
        return None, info

    def test_conveyor_episode_terminates_once_every_object_is_gone(self):
        step, info = self.first_terminated()
        self.assertIsNotNone(step)
        self.assertTrue(info["task_failed"])
        self.assertEqual(int(info["task_eval"]["num_removed_objects"]), 3)

    def test_terminate_on_failure_false_steps_to_the_end(self):
        step, info = self.first_terminated(terminate_on_failure=False)
        self.assertIsNone(step)
        self.assertTrue(
            info["task_failed"]
        )  # still reported, just not ending the episode

    def test_evaluator_options_restore_the_first_removal_rule(self):
        step, info = self.first_terminated(
            evaluator_options={"fail_when_nothing_left": False}
        )
        self.assertIsNotNone(step)
        self.assertEqual(int(info["task_eval"]["num_removed_objects"]), 1)
        default_step, _ = self.first_terminated()
        self.assertLess(step, default_step)


if __name__ == "__main__":
    unittest.main()
