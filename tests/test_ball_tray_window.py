"""Ball tray, default scoring: hold 10-15 s, then a gentle placement with the ball still on the tray."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import mujoco
import numpy as np

from abc_sim.task_eval.ball_tray import BallTrayBalancingEvaluator
from abc_sim.task_runtime.ball_tray import BallTrayBalancingRuntime
from abc_sim.task_specs import get_task_spec

REST_Z, UP_Z, LOW_Z = (
    0.009,
    0.109,
    0.025,
)  # on the table, 10 cm up, 1.6 cm up (below the 5 cm lift)
HANDLE_Y = 0.16
AWAY = [1.0, 1.0, 0.5]

XML = f"""
<mujoco><worldbody>
  <geom name="table_plane" type="plane" size="1 1 .1"/>
  <body name="ball_tray" pos="0 0 {REST_Z}"><freejoint name="ball_tray_joint"/>
    <geom type="box" size=".15 .15 .01" mass="1"/>
    <geom name="ball_tray_left_handle_collision" type="box" size=".03 .01 .01" pos="0 {HANDLE_Y} .02"/>
    <geom name="ball_tray_right_handle_collision" type="box" size=".03 .01 .01" pos="0 -{HANDLE_Y} .02"/>
    <site name="ball_tray_center_site"/></body>
  <body name="ball_tray_ball" pos="0 0 {REST_Z + 0.03}"><freejoint name="ball_tray_ball_joint"/>
    <geom type="sphere" size=".01" mass=".1"/></body>
  <body name="left_link_left_finger" pos="{AWAY[0]} {AWAY[1]} {AWAY[2]}"><freejoint name="left_finger_joint"/>
    <geom type="box" size=".01 .01 .01" mass=".1"/></body>
  <body name="right_link_left_finger" pos="-1 -1 .5"><freejoint name="right_finger_joint"/>
    <geom type="box" size=".01 .01 .01" mass=".1"/></body>
</worldbody></mujoco>
"""

PROMPT = "keep the ball balanced on the tray for 10 to 15 seconds, then gently place the tray on the table"


class BallTrayWindowTest(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)
        self.use_evaluator()

    def use_evaluator(self, **options):
        self.evaluator = BallTrayBalancingEvaluator(
            model=self.model, spec=get_task_spec("ball_tray_balancing"), **options
        )
        self.evaluator.configure_from_randomization(
            self.model, SimpleNamespace(seed=0, metadata={"hold_duration_s": 7.0})
        )
        self.sample(0, REST_Z)

    def put(self, joint, xyz):
        address = int(self.model.jnt_qposadr[self.model.joint(joint).id])
        self.data.qpos[address : address + 3] = xyz

    def sample(self, t, z, *, held=True, ball_inside=True, touch_ball=False):
        self.put("ball_tray_joint", [0, 0, z])
        ball = [0 if ball_inside else 1, 0, z + 0.03]
        self.put("ball_tray_ball_joint", ball)
        self.put("left_finger_joint", [0, HANDLE_Y, z + 0.02] if held else AWAY)
        self.put(
            "right_finger_joint", [0, -HANDLE_Y, z + 0.02] if held else [-1, -1, 0.5]
        )
        if touch_ball:
            self.put("left_finger_joint", [0, 0, z + 0.045])
        self.data.time = t
        mujoco.mj_forward(self.model, self.data)
        return self.evaluator.evaluate_model_data(self.model, self.data)

    def hold(self, start, end, step=0.5):
        result = None
        for t in np.arange(start, end + 1e-9, step):
            result = self.sample(float(t), UP_Z)
        return result

    def place(self, t, **kwargs):
        """Lower from 10 cm in 0.6 s (about 0.17 m/s) and touch down at ``t``."""
        self.sample(t - 0.1, LOW_Z, **kwargs)
        return self.sample(t, REST_Z, **kwargs)

    @staticmethod
    def score(result):
        return result.scalar_success(), round(float(result.reward[0]), 4)

    def test_hold_in_the_window_then_gentle_placement_succeeds(self):
        self.assertEqual(
            self.score(self.hold(1, 6)), (False, 0.45)
        )  # half of the 10 s minimum
        held = self.hold(6.5, 12)
        self.assertTrue(held.metrics["hold_complete"][0])
        self.assertEqual(self.score(held), (False, 0.9))
        touchdown = self.place(12.6)
        self.assertTrue(touchdown.metrics["settling"][0])
        self.assertEqual(self.score(touchdown), (False, 0.9))
        self.assertFalse(
            self.sample(13.5, REST_Z).scalar_success()
        )  # the ball must stay 1 s
        done = self.sample(13.6, REST_Z, held=False)
        self.assertEqual(self.score(done), (True, 1.0))

    def test_a_placement_within_one_control_step_still_checks_the_handles(self):
        self.hold(1, 12)
        touchdown = self.sample(12.034, REST_Z)  # lifted at the previous sample
        self.assertTrue(touchdown.metrics["handles_held"][0])
        self.assertTrue(touchdown.metrics["settling"][0])

    def test_placing_before_10_s_keeps_half_the_credit_reached_without_success(self):
        self.assertAlmostEqual(
            float(self.hold(1, 8).reward[0]), 0.63, places=4
        )  # 7 s up
        early = self.place(8.6)  # 7.5 s up: 0.675 while up, half of it kept
        self.assertEqual(self.score(early), (False, 0.3375))
        done = self.sample(9.6, REST_Z)
        self.assertTrue(done.metrics["early_placement"][0])
        self.assertEqual(self.score(done), (False, 0.3375))
        self.hold(10, 25)  # the attempt is over: lifting again changes nothing
        self.assertEqual(self.score(self.place(25.6)), (False, 0.3375))

    def test_reward_falls_after_15_s_and_a_late_placement_keeps_it(self):
        self.assertAlmostEqual(
            float(self.hold(1, 16).reward[0]), 0.9, places=4
        )  # 15 s up
        late = self.hold(16.5, 21.5)  # 20.5 s since the hold began
        self.assertAlmostEqual(float(late.reward[0]), 0.9 - 0.4 * 5.5 / 8, places=4)
        reward = float(self.place(22.1).reward[0])  # 21 s up: past the 18 s limit
        self.assertAlmostEqual(reward, 0.9 - 0.4 * 6.1 / 8, places=4)
        done = self.sample(23.1, REST_Z)
        self.assertTrue(done.metrics["late_placement"][0])
        self.assertEqual(self.score(done), (False, round(reward, 4)))
        self.assertEqual(self.score(self.sample(30, REST_Z)), (False, round(reward, 4)))

    def test_success_reward_favours_the_asked_10_to_15_s(self):
        rewards = []
        for up_until in (12, 16, 17.5, 19):  # 11.5, 15.5, 17.0, 18.5 s of air time
            self.use_evaluator()
            self.hold(1, up_until)
            self.place(up_until + 0.6)
            rewards.append(self.score(self.sample(up_until + 1.6, REST_Z)))
        expected = [
            (True, round(1.0 - 0.2 * max(a - 15, 0) / 3, 4)) for a in (11.5, 15.5, 17.0)
        ]
        self.assertEqual(rewards[:3], expected)
        self.assertFalse(rewards[3][0])  # 18.5 s is past the limit
        self.assertLess(rewards[3][1], 0.8)  # and scores below the worst success

    def test_demonstration_like_holds_succeed(self):
        # The demonstrations hold up to about 17.6 s (p95) and touch down up to about 5 s after the final drop.
        for up_until, low_until in ((18, 18.5), (12, 16.5)):
            with self.subTest(up_until=up_until, low_until=low_until):
                self.use_evaluator()
                self.hold(1, up_until)
                for t in np.arange(up_until + 0.5, low_until + 1e-9, 0.5):
                    self.sample(float(t), LOW_Z)
                touchdown = self.sample(low_until + 0.1, REST_Z)
                self.assertTrue(touchdown.metrics["settling"][0])
                done = self.sample(low_until + 1.1, REST_Z)
                self.assertTrue(done.scalar_success())
                self.assertGreaterEqual(float(done.reward[0]), 0.8)

    def test_hovering_just_above_the_table_is_a_late_placement(self):
        self.hold(1, 12)
        for t in np.arange(12.5, 20.01, 0.5):
            self.sample(
                float(t), LOW_Z
            )  # 1.6 cm up: below the lift height, not on the table
        self.sample(20.1, REST_Z)
        done = self.sample(21.1, REST_Z)
        self.assertTrue(done.metrics["late_placement"][0])
        self.assertFalse(done.scalar_success())

    def test_never_placing_decays_to_half(self):
        self.assertEqual(self.score(self.hold(1, 30, step=1.0)), (False, 0.5))

    def test_releasing_the_handles_keeps_at_most_half(self):
        for held_until, expected in ((12, 0.5), (3, 0.1125)):
            with self.subTest(held_until=held_until):
                self.use_evaluator()
                self.hold(1, held_until)
                touchdown = self.place(held_until + 0.6, held=False)
                self.assertTrue(touchdown.metrics["rough_placement"][0])
                self.assertEqual(self.score(touchdown), (False, expected))
                self.assertEqual(self.score(self.sample(30, REST_Z)), (False, expected))

    def test_ball_touched_or_lost_after_the_hold_scores_zero(self):
        for case in (
            "touched while up",
            "lost while settling",
            "lost after an early placement",
        ):
            with self.subTest(case=case):
                self.use_evaluator()
                end = 5 if case == "lost after an early placement" else 12
                self.hold(1, end)
                if case == "touched while up":
                    result = self.sample(end + 0.5, UP_Z, touch_ball=True)
                else:
                    self.assertGreater(self.place(end + 0.6).reward[0], 0.0)
                    result = self.sample(end + 1.0, REST_Z, ball_inside=False)
                self.assertTrue(result.metrics["ball_lost_after_hold"][0])
                self.assertEqual(self.score(result), (False, 0.0))
                self.assertEqual(self.score(self.sample(20, REST_Z)), (False, 0.0))

    def test_touching_the_ball_does_not_count_as_holding(self):
        self.hold(1, 5)
        self.sample(5.5, UP_Z, touch_ball=True)
        self.assertEqual(
            self.score(self.hold(6, 10)), (False, 0.36)
        )  # restarted at 6 s

    def test_flags_turn_off_each_placement_check(self):
        self.use_evaluator(require_handles_at_touchdown=False, settle_s=0.0)
        self.hold(1, 12)
        self.assertEqual(self.score(self.sample(12.1, REST_Z, held=False)), (True, 1.0))

    def test_prompts(self):
        runtime = BallTrayBalancingRuntime()
        runtime.bind(self.model, self.data)
        runtime.after_reset(
            self.model, self.data, randomization=SimpleNamespace(seed=0)
        )
        self.assertEqual(runtime.observation_prompt(), PROMPT)
        runtime.on_task_evaluation(self.sample(1, UP_Z))
        self.assertEqual(runtime.observation_prompt(), PROMPT)
        self.assertEqual(get_task_spec("ball_tray_balancing").prompt, PROMPT)
        self.use_evaluator(hold_window_s=None)
        runtime.on_task_evaluation(self.sample(0, REST_Z))
        self.assertEqual(
            runtime.observation_prompt(),
            "balance the ball in the tray for at least 7 seconds",
        )


if __name__ == "__main__":
    unittest.main()
