"""Ball tray, default scoring: hold 10-15 s, then a gentle placement with the ball still on the tray."""

from __future__ import annotations

import json
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

    def sample(
        self, t, z, *, held=True, ball_inside=True, touch_ball=False, ball_height=0.019
    ):
        self.put("ball_tray_joint", [0, 0, z])
        ball = [0 if ball_inside else 1, 0, z + ball_height]
        self.put("ball_tray_ball_joint", ball)
        self.put("left_finger_joint", [0, HANDLE_Y, z + 0.02] if held else AWAY)
        self.put(
            "right_finger_joint", [0, -HANDLE_Y, z + 0.02] if held else [-1, -1, 0.5]
        )
        if touch_ball:
            self.put("left_finger_joint", [0, 0, z + 0.029])
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

    def make_runtime(self):
        runtime = BallTrayBalancingRuntime()
        runtime.bind(self.model, self.data)
        runtime.after_reset(
            self.model, self.data, randomization=SimpleNamespace(seed=0)
        )
        runtime.on_task_evaluation(self.sample(0, REST_Z))
        runtime.after_step(self.model, self.data)
        return runtime

    def runtime_sample(self, runtime, t, z, **kwargs):
        runtime.before_step(self.model, self.data)
        result = self.sample(t, z, **kwargs)
        runtime.on_task_evaluation(result)
        runtime.after_step(self.model, self.data)
        return result

    def test_lift_is_one_time_and_hold_credit_survives_interruptions(self):
        self.assertEqual(self.score(self.sample(1, UP_Z)), (False, 0.1))
        self.assertEqual(self.score(self.sample(6, UP_Z)), (False, 0.5))
        for t, kwargs in ((6.5, {"ball_inside": False}), (7, {}), (10, {})):
            result = self.sample(t, UP_Z, **kwargs)
            self.assertEqual(self.score(result), (False, 0.5))
        self.assertEqual(result.metrics["continuous_hold_s"][0], 3)
        self.assertEqual(result.metrics["best_continuous_hold_s"][0], 5)
        self.assertEqual(result.metrics["lift_credit"][0], 0.1)

    def test_disjoint_holds_do_not_sum_and_current_hold_qualifies_success(self):
        self.hold(1, 11)
        self.sample(11.1, UP_Z, ball_inside=False)
        self.sample(11.2, UP_Z)
        result = self.sample(12, UP_Z)
        self.assertAlmostEqual(result.reward[0], 0.9)
        self.assertFalse(result.metrics["hold_complete"][0])
        self.assertTrue(result.metrics["perturbations_allowed"][0])
        self.place(12.6)
        done = self.sample(13.6, REST_Z)
        self.assertEqual(self.score(done), (False, 0.9))
        self.assertEqual(done.metrics["placement_credit"][0], 0)

    def test_dip_breaks_hold_and_cannot_farm_repeated_lifts(self):
        self.hold(1, 6)
        self.sample(6.1, LOW_Z)
        self.sample(6.2, UP_Z)
        result = self.sample(10.2, UP_Z)
        self.assertEqual(self.score(result), (False, 0.5))
        self.assertAlmostEqual(result.metrics["continuous_hold_s"][0], 4)
        self.assertEqual(result.metrics["best_continuous_hold_s"][0], 5)

    def test_target_hold_and_gentle_continuous_settle_succeed(self):
        self.assertEqual(self.score(self.hold(1, 11)), (False, 0.9))
        self.assertEqual(self.score(self.place(11.6)), (False, 0.9))
        self.assertFalse(self.sample(12.599, REST_Z).scalar_success())
        done = self.sample(12.6, REST_Z, held=False)
        self.assertEqual(self.score(done), (True, 1.0))
        self.assertTrue(done.metrics["placement_finalized"][0])
        self.assertEqual(
            self.score(self.sample(30, UP_Z, ball_inside=False)), (True, 1.0)
        )

    def test_early_safe_park_keeps_credit_and_later_qualified_hold_starts_clock(self):
        self.hold(1, 5)
        self.place(5.6)
        parked = self.sample(6.6, REST_Z)
        self.assertEqual(self.score(parked), (False, 0.42))
        self.assertTrue(parked.metrics["early_placement"][0])
        self.assertFalse(parked.metrics["placement_finalized"][0])
        self.hold(7, 17)
        self.place(17.6)
        done = self.sample(18.6, REST_Z)
        self.assertTrue(done.scalar_success())
        self.assertAlmostEqual(done.metrics["overtime_s"][0], 0.0)
        self.assertAlmostEqual(done.reward[0], 1.0, places=6)

    def test_three_short_parked_cycles_never_sum_hold_credit(self):
        for start in (1, 7, 13):
            self.hold(start, start + 4)
            self.place(start + 4.6)
            result = self.sample(start + 5.6, REST_Z)
            self.assertAlmostEqual(result.metrics["hold_credit"][0], 0.32)
            self.assertEqual(result.metrics["lift_credit"][0], 0.1)
            self.assertFalse(result.scalar_success())
        self.assertAlmostEqual(result.metrics["overtime_s"][0], 0.0)

    def test_qualified_overtime_persists_through_drop_dip_and_recovery(self):
        self.hold(1, 11)
        self.assertEqual(self.score(self.sample(17, UP_Z)), (False, 0.88))
        for t, z, kwargs in (
            (18, UP_Z, {"ball_inside": False}),
            (19, LOW_Z, {}),
            (20, UP_Z, {}),
        ):
            result = self.sample(t, z, **kwargs)
            self.assertAlmostEqual(result.metrics["overtime_s"][0], t - 16)
            self.assertAlmostEqual(result.reward[0], 0.9 - 0.02 * (t - 16), places=6)
        self.assertFalse(result.metrics["hold_complete"][0])

    def test_partial_credit_is_preserved_and_qualified_overtime_caps(self):
        self.sample(1, UP_Z)
        self.sample(2, UP_Z)
        self.sample(2.1, UP_Z, ball_inside=False)
        result = self.sample(40, UP_Z, ball_inside=False)
        self.assertEqual(self.score(result), (False, 0.18))
        self.assertAlmostEqual(result.metrics["overtime_penalty"][0], 0.0)
        self.use_evaluator()
        self.hold(1, 31, step=1)
        self.place(31.6)
        self.assertEqual(self.score(self.sample(32.6, REST_Z)), (True, 0.8))

    def test_first_qualification_after_interruption_gets_full_placement_window(self):
        self.hold(1, 5)
        self.sample(5.1, UP_Z, ball_inside=False)
        self.hold(8, 18)
        self.assertEqual(self.score(self.sample(23, UP_Z)), (False, 0.9))
        self.assertEqual(self.score(self.sample(23.5, UP_Z)), (False, 0.89))
        self.assertEqual(
            self.score(self.sample(24, UP_Z, ball_inside=False)), (False, 0.88)
        )

    def test_late_first_qualification_does_not_instantly_deduct_partial_credit(self):
        self.hold(1, 3)
        self.sample(3.1, UP_Z, ball_inside=False)
        self.assertEqual(self.score(self.sample(30, REST_Z)), (False, 0.26))
        self.hold(31, 41)
        self.assertEqual(self.score(self.sample(46, UP_Z)), (False, 0.9))
        self.assertEqual(self.score(self.sample(47, UP_Z)), (False, 0.88))

    def test_second_qualified_hold_cannot_restart_overtime_clock(self):
        self.hold(1, 11)
        self.sample(17, UP_Z)
        self.sample(18, UP_Z, ball_inside=False)
        self.hold(20, 30)
        result = self.sample(30, UP_Z)
        self.assertEqual(self.score(result), (False, 0.7))
        self.assertAlmostEqual(result.metrics["overtime_s"][0], 14)

    def test_no_lift_never_receives_credit_or_penalty(self):
        result = self.sample(300, REST_Z)
        self.assertEqual(self.score(result), (False, 0.0))
        self.assertEqual(result.metrics["overtime_penalty"][0], 0.0)

    def test_sparse_observation_uses_exact_qualification_time(self):
        self.hold(1, 5)
        self.sample(5.1, UP_Z, ball_inside=False)
        self.sample(7, UP_Z)
        result = self.sample(23, UP_Z)
        self.assertEqual(self.score(result), (False, 0.88))
        self.assertEqual(result.metrics["overtime_s"][0], 1.0)

    def test_custom_window_grace_is_upper_minus_lower_bound(self):
        self.use_evaluator(hold_window_s=(4, 6))
        self.sample(1, UP_Z)
        self.sample(5, UP_Z)
        result = self.sample(8, UP_Z)
        self.assertEqual(self.score(result), (False, 0.88))
        self.assertEqual(result.metrics["overtime_s"][0], 1.0)

    def test_valid_settling_time_is_excluded_from_overtime(self):
        self.hold(1, 15)
        self.place(15.9)
        done = self.sample(16.9, REST_Z)
        self.assertEqual(self.score(done), (True, 1.0))
        self.assertEqual(done.metrics["overtime_s"][0], 0)

    def test_failed_settle_resumes_clock_and_restarts_continuous_interval(self):
        self.hold(1, 16)
        self.place(16.6)
        gap = self.sample(17, LOW_Z, held=False)
        self.assertAlmostEqual(gap.metrics["overtime_s"][0], 1)
        self.sample(17.1, REST_Z, held=False)
        self.assertFalse(self.sample(18.099, REST_Z, held=False).scalar_success())
        done = self.sample(18.1, REST_Z, held=False)
        self.assertTrue(done.scalar_success())
        self.assertAlmostEqual(done.metrics["overtime_s"][0], 1.1)
        self.assertAlmostEqual(done.reward[0], 0.978, places=6)

    def test_full_relift_invalidates_settle_qualification(self):
        self.hold(1, 11)
        self.place(11.6)
        self.sample(12, UP_Z)
        self.place(12.6)
        done = self.sample(13.6, REST_Z)
        self.assertEqual(self.score(done), (False, 0.9))

    def test_ball_loss_during_settle_preserves_credit_without_bonus(self):
        self.hold(1, 11)
        self.place(11.6)
        self.sample(12, REST_Z, ball_inside=False)
        self.sample(12.1, REST_Z)
        done = self.sample(13.1, REST_Z)
        self.assertEqual(self.score(done), (False, 0.9))
        self.assertTrue(done.metrics["ball_lost_after_hold"][0])

    def test_grip_loss_before_touchdown_invalidates_placement(self):
        self.hold(1, 11)
        self.sample(11.2, LOW_Z, held=False)
        self.sample(11.4, REST_Z)
        self.assertEqual(self.score(self.sample(12.4, REST_Z)), (False, 0.9))

    def test_lowering_must_reach_table_within_five_seconds(self):
        self.hold(1, 11)
        self.sample(11.1, LOW_Z)
        self.sample(16.2, LOW_Z)
        self.sample(16.3, REST_Z)
        done = self.sample(17.3, REST_Z)
        self.assertFalse(done.scalar_success())
        self.assertEqual(done.metrics["placement_credit"][0], 0)
        self.assertAlmostEqual(done.reward[0], 0.894, places=6)

    def test_rough_contact_does_not_finalize_or_pause_clock(self):
        self.hold(1, 11)
        result = self.place(17, held=False)
        self.assertTrue(result.metrics["rough_placement"][0])
        self.assertFalse(result.metrics["settling"][0])
        self.assertEqual(self.score(self.sample(18, REST_Z, held=False)), (False, 0.86))

    def test_policy_done_during_settle_has_only_earned_credit(self):
        self.hold(1, 11)
        result = self.place(11.6)
        info = result.to_info(squeeze=True)
        self.assertFalse(info["success"])
        self.assertAlmostEqual(info["reward"], 0.9)
        self.assertEqual(info["placement_credit"], 0)
        self.assertFalse(info["placement_finalized"])

    def test_levitated_ball_is_not_supported(self):
        result = self.sample(1, UP_Z, ball_height=1)
        self.assertTrue(result.metrics["inside_tray"][0])
        self.assertFalse(result.metrics["ball_tray_contact"][0])
        self.assertEqual(self.score(self.sample(20, UP_Z, ball_height=1)), (False, 0.0))

    def test_table_supported_tilt_is_not_an_airborne_hold(self):
        address = int(self.model.jnt_qposadr[self.model.joint("ball_tray_joint").id])
        self.data.qpos[address + 3 : address + 7] = [
            np.cos(np.pi / 8),
            0,
            np.sin(np.pi / 8),
            0,
        ]
        result = self.sample(1, REST_Z + 0.06)
        self.assertTrue(result.metrics["tray_table_contact"][0])
        self.assertTrue(result.metrics["lifted"][0])
        self.assertFalse(result.metrics["valid_hold"][0])
        self.assertEqual(result.reward[0], 0)

    def test_ball_table_support_cannot_count_as_balancing(self):
        result = self.sample(1, UP_Z, ball_height=0.005 - UP_Z)
        self.assertTrue(result.metrics["ball_table_contact"][0])
        self.assertFalse(result.metrics["current_balanced"][0])
        self.sample(2, UP_Z)
        metrics = self.evaluator._sample_metrics_from_data(self.data)
        self.assertTrue(metrics["ball_tray_contact"])
        metrics["ball_table_contact"] = np.asarray(True)
        result = self.evaluator._result_from_metrics(metrics)
        self.assertFalse(result.metrics["valid_hold"][0])
        self.assertEqual(result.metrics["continuous_hold_s"][0], 0)

    def test_both_distinct_hands_are_required_and_grasping_ball_is_invalid(self):
        result = self.sample(1, UP_Z, held=False)
        self.assertFalse(result.metrics["both_handles_gripped"][0])
        self.assertEqual(result.reward[0], 0)
        result = self.sample(2, UP_Z, touch_ball=True)
        self.assertTrue(result.metrics["ball_gripper_contact"][0])
        self.assertEqual(result.reward[0], 0)
        self.sample(3, UP_Z, held=False)
        geom = int(self.model.body("left_link_left_finger").geomadr[0])
        self.model.geom_size[geom, 1] = 0.18
        self.put("left_finger_joint", [0, 0, UP_Z + 0.02])
        mujoco.mj_forward(self.model, self.data)
        result = self.evaluator.evaluate_model_data(self.model, self.data)
        self.assertFalse(result.metrics["both_handles_gripped"][0])

    def test_margin_only_ball_contact_does_not_supply_support(self):
        ball_geom = int(self.model.body("ball_tray_ball").geomadr[0])
        self.model.geom_margin[ball_geom] = 0.02
        result = self.sample(1, UP_Z, ball_height=0.025)
        self.assertGreater(self.data.ncon, 0)
        self.assertFalse(result.metrics["ball_tray_contact"][0])
        self.assertEqual(result.reward[0], 0)

    def test_actual_physics_contacts_support_ball_on_table(self):
        self.sample(0, REST_Z, held=False)
        self.data.qvel[:] = 0
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)
        result = self.evaluator.evaluate_model_data(self.model, self.data)
        self.assertTrue(result.metrics["ball_tray_contact"][0])
        self.assertTrue(result.metrics["tray_table_contact"][0])
        self.assertFalse(result.metrics["valid_hold"][0])
        self.assertEqual(result.reward[0], 0)

    def test_stable_dynamic_ball_reaches_target_at_production_verifier_cadence(self):
        xml = XML.replace(
            "<mujoco><worldbody>",
            '<mujoco><option timestep=".002"/><worldbody>',
        )
        xml = xml.replace(
            '<body name="ball_tray"', '<body mocap="true" name="ball_tray"'
        ).replace('<freejoint name="ball_tray_joint"/>', "")
        xml = xml.replace(
            f'pos="{AWAY[0]} {AWAY[1]} {AWAY[2]}"',
            f'pos="0 {HANDLE_Y} {UP_Z + 0.0395}"',
        ).replace('pos="-1 -1 .5"', f'pos="0 -{HANDLE_Y} {UP_Z + 0.0395}"')
        xml = xml.replace(
            '<geom type="box" size=".01 .01 .01" mass=".1"/>',
            '<geom name="left_finger_contact" type="box" size=".01 .01 .01" mass=".1"/>',
            1,
        ).replace(
            '<geom type="box" size=".01 .01 .01" mass=".1"/>',
            '<geom name="right_finger_contact" type="box" size=".01 .01 .01" mass=".1"/>',
            1,
        )
        xml = xml.replace(
            "</mujoco>",
            '<contact><pair geom1="left_finger_contact" geom2="ball_tray_left_handle_collision"/>'
            '<pair geom1="right_finger_contact" geom2="ball_tray_right_handle_collision"/>'
            '</contact><equality><weld body1="left_link_left_finger"/>'
            '<weld body1="right_link_left_finger"/></equality></mujoco>',
        )
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        evaluator = BallTrayBalancingEvaluator(
            model=model, spec=get_task_spec("ball_tray_balancing")
        )
        mujoco.mj_forward(model, data)
        evaluator.evaluate_model_data(model, data)
        data.mocap_pos[int(model.body("ball_tray").mocapid[0])] = [0, 0, UP_Z]
        address = int(model.jnt_qposadr[model.joint("ball_tray_ball_joint").id])
        data.qpos[address : address + 3] = [0, 0, UP_Z + 0.02]
        sample_counts = dict.fromkeys(
            (
                "ball_tray_contact",
                "both_handles_gripped",
                "ball_robot_contact",
                "lifted",
                "valid_hold",
            ),
            0,
        )
        for tick in range(5508):
            mujoco.mj_step(model, data)
            if (tick + 1) % 17 == 0:
                result = evaluator.evaluate_model_data(model, data)
                for key in sample_counts:
                    sample_counts[key] += int(result.metrics[key][0])
        self.assertGreaterEqual(
            result.metrics["continuous_hold_s"][0],
            10,
            {
                "sample_counts": sample_counts,
                "final_metrics": result.to_info(squeeze=True),
            },
        )
        self.assertTrue(result.metrics["hold_complete"][0])
        self.assertFalse(result.scalar_success())
        self.assertAlmostEqual(result.reward[0], 0.9)

    def test_wrist_contact_cannot_stabilize_ball_for_credit(self):
        xml = XML.replace(
            "</worldbody>",
            f'<body name="left_arm" pos="0 0 {UP_Z + 0.029}">'
            '<freejoint name="wrist_joint"/><geom type="sphere" size=".01" mass=".1"/>'
            "</body></worldbody>",
        )
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.use_evaluator()
        result = self.sample(1, UP_Z)
        self.assertTrue(result.metrics["both_handles_gripped"][0])
        self.assertFalse(result.metrics["ball_gripper_contact"][0])
        self.assertTrue(result.metrics["ball_robot_contact"][0])
        self.assertEqual(result.reward[0], 0)

    def test_worlds_repeated_timestamps_reset_and_invalid_times(self):
        rest = self.data.qpos.copy()
        self.sample(1, UP_Z)
        up = self.data.qpos.copy()
        self.evaluator.reset(nworld=2)
        self.evaluator.evaluate_qpos_batch(np.stack([rest, rest]), time_s=0)
        self.evaluator.evaluate_qpos_batch(np.stack([up, up]), time_s=1)
        result = self.evaluator.evaluate_qpos_batch(np.stack([up, up]), time_s=11)
        repeated = self.evaluator.evaluate_qpos_batch(np.stack([up, up]), time_s=11)
        np.testing.assert_array_equal(result.reward, repeated.reward)
        self.evaluator.evaluate_qpos_batch(np.stack([rest, up]), time_s=11.1)
        result = self.evaluator.evaluate_qpos_batch(np.stack([rest, up]), time_s=12.1)
        np.testing.assert_array_equal(result.success, [True, False])
        self.evaluator.reset(nworld=2)
        result = self.evaluator.evaluate_qpos_batch(np.stack([rest, rest]), time_s=0)
        np.testing.assert_array_equal(result.reward, [0, 0])
        for time_s in (-1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                self.evaluator.evaluate_qpos_batch(
                    np.stack([rest, rest]), time_s=time_s
                )

    def test_world_qualification_clocks_are_independent_and_clear_on_reset(self):
        rest = self.data.qpos.copy()
        self.sample(1, UP_Z)
        up = self.data.qpos.copy()
        self.sample(2, UP_Z, ball_inside=False)
        lost = self.data.qpos.copy()
        self.evaluator.reset(nworld=3)
        self.evaluator.evaluate_qpos_batch(np.stack([rest] * 3), time_s=0)
        for time_s in (1, 2):
            self.evaluator.evaluate_qpos_batch(np.stack([up, rest, up]), time_s=time_s)
        for time_s in (3, 11, 13):
            self.evaluator.evaluate_qpos_batch(np.stack([up, up, lost]), time_s=time_s)
        result = self.evaluator.evaluate_qpos_batch(np.stack([up, up, lost]), time_s=17)
        np.testing.assert_allclose(result.metrics["overtime_s"], [1, 0, 0])
        np.testing.assert_allclose(result.reward, [0.88, 0.9, 0.18])
        self.evaluator.evaluate_qpos_batch(np.stack([up, lost, lost]), time_s=19)
        for time_s in (20, 30):
            result = self.evaluator.evaluate_qpos_batch(
                np.stack([up, up, lost]), time_s=time_s
            )
        np.testing.assert_allclose(result.metrics["overtime_s"], [14, 12, 0])
        self.evaluator.reset(nworld=3)
        self.evaluator.evaluate_qpos_batch(np.stack([rest] * 3), time_s=0)
        for time_s in (1, 2):
            self.evaluator.evaluate_qpos_batch(np.stack([up] * 3), time_s=time_s)
        self.evaluator.evaluate_qpos_batch(np.stack([lost] * 3), time_s=3)
        result = self.evaluator.evaluate_qpos_batch(np.stack([lost] * 3), time_s=120)
        np.testing.assert_array_equal(result.metrics["overtime_s"], [0, 0, 0])
        np.testing.assert_allclose(result.reward, [0.18, 0.18, 0.18])

    def test_repeated_timestamp_state_changes_cannot_add_hold_time(self):
        self.sample(1, UP_Z)
        self.sample(5, UP_Z)
        self.sample(5, LOW_Z)
        result = self.sample(5, UP_Z)
        self.assertEqual(result.metrics["continuous_hold_s"][0], 0)
        self.assertEqual(result.metrics["best_continuous_hold_s"][0], 4)
        self.assertEqual(self.score(result), (False, 0.42))

    def test_sampled_target_uses_same_credit_and_settle_without_overtime(self):
        self.use_evaluator(hold_window_s=None)
        self.sample(1, UP_Z)
        self.assertEqual(self.score(self.sample(4.5, UP_Z)), (False, 0.5))
        self.hold(5, 31, step=1)
        self.assertEqual(self.score(self.place(31.6)), (False, 0.9))
        done = self.sample(32.6, REST_Z)
        self.assertEqual(self.score(done), (True, 1.0))
        self.assertEqual(done.metrics["overtime_s"][0], 0)
        info = done.to_info(squeeze=True)
        self.assertIsNone(info["hold_window_s"])
        self.assertEqual(json.loads(json.dumps(info, allow_nan=False)), info)

    def test_disabled_hold_window_is_json_null_for_grouped_worlds(self):
        self.use_evaluator(hold_window_s=None)
        self.evaluator.reset(nworld=2)
        result = self.evaluator.evaluate_qpos_batch(
            np.stack([self.data.qpos, self.data.qpos]), time_s=0
        )
        for squeeze in (False, True):
            info = result.to_info(squeeze=squeeze)
            self.assertIsNone(info["hold_window_s"])
            self.assertEqual(json.loads(json.dumps(info, allow_nan=False)), info)

    def test_runtime_resumes_perturbations_after_prior_complete_hold_and_loss(self):
        runtime = self.make_runtime()
        self.runtime_sample(runtime, 1, UP_Z)
        self.runtime_sample(runtime, 2, UP_Z)
        pulses = runtime.debug_state()["perturbation_count"]
        self.assertGreater(pulses, 0)
        held = self.runtime_sample(runtime, 11, UP_Z)
        self.assertFalse(held.metrics["perturbations_allowed"][0])
        self.runtime_sample(runtime, 11.1, UP_Z, ball_inside=False)
        retry = self.runtime_sample(runtime, 11.2, UP_Z)
        self.assertFalse(retry.metrics["hold_complete"][0])
        self.assertTrue(retry.metrics["perturbations_allowed"][0])
        self.assertGreater(runtime.debug_state()["perturbation_count"], pulses)
        self.assertEqual(runtime.debug_state()["state"], "counting_down")

    def test_runtime_stops_force_through_settle_and_keeps_other_forces(self):
        runtime = self.make_runtime()
        self.runtime_sample(runtime, 1, UP_Z)
        self.runtime_sample(runtime, 2, UP_Z)
        ball = self.model.body("ball_tray_ball").id
        other = np.array([0.1, 0.2, 0.3])
        self.data.xfrc_applied[ball, :3] += other
        runtime.before_step(self.model, self.data)
        self.assertGreater(np.linalg.norm(self.data.xfrc_applied[ball, :3] - other), 0)
        self.runtime_sample(runtime, 11, UP_Z)
        runtime.before_step(self.model, self.data)
        np.testing.assert_allclose(self.data.xfrc_applied[ball, :3], other)
        self.runtime_sample(runtime, 11.6, REST_Z)
        self.runtime_sample(runtime, 12.6, REST_Z)
        self.assertTrue(runtime.debug_state()["complete"])

    def test_prompts(self):
        runtime = self.make_runtime()
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
