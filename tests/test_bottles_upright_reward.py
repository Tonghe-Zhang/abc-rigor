"""Adversarial solved-contact cases for upright-bottle reward."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import TestCase, main, mock

import mujoco
import numpy as np
from abc_sim.task_eval.bottles_upright import BottlesUprightEvaluator
from abc_sim.task_specs import get_task_spec


class BottlesUprightRewardTest(TestCase):
    def setUp(self):
        # A small physics model makes target identity, table and robot contacts
        # explicit. Asset-specific stability is exercised in test_bottles_upright.
        bottles = "".join(
            f'<body name="bottle_{i}" pos="0.6 {(i - 2) * 0.3} 0.75">'
            f'<freejoint name="bottle_{i}_joint"/>'
            '<geom type="cylinder" pos="0 0 0.1" size="0.03 0.1" mass="0.05"/>'
            "</body>"
            for i in range(1, 4)
        )
        self.model = mujoco.MjModel.from_xml_string(
            '<mujoco><option timestep="0.002"/><worldbody>'
            '<geom name="table_plane" type="box" pos="0.6 0 0.73" size="0.3 0.6 0.02"/>'
            '<body name="left_link_left_finger" mocap="true"><geom name="finger" type="sphere" size="0.01"/>'
            '<body name="finger_child"><geom name="finger_child_geom" type="sphere" size="0.01"/></body></body>'
            '<body name="right_link_right_finger" mocap="true">'
            '<geom name="other_finger" type="sphere" size="0.01"/></body>'
            '<body name="distractor_1" mocap="true"><geom name="clutter" type="sphere" size="0.01"/></body>'
            f"{bottles}</worldbody></mujoco>"
        )
        self.evaluator = BottlesUprightEvaluator(model=self.model, spec=get_task_spec("lift_bottles_upright"))
        self.data = SimpleNamespace(
            qpos=self.model.qpos0.copy(), qvel=np.zeros(self.model.nv), time=0.0, contact=[], ncon=0
        )
        self.forces = []
        self.targets = [self.model.body(f"bottle_{i}").geomadr[0] for i in range(1, 4)]
        for target in self.targets:
            self.add_contact(self.model.geom("table_plane").id, target, force=0.05 * 9.81)
        self.force_patch = mock.patch("mujoco.mj_contactForce", side_effect=self.contact_force)
        self.force_patch.start()
        self.addCleanup(self.force_patch.stop)

    def add_contact(self, geom1, geom2, *, force=0.0, normal_z=1.0, z=0.75, distance=-0.0001):
        self.data.contact.append(
            SimpleNamespace(
                geom1=int(geom1),
                geom2=int(geom2),
                dist=distance,
                frame=np.array([0.0, 0.0, normal_z, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]),
                pos=np.array([0.6, 0.0, z]),
            )
        )
        self.forces.append(np.array([force, 0.0, 0.0, 0.0, 0.0, 0.0]))
        self.data.ncon = len(self.forces)

    def contact_force(self, model, data, index, out):
        out[:] = self.forces[index]

    def advance(self, steps=1):
        for _ in range(steps):
            self.data.time += self.model.opt.timestep
            self.evaluator.after_physics_step(self.model, self.data)
        return self.evaluator.evaluate_model_data(self.model, self.data)

    def score(self):
        self.evaluator.after_reset(self.model, self.data)
        return self.advance(300)

    def release_contacts(self):
        del self.data.contact[3:]
        del self.forces[3:]
        self.data.ncon = 3
        for index in range(3):
            self.forces[index] = np.array([0.05 * 9.81, 0, 0, 0, 0, 0])

    def lift_first_bottle(self):
        """Remove its tabletop load while testing a gripper-carried force pulse."""
        self.forces[0][:] = 0

    def assert_first_rejected(self):
        result = self.score()
        np.testing.assert_array_equal(result.metrics["bottle_upright_mask"], [[False, True, True]])
        self.assertFalse(result.scalar_success())
        self.assertAlmostEqual(result.scalar_reward(), 2 / 3)

    def test_released_table_supported_bottles_qualify_in_either_contact_order(self):
        contact = self.data.contact[0]
        contact.geom1, contact.geom2 = contact.geom2, contact.geom1
        contact.frame[2] = -1
        self.assertTrue(self.score().scalar_success())

    def test_gripper_contact_blocks_credit_even_with_full_table_support(self):
        self.add_contact(self.targets[0], self.model.geom("finger").id)
        self.assert_first_rejected()

    def test_clutter_prop_blocks_credit_even_with_full_table_support(self):
        self.add_contact(self.model.geom("clutter").id, self.targets[0])
        self.assert_first_rejected()

    def test_target_bottles_cannot_support_each_other(self):
        self.add_contact(self.targets[0], self.targets[1])
        np.testing.assert_array_equal(self.score().metrics["bottle_upright_mask"], [[False, False, True]])

    def test_geometric_table_contact_without_solved_support_gets_no_credit(self):
        self.forces[0] = 0.0
        self.assert_first_rejected()

    def test_partial_weight_support_gets_no_credit(self):
        self.forces[0] *= 0.75
        self.assert_first_rejected()

    def test_downward_table_contact_gets_no_credit(self):
        self.data.contact[0].frame[2] = -1
        self.assert_first_rejected()

    def test_table_contact_above_the_base_gets_no_credit(self):
        self.data.contact[0].pos[2] = 0.8
        self.assert_first_rejected()

    def test_sliding_and_spinning_bottles_get_no_credit(self):
        for axis, speed in ((0, 0.04), (5, 0.3)):
            with self.subTest(axis=axis):
                self.evaluator.reset()
                self.data.qvel[:] = 0
                self.data.qvel[axis] = speed
                self.assert_first_rejected()

    def test_nearby_noncontacting_object_does_not_block_credit(self):
        self.add_contact(self.targets[0], self.model.geom("clutter").id, distance=0.01)
        self.assertTrue(self.score().scalar_success())

    def test_force_inside_contact_margin_still_blocks_credit(self):
        self.add_contact(self.targets[0], self.model.geom("finger").id, distance=0.001, force=0.1)
        self.assert_first_rejected()

    def test_bin_bottom_support_does_not_count_as_table_support(self):
        self.data.contact[0].geom1 = self.model.geom("clutter").id
        self.assert_first_rejected()

    def test_evaluation_is_idempotent_and_returned_metrics_do_not_alias_history(self):
        result = self.score()
        before = result.to_info()
        result.metrics["bottle_sustained_seconds"][:] = 99
        result.metrics["bottle_peak_force_n"][:] = 99
        result.metrics["bottle_force_penalty"][:] = 99
        for _ in range(10):
            self.assertEqual(self.evaluator.evaluate_model_data(self.model, self.data).to_info(), before)

    def test_nonfinite_force_fails_loudly(self):
        self.forces[0] = np.nan
        with self.assertRaisesRegex(ValueError, "contact forces must be finite"):
            self.score()

    def test_force_ramp_boundaries_and_per_bottle_credit(self):
        for force, lost in ((0, 0), (9.9, 0), (10, 0), (12.5, 0.025), (15, 0.05), (20, 0.1), (80, 0.1)):
            with self.subTest(force=force):
                self.evaluator.reset()
                self.evaluator.after_reset(self.model, self.data)
                self.lift_first_bottle()
                self.add_contact(self.targets[0], self.model.geom("finger").id, force=force)
                self.advance()
                self.release_contacts()
                result = self.advance(300)
                self.assertTrue(result.scalar_success())
                self.assertAlmostEqual(result.scalar_reward(), 1 - lost)
                np.testing.assert_allclose(result.metrics["bottle_reward"], [[1 / 3 - lost, 1 / 3, 1 / 3]])
                np.testing.assert_allclose(result.metrics["bottle_force_penalty"], [[lost, 0, 0]])

    def test_peak_captures_one_tick_and_persists_through_release_and_restanding(self):
        self.evaluator.after_reset(self.model, self.data)
        self.lift_first_bottle()
        self.add_contact(self.targets[0], self.model.geom("finger").id, force=25)
        result = self.advance()
        self.assertEqual(result.metrics["bottle_peak_force_n"][0, 0], 25)
        self.release_contacts()
        result = self.advance(300)
        self.assertAlmostEqual(result.scalar_reward(), 0.9)
        address = self.model.joint("bottle_1_joint").qposadr[0]
        self.data.qpos[address + 2] += 0.1
        self.advance()
        self.data.qpos[address + 2] -= 0.1
        result = self.advance(300)
        self.assertEqual(result.metrics["bottle_peak_force_n"][0, 0], 25)
        self.assertAlmostEqual(result.scalar_reward(), 0.9)
        self.evaluator.reset()
        result = self.score()
        self.assertAlmostEqual(result.scalar_reward(), 1)
        self.assertLess(result.metrics["bottle_peak_force_n"].max(), 10)

    def test_contact_patches_and_welded_children_all_contribute(self):
        self.evaluator.after_reset(self.model, self.data)
        self.lift_first_bottle()
        for geom in ("finger", "finger", "finger_child_geom"):
            self.add_contact(self.targets[0], self.model.geom(geom).id, force=5)
        result = self.advance()
        self.assertEqual(result.metrics["bottle_contact_force_n"][0, 0], 15)
        self.release_contacts()
        self.assertAlmostEqual(self.advance(300).scalar_reward(), 0.95)

    def test_opposing_gripper_forces_add_without_cancellation(self):
        self.evaluator.after_reset(self.model, self.data)
        self.lift_first_bottle()
        self.add_contact(self.targets[0], self.model.geom("finger").id, force=8, normal_z=1)
        self.add_contact(self.targets[0], self.model.geom("other_finger").id, force=8, normal_z=-1)
        result = self.advance()
        self.assertEqual(result.metrics["bottle_contact_force_n"][0, 0], 16)
        self.release_contacts()
        self.assertAlmostEqual(self.advance(300).scalar_reward(), 0.94)

    def test_force_includes_friction_and_excludes_torque(self):
        self.evaluator.after_reset(self.model, self.data)
        self.lift_first_bottle()
        self.add_contact(self.targets[0], self.model.geom("finger").id)
        self.forces[-1][:] = [12, 9, 0, 100, 200, 300]
        result = self.advance()
        self.assertEqual(result.metrics["bottle_peak_force_n"][0, 0], 15)
        self.release_contacts()
        self.assertAlmostEqual(self.advance(300).scalar_reward(), 0.95)

    def test_table_impacts_and_other_bottles_are_also_tracked(self):
        self.evaluator.after_reset(self.model, self.data)
        self.forces[0][0] = 15
        self.add_contact(self.targets[1], self.targets[2], force=20)
        result = self.advance()
        np.testing.assert_allclose(result.metrics["bottle_peak_force_n"], [[15, 20.4905, 20.4905]])
        self.forces[0][0] = 0.05 * 9.81
        self.release_contacts()
        self.assertAlmostEqual(self.advance(300).scalar_reward(), 0.75)

    def test_gentle_credit_requires_upright_qualification(self):
        address = self.model.joint("bottle_1_joint").qposadr[0]
        self.data.qpos[address + 3 : address + 7] = [0, 1, 0, 0]
        result = self.score()
        self.assertEqual(result.metrics["bottle_reward"][0, 0], 0)
        self.assertAlmostEqual(result.scalar_reward(), 2 / 3)

    def test_all_bottles_can_finish_with_all_gentle_credit_lost(self):
        self.evaluator.after_reset(self.model, self.data)
        for target in self.targets:
            self.add_contact(target, self.model.geom("finger").id, force=21)
        self.advance()
        self.release_contacts()
        result = self.advance(300)
        self.assertTrue(result.scalar_success())
        self.assertAlmostEqual(result.scalar_reward(), 0.7)

    def test_force_threshold_configuration_is_validated(self):
        for start, end in ((-1, 20), (10, 10), (20, 10), (10, np.inf), (np.nan, 20)):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                BottlesUprightEvaluator(
                    model=self.model,
                    spec=self.evaluator.spec,
                    force_penalty_start_n=start,
                    force_penalty_end_n=end,
                )
        self.evaluator = BottlesUprightEvaluator(
            model=self.model, spec=self.evaluator.spec, force_penalty_start_n=5, force_penalty_end_n=15
        )
        self.evaluator.after_reset(self.model, self.data)
        self.lift_first_bottle()
        self.add_contact(self.targets[0], self.model.geom("finger").id, force=10)
        self.advance()
        self.release_contacts()
        self.assertAlmostEqual(self.advance(300).scalar_reward(), 0.95)


class BottlesUprightForcePhysicsTest(TestCase):
    """Check force units and compression using actual solved MuJoCo contacts."""

    def setUp(self):
        bottles = "".join(
            f'<body name="bottle_{i}" pos="0.6 {(i - 2) * 0.3} .75">'
            f'<freejoint name="bottle_{i}_joint"/>'
            '<geom type="cylinder" pos="0 0 .1" size=".03 .1" mass=".05"/>'
            f'<site name="bottle_{i}_sensor_zone" type="box" size=".04 .04 .11" pos="0 0 .1"/>'
            "</body>"
            for i in range(1, 4)
        )
        pads = "".join(
            f'<body name="{side}_finger" mocap="true" pos="{-index} 0 0">'
            '<geom type="box" size=".005 .03 .08" solref=".004 1" solimp=".99 .99 .001"/>'
            "</body>"
            for index, side in enumerate(("left", "right"), 1)
        )
        self.model = mujoco.MjModel.from_xml_string(
            '<mujoco><option timestep=".002" integrator="implicitfast"/><worldbody>'
            '<geom name="table_plane" type="box" size=".3 .6 .02" pos=".6 0 .73"/>'
            f"{pads}{bottles}</worldbody>"
            '<sensor><contact name="bottle_contacts" subtree1="bottle_1" '
            'data="found force" num="32" reduce="maxforce"/>'
            '<force name="bottle_parent_force" site="bottle_1_sensor_zone"/>'
            '<touch name="bottle_touch" site="bottle_1_sensor_zone"/></sensor></mujoco>'
        )
        self.data = mujoco.MjData(self.model)
        self.evaluator = BottlesUprightEvaluator(model=self.model, spec=get_task_spec("lift_bottles_upright"))
        mujoco.mj_forward(self.model, self.data)
        self.evaluator.after_reset(self.model, self.data)
        self.advance()

    def advance(self, steps=750):
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
            self.evaluator.after_physics_step(self.model, self.data)
        return self.evaluator.evaluate_model_data(self.model, self.data)

    def test_known_static_load_is_measured_in_newtons_and_peak_survives_unloading(self):
        body = self.model.body("bottle_1").id
        self.data.xfrc_applied[body, 2] = -15
        loaded = self.advance()
        expected = 15 + 0.05 * 9.81
        self.assertAlmostEqual(loaded.metrics["bottle_contact_force_n"][0, 0], expected, delta=0.05)
        peak = loaded.metrics["bottle_peak_force_n"][0, 0]
        self.assertGreaterEqual(peak, loaded.metrics["bottle_contact_force_n"][0, 0])
        self.assertLess(peak, 17)
        self.data.xfrc_applied[:] = 0
        released = self.advance()
        self.assertAlmostEqual(released.metrics["bottle_contact_force_n"][0, 0], 0.05 * 9.81, delta=0.005)
        self.assertEqual(released.metrics["bottle_peak_force_n"][0, 0], peak)
        self.assertTrue(released.scalar_success())
        self.assertAlmostEqual(released.metrics["bottle_reward"][0, 0], 1 / 3 - 0.1 * (peak - 10) / 10)

    def test_balanced_squeeze_is_detected_despite_small_net_force(self):
        # Opposing pads compress the cylinder by 1.5 mm at each side.
        self.data.mocap_pos[:] = [[0.5665, -0.3, 0.85], [0.6335, -0.3, 0.85]]
        squeezed = self.advance()
        self.assertGreater(squeezed.metrics["bottle_contact_force_n"][0, 0], 20)
        self.assertGreater(squeezed.metrics["bottle_peak_force_n"][0, 0], 20)
        mujoco.mj_rnePostConstraint(self.model, self.data)
        net_force = self.data.cfrc_ext[self.model.body("bottle_1").id, 3:]
        self.assertLess(np.linalg.norm(net_force), 2)
        self.assertLess(np.linalg.norm(self.data.sensor("bottle_parent_force").data), 2)
        self.assertGreater(self.data.sensor("bottle_touch").data[0], 20)
        self.data.mocap_pos[:] = [[-1, 0, 0], [-2, 0, 0]]
        released = self.advance()
        self.assertTrue(released.scalar_success())
        self.assertAlmostEqual(released.scalar_reward(), 0.9)
        np.testing.assert_allclose(released.metrics["bottle_force_penalty"], [[0.1, 0, 0]])

    def test_builtin_contact_sensor_matches_all_body_contact_magnitudes(self):
        for phase in ("rest", "loaded", "squeezed"):
            with self.subTest(phase=phase):
                self.data.xfrc_applied[self.model.body("bottle_1").id, 2] = -15 if phase == "loaded" else 0
                if phase == "squeezed":
                    self.data.mocap_pos[:] = [[0.5665, -0.3, 0.85], [0.6335, -0.3, 0.85]]
                result = self.advance()
                slots = self.data.sensor("bottle_contacts").data.reshape(32, 4)
                # The sensor's found field reports the full matching count,
                # including contacts beyond its fixed output capacity.
                self.assertLessEqual(slots[:, 0].max(), len(slots))
                sensor_total = np.linalg.norm(slots[:, 1:], axis=1).sum()
                self.assertAlmostEqual(result.metrics["bottle_contact_force_n"][0, 0], sensor_total, places=9)
                self.assertAlmostEqual(self.data.sensor("bottle_touch").data[0], slots[:, 1].sum(), places=9)


if __name__ == "__main__":
    main()
