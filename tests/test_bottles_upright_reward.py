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
            '<body name="left_link_left_finger"><geom name="finger" type="sphere" size="0.01"/></body>'
            '<body name="distractor_1"><geom name="clutter" type="sphere" size="0.01"/></body>'
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
        self.forces.append(force)
        self.data.ncon = len(self.forces)

    def contact_force(self, model, data, index, out):
        out[:] = [self.forces[index], 0.0, 0.0, 0.0, 0.0, 0.0]

    def score(self):
        self.evaluator.after_reset(self.model, self.data)
        for _ in range(300):
            self.data.time += self.model.opt.timestep
            self.evaluator.after_physics_step(self.model, self.data)
        return self.evaluator.evaluate_model_data(self.model, self.data)

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
        for _ in range(10):
            self.assertEqual(self.evaluator.evaluate_model_data(self.model, self.data).to_info(), before)

    def test_nonfinite_force_fails_loudly(self):
        self.forces[0] = np.nan
        with self.assertRaisesRegex(ValueError, "contact forces must be finite"):
            self.score()


if __name__ == "__main__":
    main()
