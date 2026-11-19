"""Dish-rack reward angle, contact and per-plate contracts."""

import unittest

import mujoco
import numpy as np
from abc_sim.task_eval.dishrack import DishrackEvaluator
from abc_sim.task_eval.registry import make_task_evaluator
from abc_sim.task_specs import get_task_spec


class DishrackTests(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string(
            "<mujoco><worldbody>"
            '<body name="dishrack"><geom type="box" size=".3 .3 .01"/></body>'
            '<body name="plate"><freejoint name="plate_joint"/>'
            '<geom type="cylinder" size=".06 .003" mass=".1"/></body>'
            '<body name="plate_1"><freejoint name="plate_joint_1"/>'
            '<geom type="cylinder" size=".06 .003" mass=".1"/></body>'
            '<body name="link_left_finger" mocap="true" pos="0 0 1">'
            '<geom type="sphere" size=".01"/></body>'
            "</worldbody></mujoco>"
        )
        self.data = mujoco.MjData(self.model)
        self.spec = get_task_spec("load_plates_into_dish_rack")
        self.evaluator = make_task_evaluator(self.model, self.spec)

    def place(self, index, tilt_degrees, *, lift=0.0):
        angle = np.deg2rad(90 - tilt_degrees)
        height = 0.01 + 0.06 * abs(np.sin(angle)) + 0.003 * abs(np.cos(angle)) - 1e-5 + lift
        joint = self.model.joint("plate_joint" if index == 0 else "plate_joint_1")
        address = joint.qposadr[0]
        self.data.qpos[address : address + 7] = [
            -0.1 + 0.2 * index,
            0,
            height,
            np.cos(angle / 2),
            np.sin(angle / 2),
            0,
            0,
        ]
        mujoco.mj_forward(self.model, self.data)

    def result(self):
        return self.evaluator.evaluate_model_data(self.model, self.data)

    def test_default_and_catalogue_allow_45_degrees_from_vertical(self):
        default = DishrackEvaluator(model=self.model, spec=self.spec)
        self.assertEqual(default.upright_normal_z_max, 2**-0.5)
        self.assertEqual(self.evaluator.upright_normal_z_max, default.upright_normal_z_max)
        for tilt in (-45, -44.9999, 0, 33.4, 44.9999, 45):
            with self.subTest(tilt=tilt):
                self.place(0, tilt)
                self.place(1, tilt)
                result = self.result()
                self.assertTrue(all(result.metrics["plate_rack_contact_mask"][0]))
                self.assertEqual(result.scalar_reward(), 1)
                self.assertTrue(result.scalar_success())

    def test_tilt_beyond_45_degrees_earns_no_upright_credit(self):
        for tilt in (-45.0001, 45.0001, 60, 90):
            with self.subTest(tilt=tilt):
                self.place(0, tilt)
                self.place(1, tilt)
                result = self.result()
                self.assertEqual(result.metrics["placement_reward"][0], 1)
                self.assertEqual(result.scalar_reward(), 0)
                self.assertFalse(result.scalar_success())

    def test_plate_credit_is_independent_and_current(self):
        self.place(0, 44)
        self.place(1, 46)
        self.assertEqual(self.result().scalar_reward(), 0.5)
        self.place(1, 44)
        self.assertEqual(self.result().scalar_reward(), 1)
        self.place(0, 46)
        self.assertEqual(self.result().scalar_reward(), 0.5)

    def test_qualifying_angle_still_requires_rack_contact(self):
        self.place(0, 40, lift=0.05)
        self.place(1, 40)
        result = self.result()
        self.assertEqual(result.metrics["plate_spatially_in_rack_mask"][0], [True, True])
        self.assertEqual(result.metrics["plate_rack_contact_mask"][0], [False, True])
        self.assertEqual(result.scalar_reward(), 0.5)

    def test_qualifying_angle_still_requires_gripper_release(self):
        self.place(0, 40)
        self.place(1, 40)
        self.data.mocap_pos[0] = self.data.xpos[self.model.body("plate").id]
        mujoco.mj_forward(self.model, self.data)
        result = self.result()
        self.assertEqual(result.metrics["plate_gripper_contact_mask"][0], [True, False])
        self.assertEqual(result.scalar_reward(), 0.5)
        self.data.mocap_pos[0] = [0, 0, 1]
        mujoco.mj_forward(self.model, self.data)
        self.assertEqual(self.result().scalar_reward(), 1)


if __name__ == "__main__":
    unittest.main()
