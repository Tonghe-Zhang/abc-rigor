"""Seeded fruit scenes and chronological, contact-backed packing and pouring."""

from __future__ import annotations

import copy
import json
import unittest
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import abc_sim
import mujoco
import numpy as np
from abc_sim.randomization.assets.fruit_bowls import (
    BOWL_COLORS,
    BOWL_SEGMENTS,
    FRUIT_CATEGORIES,
    FRUIT_DENSITY_KG_M3,
    FRUIT_RADIUS_RANGE_M,
    FruitSpec,
    build_bowl,
    build_fruit,
    fruit_variants,
    source_bowl_spec,
    target_bowl_spec,
)
from abc_sim.randomization.core import RandomizationState
from abc_sim.task_eval.pack_and_pour_fruit import PackAndPourFruitEvaluator
from abc_sim.task_specs import FRUIT_BOWL_COLOR_NAMES, get_task_spec


class PourFixture:
    """A scripted wrist positions the bowl; fruits move only through solved physics."""

    def __init__(self, *, radius=0.024, force_allowance_n=15.0):
        self.source = source_bowl_spec()
        self.target = target_bowl_spec()
        root = ET.Element("mujoco")
        ET.SubElement(root, "compiler", angle="radian")
        ET.SubElement(root, "option", timestep=".002", integrator="implicitfast")
        asset, world = ET.SubElement(root, "asset"), ET.SubElement(root, "worldbody")
        ET.SubElement(world, "geom", name="table_plane", type="plane", size="1 1 .1", pos="0 0 .75")
        for name, spec, pos in (
            ("source_bowl", self.source, ".6 -.2 .75"),
            ("target_bowl", self.target, ".6 .2 .75"),
        ):
            assets, body = build_bowl(name, spec)
            body.set("pos", pos)
            asset.extend(assets)
            world.append(body)
        ET.SubElement(world, "body", name="carrier", mocap="true", pos=".6 -.2 .75")
        rim_radius = self.source.wall / 2
        rim_center = self.source.rim_radius + rim_radius
        finger_offset = np.sqrt((0.006 + rim_radius - 0.0001) ** 2 - rim_radius**2)
        for side in ("left", "right"):
            arm = ET.SubElement(world, "body", name=f"{side}_arm", mocap="true", pos="-2 0 2")
            for suffix, x in (("left", -rim_center + finger_offset), ("right", -rim_center - finger_offset)):
                finger = ET.SubElement(
                    arm,
                    "body",
                    name=f"{side}_link_{suffix}_finger",
                    pos=f"{x} 0 {self.source.height + rim_radius}",
                )
                ET.SubElement(
                    finger, "geom", type="sphere", size=".006", friction=".6 .005 .0005", solref=".08 1", priority="2"
                )
        for i in range(1, 4):
            angle = (i - 1) * 2 * np.pi / 3
            spacing = max(0.031, 2 * radius / np.sqrt(3) + 0.001)
            slope = (self.source.rim_radius - self.source.bottom_radius) / (self.source.height - self.source.floor)
            height = max(
                self.source.floor + radius,
                self.source.floor + (spacing + radius * np.sqrt(1 + slope**2) - self.source.bottom_radius) / slope,
            )
            body = ET.SubElement(
                world,
                "body",
                name=f"fruit_{i}",
                pos=f"{0.6 + spacing * np.cos(angle)} {-0.2 + spacing * np.sin(angle)} {0.75 + height}",
            )
            ET.SubElement(body, "freejoint", name=f"fruit_{i}_joint")
            ET.SubElement(
                body,
                "geom",
                name=f"fruit_{i}_collision",
                type="sphere",
                size=str(radius),
                mass=str(800 * 4 * np.pi * radius**3 / 3),
                friction=".35 .003 .0002",
                priority="1",
                condim="6",
                solref=".004 1",
            )
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.evaluator = PackAndPourFruitEvaluator(
            model=self.model, spec=get_task_spec("pack_and_pour_fruit"), force_allowance_n=force_allowance_n
        )
        self.evaluator.after_reset(self.model, self.data)
        self.carrier = self.model.body("carrier").mocapid[0]
        self.hand = self.model.body("left_arm").mocapid[0]
        self.gripping = False

    def steps(self, count):
        for _ in range(count):
            if self.gripping:
                joint = self.data.joint("source_bowl_joint")
                joint.qpos[:3] = self.data.mocap_pos[self.carrier]
                joint.qpos[3:] = self.data.mocap_quat[self.carrier]
                joint.qvel[:] = 0
                self.data.mocap_pos[self.hand] = joint.qpos[:3]
                self.data.mocap_quat[self.hand] = joint.qpos[3:]
            mujoco.mj_step(self.model, self.data)
            self.evaluator.after_physics_step(self.model, self.data)
        return self.result()

    def result(self):
        return self.evaluator.evaluate_model_data(self.model, self.data)

    def grip(self):
        self.gripping = True
        self.data.mocap_pos[self.hand] = self.data.mocap_pos[self.carrier]
        self.data.mocap_quat[self.hand] = self.data.mocap_quat[self.carrier]
        return self.steps(150)

    def move(self, position, *, tilt=0, steps=1000):
        start = self.data.mocap_pos[self.carrier].copy()
        initial = self.data.mocap_quat[self.carrier].copy()
        final = np.array([np.cos(tilt / 2), 0, np.sin(tilt / 2), 0])
        for i in range(1, steps + 1):
            t = (1 - np.cos(np.pi * i / steps)) / 2
            self.data.mocap_pos[self.carrier] = start + (position - start) * t
            quat = initial * (1 - t) + final * t
            self.data.mocap_quat[self.carrier] = quat / np.linalg.norm(quat)
            self.steps(1)
        return self.result()

    def transport(self):
        self.steps(600)
        self.grip()
        self.move(np.array([0.6, -0.2, 0.98]), steps=750)
        self.move(np.array([0.6, 0.2, 0.98]), steps=1000)
        return self.steps(200)

    def pour(self):
        self.move(np.array([0.57, 0.2, 0.98]), tilt=np.deg2rad(110), steps=1250)
        return self.steps(2000)

    def return_bowl(self):
        self.gripping = True
        self.move(np.array([0.6, -0.2, 0.98]), steps=750)
        self.move(np.array([0.6, -0.2, 0.75]), steps=750)
        self.gripping = False
        self.data.mocap_pos[self.hand] = [-2, 0, 2]
        return self.steps(800)

    def put_fruit(self, i, xyz):
        self.data.joint(f"fruit_{i}_joint").qpos[:] = [*xyz, 1, 0, 0, 0]
        self.data.joint(f"fruit_{i}_joint").qvel[:] = 0


class PourPhysicsTest(unittest.TestCase):
    def test_physical_pack_transport_and_gravity_pour(self):
        fixture = PourFixture(force_allowance_n=1e6)
        packed = fixture.steps(600)
        self.assertAlmostEqual(packed.scalar_reward(), 0.2, places=5)
        carried = fixture.transport()
        self.assertTrue(carried.metrics["transport_complete"][0], carried.metrics)
        self.assertAlmostEqual(carried.scalar_reward(), 0.4, places=5)
        poured = fixture.pour()
        self.assertEqual(poured.metrics["num_caught_fruits"][0], 3)
        self.assertFalse(poured.scalar_success())
        self.assertEqual(poured.metrics["return_reward"][0], 0)
        self.assertGreater(poured.scalar_reward(), 0.88)
        returned = fixture.return_bowl()
        self.assertTrue(returned.scalar_success(), returned.metrics)
        self.assertGreater(returned.scalar_reward(), 0.98)
        self.assertAlmostEqual(returned.scalar_reward(), 1 - returned.metrics["force_penalty"].sum(), places=5)
        self.assertAlmostEqual(returned.scalar_reward() - poured.scalar_reward(), 0.1, places=5)
        # Picking the bowl up again revokes only the return stage.
        fixture.grip()
        lifted = fixture.move(np.array([0.6, -0.2, 0.98]), steps=200)
        self.assertEqual(lifted.metrics["return_reward"][0], 0)
        self.assertFalse(lifted.scalar_success())
        self.assertTrue(lifted.metrics["ever_success"][0])
        for key in ("packing_reward", "transport_reward", "emptying_reward", "catching_reward"):
            np.testing.assert_allclose(lifted.metrics[key], returned.metrics[key])
        fixture.evaluator.reset()
        fixture.evaluator.after_reset(fixture.model, fixture.data)
        self.assertFalse(fixture.result().metrics["source_returned"][0])
        self.assertEqual(fixture.result().metrics["source_return_settling_seconds"][0], 0)
        self.assertFalse(fixture.result().metrics["ever_success"][0])

    def test_returning_the_filled_bowl_before_pouring_has_no_return_credit(self):
        fixture = PourFixture(force_allowance_n=1e6)
        fixture.transport()
        returned = fixture.return_bowl()
        self.assertTrue(returned.metrics["bowl_ready_for_packing_mask"][0, 0])
        self.assertFalse(returned.metrics["touching_robot"][0, 3])
        self.assertAlmostEqual(returned.scalar_reward(), 0.4, places=5)
        self.assertEqual(returned.metrics["return_reward"][0], 0)

    def test_return_requires_upright_table_support_release_and_continuous_dwell(self):
        fixture = PourFixture(force_allowance_n=1e6)
        fixture.transport()
        fixture.pour()
        # An empty bowl resting upside down on its rim must not qualify.
        fixture.move(np.array([0.6, -0.2, 0.98]), tilt=np.pi, steps=500)
        fixture.move(np.array([0.6, -0.2, 0.75 + fixture.source.height + fixture.source.wall]), tilt=np.pi, steps=500)
        fixture.gripping = False
        fixture.data.mocap_pos[fixture.hand] = [-2, 0, 2]
        inverted = fixture.steps(800)
        self.assertTrue(inverted.metrics["settled_mask"][0, 3])
        self.assertGreater(inverted.metrics["source_tilt_deg"][0], 170)
        self.assertFalse(inverted.metrics["source_return_eligible"][0])
        # Position upright on the table, retaining fingertip contact.
        fixture.gripping = True
        fixture.move(np.array([0.6, -0.2, 0.98]), steps=500)
        fixture.move(np.array([0.6, -0.2, 0.75]), steps=500)
        held = fixture.steps(400)
        self.assertTrue(held.metrics["touching_robot"][0, 3])
        self.assertFalse(held.metrics["source_return_eligible"][0])
        fixture.gripping = False
        fixture.data.mocap_pos[fixture.hand] = [-2, 0, 2]
        short = fixture.steps(200)
        self.assertTrue(short.metrics["source_return_eligible"][0])
        self.assertEqual(short.metrics["return_reward"][0], 0)
        self.assertGreater(short.metrics["source_return_settling_seconds"][0], 0)
        # Touching the bowl breaks the dwell; a fresh half second is required.
        fixture.data.mocap_pos[fixture.hand] = [0.6, -0.2, 0.75]
        fixture.data.mocap_quat[fixture.hand] = [1, 0, 0, 0]
        touched = fixture.steps(1)
        self.assertTrue(touched.metrics["touching_robot"][0, 3])
        self.assertEqual(touched.metrics["source_return_settling_seconds"][0], 0)
        fixture.data.mocap_pos[fixture.hand] = [-2, 0, 2]
        self.assertEqual(fixture.steps(200).metrics["return_reward"][0], 0)
        returned = fixture.steps(400)
        self.assertAlmostEqual(returned.metrics["return_reward"][0], 0.1)
        self.assertTrue(returned.scalar_success())

    def test_return_credit_does_not_depend_on_catching_every_fruit(self):
        fixture = PourFixture(force_allowance_n=1e6)
        fixture.transport()
        fixture.pour()
        fixture.put_fruit(1, [0.4, -0.4, 0.774])
        returned = fixture.return_bowl()
        self.assertEqual(returned.metrics["num_emptied_fruits"][0], 3)
        self.assertEqual(returned.metrics["num_caught_fruits"][0], 2)
        self.assertAlmostEqual(returned.metrics["return_reward"][0], 0.1)
        self.assertAlmostEqual(returned.scalar_reward(), 1 - 0.25 / 3, places=5)
        self.assertFalse(returned.scalar_success())

    def test_direct_placement_in_the_large_bowl_earns_zero(self):
        for radius in FRUIT_RADIUS_RANGE_M:
            with self.subTest(radius=radius):
                fixture = PourFixture(radius=radius)
                for i in range(1, 4):
                    fixture.put_fruit(i, [0.6, 0.2 + (i - 2) * 2.3 * radius, 0.75 + fixture.target.floor + radius])
                result = fixture.steps(800)
                self.assertTrue(result.metrics["fruit_in_target_mask"].all())
                self.assertEqual(result.scalar_reward(), 0)
                self.assertFalse(result.scalar_success())

    def test_nesting_a_packed_bowl_without_pouring_earns_only_four_tenths(self):
        for radius in FRUIT_RADIUS_RANGE_M:
            with self.subTest(radius=radius):
                # Isolate stage weights; the default force deduction is tested separately.
                fixture = PourFixture(radius=radius, force_allowance_n=1e6)
                fixture.steps(2400)
                fixture.steps(600)
                fixture.grip()
                # Cross above the destination in motion and lower directly into it.
                # There is no stationary overhead dwell on this path.
                for i in range(1, 2001):
                    t = i / 2000
                    fixture.data.mocap_pos[fixture.carrier] = [
                        0.6,
                        -0.2 + 0.4 * t,
                        0.75 + 0.0062 * t + 0.23 * np.sin(np.pi * t),
                    ]
                    fixture.steps(1)
                fixture.gripping = False
                fixture.data.mocap_pos[fixture.hand] = [-2, 0, 2]
                result = fixture.steps(1000)
                self.assertTrue(result.metrics["source_nested_in_target"][0], result.metrics)
                self.assertTrue(result.metrics["transport_complete"][0])
                self.assertTrue(result.metrics["fruit_in_source_mask"].all())
                self.assertAlmostEqual(result.scalar_reward(), 0.4, places=4)
                self.assertEqual(result.metrics["emptying_reward"][0], 0)
                self.assertEqual(result.metrics["catching_reward"][0], 0)
                self.assertFalse(result.scalar_success())

    def test_partial_load_earns_full_transport_emptying_and_return_with_partial_catching(self):
        for count in (1, 2):
            with self.subTest(count=count):
                fixture = PourFixture(radius=0.035, force_allowance_n=1e6)
                for i in range(count + 1, 4):
                    fixture.put_fruit(i, [0.4, -0.4 + 0.1 * i, 0.785])
                # Let the initial fruit drop and rolling motion settle.
                packed = fixture.steps(2400)
                self.assertAlmostEqual(packed.scalar_reward(), count * 0.2 / 3, places=5)
                self.assertFalse(packed.metrics["packing_stage_complete"][0])
                carried = fixture.transport()
                self.assertTrue(carried.metrics["transport_complete"][0])
                self.assertAlmostEqual(carried.metrics["transport_reward"][0], 0.2)
                self.assertEqual(carried.metrics["num_carried_fruits"][0], count)
                self.assertEqual(carried.metrics["num_transported_fruits"][0], count)
                np.testing.assert_allclose(
                    carried.metrics["fruit_emptying_capacity"][0],
                    [0.25 / count] * count + [0] * (3 - count),
                )
                self.assertAlmostEqual(carried.metrics["force_gentle_credit"].sum(), 0.3)
                poured = fixture.pour()
                self.assertEqual(poured.metrics["num_emptied_fruits"][0], count)
                self.assertEqual(poured.metrics["num_caught_fruits"][0], count)
                self.assertAlmostEqual(poured.metrics["packing_reward"][0], count * 0.2 / 3)
                self.assertAlmostEqual(poured.metrics["emptying_reward"][0], 0.25)
                self.assertAlmostEqual(poured.metrics["catching_reward"][0], count * 0.25 / 3)
                returned = fixture.return_bowl()
                self.assertAlmostEqual(returned.metrics["return_reward"][0], 0.1)
                self.assertAlmostEqual(returned.scalar_reward(), 0.55 + count * 0.45 / 3, places=5)
                self.assertFalse(returned.scalar_success())
                fixture.evaluator.reset()
                fixture.evaluator.after_reset(fixture.model, fixture.data)
                reset = fixture.result()
                self.assertEqual(reset.scalar_reward(), 0)
                self.assertFalse(reset.metrics["fruit_carried_mask"].any())
                self.assertFalse(reset.metrics["fruit_transported_mask"].any())
                np.testing.assert_allclose(reset.metrics["force_gentle_credit"], [[0.045] * 3 + [0.1275, 0.0375]])

    def test_transporting_an_empty_bowl_earns_no_reward(self):
        fixture = PourFixture(force_allowance_n=1e6)
        for i in range(1, 4):
            fixture.put_fruit(i, [0.4, -0.4 + 0.1 * i, 0.774])
        result = fixture.transport()
        self.assertFalse(result.metrics["transport_complete"][0])
        self.assertEqual(result.metrics["num_carried_fruits"][0], 0)
        self.assertEqual(result.scalar_reward(), 0)
        self.assertEqual(fixture.pour().scalar_reward(), 0)
        self.assertEqual(fixture.return_bowl().scalar_reward(), 0)

    def test_spilled_fruit_cannot_shrink_the_emptying_load_or_qualify_return(self):
        fixture = PourFixture(force_allowance_n=1e6)
        fixture.put_fruit(3, [0.4, -0.4, 0.774])
        fixture.transport()
        fixture.put_fruit(2, [0.4, -0.2, 0.774])
        result = fixture.pour()
        self.assertEqual(result.metrics["num_carried_fruits"][0], 2)
        self.assertEqual(result.metrics["num_emptied_fruits"][0], 1)
        self.assertEqual(result.metrics["num_caught_fruits"][0], 1)
        self.assertAlmostEqual(result.metrics["packing_reward"][0], 2 * 0.2 / 3)
        self.assertAlmostEqual(result.metrics["emptying_reward"][0], 0.125)
        self.assertEqual(fixture.return_bowl().metrics["return_reward"][0], 0)

    def test_omitted_fruit_cannot_use_another_loads_transport_to_earn_catching(self):
        fixture = PourFixture(force_allowance_n=1e6)
        fixture.put_fruit(3, [0.4, -0.4, 0.774])
        fixture.transport()
        fixture.put_fruit(3, [0.6, 0.3, 0.780])
        fixture.steps(800)
        result = fixture.pour()
        self.assertTrue(result.metrics["fruit_in_target_mask"].all())
        self.assertEqual(result.metrics["num_caught_fruits"][0], 2)
        self.assertEqual(result.metrics["fruit_catching_reward"][0, 2], 0)
        self.assertFalse(result.metrics["fruit_transported_mask"][0, 2])
        self.assertAlmostEqual(fixture.return_bowl().scalar_reward(), 0.85, places=5)

    def test_second_load_can_complete_the_task_without_reusing_transport_proof(self):
        fixture = PourFixture(force_allowance_n=1e6)
        fixture.put_fruit(3, [0.4, -0.4, 0.774])
        fixture.transport()
        fixture.pour()
        self.assertAlmostEqual(fixture.return_bowl().scalar_reward(), 0.85, places=5)
        fixture.put_fruit(3, [0.6, -0.2, 0.75 + fixture.source.floor + 0.024])
        packed = fixture.steps(800)
        self.assertTrue(packed.metrics["packing_stage_complete"][0])
        fixture.grip()
        lifted = fixture.move(np.array([0.6, -0.2, 0.98]), steps=750)
        self.assertEqual(lifted.metrics["num_carried_fruits"][0], 3)
        self.assertEqual(lifted.metrics["num_transported_fruits"][0], 2)
        self.assertFalse(lifted.metrics["fruit_transported_mask"][0, 2])
        fixture.move(np.array([0.6, 0.2, 0.98]), steps=1000)
        transported = fixture.steps(200)
        self.assertTrue(transported.metrics["fruit_transported_mask"].all())
        fixture.pour()
        returned = fixture.return_bowl()
        self.assertTrue(returned.scalar_success(), returned.metrics)
        self.assertAlmostEqual(returned.scalar_reward(), 1, places=5)

    def test_wall_grasp_can_transport_the_filled_bowl(self):
        fixture = PourFixture(force_allowance_n=1e6)
        source = fixture.source
        z = (source.floor + source.height) / 2
        inner_radius = (source.bottom_radius + source.rim_radius) / 2
        slope = (source.rim_radius - source.bottom_radius) / (source.height - source.floor)
        offset = 0.0059 * np.sqrt(1 + slope**2)
        for suffix, x in (("left", -inner_radius + offset), ("right", -inner_radius - source.wall - offset)):
            body = fixture.model.body(f"left_link_{suffix}_finger").id
            fixture.model.body_pos[body] = [x, 0, z]
        result = fixture.transport()
        self.assertFalse(result.metrics["rim_held"][0])
        self.assertTrue(result.metrics["bowl_held"][0])
        self.assertTrue(result.metrics["transport_complete"][0], result.metrics)
        self.assertAlmostEqual(result.scalar_reward(), 0.4, places=5)

    def test_transport_accepts_base_overhang_within_receiver_radius(self):
        fixture = PourFixture(force_allowance_n=1e6)
        fixture.steps(600)
        fixture.grip()
        fixture.move(np.array([0.6, -0.2, 0.98]), steps=750)
        fixture.move(np.array([0.78, -0.2, 0.98]), steps=750)
        fixture.move(np.array([0.78, 0.2, 0.98]), steps=1000)
        result = fixture.steps(400)
        self.assertTrue(result.metrics["filled_bowl_carried"][0])
        self.assertFalse(result.metrics["source_over_target"][0])
        self.assertFalse(result.metrics["source_in_transport_zone"][0])
        self.assertFalse(result.metrics["transport_complete"][0])
        fixture.move(np.array([0.76, 0.2, 0.98]), steps=1000)
        result = fixture.steps(400)
        self.assertFalse(result.metrics["source_over_target"][0])
        self.assertTrue(result.metrics["source_in_transport_zone"][0])
        self.assertTrue(result.metrics["transport_complete"][0], result.metrics)
        self.assertAlmostEqual(result.scalar_reward(), 0.4, places=5)

    def test_supported_off_center_nesting_earns_transport(self):
        fixture = PourFixture(force_allowance_n=1e6)
        fixture.steps(600)
        fixture.grip()
        # Carry clear of the receiver rim before lowering.
        fixture.move(np.array([0.6, -0.2, 0.835]), steps=750)
        fixture.move(np.array([0.668, 0.2, 0.835]), steps=1000)
        fixture.move(np.array([0.668, 0.2, 0.7562]), steps=750)
        fixture.gripping = False
        fixture.data.mocap_pos[fixture.hand] = [-2, 0, 2]
        result = fixture.steps(1000)
        offset = np.linalg.norm(fixture.data.body("source_bowl").xpos[:2] - fixture.data.body("target_bowl").xpos[:2])
        self.assertGreater(offset, 0.06)
        self.assertTrue(result.metrics["fruit_in_source_mask"].all())
        self.assertTrue(result.metrics["source_nested_in_target"][0], result.metrics)
        self.assertAlmostEqual(result.scalar_reward(), 0.4, places=5)

    def test_released_bowl_can_finish_pouring_while_supported_by_receiver(self):
        fixture = PourFixture(force_allowance_n=1e6)
        fixture.steps(600)
        fixture.grip()
        fixture.move(np.array([0.6, -0.2, 0.98]), steps=750)
        fixture.move(np.array([0.465, 0.2, 0.90]), tilt=np.deg2rad(40), steps=1000)
        fixture.move(np.array([0.465, 0.2, 0.807]), tilt=np.deg2rad(50), steps=100)
        self.assertTrue(fixture.result().metrics["transport_complete"][0])
        self.assertEqual(fixture.result().metrics["num_emptied_fruits"][0], 0)
        fixture.gripping = False
        fixture.data.mocap_pos[fixture.hand] = [-2, 0, 2]
        departures = 0
        for _ in range(2000):
            result = fixture.steps(1)
            if result.metrics["fruit_departure_mask"].any():
                self.assertFalse(result.metrics["bowl_held"][0])
                self.assertLessEqual(result.metrics["source_receiver_support_gap_seconds"][0], 0.02)
                self.assertTrue(result.metrics["pouring_support_valid"][0])
                departures += int(result.metrics["fruit_departure_mask"].sum())
        self.assertEqual(departures, 3)
        self.assertEqual(result.metrics["num_caught_fruits"][0], 3)
        self.assertFalse(result.scalar_success())
        self.assertEqual(result.metrics["return_reward"][0], 0)
        self.assertAlmostEqual(result.scalar_reward(), 0.9, places=5)

    def test_one_fruit_can_finish_a_low_angle_pour_after_release(self):
        fixture = PourFixture(force_allowance_n=1e6)
        fixture.steps(600)
        fixture.grip()
        fixture.move(np.array([0.6, -0.2, 0.98]), steps=750)
        fixture.move(np.array([0.465, 0.2, 0.90]), tilt=np.deg2rad(40), steps=1000)
        fixture.move(np.array([0.465, 0.2, 0.807]), tilt=np.deg2rad(40), steps=500)
        fixture.gripping = False
        fixture.data.mocap_pos[fixture.hand] = [-2, 0, 2]
        fixture.steps(500)
        # Model residual fruit momentum from a rough delivery; MuJoCo solves its exit and landing.
        fixture.data.joint("fruit_1_joint").qvel[:3] = [1.1, 0, 0.9]
        departures = 0
        for _ in range(2000):
            result = fixture.steps(1)
            if result.metrics["fruit_departure_mask"].any():
                self.assertFalse(result.metrics["bowl_held"][0])
                self.assertLessEqual(result.metrics["source_receiver_support_gap_seconds"][0], 0.02)
                self.assertTrue(result.metrics["pouring_support_valid"][0])
                self.assertGreater(result.metrics["source_tilt_deg"][0], 30)
                self.assertLess(result.metrics["source_tilt_deg"][0], 45)
                departures += int(result.metrics["fruit_departure_mask"].sum())
        self.assertEqual(departures, 1)
        self.assertEqual(result.metrics["num_emptied_fruits"][0], 1)
        self.assertEqual(result.metrics["num_caught_fruits"][0], 1)
        self.assertAlmostEqual(result.scalar_reward(), 0.4 + 0.5 / 3, places=5)
        self.assertFalse(result.scalar_success())
        returned = fixture.return_bowl()
        self.assertEqual(returned.metrics["return_reward"][0], 0)

    def test_fruit_ejected_from_an_upright_bowl_is_not_a_pour(self):
        for radius in FRUIT_RADIUS_RANGE_M:
            with self.subTest(radius=radius):
                fixture = PourFixture(radius=radius, force_allowance_n=1e6)
                fixture.steps(2400)
                transported = fixture.transport()
                self.assertTrue(transported.metrics["transport_complete"][0])
                self.assertTrue(transported.metrics["fruit_source_resident_mask"].all())
                fixture.data.joint("fruit_1_joint").qvel[:3] = [2.0, 0, 2.0]
                crossings = 0
                for _ in range(1500):
                    result = fixture.steps(1)
                    crossings += int(result.metrics["fruit_crossed_mouth_mask"].sum())
                self.assertGreater(crossings, 0)
                self.assertEqual(result.metrics["num_emptied_fruits"][0], 0)
                self.assertEqual(result.metrics["num_caught_fruits"][0], 0)
                self.assertAlmostEqual(result.scalar_reward(), 0.4, places=5)

    def test_partial_packing_credit_clears_on_exit_and_reset(self):
        fixture = PourFixture()
        fixture.put_fruit(3, [0.4, -0.4, 0.774])
        self.assertAlmostEqual(fixture.steps(600).scalar_reward(), 2 * 0.2 / 3, places=5)
        fixture.put_fruit(1, [0.4, -0.2, 0.774])
        result = fixture.steps(1)
        self.assertAlmostEqual(result.scalar_reward(), 0.2 / 3, places=5)
        np.testing.assert_array_equal(result.metrics["fruit_packed_mask"], [[False, True, False]])
        fixture.evaluator.reset()
        fixture.evaluator.after_reset(fixture.model, fixture.data)
        self.assertEqual(fixture.result().scalar_reward(), 0)
        self.assertFalse(fixture.result().metrics["fruit_packed_mask"].any())

    def test_sideways_receiver_does_not_define_an_overhead_target(self):
        fixture = PourFixture()
        fixture.data.joint("source_bowl_joint").qpos[:3] = [0.6, -0.2, 0.98]
        fixture.data.joint("target_bowl_joint").qpos[:] = [0.6, 0.2, 0.98, 2**-0.5, 2**-0.5, 0, 0]
        mujoco.mj_forward(fixture.model, fixture.data)
        fixture.evaluator.reset()
        fixture.evaluator.after_reset(fixture.model, fixture.data)
        self.assertFalse(fixture.result().metrics["source_over_target"][0])
        self.assertFalse(fixture.result().metrics["source_in_transport_zone"][0])

    def test_transport_zone_requires_height_and_a_prior_filled_lift(self):
        fixture = PourFixture(force_allowance_n=1e6)
        fixture.steps(600)
        fixture.data.joint("source_bowl_joint").qpos[:3] = [0.6, 0.2, 0.73]
        mujoco.mj_forward(fixture.model, fixture.data)
        positions = fixture.data.xpos[fixture.evaluator._bodies[3:]]
        rotations = fixture.data.xmat[fixture.evaluator._bodies[3:]].reshape(2, 3, 3)
        self.assertFalse(fixture.evaluator._source_in_transport_zone(positions, rotations))
        # Moving a packed bowl directly into the receiver does not prove carriage.
        fixture.data.joint("source_bowl_joint").qpos[:3] = [0.6, 0.2, 0.7562]
        for i in range(1, 4):
            fixture.data.joint(f"fruit_{i}_joint").qpos[:3] += [0, 0.4, 0.0062]
        result = fixture.steps(800)
        self.assertTrue(result.metrics["source_nested_in_target"][0])
        self.assertFalse(result.metrics["filled_bowl_carried"][0])
        self.assertFalse(result.metrics["transport_complete"][0])
        self.assertAlmostEqual(result.scalar_reward(), 0.2, places=5)

    def test_fruits_can_qualify_at_different_times_before_pouring(self):
        fixture = PourFixture()
        third_position = fixture.data.joint("fruit_3_joint").qpos[:3].copy()
        fixture.put_fruit(3, [0.4, -0.4, 0.774])
        self.assertAlmostEqual(fixture.steps(600).scalar_reward(), 2 * 0.2 / 3, places=5)
        fixture.put_fruit(3, third_position)
        for _ in range(600):
            # The first fruit rolls in place while the last fruit settles.
            fixture.data.joint("fruit_1_joint").qvel[5] = 5
            result = fixture.steps(1)
        self.assertTrue(result.metrics["packing_stage_complete"][0], result.metrics)
        self.assertAlmostEqual(result.scalar_reward(), 0.2, places=5)
        self.assertTrue(result.metrics["fruit_armed_mask"].all())
        self.assertFalse(result.metrics["fruit_source_qualified_mask"][0, 0])
        fixture.transport()
        self.assertEqual(fixture.pour().metrics["num_caught_fruits"][0], 3)
        self.assertTrue(fixture.return_bowl().scalar_success())

    def test_retained_credit_does_not_qualify_an_unsupported_fruit_above_the_rim(self):
        fixture = PourFixture()
        third_position = fixture.data.joint("fruit_3_joint").qpos[:3].copy()
        fixture.put_fruit(3, [0.4, -0.4, 0.774])
        self.assertAlmostEqual(fixture.steps(600).scalar_reward(), 2 * 0.2 / 3, places=5)
        fixture.put_fruit(3, third_position)
        protruding = fixture.data.joint("fruit_1_joint").qpos[:3].copy()
        protruding[2] = 0.75 + fixture.source.height + 0.030
        for _ in range(600):
            # Its center is over the opening, but the pile does not support it.
            fixture.put_fruit(1, protruding)
            result = fixture.steps(1)
        self.assertTrue(result.metrics["fruit_packed_mask"].all())
        self.assertTrue(result.metrics["fruit_in_source_mask"][0, 0])
        self.assertFalse(result.metrics["fruit_source_geometric_support_mask"][0, 0])
        self.assertFalse(result.metrics["packing_stage_complete"][0])
        result = fixture.steps(800)
        self.assertTrue(result.metrics["fruit_in_source_mask"].all())
        self.assertTrue(result.metrics["packing_stage_complete"][0])

    def test_pouring_before_transport_cannot_earn_pouring_credit(self):
        for radius in FRUIT_RADIUS_RANGE_M:
            with self.subTest(radius=radius):
                fixture = PourFixture(radius=radius)
                fixture.steps(2400)
                fixture.steps(600)
                fixture.grip()
                fixture.move(np.array([0.6, -0.2, 0.98]), tilt=np.deg2rad(110))
                result = fixture.steps(800)
                self.assertFalse(result.metrics["fruit_in_source_mask"].any())
                self.assertFalse(result.metrics["transport_complete"][0])
                self.assertEqual(result.metrics["emptying_reward"][0], 0)
                self.assertEqual(result.metrics["catching_reward"][0], 0)
                self.assertEqual(fixture.return_bowl().metrics["return_reward"][0], 0)

    def test_hand_transfer_after_transport_does_not_count_as_pouring(self):
        for radius in FRUIT_RADIUS_RANGE_M:
            with self.subTest(radius=radius):
                fixture = PourFixture(radius=radius)
                fixture.steps(2400)
                fixture.transport()
                # Touch each fruit with the parked second hand before transferring it.
                right = fixture.model.body("right_arm").mocapid[0]
                for i in range(1, 4):
                    xyz = fixture.data.joint(f"fruit_{i}_joint").qpos[:3].copy()
                    finger = fixture.model.body("right_link_left_finger").id
                    fixture.data.mocap_pos[right] = xyz - fixture.model.body_pos[finger]
                    fixture.steps(1)
                    fixture.data.mocap_pos[right] = [-2, 0, 2]
                    fixture.put_fruit(i, [0.6, 0.2 + (i - 2) * 2.3 * radius, 0.75 + fixture.target.floor + radius])
                    fixture.steps(1)
                result = fixture.steps(800)
                self.assertTrue(result.metrics["fruit_in_target_mask"].all())
                self.assertEqual(result.metrics["emptying_reward"][0], 0)
                self.assertEqual(result.metrics["catching_reward"][0], 0)
                self.assertEqual(fixture.return_bowl().metrics["return_reward"][0], 0)

    def test_prior_transport_cannot_credit_dumping_away_from_receiver(self):
        for radius in FRUIT_RADIUS_RANGE_M:
            with self.subTest(radius=radius):
                fixture = PourFixture(radius=radius)
                fixture.steps(2400)
                self.assertTrue(fixture.transport().metrics["transport_complete"][0])
                fixture.move(np.array([0.6, -0.2, 0.98]))
                fixture.move(np.array([0.57, -0.2, 0.98]), tilt=np.deg2rad(110), steps=1250)
                result = fixture.steps(2000)
                self.assertFalse(result.metrics["fruit_source_resident_mask"].any())
                self.assertEqual(result.metrics["emptying_reward"][0], 0)
                self.assertEqual(result.metrics["catching_reward"][0], 0)
                self.assertEqual(fixture.return_bowl().metrics["return_reward"][0], 0)

    def test_landing_credit_is_per_fruit_and_revoked_when_one_leaves(self):
        fixture = PourFixture()
        fixture.transport()
        fixture.pour()
        completed = fixture.return_bowl()
        self.assertTrue(completed.scalar_success(), completed.metrics)
        fixture.put_fruit(1, [0.4, -0.4, 0.774])
        result = fixture.steps(600)
        self.assertEqual(result.metrics["num_caught_fruits"][0], 2)
        self.assertAlmostEqual(
            completed.metrics["catching_reward"][0] - result.metrics["catching_reward"][0],
            completed.metrics["fruit_catching_reward"][0, 0],
            places=5,
        )
        self.assertEqual(result.metrics["num_emptied_fruits"][0], 3)
        self.assertFalse(result.scalar_success())

    def test_force_dose_integrates_all_entities_and_queries_do_not_advance_time(self):
        fixture = PourFixture()
        fixture.steps(600)
        fixture.data.xfrc_applied[fixture.model.body("source_bowl").id, 2] = -100
        before = fixture.evaluator._force_dose_ns.copy()
        result = fixture.steps(1)
        expected = (
            before
            + np.maximum(result.metrics["contact_force_n"][0] - fixture.evaluator.force_threshold_n, 0)
            * fixture.model.opt.timestep
        )
        np.testing.assert_allclose(result.metrics["force_dose_ns"][0], expected)
        self.assertGreater(expected[3], 0)
        for _ in range(100):
            np.testing.assert_array_equal(fixture.result().metrics["force_dose_ns"], result.metrics["force_dose_ns"])
        fixture.data.xfrc_applied[:] = 0
        later = fixture.steps(500)
        self.assertTrue((later.metrics["force_dose_ns"] >= result.metrics["force_dose_ns"]).all())
        fixture.evaluator.reset()
        fixture.evaluator.after_reset(fixture.model, fixture.data)
        np.testing.assert_array_equal(fixture.result().metrics["force_dose_ns"], np.zeros((1, 5)))
        self.assertFalse(fixture.result().metrics["packing_stage_complete"][0])

    def test_observer_requires_every_physics_tick_and_native_contacts(self):
        fixture = PourFixture()
        with self.assertRaises(NotImplementedError):
            fixture.evaluator.evaluate_qpos_batch(fixture.data.qpos[None])
        with self.assertRaises(ValueError):
            fixture.evaluator.after_physics_step(fixture.model, fixture.data)
        mujoco.mj_step(fixture.model, fixture.data, nstep=2)
        with self.assertRaises(ValueError):
            fixture.evaluator.after_physics_step(fixture.model, fixture.data)

    def test_exhausted_bowl_force_credit_is_permanent_and_only_affects_earned_stages(self):
        fixture = PourFixture()
        fixture.steps(600)
        fixture.data.xfrc_applied[fixture.model.body("source_bowl").id, 2] = -100
        packed = fixture.steps(100)
        self.assertGreater(packed.metrics["force_dose_ns"][0, 3], fixture.evaluator.force_dose_budget_ns)
        self.assertAlmostEqual(packed.scalar_reward(), 0.2, places=5)
        fixture.data.xfrc_applied[:] = 0
        carried = fixture.transport()
        self.assertTrue(carried.metrics["transport_complete"][0])
        self.assertAlmostEqual(carried.scalar_reward(), 0.34, places=5)
        self.assertAlmostEqual(carried.metrics["force_penalty"][0, 3], 0.1275)
        fixture.pour()
        returned = fixture.return_bowl()
        self.assertAlmostEqual(returned.metrics["return_reward"][0], 0.07)
        self.assertTrue(returned.scalar_success())
        self.assertAlmostEqual(returned.scalar_reward(), 1 - returned.metrics["force_penalty"].sum(), places=5)

    def test_smallest_and_largest_fruits_can_pack_and_pour(self):
        for radius in (*FRUIT_RADIUS_RANGE_M, 0.035):
            with self.subTest(radius=radius):
                fixture = PourFixture(radius=radius, force_allowance_n=1e6)
                packed = fixture.steps(800)
                self.assertAlmostEqual(packed.scalar_reward(), 0.2, places=5)
                self.assertTrue(packed.metrics["fruit_source_geometric_support_mask"].all())
                centers = fixture.data.geom_xpos[fixture.evaluator._fruit_geoms]
                source = fixture.data.body("source_bowl").xpos
                self.assertTrue(np.any(centers[:, 2] + radius > source[2] + fixture.source.height))
                self.assertTrue(fixture.transport().metrics["transport_complete"][0])
                result = fixture.pour()
                self.assertEqual(result.metrics["num_caught_fruits"][0], 3, result.metrics)
                self.assertTrue(fixture.return_bowl().scalar_success())

    def test_fruit_stacked_on_other_fruit_above_the_rim_can_qualify(self):
        fixture = PourFixture(radius=0.035, force_allowance_n=1e6)
        for i in range(1, 4):
            fixture.put_fruit(i, [0.6, -0.2, 0.75 + fixture.source.floor + 0.035 + (i - 1) * 0.070])
        packed = fixture.steps(800)
        self.assertTrue(packed.metrics["fruit_source_supported_mask"].all(), packed.metrics)
        self.assertTrue(packed.metrics["packing_stage_complete"][0])
        self.assertAlmostEqual(packed.scalar_reward(), 0.2)

    def test_rim_grasp_requires_opposing_fingers_and_expires_on_release(self):
        fixture = PourFixture()
        fixture.steps(600)
        fixture.grip()
        self.assertTrue(fixture.result().metrics["rim_held"][0])
        self.assertTrue(fixture.result().metrics["bowl_held"][0])
        fixture.gripping = False
        fixture.data.mocap_pos[fixture.hand] = [-2, 0, 2]
        fixture.steps(20)
        self.assertFalse(fixture.result().metrics["rim_held"][0])
        self.assertFalse(fixture.result().metrics["bowl_held"][0])
        # A single fingertip touching the rim cannot acquire a hold.
        fixture.data.mocap_pos[fixture.hand] = [0.6, -0.2, 0.75]
        second = fixture.model.body("left_link_right_finger").id
        fixture.model.body_pos[second, 0] = -0.30
        fixture.steps(100)
        self.assertFalse(fixture.result().metrics["rim_held"][0])
        self.assertFalse(fixture.result().metrics["bowl_held"][0])

    def test_grasp_acquisition_tolerates_brief_finger_contact_gaps(self):
        fixture = PourFixture()
        fixture.steps(600)
        fixture.gripping = True
        finger = fixture.model.body("left_link_right_finger").id
        position = fixture.model.body_pos[finger].copy()
        for step in range(150):
            # A 2 ms contact gap every 20 ms represents a changing contact manifold.
            fixture.model.body_pos[finger] = [-0.3, 0, 0.083] if step % 10 == 0 else position
            fixture.steps(1)
        self.assertTrue(fixture.result().metrics["bowl_held"][0])
        self.assertTrue(fixture.result().metrics["rim_held"][0])


class BowlMaterialTest(unittest.TestCase):
    @staticmethod
    def compile_bowl(spec):
        root = ET.Element("mujoco")
        assets, body = build_bowl("bowl", spec)
        ET.SubElement(root, "asset").extend(assets)
        ET.SubElement(root, "worldbody").append(body)
        return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))

    def test_mass_matches_polypropylene_volume_of_floor_walls_and_rim(self):
        for spec in (source_bowl_spec(), target_bowl_spec()):
            with self.subTest(spec=spec):
                model = self.compile_bowl(spec)
                self.assertEqual(spec.density, 900)
                angle = np.pi / BOWL_SEGMENTS
                floor_volume = np.pi * ((spec.bottom_radius + spec.wall) / np.cos(angle)) ** 2 * spec.floor
                wall_volume = (
                    BOWL_SEGMENTS
                    * np.tan(angle)
                    * (spec.height - spec.floor)
                    * (spec.wall * (spec.bottom_radius + spec.rim_radius) + spec.wall**2)
                )
                capsule_radius = spec.wall / 2
                capsule_length = 2 * (spec.rim_radius + capsule_radius) * np.tan(angle)
                rim_volume = BOWL_SEGMENTS * np.pi * (capsule_radius**2 * capsule_length + 4 / 3 * capsule_radius**3)
                expected = spec.density * (floor_volume + wall_volume + rim_volume)
                np.testing.assert_allclose(model.body("bowl").mass[0], expected, rtol=1e-6)
                self.assertTrue((model.body("bowl").inertia > 0).all())
                self.assertGreater(model.body("bowl").ipos[2], spec.floor)
                self.assertLess(model.body("bowl").ipos[2], spec.height)

    def test_mass_and_inertia_follow_density_and_size(self):
        spec = source_bowl_spec()
        reference = self.compile_bowl(spec).body("bowl")
        denser = self.compile_bowl(replace(spec, density=1800)).body("bowl")
        np.testing.assert_allclose(denser.mass, reference.mass * 2)
        np.testing.assert_allclose(denser.inertia, reference.inertia * 2)
        np.testing.assert_allclose(denser.ipos, reference.ipos, atol=1e-12)
        scale = 0.8
        smaller = self.compile_bowl(
            replace(
                spec,
                bottom_radius=spec.bottom_radius * scale,
                rim_radius=spec.rim_radius * scale,
                height=spec.height * scale,
                floor=spec.floor * scale,
                wall=spec.wall * scale,
            )
        ).body("bowl")
        np.testing.assert_allclose(smaller.mass, reference.mass * scale**3, rtol=1e-6)
        # Thin float32 wall meshes incur compiler roundoff in their inertia frames.
        np.testing.assert_allclose(smaller.inertia, reference.inertia * scale**5, rtol=1e-4)
        np.testing.assert_allclose(smaller.ipos, reference.ipos * scale, atol=1e-8)
        for density in (0, -900, np.nan):
            with self.assertRaises(ValueError):
                replace(spec, density=density)


class PourSceneTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = abc_sim.make_env(task="pack_and_pour_fruit", render_cameras=False)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_seeded_colors_fruits_poses_and_prompt_match(self):
        colors, categories, layouts, prompts = set(), set(), set(), set()
        self.assertEqual(tuple(BOWL_COLORS), FRUIT_BOWL_COLOR_NAMES)
        for seed in range(32):
            self.env.reset(seed=seed)
            state = self.env._last_randomization
            meta = state.metadata
            colors.update((meta["source_color"], meta["target_color"]))
            categories.update(item["category"] for item in meta["fruits"])
            layouts.add(tuple(self.env.data.qpos))
            prompts.add(meta["prompt_index"])
            self.assertNotEqual(meta["source_color"], meta["target_color"])
            self.assertIn(f"{meta['source_color']} bowl", self.env.prompt)
            self.assertIn(f"large {meta['target_color']} bowl", self.env.prompt)
            self.assertIn(f"empty {meta['source_color']} bowl", self.env.prompt)
            self.assertIn("table", self.env.prompt)
            self.assertEqual(get_task_spec(self.env.prompt).name, "pack_and_pour_fruit")
            self.assertEqual(len(meta["fruits"]), 3)
            for i, fruit in enumerate(meta["fruits"], 1):
                radius = fruit["radius"]
                self.assertGreaterEqual(radius, FRUIT_RADIUS_RANGE_M[0])
                self.assertLessEqual(radius, FRUIT_RADIUS_RANGE_M[1])
                mass = FRUIT_DENSITY_KG_M3 * 4 * np.pi * radius**3 / 3
                self.assertAlmostEqual(self.env.model.body(f"fruit_{i}").mass[0], mass)
                np.testing.assert_allclose(self.env.model.body(f"fruit_{i}").inertia, 2 * mass * radius**2 / 5)
            self.assertEqual(len(state.object_states), 5)
            self.assertTrue(self.env._task_randomizer._pairwise_ok(state.object_states))
            self.assertTrue(self.env._task_randomizer._contacts_ok(self.env.model, self.env.data))
            self.assertEqual(self.env.evaluate_task().scalar_reward(), 0)
        self.assertEqual(colors, set(BOWL_COLORS))
        self.assertEqual(categories, set(FRUIT_CATEGORIES))
        self.assertEqual(len(layouts), 32)
        self.assertGreaterEqual(len(prompts), 5)

    def test_static_scene_and_randomization_use_the_same_plastic_bowls(self):
        scene = Path(abc_sim.__file__).parent / "models" / "yam_pack_and_pour_fruit_scene.xml"
        baseline = mujoco.MjModel.from_xml_path(str(scene))
        self.env.reset(seed=6)
        for name, spec in (("source_bowl", source_bowl_spec()), ("target_bowl", target_bowl_spec())):
            np.testing.assert_allclose(baseline.body(name).mass, self.env.model.body(name).mass)
            np.testing.assert_allclose(baseline.body(name).inertia, self.env.model.body(name).inertia)
            np.testing.assert_allclose(self.env.model.site(f"{name}_cavity_rim").pos, [0, 0, spec.height])
            self.assertEqual(self.env.model.site(f"{name}_cavity_rim").size[0], spec.rim_radius)
        self.assertAlmostEqual(self.env.model.body("source_bowl").mass[0], 0.079662, places=5)
        self.assertAlmostEqual(self.env.model.body("target_bowl").mass[0], 0.330931, places=5)

    def test_replay_and_independent_instances_are_identical(self):
        self.env.reset(seed=9)
        qpos, vertices, texture = (
            self.env.data.qpos.copy(),
            self.env.model.mesh_vert.copy(),
            self.env.model.tex_data.copy(),
        )
        state = json.loads(json.dumps(self.env._last_randomization.to_dict()))
        self.env.reset(seed=4)
        self.env._task_randomizer.apply(self.env.model, self.env.data, RandomizationState.from_dict(state))
        np.testing.assert_array_equal(self.env.data.qpos, qpos)
        np.testing.assert_array_equal(self.env.model.mesh_vert, vertices)
        np.testing.assert_array_equal(self.env.model.tex_data, texture)
        self.assertEqual(self.env._last_randomization.to_dict(), state)
        other = abc_sim.make_env(task="pack_and_pour_fruit", render_cameras=False)
        try:
            other.reset(seed=9)
            np.testing.assert_array_equal(other.data.qpos, qpos)
            self.assertEqual(other.prompt, self.env.prompt)
        finally:
            other.close()

    def test_invalid_options_and_replay_do_not_mutate_the_scene(self):
        self.env.reset(seed=2)
        for options in (
            {"unknown": 1},
            {"fruits": []},
            {"source_color": "bad"},
            {"source_color": "blue", "target_color": "blue"},
            {"prompt_index": 6},
        ):
            with self.assertRaises(ValueError):
                self.env.reset(seed=0, options={"randomization": options})
        self.env.reset(seed=2)
        state = copy.deepcopy(self.env._last_randomization)
        state.metadata["prompt"] = "wrong colors"
        model, qpos = self.env.model, self.env.data.qpos.copy()
        with self.assertRaises(ValueError):
            self.env._task_randomizer.apply(model, self.env.data, state)
        self.assertIs(self.env.model, model)
        np.testing.assert_array_equal(self.env.data.qpos, qpos)

    def test_all_fruit_visuals_fit_their_spherical_collision_bounds(self):
        count = 0
        for category in FRUIT_CATEGORIES:
            for variant in fruit_variants(category):
                count += 1
                assets, body = build_fruit(FruitSpec(category, variant), 1)
                root = ET.Element("mujoco")
                ET.SubElement(root, "asset").extend(assets)
                ET.SubElement(root, "worldbody").append(body)
                model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
                data = mujoco.MjData(model)
                mujoco.mj_forward(model, data)
                center = data.geom("fruit_1_collision").xpos
                for g in range(model.ngeom):
                    if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                        mesh = model.geom_dataid[g]
                        start, n = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
                        vertices = (
                            model.mesh_vert[start : start + n] @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
                        )
                        self.assertLessEqual(np.linalg.norm(vertices - center, axis=1).max(), 0.035001)
        self.assertGreaterEqual(count, 12)


if __name__ == "__main__":
    unittest.main()
