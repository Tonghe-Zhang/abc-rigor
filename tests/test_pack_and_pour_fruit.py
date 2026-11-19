"""Seeded fruit scenes and chronological, contact-backed packing and pouring."""

from __future__ import annotations

import copy
import json
import unittest
import xml.etree.ElementTree as ET

import abc_sim
import mujoco
import numpy as np
from abc_sim.randomization.assets.fruit_bowls import (
    BOWL_COLORS,
    FRUIT_CATEGORIES,
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
        root = ET.Element("mujoco")
        ET.SubElement(root, "compiler", angle="radian")
        ET.SubElement(root, "option", timestep=".002", integrator="implicitfast")
        asset, world = ET.SubElement(root, "asset"), ET.SubElement(root, "worldbody")
        ET.SubElement(world, "geom", name="table_plane", type="plane", size="1 1 .1", pos="0 0 .75")
        for name, spec, pos in (
            ("source_bowl", source_bowl_spec(), ".6 -.2 .75"),
            ("target_bowl", target_bowl_spec(), ".6 .2 .75"),
        ):
            assets, body = build_bowl(name, spec)
            body.set("pos", pos)
            asset.extend(assets)
            world.append(body)
        ET.SubElement(world, "body", name="carrier", mocap="true", pos=".6 -.2 .75")
        for side in ("left", "right"):
            arm = ET.SubElement(world, "body", name=f"{side}_arm", mocap="true", pos="-2 0 2")
            for suffix, x in (("left", -0.09962), ("right", -0.11638)):
                finger = ET.SubElement(arm, "body", name=f"{side}_link_{suffix}_finger", pos=f"{x} 0 .083")
                ET.SubElement(
                    finger, "geom", type="sphere", size=".006", friction=".6 .005 .0005", solref=".08 1", priority="2"
                )
        for i in range(1, 4):
            angle = (i - 1) * 2 * np.pi / 3
            body = ET.SubElement(
                world,
                "body",
                name=f"fruit_{i}",
                pos=f"{0.6 + 0.031 * np.cos(angle)} {-0.2 + 0.031 * np.sin(angle)} {0.756 + radius}",
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

    def put_fruit(self, i, xyz):
        self.data.joint(f"fruit_{i}_joint").qpos[:] = [*xyz, 1, 0, 0, 0]
        self.data.joint(f"fruit_{i}_joint").qvel[:] = 0


class PourPhysicsTest(unittest.TestCase):
    def test_physical_pack_transport_and_gravity_pour(self):
        fixture = PourFixture()
        packed = fixture.steps(600)
        self.assertAlmostEqual(packed.scalar_reward(), 0.3, places=5)
        carried = fixture.transport()
        self.assertTrue(carried.metrics["transport_complete"][0], carried.metrics)
        self.assertAlmostEqual(carried.scalar_reward(), 0.6, places=5)
        poured = fixture.pour()
        self.assertTrue(poured.scalar_success(), poured.metrics)
        self.assertGreater(poured.scalar_reward(), 0.98)
        self.assertAlmostEqual(poured.scalar_reward(), 1 - poured.metrics["force_penalty"].sum(), places=5)

    def test_direct_placement_in_the_large_bowl_earns_zero(self):
        fixture = PourFixture()
        for i in range(1, 4):
            fixture.put_fruit(i, [0.6, 0.2 + (i - 2) * 0.06, 0.780])
        result = fixture.steps(800)
        self.assertTrue(result.metrics["fruit_in_target_mask"].all())
        self.assertEqual(result.scalar_reward(), 0)
        self.assertFalse(result.scalar_success())

    def test_nesting_a_packed_bowl_without_pouring_earns_only_six_tenths(self):
        # Isolate stage weights; the default force deduction is tested separately.
        fixture = PourFixture(force_allowance_n=1e6)
        fixture.steps(600)
        fixture.grip()
        # Cross above the destination in motion and lower directly into it.
        # There is no stationary overhead dwell on this path.
        for i in range(1, 1001):
            t = i / 1000
            fixture.data.mocap_pos[fixture.carrier] = [
                0.6,
                -0.2 + 0.4 * t,
                0.75 + 0.0062 * t + 0.23 * np.sin(np.pi * t),
            ]
            fixture.steps(1)
        self.assertFalse(fixture.result().metrics["transport_complete"][0])
        fixture.gripping = False
        fixture.data.mocap_pos[fixture.hand] = [-2, 0, 2]
        result = fixture.steps(1000)
        self.assertTrue(result.metrics["source_nested_in_target"][0], result.metrics)
        self.assertTrue(result.metrics["transport_complete"][0])
        self.assertTrue(result.metrics["fruit_in_source_mask"].all())
        self.assertAlmostEqual(result.scalar_reward(), 0.6, places=4)
        self.assertEqual(result.metrics["emptying_reward"][0], 0)
        self.assertEqual(result.metrics["catching_reward"][0], 0)
        self.assertFalse(result.scalar_success())

    def test_two_packed_fruits_receive_partial_credit_but_cannot_unlock_transport(self):
        fixture = PourFixture()
        fixture.put_fruit(3, [0.4, -0.4, 0.774])
        result = fixture.steps(600)
        self.assertAlmostEqual(result.scalar_reward(), 0.2, places=5)
        self.assertFalse(result.metrics["packing_stage_complete"][0])
        result = fixture.transport()
        np.testing.assert_array_equal(result.metrics["fruit_in_source_mask"], [[True, True, False]])
        self.assertAlmostEqual(result.scalar_reward(), 0.2, places=5)
        self.assertFalse(result.metrics["transport_complete"][0])
        self.assertEqual(result.metrics["emptying_reward"][0], 0)
        self.assertEqual(result.metrics["catching_reward"][0], 0)

    def test_partial_packing_credit_clears_on_exit_and_reset(self):
        fixture = PourFixture()
        fixture.put_fruit(3, [0.4, -0.4, 0.774])
        self.assertAlmostEqual(fixture.steps(600).scalar_reward(), 0.2, places=5)
        fixture.put_fruit(1, [0.4, -0.2, 0.774])
        result = fixture.steps(1)
        self.assertAlmostEqual(result.scalar_reward(), 0.1, places=5)
        np.testing.assert_array_equal(result.metrics["fruit_packed_mask"], [[False, True, False]])
        fixture.evaluator.reset()
        fixture.evaluator.after_reset(fixture.model, fixture.data)
        self.assertEqual(fixture.result().scalar_reward(), 0)
        self.assertFalse(fixture.result().metrics["fruit_packed_mask"].any())

    def test_fruits_can_qualify_at_different_times_before_pouring(self):
        fixture = PourFixture()
        third_position = fixture.data.joint("fruit_3_joint").qpos[:3].copy()
        fixture.put_fruit(3, [0.4, -0.4, 0.774])
        self.assertAlmostEqual(fixture.steps(600).scalar_reward(), 0.2, places=5)
        fixture.put_fruit(3, third_position)
        for _ in range(600):
            # The first fruit rolls in place while the last fruit settles.
            fixture.data.joint("fruit_1_joint").qvel[5] = 5
            result = fixture.steps(1)
        self.assertTrue(result.metrics["packing_stage_complete"][0], result.metrics)
        self.assertAlmostEqual(result.scalar_reward(), 0.3, places=5)
        self.assertTrue(result.metrics["fruit_armed_mask"].all())
        self.assertFalse(result.metrics["fruit_source_qualified_mask"][0, 0])
        fixture.transport()
        self.assertTrue(fixture.pour().scalar_success())

    def test_retained_credit_does_not_relax_full_packing_containment(self):
        fixture = PourFixture()
        third_position = fixture.data.joint("fruit_3_joint").qpos[:3].copy()
        fixture.put_fruit(3, [0.4, -0.4, 0.774])
        self.assertAlmostEqual(fixture.steps(600).scalar_reward(), 0.2, places=5)
        fixture.put_fruit(3, third_position)
        for _ in range(600):
            # Its center is inside, but the top protrudes above the bowl's mouth.
            fixture.put_fruit(1, [0.6, -0.2, 0.820])
            result = fixture.steps(1)
        self.assertTrue(result.metrics["fruit_packed_mask"].all())
        self.assertFalse(result.metrics["fruit_in_source_mask"][0, 0])
        self.assertFalse(result.metrics["packing_stage_complete"][0])
        result = fixture.steps(800)
        self.assertTrue(result.metrics["fruit_in_source_mask"].all())
        self.assertTrue(result.metrics["packing_stage_complete"][0])

    def test_pouring_before_transport_cannot_earn_the_last_four_tenths(self):
        fixture = PourFixture()
        fixture.steps(600)
        fixture.grip()
        fixture.move(np.array([0.6, -0.2, 0.98]), tilt=np.deg2rad(110))
        result = fixture.steps(800)
        self.assertFalse(result.metrics["fruit_in_source_mask"].any())
        self.assertFalse(result.metrics["transport_complete"][0])
        self.assertEqual(result.metrics["emptying_reward"][0], 0)
        self.assertEqual(result.metrics["catching_reward"][0], 0)

    def test_hand_transfer_after_transport_does_not_count_as_pouring(self):
        fixture = PourFixture()
        fixture.transport()
        # Touch each fruit with the parked second hand before transferring it.
        right = fixture.model.body("right_arm").mocapid[0]
        for i in range(1, 4):
            xyz = fixture.data.joint(f"fruit_{i}_joint").qpos[:3].copy()
            fixture.data.mocap_pos[right] = xyz + [0.09962, 0, -0.083]
            fixture.steps(1)
            fixture.data.mocap_pos[right] = [-2, 0, 2]
            fixture.put_fruit(i, [0.6, 0.2 + (i - 2) * 0.06, 0.780])
            fixture.steps(1)
        result = fixture.steps(800)
        self.assertTrue(result.metrics["fruit_in_target_mask"].all())
        self.assertEqual(result.metrics["emptying_reward"][0], 0)
        self.assertEqual(result.metrics["catching_reward"][0], 0)

    def test_landing_credit_is_per_fruit_and_revoked_when_one_leaves(self):
        fixture = PourFixture()
        fixture.transport()
        completed = fixture.pour()
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
        self.assertAlmostEqual(packed.scalar_reward(), 0.3, places=5)
        fixture.data.xfrc_applied[:] = 0
        carried = fixture.transport()
        self.assertTrue(carried.metrics["transport_complete"][0])
        self.assertAlmostEqual(carried.scalar_reward(), 0.51, places=5)
        self.assertAlmostEqual(carried.metrics["force_penalty"][0, 3], 0.12)

    def test_smallest_and_largest_fruits_can_pack_and_pour(self):
        for radius in (0.021, 0.025):
            with self.subTest(radius=radius):
                fixture = PourFixture(radius=radius)
                self.assertTrue(fixture.transport().metrics["transport_complete"][0])
                result = fixture.pour()
                self.assertTrue(result.scalar_success(), result.metrics)

    def test_rim_grasp_requires_opposing_fingers_and_expires_on_release(self):
        fixture = PourFixture()
        fixture.steps(600)
        fixture.grip()
        self.assertTrue(fixture.result().metrics["rim_held"][0])
        fixture.gripping = False
        fixture.data.mocap_pos[fixture.hand] = [-2, 0, 2]
        fixture.steps(20)
        self.assertFalse(fixture.result().metrics["rim_held"][0])
        # A single fingertip touching the rim cannot acquire a hold.
        fixture.data.mocap_pos[fixture.hand] = [0.6, -0.2, 0.75]
        second = fixture.model.body("left_link_right_finger").id
        fixture.model.body_pos[second, 0] = -0.30
        fixture.steps(100)
        self.assertFalse(fixture.result().metrics["rim_held"][0])


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
            self.assertEqual(get_task_spec(self.env.prompt).name, "pack_and_pour_fruit")
            self.assertEqual(len(meta["fruits"]), 3)
            self.assertEqual(len(state.object_states), 5)
            self.assertTrue(self.env._task_randomizer._pairwise_ok(state.object_states))
            self.assertTrue(self.env._task_randomizer._contacts_ok(self.env.model, self.env.data))
            self.assertEqual(self.env.evaluate_task().scalar_reward(), 0)
        self.assertEqual(colors, set(BOWL_COLORS))
        self.assertEqual(categories, set(FRUIT_CATEGORIES))
        self.assertEqual(len(layouts), 32)
        self.assertGreaterEqual(len(prompts), 5)

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
                        self.assertLessEqual(np.linalg.norm(vertices - center, axis=1).max(), 0.024001)
        self.assertGreaterEqual(count, 12)


if __name__ == "__main__":
    unittest.main()
