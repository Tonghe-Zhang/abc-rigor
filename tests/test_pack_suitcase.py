"""Suitcase geometry, seeded scenes, physical packing, closure and replay."""

from __future__ import annotations

import copy
import json
import unittest
import xml.etree.ElementTree as ET
from itertools import product

import abc_sim
import mujoco
import numpy as np
from abc_sim.randomization.assets.suitcase import (
    CASE_FLOOR_Z,
    CASE_HALF_XY,
    CASE_INNER_XY,
    CASE_RIM_Z,
    HANDLE_STYLES,
    LID_LIMIT_RAD,
    PACKING_CATEGORIES,
    SHELL_DETAILS,
    SHELL_PALETTES,
    SHELL_PATTERNS,
    STRAP_LAYOUTS,
    SuitcaseStyle,
    SuitcaseTexture,
    build_packing_object,
    build_suitcase,
    packing_object_dimensions,
    packing_variants,
)
from abc_sim.randomization.assets.tabletop_distractors import DistractorSpec
from abc_sim.randomization.core import RandomizationState
from abc_sim.task_eval.pack_suitcase import PackSuitcaseEvaluator
from abc_sim.task_specs import PACK_SUITCASE_PROMPTS, get_task_spec


class SuitcaseSceneTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = abc_sim.make_env(task="pack_suitcase", render_cameras=False)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def setUp(self):
        self.env.reset(seed=0)

    def test_seeded_scenes_have_three_objects_open_lid_and_diverse_textures(self):
        textures, prompts, categories, layouts = set(), set(), set(), set()
        handles, straps, details, widths, walls = set(), set(), set(), set(), set()
        for seed in range(32):
            self.env.reset(seed=seed)
            state = self.env._last_randomization
            textures.add(tuple(state.metadata["texture"].values()))
            prompts.add(self.env.prompt)
            categories.update(obj["category"] for obj in state.metadata["objects"])
            style = state.metadata["style"]
            handles.add(style["handle"])
            straps.add(style["straps"])
            details.add(style["detail"])
            widths.add(style["width"])
            walls.add(style["wall_thickness"])
            layouts.add(tuple(self.env.data.qpos))
            free = [
                self.env.model.joint(i).name
                for i in range(self.env.model.njnt)
                if self.env.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE
            ]
            self.assertEqual(set(free), {"suitcase_joint", *(f"packing_object_{i}_joint" for i in range(1, 4))})
            self.assertTrue(self.env._task_randomizer._pairwise_ok(state.object_states))
            self.assertTrue(self.env._task_randomizer._contacts_ok(self.env.model, self.env.data))
            self.assertGreaterEqual(np.rad2deg(self.env.data.joint("suitcase_hinge").qpos[0]), 105)
            self.assertLessEqual(np.rad2deg(self.env.data.joint("suitcase_hinge").qpos[0]), 110)
            result = self.env.evaluate_task()
            self.assertEqual(result.metrics["packing_reward"][0], 0)
            self.assertEqual(result.metrics["release_reward"][0], 0)
            self.assertLessEqual(result.scalar_reward(), 0.3 * 5 / 107)
            self.assertEqual(get_task_spec(self.env.prompt).name, "pack_suitcase")
        self.assertGreaterEqual(len(textures), 16)
        self.assertEqual(categories, set(PACKING_CATEGORIES))
        self.assertEqual(prompts, set(PACK_SUITCASE_PROMPTS))
        self.assertEqual(len(layouts), 32)
        self.assertEqual(handles, set(HANDLE_STYLES))
        self.assertEqual(straps, set(STRAP_LAYOUTS))
        self.assertEqual(details, set(SHELL_DETAILS))
        self.assertEqual(len(widths), 32)
        self.assertEqual(len(walls), 32)

    def test_same_seed_and_serialized_replay_restore_geometry_texture_pose_and_prompt(self):
        self.env.reset(seed=9)
        qpos = self.env.data.qpos.copy()
        textures = self.env.model.tex_data.copy()
        geom_sizes = self.env.model.geom_size.copy()
        vertices = self.env.model.mesh_vert.copy()
        saved = json.loads(json.dumps(self.env._last_randomization.to_dict()))
        self.env.reset(seed=4)
        self.env._task_randomizer.apply(self.env.model, self.env.data, RandomizationState.from_dict(saved))
        np.testing.assert_array_equal(self.env.data.qpos, qpos)
        np.testing.assert_array_equal(self.env.model.tex_data, textures)
        np.testing.assert_array_equal(self.env.model.geom_size, geom_sizes)
        np.testing.assert_array_equal(self.env.model.mesh_vert, vertices)
        self.assertEqual(self.env._last_randomization.to_dict(), saved)
        other = abc_sim.make_env(task="pack_suitcase", render_cameras=False)
        try:
            other.reset(seed=9)
            np.testing.assert_array_equal(other.data.qpos, qpos)
            np.testing.assert_array_equal(other.model.tex_data, textures)
            self.assertEqual(other.prompt, self.env.prompt)
        finally:
            other.close()

    def test_bad_replay_and_reset_options_fail_before_scene_mutation(self):
        saved = self.env._last_randomization.to_dict()
        for key, value in (
            ("objects", []),
            ("texture", {"palette": "bad", "pattern": "solid"}),
            ("lid_angle_deg", 111),
            ("style", {"wall_thickness": 0.04}),
        ):
            state = RandomizationState.from_dict(copy.deepcopy(saved))
            state.metadata[key] = value
            qpos, model = self.env.data.qpos.copy(), self.env.model
            with self.assertRaises(ValueError):
                self.env._task_randomizer.apply(model, self.env.data, state)
            self.assertIs(self.env.model, model)
            np.testing.assert_array_equal(self.env.data.qpos, qpos)
        for options in ({"unknown": 1}, {"objects": []}, {"lid_angle_deg": float("nan")}, {"lid_angle_deg": 0}):
            with self.assertRaises(ValueError):
                self.env.reset(seed=0, options={"randomization": options})

    def test_open_lid_stays_open_and_limit_is_passive(self):
        self.env.advance_physics(1500)
        model, data = self.env.model, self.env.data
        joint = model.joint("suitcase_hinge").id
        np.testing.assert_allclose(model.jnt_range[joint], [0, LID_LIMIT_RAD])
        self.assertNotIn(joint, model.actuator_trnid[:, 0])
        self.assertGreater(data.joint("suitcase_hinge").qpos[0], np.deg2rad(104))
        self.assertLess(data.joint("suitcase_hinge").qpos[0], LID_LIMIT_RAD + 1e-4)
        self.assertAlmostEqual(data.joint("suitcase_joint").qpos[2], 0.75, places=3)
        result = self.env.evaluate_task()
        self.assertEqual(result.metrics["packing_reward"][0], 0)
        self.assertEqual(result.metrics["release_reward"][0], 0)
        self.assertLessEqual(result.scalar_reward(), 0.3 * 6 / 107)

    def test_all_object_variants_compile_settle_and_fit_three_across(self):
        count = 0
        for category in PACKING_CATEGORIES:
            for variant in packing_variants(category):
                count += 1
                spec = DistractorSpec(category, variant)
                _, dimensions = packing_object_dimensions(spec)
                self.assertLessEqual(max(dimensions), 0.08000001)
                self.assertLess(dimensions[2], CASE_RIM_Z - CASE_FLOOR_Z - 0.01)
                self.assertLess(3 * dimensions[1] + 0.03, 2 * CASE_INNER_XY[1])
                assets, body = build_packing_object(spec, 1)
                root = ET.Element("mujoco")
                ET.SubElement(root, "option", integrator="implicitfast")
                ET.SubElement(root, "asset").extend(assets)
                world = ET.SubElement(root, "worldbody")
                ET.SubElement(world, "geom", type="plane", size="1 1 0.1", pos="0 0 0.75")
                world.append(body)
                model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
                data = mujoco.MjData(model)
                mujoco.mj_step(model, data, nstep=500)
                self.assertTrue(np.isfinite(data.qpos).all())
                self.assertAlmostEqual(data.qpos[2], 0.75, delta=0.002)
                self.assertLess(np.linalg.norm(data.qvel), 0.05)
        self.assertGreater(count, 20)

    def test_every_texture_compiles(self):
        signatures = set()
        for palette in SHELL_PALETTES:
            for pattern in SHELL_PATTERNS:
                assets, body = build_suitcase(SuitcaseTexture(palette, pattern))
                root = ET.Element("mujoco")
                ET.SubElement(root, "compiler", angle="radian")
                ET.SubElement(root, "asset").extend(assets)
                ET.SubElement(root, "worldbody").append(body)
                model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
                signatures.add((model.tex_data.tobytes(), model.mesh_texcoord.tobytes()))
        self.assertEqual(len(signatures), len(SHELL_PALETTES) * len(SHELL_PATTERNS))

    def test_all_construction_styles_fit_three_maximum_size_objects_and_close(self):
        for handle, straps, detail in product(HANDLE_STYLES, STRAP_LAYOUTS, SHELL_DETAILS):
            with self.subTest(handle=handle, straps=straps, detail=detail):
                style = SuitcaseStyle(
                    width=0.19,
                    length=0.29,
                    height=0.08,
                    wall_thickness=0.012,
                    floor_thickness=0.014,
                    lid_thickness=0.020,
                    handle=handle,
                    straps=straps,
                    detail=detail,
                )
                root = ET.Element("mujoco")
                ET.SubElement(root, "compiler", angle="radian")
                ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast")
                asset, world = ET.SubElement(root, "asset"), ET.SubElement(root, "worldbody")
                ET.SubElement(world, "geom", name="table_plane", type="plane", size="1 1 0.1", pos="0 0 0.75")
                for side in ("left", "right"):
                    body = ET.SubElement(world, "body", name=f"{side}_arm", pos="-2 0 2")
                    ET.SubElement(body, "geom", type="sphere", size="0.01")
                assets, case = build_suitcase(SuitcaseTexture("teal", "checker"), style)
                asset.extend(assets)
                world.append(case)
                for i in range(1, 4):
                    body = ET.SubElement(
                        world,
                        "body",
                        name=f"packing_object_{i}",
                        pos=f"0.62 {(i - 2) * 0.087} {0.75 + style.floor_thickness + 0.0275}",
                    )
                    ET.SubElement(body, "freejoint", name=f"packing_object_{i}_joint")
                    ET.SubElement(body, "geom", type="box", size="0.04 0.04 0.0275", mass="0.02")
                model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
                data = mujoco.MjData(model)
                mujoco.mj_forward(model, data)
                evaluator = PackSuitcaseEvaluator(model=model, spec=get_task_spec("pack_suitcase"))
                evaluator.after_reset(model, data)
                for _ in range(600):
                    mujoco.mj_step(model, data)
                    evaluator.after_physics_step(model, data)
                result = evaluator.evaluate_model_data(model, data)
                self.assertTrue(result.scalar_success())
                self.assertAlmostEqual(result.scalar_reward(), 1)
                self.assertEqual(
                    len([g for g in range(model.ngeom) if model.geom(g).name.startswith("suitcase_lid_strap")]),
                    {"none": 0, "single": 1, "double": 2, "cross": 2}[straps],
                )
                data.joint("packing_object_2_joint").qpos[0] += 0.045
                mujoco.mj_forward(model, data)
                mujoco.mj_step(model, data)
                evaluator.after_physics_step(model, data)
                self.assertFalse(evaluator.evaluate_model_data(model, data).metrics["object_contained_mask"][0, 1])


class SuitcaseRewardTest(unittest.TestCase):
    """Use real contacts with movable probe fingers for release regressions."""

    def setUp(self):
        root = ET.Element("mujoco")
        ET.SubElement(root, "compiler", angle="radian")
        ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast")
        asset = ET.SubElement(root, "asset")
        world = ET.SubElement(root, "worldbody")
        ET.SubElement(world, "geom", name="table_plane", type="box", pos="0.6 0 0.745", size="0.3 0.65 0.005")
        for side in ("left", "right"):
            robot = ET.SubElement(world, "body", name=f"{side}_arm", mocap="true", pos="-2 0 2")
            ET.SubElement(robot, "geom", type="sphere", size="0.012")
        assets, case = build_suitcase(SuitcaseTexture("navy", "solid"))
        case.set("pos", "0.62 0 0.75")
        asset.extend(assets)
        world.append(case)
        for i in range(1, 4):
            assets, body = build_packing_object(DistractorSpec("bar_soap", packing_variants("bar_soap")[0]), i)
            body.set("pos", f"0.4 {(i - 2) * 0.2} 0.75")
            asset.extend(assets)
            world.append(body)
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        self.data = mujoco.MjData(self.model)
        self.data.joint("suitcase_hinge").qpos[0] = np.deg2rad(107)
        mujoco.mj_forward(self.model, self.data)
        self.evaluator = PackSuitcaseEvaluator(model=self.model, spec=get_task_spec("pack_suitcase"))
        self.evaluator.after_reset(self.model, self.data)

    def steps(self, count):
        for _ in range(count):
            mujoco.mj_step(self.model, self.data)
            self.evaluator.after_physics_step(self.model, self.data)
        return self.result()

    def result(self):
        return self.evaluator.evaluate_model_data(self.model, self.data)

    def pack(self, count=3, close=False):
        for i in range(1, count + 1):
            self.data.joint(f"packing_object_{i}_joint").qpos[:] = [0.62, (i - 2) * 0.09, 0.766, 1, 0, 0, 0]
        if close:
            self.data.joint("suitcase_hinge").qpos[0] = 0
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def observe_angle(self, angle_deg, ticks=1, hinge_speed=0):
        """Hold a controlled pose and solve contacts at each observed tick."""
        for _ in range(ticks):
            self.data.joint("suitcase_hinge").qpos[0] = np.deg2rad(angle_deg)
            self.data.qvel[:] = 0
            self.data.joint("suitcase_hinge").qvel[0] = hinge_speed
            self.data.time += self.model.opt.timestep
            mujoco.mj_forward(self.model, self.data)
            self.evaluator.after_physics_step(self.model, self.data)
        return self.result()

    def test_angle_credit_is_linear_from_native_open_limit_to_three_degrees(self):
        self.pack(close=True)
        self.steps(700)
        for angle, fraction in ((110, 0), (56.5, 0.5), (3.1, 106.9 / 107), (3, 1), (2.9, 1), (0, 1)):
            with self.subTest(angle=angle):
                result = self.observe_angle(angle, hinge_speed=0.2)
                self.assertTrue(result.metrics["suitcase_supported_and_settled"][0])
                self.assertAlmostEqual(result.metrics["lid_closure_fraction"][0], fraction)
                multiplier = 1 - result.metrics["force_penalty"][0, 3] / 0.4
                self.assertAlmostEqual(result.metrics["lid_reward"][0], 0.3 * fraction * multiplier)
                self.assertEqual(result.metrics["release_reward"][0], 0)
                self.assertFalse(result.scalar_success())
        for angle in (3, 56.5, 110, 111):
            positive = self.observe_angle(angle).metrics["lid_closure_fraction"][0]
            negative = self.observe_angle(-angle).metrics["lid_closure_fraction"][0]
            self.assertAlmostEqual(positive, negative)

    def test_angle_credit_survives_robot_contact_but_release_requires_closed_dwell(self):
        self.pack(close=True)
        self.assertTrue(self.steps(700).scalar_success())
        self.data.mocap_pos[0] = self.data.geom_xpos[self.model.geom("suitcase_front").id]
        for angle in (110, 56.5, 0):
            result = self.observe_angle(angle)
            self.assertTrue(result.metrics["suitcase_touching_robot"][0])
            self.assertTrue(result.metrics["suitcase_supported_and_settled"][0])
            fraction = result.metrics["lid_closure_fraction"][0]
            multiplier = 1 - result.metrics["force_penalty"][0, 3] / 0.4
            self.assertAlmostEqual(result.metrics["lid_reward"][0], 0.3 * fraction * multiplier)
            self.assertEqual(result.metrics["release_reward"][0], 0)
            self.assertFalse(result.scalar_success())
        self.data.mocap_pos[0] = [-2, 0, 2]
        self.assertEqual(self.observe_angle(110, ticks=300).metrics["release_reward"][0], 0)
        self.assertEqual(self.observe_angle(0, ticks=200).metrics["release_reward"][0], 0)
        self.assertEqual(self.observe_angle(20).metrics["release_reward"][0], 0)
        self.assertEqual(self.observe_angle(0, ticks=200).metrics["release_reward"][0], 0)
        for _ in range(100):
            self.assertEqual(self.result().metrics["release_reward"][0], 0)
        result = self.observe_angle(0, ticks=100)
        self.assertAlmostEqual(result.metrics["release_reward"][0], 0.1 - 0.25 * result.metrics["force_penalty"][0, 3])
        self.assertTrue(result.scalar_success())
        self.data.mocap_pos[0] = self.data.geom_xpos[self.model.geom("suitcase_front").id]
        self.assertEqual(self.observe_angle(0).metrics["release_reward"][0], 0)
        self.data.mocap_pos[0] = [-2, 0, 2]
        self.assertEqual(self.observe_angle(0, ticks=200).metrics["release_reward"][0], 0)
        self.evaluator.reset()
        self.evaluator.after_reset(self.model, self.data)
        self.assertEqual(self.result().metrics["release_reward"][0], 0)
        np.testing.assert_array_equal(self.result().metrics["force_dose_ns"], 0)

    def test_partial_credit_then_full_packed_closed_success(self):
        for count in range(4):
            self.pack(count)
            result = self.steps(600)
            self.assertAlmostEqual(result.metrics["packing_reward"][0], count * 0.1)
            self.assertAlmostEqual(result.scalar_reward(), count * 0.1 + result.metrics["lid_reward"][0])
            self.assertEqual(result.metrics["release_reward"][0], 0)
            self.assertFalse(self.result().scalar_success())
        self.data.joint("suitcase_hinge").qpos[0] = 0
        result = self.steps(600)
        self.assertAlmostEqual(result.scalar_reward(), 1 - result.metrics["force_penalty"].sum())
        self.assertGreater(result.scalar_reward(), 0.99)
        self.assertTrue(self.result().scalar_success())

    def test_closure_and_staying_credit_are_independent_of_packing_count(self):
        for count in range(4):
            with self.subTest(count=count):
                self.setUp()
                self.pack(count, close=True)
                result = self.steps(700)
                penalty = result.metrics["force_penalty"][0]
                self.assertAlmostEqual(result.metrics["packing_reward"][0], count * 0.1 - penalty[:count].sum())
                self.assertAlmostEqual(result.metrics["lid_reward"][0], 0.3 - 0.75 * penalty[3])
                self.assertAlmostEqual(result.metrics["release_reward"][0], 0.1 - 0.25 * penalty[3])
                self.assertAlmostEqual(result.metrics["suitcase_staying_reward"][0], 0.3)
                self.assertAlmostEqual(result.scalar_reward(), 0.7 + count * 0.1 - penalty[:count].sum() - penalty[3])
                self.assertEqual(result.scalar_success(), count == 3)

    def test_losing_packing_credit_keeps_a_properly_closed_cases_credit(self):
        self.pack(close=True)
        self.assertTrue(self.steps(700).scalar_success())
        self.data.joint("packing_object_1_joint").qpos[:] = [0.4, -0.2, 0.77, 1, 0, 0, 0]
        result = self.steps(1)
        self.assertAlmostEqual(result.metrics["packing_reward"][0], 0.2 - result.metrics["force_penalty"][0, 1:3].sum())
        self.assertAlmostEqual(result.metrics["lid_reward"][0], 0.3 - 0.75 * result.metrics["force_penalty"][0, 3])
        self.assertAlmostEqual(result.metrics["suitcase_staying_reward"][0], 0.3)
        self.assertFalse(result.scalar_success())

    def test_current_credit_revokes_on_removal_or_reopening_and_history_resets(self):
        self.pack(close=True)
        self.assertTrue(self.steps(700).scalar_success())
        self.data.joint("suitcase_hinge").qpos[0] = np.deg2rad(30)
        self.assertFalse(self.steps(1).scalar_success())
        self.assertAlmostEqual(self.result().metrics["packing_reward"][0], 0.3)
        self.assertEqual(self.result().metrics["release_reward"][0], 0)
        self.assertTrue(self.result().metrics["ever_success"][0])
        self.data.joint("packing_object_1_joint").qpos[0] = 0.4
        self.assertAlmostEqual(self.steps(1).metrics["packing_reward"][0], 0.2)
        self.evaluator.reset()
        self.evaluator.after_reset(self.model, self.data)
        self.assertFalse(self.result().metrics["ever_success"][0])
        self.assertEqual(self.result().metrics["packing_reward"][0], 0)
        self.assertEqual(self.result().metrics["release_reward"][0], 0)

    def test_full_bounds_reject_overhang_above_lid_and_below_floor(self):
        self.pack()
        self.assertAlmostEqual(self.steps(600).metrics["packing_reward"][0], 0.3)
        original = self.data.joint("packing_object_1_joint").qpos.copy()
        for displacement in ((0.075, 0, 0), (0, 0, 0.08), (0, 0, -0.06)):
            self.data.joint("packing_object_1_joint").qpos[:] = original
            self.data.joint("packing_object_1_joint").qpos[:3] += displacement
            self.data.qvel[:] = 0
            result = self.steps(1)
            self.assertFalse(result.metrics["object_contained_mask"][0, 0])
            self.assertFalse(result.metrics["object_packed_mask"][0, 0])

    def place_first_object_near_walls(self, sides, clearance):
        geom = self.evaluator._bounds_geoms[0][0]
        case = self.evaluator._case
        rotation = self.data.xmat[case].reshape(3, 3)
        center = rotation.T @ (self.data.geom_xpos[geom] - self.data.xpos[case])
        extent = np.abs(rotation.T @ self.data.geom_xmat[geom].reshape(3, 3)) @ self.model.geom_size[geom]
        offset = np.zeros(3)
        for axis, side in enumerate(sides):
            if side:
                target = side * (CASE_INNER_XY[axis] - extent[axis] - clearance)
                offset[axis] = target - center[axis]
        self.data.joint("packing_object_1_joint").qpos[:3] += rotation @ offset
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def test_wall_and_corner_contact_qualifies_and_retains_packing(self):
        sides = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
        for side, clearance in product(sides, (0.0005, 0.0, -0.0005)):
            with self.subTest(side=side, clearance=clearance):
                self.setUp()
                self.pack(1)
                self.place_first_object_near_walls(side, clearance)
                result = self.steps(700)
                self.assertTrue(result.metrics["object_packed_mask"][0, 0])
                self.assertAlmostEqual(result.metrics["packing_reward"][0], 0.1 - result.metrics["force_penalty"][0, 0])
                self.place_first_object_near_walls(side, clearance)
                for _ in range(10):
                    result = self.steps(1)
                    self.assertTrue(result.metrics["object_packed_mask"][0, 0])

    def test_closed_case_keeps_credit_near_wall_and_rejects_actual_exit(self):
        self.pack(close=True)
        self.assertTrue(self.steps(700).scalar_success())
        self.place_first_object_near_walls((-1, 0), 0.0005)
        result = self.steps(700)
        self.assertTrue(result.scalar_success())
        self.assertAlmostEqual(result.scalar_reward(), 1 - result.metrics["force_penalty"].sum())
        for side in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            with self.subTest(side=side):
                self.setUp()
                self.pack(close=True)
                self.assertTrue(self.steps(700).scalar_success())
                self.place_first_object_near_walls(side, -0.0015)
                result = self.steps(1)
                self.assertFalse(result.metrics["object_contained_mask"][0, 0])
                self.assertFalse(result.metrics["object_packed_mask"][0, 0])
                self.assertFalse(result.scalar_success())

    def test_airborne_object_in_cavity_needs_floor_support(self):
        self.pack(1)
        self.data.joint("packing_object_1_joint").qpos[2] += 0.012
        result = self.steps(1)
        self.assertTrue(result.metrics["object_contained_mask"][0, 0])
        self.assertFalse(result.metrics["object_supported_mask"][0, 0])
        self.assertEqual(result.metrics["packing_reward"][0], 0)
        self.assertEqual(result.metrics["release_reward"][0], 0)

    def test_object_contact_preserves_case_credit_but_case_contact_revokes_release_and_pose(self):
        self.pack(close=True)
        self.assertTrue(self.steps(700).scalar_success())
        # Reach through the lid with a test probe to isolate item release.
        collision = self.evaluator._bounds_geoms[0][0]
        center = self.data.geom_xpos[collision].copy()
        self.data.mocap_pos[0] = center
        result = self.steps(1)
        self.assertTrue(result.metrics["object_touching_robot"][0, 0])
        self.assertTrue(result.metrics["object_packed_mask"][0, 0])
        self.assertFalse(result.metrics["suitcase_touching_robot"][0])
        self.assertAlmostEqual(result.metrics["suitcase_staying_reward"][0], 0.3)
        self.assertAlmostEqual(result.metrics["lid_reward"][0], 0.3 - 0.75 * result.metrics["force_penalty"][0, 3])
        self.assertAlmostEqual(result.metrics["packing_reward"][0], 0.3 - result.metrics["force_penalty"][0, :3].sum())
        self.assertFalse(result.scalar_success())
        self.data.mocap_pos[0] = [-2, 0, 2]
        self.pack(close=True)
        self.assertTrue(self.steps(700).scalar_success())
        self.data.mocap_pos[0] = self.data.xpos[self.model.body("suitcase_lid").id] + [-CASE_HALF_XY[0], 0, 0.025]
        result = self.steps(1)
        self.assertTrue(result.metrics["suitcase_touching_robot"][0])
        self.assertEqual(result.metrics["suitcase_staying_reward"][0], 0)
        self.assertFalse(result.scalar_success())

    def test_dwell_requires_real_time_and_every_tick(self):
        self.pack(close=True)
        self.assertFalse(self.steps(100).scalar_success())
        for _ in range(500):
            self.assertFalse(self.result().scalar_success())
        self.assertTrue(self.steps(650).scalar_success())
        with self.assertRaisesRegex(ValueError, "every physics step"):
            self.evaluator.after_physics_step(self.model, self.data)
        mujoco.mj_step(self.model, self.data, nstep=2)
        with self.assertRaisesRegex(ValueError, "every physics step"):
            self.evaluator.after_physics_step(self.model, self.data)
        with self.assertRaises(NotImplementedError):
            self.evaluator.evaluate_qpos_batch(self.data.qpos[None])
        with self.assertRaises(NotImplementedError):
            self.evaluator.reset(nworld=16)

    def test_case_table_support_is_required_for_closure_but_not_packing_retention(self):
        self.pack(close=True)
        self.assertTrue(self.steps(700).scalar_success())
        self.data.joint("suitcase_joint").qpos[2] += 0.1
        for i in range(1, 4):
            self.data.joint(f"packing_object_{i}_joint").qpos[2] += 0.1
        result = self.steps(1)
        self.assertFalse(result.metrics["suitcase_supported_and_settled"][0])
        self.assertAlmostEqual(result.scalar_reward(), 0.3, places=3)
        self.assertEqual(result.metrics["suitcase_staying_reward"][0], 0)
        self.assertAlmostEqual(result.metrics["packing_reward"][0], 0.3)
        self.assertEqual(result.metrics["lid_reward"][0], 0)

    def test_packed_resting_case_does_not_accumulate_force_penalty(self):
        self.pack(close=True)
        self.steps(700)
        initial = self.result().metrics["force_dose_ns"].copy()
        result = self.steps(2000)
        np.testing.assert_array_equal(result.metrics["force_dose_ns"], initial)
        np.testing.assert_array_equal(result.metrics["excess_force_n"], 0)
        weights = self.model.body_subtreemass[[*self.evaluator._bodies, self.evaluator._case]] * 9.81
        self.assertAlmostEqual(self.evaluator.force_threshold_n[3], weights[3] + 2 * weights[:3].sum() + 15)
        self.assertAlmostEqual(result.metrics["contact_force_n"][0, 3], weights[3] + 2 * weights[:3].sum(), delta=0.03)

    def test_case_force_dose_spends_only_case_credit_and_persists_until_reset(self):
        self.pack(close=True)
        self.steps(700)
        self.data.xfrc_applied[self.evaluator._case, 2] = -50
        result = self.steps(600)
        self.assertAlmostEqual(result.metrics["force_penalty"][0, 3], 0.12)
        np.testing.assert_allclose(result.metrics["force_penalty"][0, :3], 0, atol=1e-8)
        self.data.xfrc_applied[:] = 0
        result = self.steps(700)
        self.assertTrue(result.scalar_success())
        self.assertAlmostEqual(result.scalar_reward(), 0.88)
        self.assertAlmostEqual(result.metrics["lid_reward"][0], 0.21)
        self.assertAlmostEqual(result.metrics["release_reward"][0], 0.07)
        dose = result.metrics["force_dose_ns"].copy()
        for _ in range(100):
            np.testing.assert_array_equal(self.result().metrics["force_dose_ns"], dose)
        self.steps(1000)
        np.testing.assert_array_equal(self.result().metrics["force_dose_ns"], dose)
        self.evaluator.reset()
        self.evaluator.after_reset(self.model, self.data)
        np.testing.assert_array_equal(self.result().metrics["force_dose_ns"], 0)

    def test_pressing_object_charges_object_and_case_contact_loads(self):
        self.pack(close=True)
        self.steps(700)
        body = int(self.model.geom_bodyid[self.evaluator._bounds_geoms[0][0]])
        self.data.xfrc_applied[body, 2] = -20
        expected_dose = self.result().metrics["force_dose_ns"][0].copy()
        for _ in range(600):
            result = self.steps(1)
            expected_dose += (
                np.maximum(result.metrics["contact_force_n"][0] - self.evaluator.force_threshold_n, 0) * 0.002
            )
        np.testing.assert_allclose(result.metrics["force_dose_ns"][0], expected_dose, atol=1e-9)
        object_weight = self.model.body_subtreemass[body] * 9.81
        self.assertAlmostEqual(result.metrics["contact_force_n"][0, 0], object_weight + 20, delta=0.03)
        self.assertAlmostEqual(result.metrics["force_penalty"][0, 0], 0.03)
        self.assertAlmostEqual(result.metrics["force_penalty"][0, 3], 0.12)
        self.data.xfrc_applied[:] = 0
        result = self.steps(700)
        self.assertTrue(result.scalar_success())
        self.assertAlmostEqual(result.scalar_reward(), 0.85)

    def pose(self, displacement=(0, 0, 0), rotation_deg=0, axis=(0, 0, 1)):
        self.data.joint("suitcase_hinge").qpos[0] = 0
        joint = self.data.joint("suitcase_joint")
        joint.qpos[:3] = self.evaluator._initial_case_pos + displacement
        rotation = np.empty(4)
        mujoco.mju_axisAngle2Quat(rotation, np.asarray(axis, dtype=float), np.deg2rad(rotation_deg))
        mujoco.mju_mulQuat(joint.qpos[3:], self.evaluator._initial_case_quat, rotation)
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        return self.steps(1)

    def test_staying_credit_requires_closed_lid_and_revokes_on_reopening(self):
        self.assertEqual(self.result().metrics["packing_reward"][0], 0)
        self.assertEqual(self.result().metrics["release_reward"][0], 0)
        self.pack()
        self.steps(600)
        for angle, closed in ((107, False), (3.1, False), (2.9, True), (0, True), (20, False)):
            with self.subTest(angle=angle):
                self.data.joint("suitcase_hinge").qpos[0] = np.deg2rad(angle)
                self.data.qvel[:] = 0
                mujoco.mj_forward(self.model, self.data)
                result = self.steps(1)
                self.assertEqual(result.metrics["lid_is_closed"][0], closed)
                self.assertEqual(result.metrics["suitcase_staying_reward"][0], 0)
                np.testing.assert_array_equal(result.metrics["object_packed_mask"], True)
                self.assertAlmostEqual(
                    result.metrics["packing_reward"][0], 0.3 - result.metrics["force_penalty"][0, :3].sum()
                )
        self.data.joint("suitcase_hinge").qpos[0] = 0
        self.data.qvel[:] = 0
        self.assertEqual(self.steps(100).metrics["suitcase_staying_reward"][0], 0)
        self.assertAlmostEqual(self.steps(700).metrics["suitcase_staying_reward"][0], 0.3)
        self.data.joint("suitcase_hinge").qpos[0] = np.deg2rad(20)
        self.assertEqual(self.steps(1).metrics["suitcase_staying_reward"][0], 0)
        self.evaluator.reset()
        self.evaluator.after_reset(self.model, self.data)
        self.assertEqual(self.result().metrics["suitcase_staying_reward"][0], 0)

    def test_staying_credit_translation_deadband_taper_and_distance(self):
        for distance, fraction in ((0, 1), (0.049, 1), (0.05, 1), (0.125, 0.5), (0.2, 0), (0.25, 0)):
            with self.subTest(distance=distance):
                result = self.pose((distance, 0, 0))
                self.assertAlmostEqual(result.metrics["suitcase_translation_m"][0], distance)
                self.assertAlmostEqual(result.metrics["suitcase_staying_fraction"][0], fraction)
                self.assertEqual(result.metrics["suitcase_staying_reward"][0], 0)
        result = self.pose((0.04, 0.04, 0))
        distance = np.sqrt(2) * 0.04
        self.assertAlmostEqual(result.metrics["suitcase_translation_m"][0], distance)
        self.assertAlmostEqual(result.metrics["suitcase_staying_fraction"][0], (0.2 - distance) / 0.15)
        self.assertEqual(result.metrics["packing_reward"][0], 0)
        self.assertEqual(result.metrics["release_reward"][0], 0)

    def test_staying_credit_rotation_deadband_taper_and_worst_fraction(self):
        for axis in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            for angle, fraction in ((0, 1), (5, 1), (25, 0.5), (45, 0), (90, 0), (180, 0)):
                with self.subTest(axis=axis, angle=angle):
                    result = self.pose(rotation_deg=angle, axis=axis)
                    self.assertAlmostEqual(result.metrics["suitcase_rotation_deg"][0], angle)
                    self.assertAlmostEqual(result.metrics["suitcase_staying_fraction"][0], fraction)
                    self.assertEqual(result.metrics["suitcase_staying_reward"][0], 0)
        for displacement, angle, fraction in ((0.125, 15, 0.5), (0.08, 25, 0.5), (0.2, 0, 0), (0, 45, 0)):
            result = self.pose((displacement, 0, 0), rotation_deg=angle)
            self.assertAlmostEqual(result.metrics["suitcase_staying_fraction"][0], fraction)

    def test_staying_credit_uses_reset_pose_and_shortest_quaternion_rotation(self):
        self.pose((0.1, 0.04, 0), rotation_deg=170)
        self.evaluator.reset()
        self.evaluator.after_reset(self.model, self.data)
        self.assertEqual(self.result().metrics["suitcase_staying_fraction"][0], 1)
        self.assertEqual(self.result().metrics["suitcase_staying_reward"][0], 0)
        result = self.pose(rotation_deg=15)
        self.assertAlmostEqual(result.metrics["suitcase_rotation_deg"][0], 15)
        self.assertAlmostEqual(result.metrics["suitcase_staying_fraction"][0], 0.75)
        self.data.joint("suitcase_joint").qpos[3:] *= -1
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        result = self.steps(1)
        self.assertAlmostEqual(result.metrics["suitcase_rotation_deg"][0], 15, places=5)
        self.pose((0.2, 0, 0))
        self.assertAlmostEqual(self.result().metrics["suitcase_staying_reward"][0], 0)
        self.assertEqual(self.pose().metrics["suitcase_staying_fraction"][0], 1)
        self.evaluator.reset()
        self.evaluator.after_reset(self.model, self.data)
        self.assertEqual(self.result().metrics["suitcase_staying_fraction"][0], 1)
        self.assertEqual(self.result().metrics["suitcase_staying_reward"][0], 0)

    def test_displaced_packed_closed_case_keeps_handling_credit_but_loses_pose_credit(self):
        self.pack(close=True)
        self.assertTrue(self.steps(700).scalar_success())
        for distance, pose_reward in ((0.125, 0.15), (0.2, 0), (0, 0.3)):
            old_pos = self.data.joint("suitcase_joint").qpos[:3].copy()
            new_pos = self.evaluator._initial_case_pos + [0, distance, 0]
            translation = new_pos - old_pos
            for name in ("suitcase_joint", *(f"packing_object_{i}_joint" for i in range(1, 4))):
                self.data.joint(name).qpos[:3] += translation
            self.data.qvel[:] = 0
            mujoco.mj_forward(self.model, self.data)
            result = self.steps(700)
            self.assertAlmostEqual(result.metrics["packing_reward"][0], 0.3)
            self.assertAlmostEqual(result.metrics["lid_reward"][0], 0.3 - 0.75 * result.metrics["force_penalty"][0, 3])
            self.assertAlmostEqual(result.metrics["suitcase_staying_reward"][0], pose_reward, places=4)
            self.assertAlmostEqual(
                result.scalar_reward(), 0.7 + pose_reward - result.metrics["force_penalty"].sum(), places=4
            )
            self.assertEqual(result.scalar_success(), distance == 0)

    def test_packing_credit_survives_case_motion_and_clears_only_on_exit(self):
        self.pack()
        self.steps(600)
        self.assertAlmostEqual(self.result().metrics["packing_reward"][0], 0.3)
        dofs = [self.evaluator._case_dof, *self.evaluator._dofs]
        for dof in dofs:
            self.data.qvel[dof + 1] = 0.5
        result = self.steps(10)
        self.assertFalse(result.metrics["suitcase_supported_and_settled"][0])
        np.testing.assert_array_equal(result.metrics["object_contained_mask"], True)
        np.testing.assert_array_equal(result.metrics["object_packed_mask"], True)
        self.assertAlmostEqual(result.metrics["packing_reward"][0], 0.3)
        self.data.joint("packing_object_1_joint").qpos[0] -= 0.2
        self.data.qvel[:] = 0
        result = self.steps(1)
        self.assertAlmostEqual(result.metrics["packing_reward"][0], 0.2)
        self.assertFalse(result.metrics["object_packed_mask"][0, 0])
        self.pack(1)
        self.assertFalse(self.steps(1).metrics["object_packed_mask"][0, 0])
        self.evaluator.reset()
        self.evaluator.after_reset(self.model, self.data)
        np.testing.assert_array_equal(self.result().metrics["object_packed_mask"], False)

    def test_force_configuration_rejects_invalid_values(self):
        for kwargs in (
            {"object_force_allowance_n": -1},
            {"case_extra_force_allowance_n": float("nan")},
            {"force_dose_budget_ns": 0},
        ):
            with self.assertRaises(ValueError):
                PackSuitcaseEvaluator(model=self.model, spec=get_task_spec("pack_suitcase"), **kwargs)


if __name__ == "__main__":
    unittest.main()
