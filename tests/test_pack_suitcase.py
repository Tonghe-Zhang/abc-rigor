"""Suitcase geometry, seeded scenes, physical packing, closure and replay."""

from __future__ import annotations

import copy
import json
import unittest
import xml.etree.ElementTree as ET

import abc_sim
import mujoco
import numpy as np
from abc_sim.randomization.assets.suitcase import (
    CASE_FLOOR_Z,
    CASE_INNER_XY,
    CASE_RIM_Z,
    LID_LIMIT_RAD,
    PACKING_CATEGORIES,
    SHELL_PALETTES,
    SHELL_PATTERNS,
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
        for seed in range(32):
            self.env.reset(seed=seed)
            state = self.env._last_randomization
            textures.add(tuple(state.metadata["texture"].values()))
            prompts.add(self.env.prompt)
            categories.update(obj["category"] for obj in state.metadata["objects"])
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
            self.assertEqual(self.env.evaluate_task().scalar_reward(), 0)
            self.assertEqual(get_task_spec(self.env.prompt).name, "pack_suitcase")
        self.assertGreaterEqual(len(textures), 16)
        self.assertEqual(categories, set(PACKING_CATEGORIES))
        self.assertEqual(prompts, set(PACK_SUITCASE_PROMPTS))
        self.assertEqual(len(layouts), 32)

    def test_same_seed_and_serialized_replay_restore_geometry_texture_pose_and_prompt(self):
        self.env.reset(seed=9)
        qpos = self.env.data.qpos.copy()
        textures = self.env.model.tex_data.copy()
        saved = json.loads(json.dumps(self.env._last_randomization.to_dict()))
        self.env.reset(seed=4)
        self.env._task_randomizer.apply(self.env.model, self.env.data, RandomizationState.from_dict(saved))
        np.testing.assert_array_equal(self.env.data.qpos, qpos)
        np.testing.assert_array_equal(self.env.model.tex_data, textures)
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
        self.assertEqual(self.env.evaluate_task().scalar_reward(), 0)

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
            self.data.joint(f"packing_object_{i}_joint").qpos[:] = [0.62, (i - 2) * 0.105, 0.766, 1, 0, 0, 0]
        if close:
            self.data.joint("suitcase_hinge").qpos[0] = 0
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def test_partial_credit_then_full_packed_closed_success(self):
        for count in range(4):
            self.pack(count)
            self.assertAlmostEqual(self.steps(600).scalar_reward(), count / 6)
            self.assertFalse(self.result().scalar_success())
        self.data.joint("suitcase_hinge").qpos[0] = 0
        result = self.steps(600)
        self.assertAlmostEqual(result.scalar_reward(), 1 - result.metrics["force_penalty"].sum())
        self.assertGreater(result.scalar_reward(), 0.99)
        self.assertTrue(self.result().scalar_success())

    def test_closed_empty_or_missing_object_never_gets_closure_credit(self):
        self.pack(0, close=True)
        self.assertEqual(self.steps(700).scalar_reward(), 0)
        self.pack(2, close=True)
        result = self.steps(700)
        self.assertAlmostEqual(result.scalar_reward(), 1 / 3)
        self.assertFalse(result.scalar_success())

    def test_current_credit_revokes_on_removal_or_reopening_and_history_resets(self):
        self.pack(close=True)
        self.assertTrue(self.steps(700).scalar_success())
        self.data.joint("suitcase_hinge").qpos[0] = np.deg2rad(30)
        self.assertFalse(self.steps(1).scalar_success())
        self.assertEqual(self.result().scalar_reward(), 0.5)
        self.assertTrue(self.result().metrics["ever_success"][0])
        self.data.joint("packing_object_1_joint").qpos[0] = 0.4
        self.assertAlmostEqual(self.steps(1).scalar_reward(), 1 / 3)
        self.evaluator.reset()
        self.evaluator.after_reset(self.model, self.data)
        self.assertFalse(self.result().metrics["ever_success"][0])
        self.assertEqual(self.result().scalar_reward(), 0)

    def test_full_bounds_reject_overhang_above_lid_and_below_floor(self):
        self.pack()
        self.assertEqual(self.steps(600).scalar_reward(), 0.5)
        original = self.data.joint("packing_object_1_joint").qpos.copy()
        for displacement in ((0.11, 0, 0), (0, 0, 0.08), (0, 0, -0.06)):
            self.data.joint("packing_object_1_joint").qpos[:] = original
            self.data.joint("packing_object_1_joint").qpos[:3] += displacement
            self.data.qvel[:] = 0
            result = self.steps(1)
            self.assertFalse(result.metrics["object_contained_mask"][0, 0])
            self.assertFalse(result.metrics["object_packed_mask"][0, 0])

    def test_airborne_object_in_cavity_needs_floor_support(self):
        self.pack(1)
        self.data.joint("packing_object_1_joint").qpos[2] += 0.012
        result = self.steps(1)
        self.assertTrue(result.metrics["object_contained_mask"][0, 0])
        self.assertFalse(result.metrics["object_supported_mask"][0, 0])
        self.assertEqual(result.scalar_reward(), 0)

    def test_touching_robot_revokes_object_and_closed_case_credit(self):
        self.pack(close=True)
        self.assertTrue(self.steps(700).scalar_success())
        # Reach through the lid with a test probe to isolate item release.
        collision = self.evaluator._bounds_geoms[0][0]
        center = self.data.geom_xpos[collision].copy()
        self.data.mocap_pos[0] = center
        result = self.steps(1)
        self.assertTrue(result.metrics["object_touching_robot"][0, 0])
        self.assertFalse(result.scalar_success())
        self.data.mocap_pos[0] = [-2, 0, 2]
        self.pack(close=True)
        self.assertTrue(self.steps(700).scalar_success())
        self.data.mocap_pos[0] = self.data.xpos[self.model.body("suitcase_lid").id] + [-0.13, 0, 0.025]
        result = self.steps(1)
        self.assertTrue(result.metrics["suitcase_touching_robot"][0])
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

    def test_case_must_rest_on_table(self):
        self.pack(close=True)
        self.assertTrue(self.steps(700).scalar_success())
        self.data.joint("suitcase_joint").qpos[2] += 0.1
        for i in range(1, 4):
            self.data.joint(f"packing_object_{i}_joint").qpos[2] += 0.1
        result = self.steps(1)
        self.assertFalse(result.metrics["suitcase_supported_and_settled"][0])
        self.assertEqual(result.scalar_reward(), 0)

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
        self.assertAlmostEqual(result.metrics["force_penalty"][0, 3], 0.15)
        np.testing.assert_allclose(result.metrics["force_penalty"][0, :3], 0, atol=1e-8)
        self.data.xfrc_applied[:] = 0
        result = self.steps(700)
        self.assertTrue(result.scalar_success())
        self.assertAlmostEqual(result.scalar_reward(), 0.85)
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
        self.assertAlmostEqual(result.metrics["force_penalty"][0, 0], 0.05)
        self.assertAlmostEqual(result.metrics["force_penalty"][0, 3], 0.15)
        self.data.xfrc_applied[:] = 0
        result = self.steps(700)
        self.assertTrue(result.scalar_success())
        self.assertAlmostEqual(result.scalar_reward(), 0.8)

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
