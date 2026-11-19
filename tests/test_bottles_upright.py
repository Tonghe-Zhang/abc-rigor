"""Upright-bottle geometry, language, scoring, and replay regressions."""

from __future__ import annotations

import copy
import json
import unittest
import xml.etree.ElementTree as ET

import abc_sim
import mujoco
import numpy as np
from abc_sim.randomization.assets.tabletop_distractors import (
    DISTRACTOR_CATEGORIES,
    TRASH_BIN_VARIANTS,
    DistractorSpec,
    build_distractor,
    distractor_half_extents,
    distractor_variants,
)
from abc_sim.randomization.core import RandomizationState
from abc_sim.randomization.tasks.bottles_upright import UPRIGHT_BOTTLE_VARIANTS
from abc_sim.task_specs import LIFT_BOTTLES_UPRIGHT_PROMPTS, get_task_spec


class BottlesUprightTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = abc_sim.make_env(task="lift_bottles_upright", render_cameras=False)

    @property
    def addrs(self):
        return [self.env.model.jnt_qposadr[self.env.model.joint(f"bottle_{i}_joint").id] for i in range(1, 4)]

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def setUp(self):
        self.env.reset(seed=0)

    def test_exactly_three_bottles_and_one_to_three_distractors(self):
        free = [
            self.env.model.joint(i).name
            for i in range(self.env.model.njnt)
            if self.env.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE
        ]
        count = self.env._last_randomization.metadata["distractor_count"]
        self.assertIn(count, (1, 2, 3))
        self.assertEqual(
            free,
            [
                *(f"bottle_{i}_joint" for i in range(1, 4)),
                *(f"distractor_{i}_joint" for i in range(1, count + 1)),
            ],
        )
        self.assertEqual(mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, "bin_container"), -1)
        self.assertEqual(self.env.model.geom("table_plane").type, mujoco.mjtGeom.mjGEOM_BOX)

    def test_seeded_sideways_layouts_and_all_prompts(self):
        prompts, layouts, variants, counts = set(), set(), set(), set()
        randomizer = self.env._task_randomizer
        for seed in (*range(32), 112):
            self.env.reset(seed=seed)
            state = self.env._last_randomization
            prompts.add(self.env.prompt)
            variants.update(state.metadata["bottle_variants"])
            counts.add(state.metadata["distractor_count"])
            layouts.add(tuple(self.env.data.qpos[self.addrs]))
            self.assertTrue(randomizer._bounds_ok(state.object_states))
            self.assertTrue(randomizer._pairwise_ok(state.object_states))
            self.assertTrue(randomizer._contacts_ok(self.env.model, self.env.data))
            for addr in self.addrs:
                quat = self.env.data.qpos[addr + 3 : addr + 7]
                self.assertAlmostEqual(1 - 2 * (quat[1] ** 2 + quat[2] ** 2), 0)
            self.assertEqual(self.env.evaluate_task().scalar_reward(), 0)
        self.assertEqual(prompts, set(LIFT_BOTTLES_UPRIGHT_PROMPTS))
        self.assertEqual(len(layouts), 33)
        self.assertEqual(variants, set(UPRIGHT_BOTTLE_VARIANTS))
        self.assertEqual(counts, {1, 2, 3})
        for prompt in prompts:
            self.assertEqual(get_task_spec(prompt).name, "lift_bottles_upright")

    def _stand(self, count=3):
        # Isolate scoring from the robot and randomized clutter. The standing
        # bottle is taller than its initial lying footprint's arm clearance.
        for index in range(self.env._last_randomization.metadata["distractor_count"]):
            self.env.data.joint(f"distractor_{index + 1}_joint").qpos[:3] = [-2 - index, 0, -2]
        for index, addr in enumerate(self.addrs[:count]):
            self.env.data.qpos[addr : addr + 7] = [0.76, (index - 1) * 0.36, 0.75, 1, 0, 0, 0]
        self.env.data.qvel[:] = 0
        mujoco.mj_forward(self.env.model, self.env.data)

    def test_partial_credit_success_and_history_reset(self):
        for count in range(4):
            self._stand(count)
            self.env.advance_physics(750)
            result = self.env.evaluate_task()
            self.assertAlmostEqual(result.scalar_reward(), count / 3)
            self.assertEqual(result.scalar_success(), count == 3)
        self.env.data.qpos[self.addrs[0] + 3 : self.addrs[0] + 7] = [0, 1, 0, 0]
        self.env.advance_physics(1)
        result = self.env.evaluate_task()
        self.assertFalse(result.scalar_success())
        self.assertTrue(result.metrics["ever_success"][0])
        self.env.reset(seed=0)
        self.assertFalse(self.env.evaluate_task().metrics["ever_success"][0])

    def test_airborne_below_table_off_table_inverted_and_tilted_fail(self):
        bad_poses = [
            [0.57, 0, 0.85, 1, 0, 0, 0],
            [0.57, 0, 0.65, 1, 0, 0, 0],
            [0.95, 0, 0.75, 1, 0, 0, 0],
            [0.57, 0.7, 0.75, 1, 0, 0, 0],
            [0.57, 0, 0.75, 0, 1, 0, 0],
            [0.57, 0, 0.75, np.cos(np.pi / 8), np.sin(np.pi / 8), 0, 0],
        ]
        for pose in bad_poses:
            with self.subTest(pose=pose):
                self.env.reset(seed=0)
                self._stand()
                self.env.advance_physics(750)
                self.assertTrue(self.env.evaluate_task().scalar_success())
                self.env.data.qpos[self.addrs[0] : self.addrs[0] + 7] = pose
                self.env.advance_physics(1)
                result = self.env.evaluate_task()
                self.assertFalse(result.scalar_success())
                self.assertEqual(result.metrics["num_upright_bottles"][0], 2)

    def test_standing_and_sideways_poses_survive_physics(self):
        for start in range(0, len(UPRIGHT_BOTTLE_VARIANTS), 3):
            variants = [UPRIGHT_BOTTLE_VARIANTS[i % len(UPRIGHT_BOTTLE_VARIANTS)] for i in range(start, start + 3)]
            options = {
                "randomization": {
                    "bottle_variants": variants,
                    "distractors": [{"category": "sponge", "variant": distractor_variants("sponge")[0]}],
                }
            }
            for standing in (False, True):
                with self.subTest(variants=variants, standing=standing):
                    self.env.reset(seed=0, options=options)
                    if standing:
                        self._stand()
                    # Rounded bottle bases can rock before satisfying the
                    # velocity threshold and starting their settling dwell.
                    self.env.advance_physics(1500)
                    self.assertEqual(self.env.evaluate_task().scalar_success(), standing)
                    for addr in self.addrs:
                        self.assertGreater(self.env.data.qpos[addr + 2], 0.74)

    def test_seed_and_serialized_replay_restore_pose_prompt_and_score(self):
        self.env.reset(seed=11)
        qpos = self.env.data.qpos.copy()
        saved = json.loads(json.dumps(self.env._last_randomization.to_dict()))
        self.env.reset(seed=12)
        self.env._task_randomizer.apply(self.env.model, self.env.data, RandomizationState.from_dict(saved))
        np.testing.assert_array_equal(self.env.data.qpos, qpos)
        self.assertEqual(self.env._last_randomization.to_dict(), saved)
        self.assertEqual(self.env.prompt, LIFT_BOTTLES_UPRIGHT_PROMPTS[11])
        self.assertEqual(self.env.evaluate_task().scalar_reward(), 0)
        self.env.reset(seed=11)
        np.testing.assert_array_equal(self.env.data.qpos, qpos)

    def test_all_distractor_assets_compile_and_settle(self):
        self.assertGreaterEqual(len(DISTRACTOR_CATEGORIES), 20)
        self.assertNotIn("water_bottle", DISTRACTOR_CATEGORIES)
        for category in DISTRACTOR_CATEGORIES:
            for variant in distractor_variants(category):
                with self.subTest(category=category, variant=variant):
                    spec = DistractorSpec(category, variant)
                    half = np.asarray(distractor_half_extents(spec))
                    self.assertTrue(np.isfinite(half).all())
                    self.assertTrue((half > 0).all())
                    self.assertLess(np.linalg.norm(half), 0.25)
                    assets, body = build_distractor(spec, 1)
                    root = ET.Element("mujoco")
                    ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast")
                    ET.SubElement(root, "asset").extend(assets)
                    world = ET.SubElement(root, "worldbody")
                    ET.SubElement(world, "geom", type="plane", size="1 1 0.1", pos="0 0 0.75")
                    world.append(body)
                    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
                    data = mujoco.MjData(model)
                    for _ in range(500):
                        mujoco.mj_step(model, data)
                    self.assertTrue(np.isfinite(data.qpos).all())
                    self.assertGreater(data.qpos[2], 0.74)
                    self.assertLess(data.qpos[2], 0.76)
                    self.assertLess(np.linalg.norm(data.qvel), 0.05)

    def test_trash_bins_are_distractors_and_assets_can_be_pinned(self):
        for variant in TRASH_BIN_VARIANTS:
            options = {
                "randomization": {
                    "bottle_variants": ["bottle_5", "bottle_11", "bottle_18"],
                    "distractors": [
                        {"category": "trash_bin", "variant": variant},
                        {"category": "apple", "variant": distractor_variants("apple")[0]},
                        {"category": "sponge", "variant": distractor_variants("sponge")[0]},
                    ],
                }
            }
            self.env.reset(seed=0, options=options)
            state = self.env._last_randomization
            self.assertEqual(state.metadata["distractors"], options["randomization"]["distractors"])
            self.assertEqual(state.metadata["bottle_variants"], options["randomization"]["bottle_variants"])
            self.assertEqual(self.env.evaluate_task().scalar_reward(), 0)
            self.assertTrue(self.env._task_randomizer._contacts_ok(self.env.model, self.env.data))
            nq, meshes = self.env.model.nq, self.env.model.nmesh
            self.env.reset(seed=1, options=options)
            self.assertEqual((self.env.model.nq, self.env.model.nmesh), (nq, meshes))
            self._stand()
            self.env.advance_physics(750)
            self.assertTrue(self.env.evaluate_task().scalar_success())
            addr = self.env.model.jnt_qposadr[self.env.model.joint("distractor_1_joint").id]
            self.env.data.qpos[addr + 2] = -1
            self.env.advance_physics(1)
            self.assertTrue(self.env.evaluate_task().scalar_success())

    def test_bad_asset_replay_is_rejected_before_mutation(self):
        saved = self.env._last_randomization.to_dict()
        qpos, model = self.env.data.qpos.copy(), self.env.model
        for field, value in (
            ("bottle_variants", ["bottle_0"] * 3),
            ("distractors", [{"category": "water_bottle", "variant": "bottle_17"}]),
            ("distractor_count", 99),
        ):
            state = RandomizationState.from_dict(copy.deepcopy(saved))
            state.metadata[field] = value
            with self.assertRaises(ValueError):
                self.env._task_randomizer.apply(self.env.model, self.env.data, state)
            self.assertIs(self.env.model, model)
            np.testing.assert_array_equal(self.env.data.qpos, qpos)

    def test_independent_instances_share_the_same_seeded_scene(self):
        self.env.reset(seed=42)
        other = abc_sim.make_env(task="lift_bottles_upright", render_cameras=False)
        try:
            other.reset(seed=41)
            other.reset(seed=42)
            np.testing.assert_array_equal(other.data.qpos, self.env.data.qpos)
            self.assertEqual(other._last_randomization.to_dict(), self.env._last_randomization.to_dict())
        finally:
            other.close()

    def test_incomplete_replay_is_rejected_before_mutation(self):
        state = copy.deepcopy(self.env._last_randomization)
        state.object_states.pop("bottle_3_joint")
        qpos = self.env.data.qpos.copy()
        with self.assertRaises(ValueError):
            self.env._task_randomizer.apply(self.env.model, self.env.data, state)
        np.testing.assert_array_equal(self.env.data.qpos, qpos)

    def test_qpos_only_scoring_is_explicitly_unsupported(self):
        with self.assertRaisesRegex(NotImplementedError, "solved contacts"):
            self.env._task_evaluator.evaluate_qpos_batch(self.env.data.qpos[None, :])
        with self.assertRaisesRegex(NotImplementedError, "one native physics observer"):
            self.env._task_evaluator.reset(nworld=16)

    def test_dwell_requires_physics_time_and_resets_after_an_interruption(self):
        self._stand()
        self.env.advance_physics(200)
        for _ in range(500):
            self.assertFalse(self.env.evaluate_task().scalar_success())
        self.env.advance_physics(550)
        self.assertTrue(self.env.evaluate_task().scalar_success())
        original = self.env.data.qpos[self.addrs[0] : self.addrs[0] + 7].copy()
        self.env.data.qpos[self.addrs[0] + 2] += 0.05
        self.env.advance_physics(1)
        self.assertFalse(self.env.evaluate_task().scalar_success())
        self.env.data.qpos[self.addrs[0] : self.addrs[0] + 7] = original
        self.env.data.qvel[:] = 0
        self.env.advance_physics(200)
        self.assertFalse(self.env.evaluate_task().scalar_success())
        self.env.advance_physics(550)
        self.assertTrue(self.env.evaluate_task().scalar_success())

    def test_observer_rejects_skipped_and_duplicate_ticks(self):
        evaluator = self.env._task_evaluator
        with self.assertRaisesRegex(ValueError, "every physics step exactly once"):
            evaluator.after_physics_step(self.env.model, self.env.data)
        mujoco.mj_step(self.env.model, self.env.data, nstep=2)
        with self.assertRaisesRegex(ValueError, "every physics step exactly once"):
            evaluator.after_physics_step(self.env.model, self.env.data)


if __name__ == "__main__":
    unittest.main()
