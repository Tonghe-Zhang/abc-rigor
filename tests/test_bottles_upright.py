"""Upright-bottle geometry, language, scoring, and replay regressions."""

from __future__ import annotations

import copy
import unittest

import abc_sim
import mujoco
import numpy as np
from abc_sim.randomization.core import RandomizationState
from abc_sim.task_specs import LIFT_BOTTLES_UPRIGHT_PROMPTS, get_task_spec


class BottlesUprightTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = abc_sim.make_env(task="lift_bottles_upright", render_cameras=False)
        cls.addrs = [cls.env.model.jnt_qposadr[cls.env.model.joint(f"bottle_{i}_joint").id] for i in range(1, 4)]

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def setUp(self):
        self.env.reset(seed=0)

    def test_exactly_three_bottles_no_bin(self):
        free = [
            self.env.model.joint(i).name
            for i in range(self.env.model.njnt)
            if self.env.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE
        ]
        self.assertEqual(free, [f"bottle_{i}_joint" for i in range(1, 4)])
        self.assertEqual(mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, "bin_container"), -1)
        self.assertEqual(self.env.model.geom("table_plane").type, mujoco.mjtGeom.mjGEOM_BOX)

    def test_seeded_sideways_layouts_and_all_prompts(self):
        prompts, layouts = set(), set()
        randomizer = self.env._task_randomizer
        for seed in range(32):
            self.env.reset(seed=seed)
            state = self.env._last_randomization
            prompts.add(self.env.prompt)
            layouts.add(tuple(self.env.data.qpos[self.addrs]))
            self.assertTrue(randomizer._bounds_ok(state.object_states))
            self.assertTrue(randomizer._pairwise_ok(state.object_states))
            self.assertTrue(randomizer._contacts_ok(self.env.model, self.env.data))
            for addr in self.addrs:
                quat = self.env.data.qpos[addr + 3 : addr + 7]
                self.assertAlmostEqual(1 - 2 * (quat[1] ** 2 + quat[2] ** 2), 0)
            self.assertEqual(self.env.evaluate_task().scalar_reward(), 0)
        self.assertEqual(prompts, set(LIFT_BOTTLES_UPRIGHT_PROMPTS))
        self.assertEqual(len(layouts), 32)
        for prompt in prompts:
            self.assertEqual(get_task_spec(prompt).name, "lift_bottles_upright")

    def _stand(self, count=3):
        for index, addr in enumerate(self.addrs[:count]):
            self.env.data.qpos[addr : addr + 7] = [0.57, -0.3 + index * 0.3, 0.75, 1, 0, 0, 0]
        self.env.data.qvel[:] = 0
        mujoco.mj_forward(self.env.model, self.env.data)

    def test_partial_credit_success_and_history_reset(self):
        for count in range(4):
            self._stand(count)
            result = self.env.evaluate_task()
            self.assertAlmostEqual(result.scalar_reward(), count / 3)
            self.assertEqual(result.scalar_success(), count == 3)
        self.env.data.qpos[self.addrs[0] + 3 : self.addrs[0] + 7] = [0, 1, 0, 0]
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
                self._stand()
                self.env.data.qpos[self.addrs[0] : self.addrs[0] + 7] = pose
                result = self.env.evaluate_task()
                self.assertFalse(result.scalar_success())
                self.assertEqual(result.metrics["num_upright_bottles"][0], 2)

    def test_standing_and_sideways_poses_survive_physics(self):
        for standing in (False, True):
            self.env.reset(seed=0)
            if standing:
                self._stand()
            for _ in range(500):
                mujoco.mj_step(self.env.model, self.env.data)
            self.assertEqual(self.env.evaluate_task().scalar_success(), standing)
            for addr in self.addrs:
                self.assertGreater(self.env.data.qpos[addr + 2], 0.74)

    def test_seed_and_serialized_replay_restore_pose_prompt_and_score(self):
        self.env.reset(seed=11)
        qpos = self.env.data.qpos.copy()
        saved = self.env._last_randomization.to_dict()
        self.env.reset(seed=12)
        self.env._task_randomizer.apply(self.env.model, self.env.data, RandomizationState.from_dict(saved))
        np.testing.assert_allclose(self.env.data.qpos, qpos)
        self.assertEqual(self.env.prompt, LIFT_BOTTLES_UPRIGHT_PROMPTS[11])
        self.assertEqual(self.env.evaluate_task().scalar_reward(), 0)
        self.env.reset(seed=11)
        np.testing.assert_allclose(self.env.data.qpos, qpos)

    def test_incomplete_replay_is_rejected_before_mutation(self):
        state = copy.deepcopy(self.env._last_randomization)
        state.object_states.pop("bottle_3_joint")
        qpos = self.env.data.qpos.copy()
        with self.assertRaises(ValueError):
            self.env._task_randomizer.apply(self.env.model, self.env.data, state)
        np.testing.assert_array_equal(self.env.data.qpos, qpos)

    def test_batched_pose_scoring(self):
        initial = self.env.data.qpos.copy()
        self._stand()
        result = self.env._task_evaluator.evaluate_qpos_batch(np.stack([initial, self.env.data.qpos]))
        np.testing.assert_array_equal(result.success, [False, True])
        np.testing.assert_array_equal(result.reward, [0, 1])


if __name__ == "__main__":
    unittest.main()
