"""Rejected manipulation resets preserve live physics and reward observation."""

from __future__ import annotations

import copy
import unittest

import abc_sim
import mujoco
import numpy as np
from abc_sim.randomization.assets.fruit_bowls import BOWL_COLORS
from abc_sim.randomization.tasks.pack_and_pour_fruit import PackAndPourFruitRandomizer

TASKS = {
    "lift_bottles_upright": (
        {"force_penalty_start_n": 0},
        ({"bottle_variants": ["invalid"] * 3}, {"distractors": []}),
    ),
    "pack_suitcase": (
        {"object_force_allowance_n": 0},
        ({"lid_angle_deg": 111}, {"objects": []}, {"texture": {"palette": "bad", "pattern": "solid"}}),
    ),
    "pack_and_pour_fruit": (
        {"force_allowance_n": 0},
        (
            {"source_color": "bad"},
            {"target_color": "bad"},
            {"source_color": "red", "target_color": "red"},
            {"fruits": []},
            {"prompt_index": 6},
        ),
    ),
}


class ManipulationResetTest(unittest.TestCase):
    def test_each_pinned_bowl_color_leaves_a_distinct_valid_partner_for_every_seed(self):
        randomizer = PackAndPourFruitRandomizer()
        for key in ("source_color", "target_color"):
            for color in BOWL_COLORS:
                request = {key: color}
                randomizer.prepare_reset(request=request)
                for seed in range(32):
                    with self.subTest(key=key, color=color, seed=seed):
                        _, source, target, _ = randomizer._identities(request, np.random.default_rng(seed))
                        self.assertEqual(source if key == "source_color" else target, color)
                        self.assertIn(source, BOWL_COLORS)
                        self.assertIn(target, BOWL_COLORS)
                        self.assertNotEqual(source, target)

    def test_pinned_receiver_colors_reset_deterministically_without_losing_observation(self):
        env = abc_sim.make_env(task="pack_and_pour_fruit", render_cameras=False)
        try:
            for color in BOWL_COLORS:
                with self.subTest(color=color):
                    request = {"target_color": color}
                    env.reset(seed=1, options={"randomization": request})
                    expected = copy.deepcopy(env._last_randomization)
                    self.assertEqual(expected.metadata["target_color"], color)
                    self.assertNotEqual(expected.metadata["source_color"], color)
                    env.advance_physics(1)
                    self.assertAlmostEqual(env._task_evaluator._previous_time, env.data.time)
                    env.reset(seed=1, options={"randomization": request})
                    self.assertEqual(env._last_randomization, expected)
                    self.assertEqual(env.evaluate_task().scalar_reward(), 0)
        finally:
            env.close()

    def assert_rejected_without_mutation(self, env, **kwargs):
        model, data = env.model, env.data
        observer, evaluator = env._physics_observer, env._task_evaluator
        arrays = {
            name: getattr(data, name).copy() for name in ("qpos", "qvel", "ctrl", "xfrc_applied", "qacc_warmstart")
        }
        time, prompt = data.time, env.prompt
        randomization = copy.deepcopy(env._last_randomization)
        history = evaluator._force_dose_ns.copy()
        result = None if observer is None else env.evaluate_task()
        with self.assertRaises(ValueError):
            env.reset(seed=19, **kwargs)
        self.assertIs(env.model, model)
        self.assertIs(env.data, data)
        self.assertIs(env._physics_observer, observer)
        self.assertIs(env._task_evaluator, evaluator)
        self.assertEqual(data.time, time)
        self.assertEqual(env.prompt, prompt)
        self.assertEqual(env._last_randomization, randomization)
        for name, expected in arrays.items():
            np.testing.assert_array_equal(getattr(data, name), expected)
        np.testing.assert_array_equal(evaluator._force_dose_ns, history)
        if result is not None:
            current = env.evaluate_task()
            np.testing.assert_array_equal(current.reward, result.reward)
            np.testing.assert_array_equal(current.success, result.success)
            self.assertEqual(current.metrics.keys(), result.metrics.keys())
            for name in result.metrics:
                np.testing.assert_array_equal(current.metrics[name], result.metrics[name])

    def test_nonrandomized_resets_reject_fresh_and_previously_sampled_models(self):
        for task in TASKS:
            with self.subTest(task=task):
                env = abc_sim.make_env(task=task, render_cameras=False)
                try:
                    self.assert_rejected_without_mutation(env, randomize=False)
                    env.reset(seed=7)
                    env.advance_physics(750)
                    self.assertEqual(env.evaluate_task().scalar_reward(), 0)
                    self.assert_rejected_without_mutation(env, randomize=False)
                    env.advance_physics(10)
                    self.assertEqual(env.evaluate_task().scalar_reward(), 0)
                    self.assertAlmostEqual(env._task_evaluator._previous_time, env.data.time)
                finally:
                    env.close()

    def test_invalid_options_preserve_history_and_force_dose_keeps_advancing(self):
        for task, (force_options, invalid_requests) in TASKS.items():
            with self.subTest(task=task):
                env = abc_sim.make_env(task=task, render_cameras=False, evaluator_options=force_options)
                try:
                    env.reset(seed=7)
                    env.advance_physics(250)
                    for request in (*invalid_requests, {"unknown": 1}):
                        self.assert_rejected_without_mutation(env, options={"randomization": request})
                        before = env._task_evaluator._force_dose_ns.copy()
                        env.advance_physics(10)
                        self.assertTrue(np.any(env._task_evaluator._force_dose_ns > before))
                        self.assertAlmostEqual(env._task_evaluator._previous_time, env.data.time)
                finally:
                    env.close()

    def test_rejected_reset_cannot_freeze_a_previously_successful_reward(self):
        env = abc_sim.make_env(task="lift_bottles_upright", render_cameras=False)
        try:
            env.reset(seed=7)
            for i in range(1, 4):
                env.data.joint(f"bottle_{i}_joint").qpos[:] = [0.6, (i - 2) * 0.20, 0.75, 1, 0, 0, 0]
            # Move clutter clear of the intentionally successful arrangement.
            for joint in env._task_randomizer._distractors:
                env.data.joint(joint).qpos[:3] = [-2, 0, 1]
            env.data.qvel[:] = 0
            mujoco.mj_forward(env.model, env.data)
            env.advance_physics(750)
            self.assertTrue(env.evaluate_task().scalar_success())
            for kwargs in ({"randomize": False}, {"options": {"randomization": {"bottle_variants": []}}}):
                self.assert_rejected_without_mutation(env, **kwargs)
            env.data.joint("bottle_1_joint").qpos[2] += 0.2
            env.advance_physics(1)
            self.assertFalse(env.evaluate_task().scalar_success())
            self.assertLess(env.evaluate_task().scalar_reward(), 0.7)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
