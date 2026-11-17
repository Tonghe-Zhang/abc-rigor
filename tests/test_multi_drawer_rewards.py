"""Ordered drawer retrieval and episode-unique exploration reward regressions."""

from types import SimpleNamespace
import unittest

import mujoco
import numpy as np

from abc_sim.env import MuJoCoYAMEnv
from abc_sim.randomization.requests import MultiDrawerSearchResetRequest
from abc_sim.randomization.tasks.multi_drawer_search import MultiDrawerSearchRandomizer
from abc_sim.task_eval.multi_drawer import (
    MultiDrawerSearchTargetInBinEvaluator,
    ShapedMultiDrawerSearchEvaluator,
)
from abc_sim.task_runtime.multi_drawer import MultiDrawerSearchRuntime
from abc_sim.task_specs import get_task_spec


def _model():
    drawers = "".join(
        f'<body pos="{i} 0 0"><joint name="drawer{i}" type="slide" range="0 0.216"/>'
        '<geom size=".01" mass="1" contype="0" conaffinity="0"/></body>'
        for i in range(6)
    )
    return mujoco.MjModel.from_xml_string(f"""
        <mujoco><option gravity="0 0 0"/><worldbody>{drawers}
          <body name="first" pos="1 0 0">
            <freejoint name="drawer_search_target_joint"/><geom size=".01" mass="1" contype="0" conaffinity="0"/>
          </body>
          <body name="second" pos="2 0 0">
            <freejoint name="second_joint"/><geom size=".01" mass="1" contype="0" conaffinity="0"/>
          </body>
          <body><freejoint name="drawer_search_goal_bin_joint"/>
            <geom size=".01" mass="1" contype="0" conaffinity="0"/>
            <site name="drawer_search_goal_region" type="box" size=".056 .052 .05"/>
          </body>
        </worldbody></mujoco>
    """)


class DrawerRewardsTest(unittest.TestCase):
    def setUp(self):
        self.model = _model()
        self.data = mujoco.MjData(self.model)
        self.metadata = {
            "active_drawers": [f"d{i}" for i in range(6)],
            "drawer_joints": [f"drawer{i}" for i in range(6)],
            "target_sequence": [
                {
                    "joint": "drawer_search_target_joint",
                    "body": "first",
                    "label": "apple",
                    "drawer": "d4",
                },
                {
                    "joint": "second_joint",
                    "body": "second",
                    "label": "lemon",
                    "drawer": "d5",
                },
            ],
        }
        self.randomization = SimpleNamespace(metadata=self.metadata)
        self.spec = get_task_spec("multi_drawer_search_2_shaped")
        self.evaluator = self.make_evaluator(ShapedMultiDrawerSearchEvaluator)

    def make_evaluator(self, cls):
        evaluator = cls(model=self.model, spec=self.spec)
        evaluator.configure_from_randomization(self.model, self.randomization)
        return evaluator

    def score(self):
        return self.evaluator.evaluate_qpos_batch(self.data.qpos[None])

    def open(self, index):
        self.data.qpos[self.model.joint(f"drawer{index}").qposadr[0]] = 0.15
        return self.score()

    def deposit(self, joint):
        address = self.model.joint(joint).qposadr[0]
        self.data.qpos[address : address + 3] = 0
        return self.score()

    def test_completed_target_is_not_scored_twice(self):
        self.evaluator = self.make_evaluator(MultiDrawerSearchTargetInBinEvaluator)
        result = self.deposit("drawer_search_target_joint")
        self.assertAlmostEqual(result.reward[0], 0.5)
        self.assertFalse(result.success[0])
        np.testing.assert_array_equal(result.metrics["target_pos"], [[2, 0, 0]])
        self.assertAlmostEqual(self.score().reward[0], 0.5)

    def test_threshold_reopens_gate_and_uncredited_openings(self):
        self.assertEqual(self.score().reward[0], 0)
        self.data.qpos[self.model.joint("drawer0").qposadr[0]] = 0.107
        self.assertEqual(self.score().reward[0], 0)
        self.assertAlmostEqual(self.open(0).reward[0], 0.05)
        self.data.qpos[self.model.joint("drawer0").qposadr[0]] = 0
        self.score()
        self.assertAlmostEqual(self.open(0).reward[0], 0.05)
        self.assertAlmostEqual(self.open(4).reward[0], 0.10)
        result = self.open(5)
        self.assertAlmostEqual(result.reward[0], 0.10)
        self.assertEqual(result.metrics["opened_drawer_count"][0], 3)
        self.assertAlmostEqual(
            self.deposit("drawer_search_target_joint").reward[0], 0.475
        )
        # The second target's drawer was already opened without credit.
        self.assertAlmostEqual(self.open(1).reward[0], 0.475)
        result = self.deposit("second_joint")
        self.assertEqual(result.reward[0], 1)
        self.assertTrue(result.success[0])

    def test_credit_cap_and_batch_state_reset(self):
        for i in range(4):
            self.open(i)
        self.assertAlmostEqual(self.open(4).metrics["drawer_opening_bonus"][0], 0.2)
        self.evaluator.reset(nworld=2)
        batch = np.stack([self.data.qpos, self.model.qpos0])
        result = self.evaluator.evaluate_qpos_batch(batch)
        np.testing.assert_allclose(result.metrics["drawer_opening_bonus"], [0.2, 0])
        self.evaluator.reset()
        self.data.qpos[:] = self.model.qpos0
        self.assertEqual(self.score().reward[0], 0)

    def test_credit_is_retained_and_reenabled_for_unseen_next_target(self):
        self.open(4)
        self.open(0)  # Ineligible, but now remembered.
        self.deposit("drawer_search_target_joint")
        self.assertAlmostEqual(self.open(0).metrics["drawer_opening_bonus"][0], 0.05)
        self.assertAlmostEqual(self.open(1).metrics["drawer_opening_bonus"][0], 0.10)
        self.assertAlmostEqual(self.open(5).metrics["drawer_opening_bonus"][0], 0.15)

    def test_chronological_replay_reconstructs_episode_history(self):
        states, expected = [], []
        for mutation in (
            lambda: self.open(4),
            lambda: self.open(0),
            lambda: self.deposit("drawer_search_target_joint"),
            lambda: self.open(1),
            lambda: self.open(5),
            lambda: self.deposit("second_joint"),
        ):
            result = mutation()
            states.append(self.data.qpos.copy())
            expected.append(result.reward.copy())
        replay = self.make_evaluator(ShapedMultiDrawerSearchEvaluator)
        actual = [replay.evaluate_qpos_batch(qpos[None]).reward for qpos in states]
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(
            replay._opened_drawers, self.evaluator._opened_drawers
        )

    def test_runtime_and_reward_share_progress_at_each_tick(self):
        runtime = MultiDrawerSearchRuntime()
        runtime.after_reset(self.model, self.data, randomization=self.randomization)
        env = MuJoCoYAMEnv.__new__(MuJoCoYAMEnv)
        env.model, env.data = self.model, self.data
        env._task_evaluator = self.evaluator
        env._task_runtime = runtime
        env._physics_evaluator = None
        env._physics_observer = None
        env._step_updates = True
        env._step_update_interval = 1
        env._control_decimation = 17
        self.deposit("drawer_search_target_joint")
        env.advance_physics(1)
        self.assertEqual(runtime.debug_state()["completed_target_count"], 1)
        self.assertEqual(env.prompt, "find the lemon")
        # A transient deposit remains completed for both consumers.
        self.data.qpos[self.model.joint("drawer_search_target_joint").qposadr[0]] = 1
        env.advance_physics(1)
        self.assertEqual(runtime.debug_state()["completed_target_count"], 1)
        self.assertEqual(self.score().metrics["completed_target_count"][0], 1)

    def test_native_sampler_default_and_explicit_count(self):
        randomizer = MultiDrawerSearchRandomizer()
        default_counts = set()
        for seed in range(25):
            _, default = randomizer._sample_selections(np.random.default_rng(seed))
            _, explicit = randomizer._sample_selections(
                np.random.default_rng(seed), target_count=2
            )
            default_counts.add(len(default["target_sequence"]))
            self.assertEqual(len(explicit["target_sequence"]), 2)
        self.assertEqual(default_counts, {2, 3, 4})
        self.assertEqual(self.spec.reset_request_options(), {"target_count": 2})
        self.assertIsNone(get_task_spec("multi_drawer_search").reset_request_options())
        for count in (0, 5, True, 2.5):
            with self.assertRaises(ValueError):
                MultiDrawerSearchResetRequest(target_count=count)


if __name__ == "__main__":
    unittest.main()
