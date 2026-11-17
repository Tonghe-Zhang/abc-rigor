"""Ordered drawer retrieval and episode-unique exploration reward regressions."""

import copy
from types import SimpleNamespace
import unittest
from unittest import mock

import mujoco
import numpy as np

from abc_sim.config import RobotSystemConfig
from abc_sim.env import MuJoCoYAMEnv
from abc_sim.randomization.core import RandomizationState, SceneRandomizer
from abc_sim.randomization.requests import MultiDrawerSearchResetRequest
from abc_sim.randomization.tasks.multi_drawer_search import MultiDrawerSearchRandomizer
from abc_sim.task_eval.multi_drawer import (
    MultiDrawerSearchTargetInBinEvaluator,
    ShapedMultiDrawerSearchEvaluator,
)
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
            "target_count": 2,
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

    def make_env(self, task="multi_drawer_search_2_shaped"):
        with mock.patch.object(
            MuJoCoYAMEnv, "setup_model", lambda env: env._bind_model(self.model)
        ):
            env = MuJoCoYAMEnv(RobotSystemConfig(), render_cameras=False)
        env.get_init_q = lambda: np.zeros(0)
        env.set_task(task)
        metadata = self.metadata

        class FixtureRandomizer(SceneRandomizer):
            def randomize(self, model, data, seed=None, request=None):
                return RandomizationState(
                    seed=seed or 0,
                    object_states={},
                    metadata=copy.deepcopy(metadata),
                )

        env._task_randomizer = FixtureRandomizer()
        env._task_randomizer.bind_env(env)
        return env

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

    def test_simultaneous_deposit_and_opening_use_previous_target_eligibility(self):
        for opened_source, expected_bonus, expected_reward in (
            (5, 0.10, 0.475),  # Eligible first target, ineligible next target.
            (4, 0.05, 0.425),  # Ineligible first target, eligible next target.
        ):
            with self.subTest(opened_source=opened_source):
                self.data.qpos[:] = self.model.qpos0
                self.evaluator.reset()
                self.assertAlmostEqual(self.open(opened_source).reward[0], 0.05)
                target = self.model.joint("drawer_search_target_joint").qposadr[0]
                self.data.qpos[target : target + 3] = 0
                self.data.qpos[self.model.joint("drawer0").qposadr[0]] = 0.15
                result = self.score()
                self.assertEqual(result.metrics["completed_target_count"][0], 1)
                self.assertEqual(result.metrics["opened_drawer_count"][0], 2)
                self.assertAlmostEqual(
                    result.metrics["drawer_opening_bonus"][0], expected_bonus
                )
                self.assertAlmostEqual(result.reward[0], expected_reward)
                # The next observation uses the newly selected target.
                result = self.open(1)
                self.assertAlmostEqual(result.metrics["drawer_opening_bonus"][0], 0.10)
                self.assertAlmostEqual(result.reward[0], 0.475)

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
        env = self.make_env()
        env.reset(seed=4)
        target = env.model.joint("drawer_search_target_joint")
        env.data.qpos[target.qposadr[0]] = -0.08
        env.data.qvel[target.dofadr[0]] = 10
        # Physics carries the target through the bin and out within this action.
        with (
            mock.patch.object(
                env, "_update_step_task_eval", wraps=env._update_step_task_eval
            ) as observe,
            mock.patch(
                "abc_sim.env.mujoco.mj_fwdPosition", wraps=mujoco.mj_fwdPosition
            ) as refresh,
        ):
            _, reward, _, _, info = env.step(np.zeros(0), render_obs=False)
        self.assertEqual(observe.call_count, 17)
        refresh.assert_not_called()
        self.assertGreater(env.data.qpos[target.qposadr[0]], 0.056)
        self.assertEqual(env.task_runtime_debug_state()["completed_target_count"], 1)
        self.assertEqual(env.prompt, "find the lemon")
        self.assertAlmostEqual(reward, 0.375)
        self.assertEqual(info["task_eval"]["completed_target_count"], 1)
        self.assertFalse(info["task_success"])

    def test_nonrandomized_reset_shares_retained_episode_definition(self):
        for task in ("multi_drawer_search", "multi_drawer_search_2_shaped"):
            with self.subTest(task=task):
                env = self.make_env(task)
                env.reset(seed=4)
                env.data.qpos[
                    env.model.joint("drawer_search_target_joint").qposadr[0]
                ] = 0
                env.advance_physics(1)
                self.assertEqual(
                    env.task_runtime_debug_state()["completed_target_count"], 1
                )
                _, info = env.reset(randomize=False)
                self.assertNotIn("randomization", info)
                state = env.task_runtime_debug_state()
                self.assertEqual(state["target_count"], 2)
                self.assertEqual(state["completed_target_count"], 0)
                self.assertEqual(env._step_task_eval.metrics["target_count"][0], 2)
                self.assertEqual(
                    env._step_task_eval.metrics["completed_target_count"][0], 0
                )
                self.assertEqual(env._step_task_eval.reward[0], 0)
                self.assertEqual(env.prompt, "find the apple")

    def test_bound_replay_configures_new_drawer_sequence_and_history(self):
        env = self.make_env()
        env.reset(seed=4)
        env.data.qpos[env.model.joint("drawer0").qposadr[0]] = 0.15
        env.advance_physics(1)
        self.assertAlmostEqual(
            env._step_task_eval.metrics["drawer_opening_bonus"][0], 0.05
        )
        metadata = copy.deepcopy(self.metadata)
        metadata["target_sequence"].reverse()
        state = RandomizationState(seed=8, object_states={}, metadata=metadata)
        randomizer = SceneRandomizer()
        randomizer.bind_env(env)
        env.data.qpos[:] = env.model.qpos0
        randomizer.apply(env.model, env.data, state)
        self.assertIs(env._episode_restorer, randomizer)
        self.assertEqual(env.prompt, "find the lemon")
        self.assertEqual(env._step_task_eval.metrics["drawer_opening_bonus"][0], 0)
        self.assertEqual(env.task_runtime_debug_state()["completed_target_count"], 0)
        env.reset(randomize=False)
        self.assertEqual(env.prompt, "find the lemon")

    def test_registry_sampler_is_cloned_and_bound_for_retained_reset(self):
        env = self.make_env()
        prototype = type(env._task_randomizer)()
        del env._task_randomizer
        with mock.patch("abc_sim.env.get_task_randomizer", return_value=prototype):
            env.reset(seed=4)
        self.assertIsNone(prototype._env_ref)
        self.assertIsNot(env._episode_restorer, prototype)
        self.assertIs(env._episode_restorer._env_ref, env)
        self.assertIs(env._episode_restorer, env._task_randomizer)
        sampler = env._task_randomizer
        env.data.qpos[env.model.joint("drawer_search_target_joint").qposadr[0]] = 0
        env.advance_physics(1)
        env.reset(randomize=False)
        self.assertEqual(env.prompt, "find the apple")
        self.assertEqual(env.task_runtime_debug_state()["completed_target_count"], 0)
        self.assertEqual(env._step_task_eval.reward[0], 0)
        env.reset(seed=5)
        self.assertIs(env._task_randomizer, sampler)
        self.assertIs(env._episode_restorer, sampler)

    def test_named_task_dispatch_preserves_exact_count_for_empty_and_typed_options(
        self,
    ):
        class SelectionCaptured(Exception):
            pass

        for request in (
            None,
            {},
            {"target_count": None},
            {"target_count": 2},
            MultiDrawerSearchResetRequest(),
            MultiDrawerSearchResetRequest(target_count=2),
        ):
            with self.subTest(request=request):
                env = self.make_env()
                randomizer = MultiDrawerSearchRandomizer()
                env._task_randomizer = randomizer
                select = randomizer._sample_selections
                captured = []

                def capture(rng, **kwargs):
                    _, selection = select(rng, **kwargs)
                    captured.append(selection["target_sequence"])
                    raise SelectionCaptured

                with mock.patch.object(
                    randomizer, "_sample_selections", side_effect=capture
                ):
                    with self.assertRaises(SelectionCaptured):
                        env.reset(seed=0, options={"randomization": request})
                self.assertEqual(len(captured[0]), 2)

    def test_conflicting_task_count_is_rejected_before_reset_mutation(self):
        for request in (
            {"target_count": 1},
            {"target_count": 3},
            {"target_count": 4},
            {"target_count": True},
            {"target_count": 2.0},
            MultiDrawerSearchResetRequest(target_count=3),
        ):
            with self.subTest(request=request):
                env = self.make_env()
                env.reset(seed=4)
                env.data.time = 12
                env.data.qpos[0] = 0.15
                qpos = env.data.qpos.copy()
                observer = env._step_task_eval
                rng_state = copy.deepcopy(env.np_random.bit_generator.state)
                with mock.patch.object(env._task_randomizer, "randomize") as sample:
                    with self.assertRaisesRegex(ValueError, "requires target_count=2"):
                        env.reset(seed=9, options={"randomization": request})
                    sample.assert_not_called()
                np.testing.assert_array_equal(env.data.qpos, qpos)
                self.assertEqual(env.data.time, 12)
                self.assertIs(env._step_task_eval, observer)
                self.assertEqual(env.np_random.bit_generator.state, rng_state)

    def test_shaped_replay_rejects_conflicting_target_count(self):
        metadata = copy.deepcopy(self.metadata)
        metadata["target_sequence"] = metadata["target_sequence"][:1]
        with self.assertRaisesRegex(ValueError, "target sequence length"):
            self.evaluator.configure_from_randomization(
                self.model, SimpleNamespace(metadata=metadata)
            )

    def test_replay_task_preflight_precedes_transient_reset(self):
        env = self.make_env()
        env.reset(seed=4)
        env.data.time = 12
        env.data.qvel[:] = 0.3
        randomizer = SceneRandomizer()
        randomizer.bind_env(env)
        for count in (3, None):
            with self.subTest(count=count):
                metadata = copy.deepcopy(self.metadata)
                if count is None:
                    del metadata["target_count"]
                else:
                    metadata["target_count"] = count
                state = RandomizationState(seed=0, object_states={}, metadata=metadata)
                with mock.patch.object(
                    env, "initialize_randomization", wraps=env.initialize_randomization
                ) as initialize:
                    with self.assertRaises(ValueError):
                        randomizer.apply(env.model, env.data, state)
                    initialize.assert_not_called()
                self.assertEqual(env.data.time, 12)
                np.testing.assert_array_equal(env.data.qvel, 0.3)

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
        native = get_task_spec("multi_drawer_search")
        self.assertIs(get_task_spec(native.prompt), native)
        for count in (1, 2, 3, 4):
            self.assertEqual(
                get_task_spec("multi_drawer_search").resolve_reset_request(
                    {"target_count": count}
                ),
                {"target_count": count},
            )
        for count in (0, 5, True, 2.5):
            with self.assertRaises(ValueError):
                MultiDrawerSearchResetRequest(target_count=count)


if __name__ == "__main__":
    unittest.main()
