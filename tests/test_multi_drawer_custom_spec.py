"""Public drawer lifecycle with optional reward exposure and shared progress."""

import copy
import gc
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

import abc_sim
from abc_sim.randomization.core import RandomizationState
from abc_sim.randomization.tasks.multi_drawer_search import MultiDrawerSearchRandomizer
from abc_sim.task_eval import registry as evaluator_registry
from abc_sim.task_specs import SimTaskSpec
from tests.test_multi_drawer_scene import _model
from tests.test_spelling_goals import blocks_xml


def drawer_xml():
    """Serialize the complete small drawer fixture with its policy robot joints."""
    model = _model()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "drawer.xml"
        mujoco.mj_saveLastXML(str(path), model)
        root = ET.parse(path).getroot()
    robots = ET.fromstring(blocks_xml())
    for body in robots.find("worldbody"):
        if body.find("joint") is not None:
            root.find("worldbody").append(copy.deepcopy(body))
    root.append(copy.deepcopy(robots.find("actuator")))
    return ET.tostring(root, encoding="unicode")


class DrawerCustomSpecTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.xml = drawer_xml()

    def make_env(self, task, enabled, **kwargs):
        env = abc_sim.make_env(
            task,
            scene_xml_string=self.xml,
            enable_task_randomizer=enabled,
            render_cameras=False,
            camera_height=4,
            camera_width=4,
            control_decimation=3,
            **kwargs,
        )
        self.addCleanup(env.close)
        return env

    def assert_initial_episode(self, env, record, qpos):
        np.testing.assert_array_equal(env.data.qpos, qpos)
        self.assertEqual(env.data.time, 0)
        self.assertEqual(env.cur_step, 0)
        self.assertIsNone(env.evaluate_task())
        self.assertIsNone(env._task_evaluator)
        self.assertEqual(env.task_runtime_debug_state()["completed_target_count"], 0)
        self.assertEqual(
            env.prompt, f"find the {record['metadata']['target_sequence'][0]['label']}"
        )
        self.assertEqual(
            json.loads(json.dumps(env._last_randomization.to_dict())), record
        )

    @staticmethod
    def deposit(env, target):
        mujoco.mj_forward(env.model, env.data)
        goal = env.data.site("drawer_search_goal_region").xpos.copy()
        address = env.model.joint(target["joint"]).qposadr[0]
        env.data.qpos[address : address + 3] = goal
        dof = env.model.joint(target["joint"]).dofadr[0]
        env.data.qvel[dof : dof + 6] = 0

    @staticmethod
    def restore_target(env, target, initial_qpos):
        address = env.model.joint(target["joint"]).qposadr[0]
        env.data.qpos[address : address + 7] = initial_qpos[address : address + 7]
        dof = env.model.joint(target["joint"]).dofadr[0]
        env.data.qvel[dof : dof + 6] = 0

    def test_unscored_custom_task_reset_replay_and_ordered_prompt_progress(self):
        task = SimTaskSpec(
            name="drawer_preview",
            env_task="multi_drawer_search",
            prompt="find the target",
        )
        for enabled in (True, False):
            with self.subTest(initial_sampler=enabled):
                env = self.make_env(
                    task, enabled, terminate_on_success=True, max_episode_steps=2
                )
                self.assertEqual(hasattr(env, "_task_randomizer"), enabled)
                _, info = env.reset(
                    seed=6, options={"randomization": {"target_count": 2}}
                )
                record = json.loads(json.dumps(info["randomization"].to_dict()))
                saved = RandomizationState.from_dict(record)
                initial_qpos = env.data.qpos.copy()
                self.assert_initial_episode(env, record, initial_qpos)
                first, second = record["metadata"]["target_sequence"]
                first_prompt, second_prompt = (
                    f"find the {target['label']}" for target in (first, second)
                )

                self.deposit(env, second)
                env.advance_physics(1)
                self.assertEqual(
                    env.task_runtime_debug_state()["completed_target_count"], 0
                )
                self.assertEqual(env.prompt, first_prompt)
                self.restore_target(env, second, initial_qpos)
                self.deposit(env, first)
                env.advance_physics(1)
                self.assertEqual(
                    env.task_runtime_debug_state()["completed_target_count"], 1
                )
                self.assertEqual(env.get_obs()["prompt"], second_prompt)
                self.restore_target(env, first, initial_qpos)
                env.advance_physics(1)
                self.assertEqual(
                    env.task_runtime_debug_state()["completed_target_count"], 1
                )
                self.deposit(env, second)
                _, reward, terminated, truncated, step_info = env.step(
                    env.get_state(), render_obs=False
                )
                self.assertEqual(reward, 0)
                self.assertFalse(terminated)
                self.assertFalse(truncated)
                self.assertIsNone(env.evaluate_task())
                for key in ("task_reward", "task_success", "task_eval"):
                    self.assertNotIn(key, step_info)
                self.assertEqual(
                    env.task_runtime_debug_state()["completed_target_count"], 2
                )
                self.assertTrue(env.task_runtime_debug_state()["complete"])
                self.assertEqual(env.get_obs()["prompt"], "finished")
                _, reward, terminated, truncated, step_info = env.step(
                    env.get_state(), render_obs=False
                )
                self.assertEqual(reward, 0)
                self.assertFalse(terminated)
                self.assertTrue(truncated)
                self.assertEqual(step_info["episode_step"], 2)
                self.assertTrue(step_info["time_limit_reached"])
                for key in ("task_reward", "task_success", "task_eval"):
                    self.assertNotIn(key, step_info)

                owner = env._episode_restorer
                env.reset(randomize=False)
                self.assert_initial_episode(env, record, initial_qpos)
                self.assertIs(env._episode_restorer, owner)
                env.reset(seed=0, options={"randomization": {"target_count": 2}})
                self.assertNotEqual(
                    env._last_randomization.metadata["object_set"],
                    record["metadata"]["object_set"],
                )
                owner.apply(env.model, env.data, saved)
                self.assert_initial_episode(env, record, initial_qpos)

                fresh = self.make_env(task, enabled)
                restorer = MultiDrawerSearchRandomizer()
                restorer.bind_env(fresh)
                restorer.apply(fresh.model, fresh.data, saved)
                self.assert_initial_episode(fresh, record, initial_qpos)
                self.assertEqual(hasattr(fresh, "_task_randomizer"), enabled)
                self.deposit(fresh, first)
                fresh.advance_physics(1)
                self.assertEqual(fresh.prompt, second_prompt)
                fresh.reset(randomize=False)
                self.assert_initial_episode(fresh, record, initial_qpos)
                self.assertIs(fresh._episode_restorer, restorer)

    def test_named_tasks_share_one_history_evaluation_per_physics_tick(self):
        for task, expected_reward in (
            ("multi_drawer_search", 0.5),
            ("multi_drawer_search_2_shaped", 0.375),
        ):
            with self.subTest(task=task):
                env = self.make_env(task, True)
                if task == "multi_drawer_search":
                    initial = env.evaluate_task()
                    self.assertEqual(initial.num_worlds, 1)
                    self.assertIs(env.evaluate_task(), initial)
                else:
                    with self.assertRaisesRegex(
                        RuntimeError, "requires reset randomization metadata"
                    ):
                        env.evaluate_task()
                _, info = env.reset(
                    seed=6, options={"randomization": {"target_count": 2}}
                )
                first, second = info["randomization"].metadata["target_sequence"]
                self.assertIs(env._step_evaluator, env._task_evaluator)
                self.deposit(env, first)
                with mock.patch.object(
                    env._step_evaluator,
                    "evaluate_qpos_batch",
                    wraps=env._step_evaluator.evaluate_qpos_batch,
                ) as evaluate:
                    for expected_calls in (3, 6):
                        _, reward, _, _, step_info = env.step(
                            env.get_state(), render_obs=False
                        )
                        self.assertEqual(evaluate.call_count, expected_calls)
                        self.assertAlmostEqual(reward, expected_reward)
                        self.assertEqual(step_info["task_reward"], reward)
                        self.assertEqual(
                            step_info["task_eval"]["completed_target_count"], 1
                        )
                        self.assertEqual(env.prompt, f"find the {second['label']}")
                        for _ in range(3):
                            self.assertAlmostEqual(
                                env.evaluate_task().scalar_reward(), expected_reward
                            )
                            cached_reward, cached_info = env._task_step_info()
                            self.assertAlmostEqual(cached_reward, expected_reward)
                            self.assertEqual(
                                cached_info["task_eval"]["completed_target_count"], 1
                            )
                        self.assertEqual(evaluate.call_count, expected_calls)

    def test_fresh_fixed_count_rejects_before_state_changes_then_accepts_recordings(
        self,
    ):
        for scoring in (None, "multi_drawer_target_in_bin"):
            for enabled in (False, True):
                with self.subTest(scoring=scoring, initial_sampler=enabled):
                    task = SimTaskSpec(
                        name="fixed_two_preview",
                        env_task="multi_drawer_search",
                        prompt="find two targets",
                        evaluator_name=scoring,
                        reset_constraints=(("target_count", 2),),
                    )
                    env = self.make_env(task, enabled)
                    env.np_random = np.random.default_rng(89)
                    env.data.qvel[:] = 0.3
                    env.data.ctrl[:] = 0.2
                    env.data.time = 12
                    env.cur_step = 7

                    def observer(model, data):
                        return None

                    env._physics_observer = observer
                    physical = {
                        name: getattr(env.data, name).copy()
                        for name in (
                            "qpos",
                            "qvel",
                            "ctrl",
                            "qacc_warmstart",
                            "xfrc_applied",
                            "qfrc_applied",
                        )
                    }
                    rng = copy.deepcopy(env.np_random.bit_generator.state)
                    runtime = env.task_runtime_debug_state()
                    model, data = env.model, env.data
                    source, controller = env._step_evaluator, env._task_runtime
                    sampler = getattr(env, "_task_randomizer", None)
                    attempts = [
                        lambda: env.reset(seed=123, randomize=False),
                        lambda: env.finalize_randomization(None, restorer=None),
                    ]
                    if scoring is not None:
                        attempts.extend(
                            [
                                env.evaluate_task,
                                lambda: source.evaluate_qpos_batch(env.data.qpos[None]),
                            ]
                        )
                    for attempt in attempts:
                        with self.assertRaisesRegex(
                            ValueError, "requires target_count=2"
                        ):
                            attempt()
                        for name, values in physical.items():
                            np.testing.assert_array_equal(
                                getattr(env.data, name), values
                            )
                        self.assertEqual(env.data.time, 12)
                        self.assertEqual(env.cur_step, 7)
                        self.assertEqual(env.np_random.bit_generator.state, rng)
                        self.assertEqual(env.task_runtime_debug_state(), runtime)
                        self.assertEqual(env.prompt, task.prompt)
                        self.assertIs(env.model, model)
                        self.assertIs(env.data, data)
                        self.assertIs(env._step_evaluator, source)
                        self.assertIs(env._task_runtime, controller)
                        self.assertIs(env._physics_observer, observer)
                        self.assertIsNone(source._episode)
                        self.assertIsNone(env._last_randomization)
                        self.assertIsNone(env._episode_restorer)
                        self.assertIsNone(env._step_task_eval)
                        self.assertIs(getattr(env, "_task_randomizer", None), sampler)
                        np.testing.assert_array_equal(
                            source._completed_target_count, [0]
                        )

                    _, info = env.reset(seed=6)
                    record = json.loads(json.dumps(info["randomization"].to_dict()))
                    initial = env.data.qpos.copy()
                    self.assertEqual(env.task_runtime_debug_state()["target_count"], 2)
                    self.deposit(env, record["metadata"]["target_sequence"][0])
                    env.advance_physics(1)
                    self.assertEqual(
                        env.task_runtime_debug_state()["completed_target_count"], 1
                    )
                    env.reset(randomize=False)
                    np.testing.assert_array_equal(env.data.qpos, initial)
                    self.assertEqual(
                        env.task_runtime_debug_state()["completed_target_count"], 0
                    )
                    owner = env._episode_restorer
                    env.reset(seed=0)
                    owner.apply(
                        env.model, env.data, RandomizationState.from_dict(record)
                    )
                    np.testing.assert_array_equal(env.data.qpos, initial)
                    self.assertEqual(env.task_runtime_debug_state()["target_count"], 2)
                    self.assertEqual(
                        env.task_runtime_debug_state()["completed_target_count"], 0
                    )

    def test_compatible_compiled_episode_uses_native_single_target(self):
        for scoring in (None, "multi_drawer_target_in_bin"):
            for fixed_count in (False, True):
                for enabled in (False, True):
                    with self.subTest(
                        scoring=scoring,
                        fixed_count=fixed_count,
                        initial_sampler=enabled,
                    ):
                        task = SimTaskSpec(
                            name="compatible_preview",
                            env_task="multi_drawer_search",
                            prompt="find the target",
                            evaluator_name=scoring,
                            reset_request=(("target_count", 2),)
                            if not fixed_count
                            else (),
                            reset_constraints=(("target_count", 1),)
                            if fixed_count
                            else (),
                        )
                        env = self.make_env(task, enabled)
                        if scoring is not None:
                            self.assertEqual(
                                env.evaluate_task().metrics["target_count"][0], 1
                            )
                        _, info = env.reset(seed=12, randomize=False)
                        self.assertNotIn("randomization", info)
                        self.assertIsNone(env._last_randomization)
                        self.assertIsNone(env._episode_restorer)
                        self.assertEqual(
                            env.task_runtime_debug_state()["target_count"], 1
                        )
                        self.assertEqual(env.prompt, "find the target object")
                        self.deposit(env, {"joint": "drawer_search_target_joint"})
                        env.advance_physics(1)
                        self.assertEqual(env.prompt, "finished")
                        self.assertTrue(env.task_runtime_debug_state()["complete"])
                        if scoring is None:
                            self.assertIsNone(env.evaluate_task())
                        else:
                            self.assertEqual(env.evaluate_task().scalar_reward(), 1)
                            self.assertTrue(env.evaluate_task().scalar_success())
                        env.reset(randomize=False)
                        self.assertEqual(env.prompt, "find the target object")
                        self.assertEqual(
                            env.task_runtime_debug_state()["completed_target_count"], 0
                        )

    def test_incompatible_reward_source_rejects_as_a_configuration_error(self):
        class NoDrawerProgressEvaluator:
            def __init__(self, *, model, spec):
                self.spec = spec

            def reset(self, *, nworld=1):
                pass

            def evaluate_qpos_batch(self, qpos_batch):
                raise AssertionError("An incompatible evaluator must never run.")

        task = SimTaskSpec(
            name="drawer_incompatible",
            env_task="multi_drawer_search",
            prompt="find the target",
            evaluator_name="test_only_no_drawer_progress",
        )
        with mock.patch.dict(
            evaluator_registry._TASK_EVALUATORS,
            {"test_only_no_drawer_progress": NoDrawerProgressEvaluator},
        ):
            for enabled in (True, False):
                with self.subTest(initial_sampler=enabled):
                    with self.assertRaisesRegex(ValueError, "drawer progress"):
                        self.make_env(task, enabled)

    def test_incompatible_randomizer_binding_leaves_the_restorer_unbound(self):
        task = SimTaskSpec(
            name="non_drawer_preview",
            env_task="bottles",
            prompt="inspect the objects",
        )
        env = self.make_env(task, False)
        restorer = MultiDrawerSearchRandomizer()
        with self.assertRaisesRegex(ValueError, "drawer progress"):
            restorer.bind_env(env)
        self.assertIsNone(restorer._env_ref)
        self.assertIsNone(restorer._base_scene_xml_string)
        self.assertIsNone(env._episode_restorer)
        self.assertFalse(hasattr(env, "_task_randomizer"))

    def test_native_template_fixed_task_rejects_fallback_then_assembles(self):
        for scoring in (None, "multi_drawer_target_in_bin"):
            task = SimTaskSpec(
                name="drawer_preview",
                env_task="multi_drawer_search",
                prompt="find the target",
                evaluator_name=scoring,
                reset_constraints=(("target_count", 2),),
            )
            for enabled in (True, False):
                with self.subTest(scoring=scoring, initial_sampler=enabled):
                    env = abc_sim.make_env(
                        task,
                        enable_task_randomizer=enabled,
                        render_cameras=False,
                        camera_height=4,
                        camera_width=4,
                    )
                    try:
                        template_id = mujoco.mj_name2id(
                            env.model, mujoco.mjtObj.mjOBJ_BODY, "drawer_body"
                        )
                        self.assertEqual(template_id >= 0, not enabled)
                        model, data, source = (
                            env.model,
                            env.data,
                            env._step_evaluator,
                        )
                        env.np_random = np.random.default_rng(89)
                        rng = copy.deepcopy(env.np_random.bit_generator.state)
                        initial_qpos = env.data.qpos.copy()
                        env.data.time = 12
                        with self.assertRaisesRegex(
                            ValueError, "requires target_count=2"
                        ):
                            env.reset(seed=123, randomize=False)
                        self.assertIs(env.model, model)
                        self.assertIs(env.data, data)
                        self.assertIs(env._step_evaluator, source)
                        self.assertEqual(env.np_random.bit_generator.state, rng)
                        self.assertEqual(env.data.time, 12)
                        np.testing.assert_array_equal(env.data.qpos, initial_qpos)
                        self.assertIsNone(env._last_randomization)
                        self.assertIsNone(env._episode_restorer)
                        self.assertEqual(hasattr(env, "_task_randomizer"), enabled)

                        _, info = env.reset(seed=6)
                        self.assertEqual(
                            mujoco.mj_name2id(
                                env.model, mujoco.mjtObj.mjOBJ_BODY, "drawer_body"
                            ),
                            -1,
                        )
                        first, second = info["randomization"].metadata[
                            "target_sequence"
                        ]
                        self.assertEqual(env.prompt, f"find the {first['label']}")
                        self.deposit(env, first)
                        env.advance_physics(1)
                        self.assertEqual(
                            env.task_runtime_debug_state()["completed_target_count"],
                            1,
                        )
                        self.assertEqual(
                            env.get_obs()["prompt"], f"find the {second['label']}"
                        )
                        if scoring is None:
                            self.assertIsNone(env.evaluate_task())
                            self.assertIsNone(env._task_evaluator)
                        else:
                            self.assertEqual(
                                env.evaluate_task().metrics["completed_target_count"][
                                    0
                                ],
                                1,
                            )
                    finally:
                        env.close()
                        del env
                        gc.collect()


if __name__ == "__main__":
    unittest.main()
