"""Goal-site containment through public drawer scoring and prompt progression."""

import copy
import json
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

import abc_sim
from abc_sim.randomization.core import RandomizationState
from abc_sim.randomization.tasks.multi_drawer_search import MultiDrawerSearchRandomizer
from abc_sim.task_eval.multi_drawer import MultiDrawerSearchTargetInBinEvaluator
from abc_sim.task_specs import SimTaskSpec
from tests.test_multi_drawer_custom_spec import drawer_xml
from tests.test_multi_drawer_scene import misplaced_goal


TASKS = (
    "multi_drawer_search",
    "multi_drawer_search_2_shaped",
    SimTaskSpec(
        name="drawer_preview",
        env_task="multi_drawer_search",
        prompt="find the target",
    ),
)


class DrawerGoalTransformTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = ET.fromstring(drawer_xml())
        site = root.find(".//site[@name='drawer_search_goal_region']")
        site.set("size", ".1 .02 .05")
        site.set("pos", ".03 -.01 .015")
        site.set("quat", "1 0 0 0")
        cls.identity_xml = ET.tostring(root, encoding="unicode")
        site.set("quat", f"{2**-0.5} 0 0 {2**-0.5}")
        cls.rotated_xml = ET.tostring(root, encoding="unicode")

    def make_env(self, task, xml):
        env = abc_sim.make_env(
            task,
            scene_xml_string=xml,
            render_cameras=False,
            camera_height=4,
            camera_width=4,
            control_decimation=1,
        )
        self.addCleanup(env.close)
        return env

    def position_target_and_check_site_oracle(self, env, target, bin_offset, expected):
        bin_address = env.model.joint("drawer_search_goal_bin_joint").qposadr[0]
        axis = np.array([1, 2, 3]) / np.sqrt(14)
        env.data.qpos[bin_address : bin_address + 7] = [
            0.37,
            -0.19,
            0.8,
            np.cos(0.35),
            *(axis * np.sin(0.35)),
        ]
        env.data.qvel[:] = 0
        mujoco.mj_forward(env.model, env.data)
        goal_site = env.data.site("drawer_search_goal_region")
        bin_rotation = env.data.body("drawer_search_goal_bin").xmat.reshape(3, 3)
        point = goal_site.xpos + bin_rotation @ bin_offset
        address = env.model.joint(target["joint"]).qposadr[0]
        env.data.qpos[address : address + 3] = point
        mujoco.mj_forward(env.model, env.data)

        # MuJoCo's independently evaluated site transform supplies the oracle.
        goal_site = env.data.site("drawer_search_goal_region")
        site_local = goal_site.xmat.reshape(3, 3).T @ (
            env.data.body(target["body"]).xpos - goal_site.xpos
        )
        inside = bool(
            np.all(
                np.abs(site_local) <= env.model.site("drawer_search_goal_region").size
            )
        )
        self.assertEqual(inside, expected)

    def assert_one_observation(self, env, targets, expected):
        source = env._step_evaluator
        if env._task_evaluator is not None:
            self.assertIs(source, env._task_evaluator)
        with (
            mock.patch.object(
                mujoco,
                "mj_fwdPosition",
                side_effect=AssertionError(
                    "Qpos observation must not refresh positions."
                ),
            ),
            mock.patch.object(
                source, "evaluate_qpos_batch", wraps=source.evaluate_qpos_batch
            ) as observe,
        ):
            _, reward, _, _, info = env.step(env.get_state(), render_obs=False)
            result = env.evaluate_task()
            self.assertEqual(observe.call_count, 1)
        self.assertEqual(
            env.task_runtime_debug_state()["completed_target_count"], int(expected)
        )
        self.assertEqual(env.prompt, f"find the {targets[int(expected)]['label']}")
        if env._task_evaluator is None:
            self.assertIsNone(result)
            self.assertEqual(reward, 0)
            for key in ("task_reward", "task_success", "task_eval"):
                self.assertNotIn(key, info)
        else:
            self.assertEqual(info["task_eval"]["completed_target_count"], int(expected))
            self.assertEqual(
                bool(result.metrics["target_sequence_in_bin"][0, 0]), expected
            )
            self.assertEqual(result.scalar_reward(), reward)

    def test_local_site_rotation_and_bin_pose_match_the_site_oracle(self):
        for task in TASKS:
            for rotated in (False, True):
                with self.subTest(task=task, rotated_site=rotated):
                    env = self.make_env(
                        task, self.rotated_xml if rotated else self.identity_xml
                    )
                    for bin_offset, identity_inside in (
                        ((0.08, 0, 0), True),
                        ((0, 0.08, 0), False),
                    ):
                        expected = not identity_inside if rotated else identity_inside
                        with self.subTest(bin_offset=bin_offset, expected=expected):
                            _, info = env.reset(
                                seed=6,
                                options={"randomization": {"target_count": 2}},
                            )
                            targets = info["randomization"].metadata["target_sequence"]
                            self.position_target_and_check_site_oracle(
                                env, targets[0], bin_offset, expected
                            )
                            self.assert_one_observation(env, targets, expected)

    def test_public_model_reload_and_replay_refresh_the_goal_transform(self):
        for task in TASKS:
            with self.subTest(task=task):
                env = self.make_env(task, self.identity_xml)
                _, info = env.reset(
                    seed=6, options={"randomization": {"target_count": 2}}
                )
                record = json.loads(json.dumps(info["randomization"].to_dict()))
                targets = record["metadata"]["target_sequence"]
                restorer = env._episode_restorer
                for xml, expected in (
                    (self.rotated_xml, False),
                    (self.identity_xml, True),
                ):
                    with self.subTest(expected=expected):
                        previous_model = env.model
                        previous_source = env._step_evaluator
                        env.reload_from_xml(xml)
                        self.assertIsNot(env.model, previous_model)
                        self.assertIsNot(env._step_evaluator, previous_source)
                        restored = RandomizationState.from_dict(copy.deepcopy(record))
                        restorer.apply(env.model, env.data, restored)
                        self.assertIs(env._episode_restorer, restorer)
                        self.position_target_and_check_site_oracle(
                            env, targets[0], (0.08, 0, 0), expected
                        )
                        self.assert_one_observation(env, targets, expected)

    def test_compiled_entry_points_reject_wrong_goal_owner_without_mutation(self):
        for task in (TASKS[0], TASKS[2]):
            for enabled in (False, True):
                with self.subTest(task=task, initial_sampler=enabled):
                    env = abc_sim.make_env(
                        task,
                        scene_xml_string=self.identity_xml,
                        enable_task_randomizer=enabled,
                        render_cameras=False,
                    )
                    self.addCleanup(env.close)
                    source = env._step_evaluator
                    episode = source.prepare_randomization(env.model, None)
                    site = env.model.site("drawer_search_goal_region").id
                    owner = int(env.model.site_bodyid[site])
                    env.model.site_bodyid[site] = 0
                    env.np_random = np.random.default_rng(89)
                    rng = copy.deepcopy(env.np_random.bit_generator.state)
                    env.data.qvel[:] = 0.3
                    env.data.ctrl[:] = 0.2
                    env.data.time = 12
                    env.cur_step = 7
                    physical = {
                        name: getattr(env.data, name).copy()
                        for name in ("qpos", "qvel", "ctrl")
                    }
                    history = env.task_runtime_debug_state()
                    sampler = getattr(env, "_task_randomizer", None)
                    attempts = [
                        lambda: env.reset(seed=123, randomize=False),
                        lambda: env.finalize_randomization(None, restorer=None),
                        lambda: source.prepare_randomization(env.model, None),
                        lambda: source.configure_episode(env.model, episode),
                        lambda: source.evaluate_qpos_batch(env.data.qpos[None]),
                    ]
                    if env._task_evaluator is not None:
                        attempts.append(env.evaluate_task)
                    for attempt in attempts:
                        with self.assertRaisesRegex(
                            ValueError, "goal region must belong"
                        ):
                            attempt()
                        for name, values in physical.items():
                            np.testing.assert_array_equal(
                                getattr(env.data, name), values
                            )
                        self.assertEqual(env.data.time, 12)
                        self.assertEqual(env.cur_step, 7)
                        self.assertEqual(env.np_random.bit_generator.state, rng)
                        self.assertEqual(env.task_runtime_debug_state(), history)
                        self.assertIsNone(source._episode)
                        self.assertIsNone(env._step_task_eval)
                        self.assertIsNone(env._last_randomization)
                        self.assertIsNone(env._episode_restorer)
                        self.assertIs(getattr(env, "_task_randomizer", None), sampler)
                    restorer = MultiDrawerSearchRandomizer()
                    with self.assertRaisesRegex(ValueError, "goal region must belong"):
                        restorer.bind_env(env)
                    self.assertIsNone(restorer._env_ref)
                    self.assertIsNone(restorer._base_scene_xml_string)
                    env.model.site_bodyid[site] = owner
                    env.reset(randomize=False)
                    self.assertEqual(env.task_runtime_debug_state()["target_count"], 1)

    def test_candidate_and_replay_goal_validation_preserves_live_episode(self):
        bad_xml = misplaced_goal(self.identity_xml)
        bad_model = mujoco.MjModel.from_xml_string(bad_xml)
        for task in TASKS:
            with self.subTest(task=task):
                env = self.make_env(task, self.identity_xml)
                _, info = env.reset(
                    seed=6, options={"randomization": {"target_count": 2}}
                )
                saved = info["randomization"]
                source = env._step_evaluator
                episode = source.prepare_randomization(env.model, saved)
                model, data = env.model, env.data
                owner = env._episode_restorer
                evaluation = env._step_task_eval
                history = env.task_runtime_debug_state()
                initial_episode = source._episode
                rng = copy.deepcopy(env.np_random.bit_generator.state)
                xml = env._scene_xml_string
                env.data.qvel[:] = 0.3
                env.data.ctrl[:] = 0.2
                env.data.time = 12
                env.cur_step = 7
                physical = {
                    name: getattr(env.data, name).copy()
                    for name in ("qpos", "qvel", "ctrl")
                }
                attempts = [
                    lambda: source.prepare_randomization(bad_model, saved),
                    lambda: source.configure_episode(bad_model, episode),
                    lambda: env.reload_from_xml(bad_xml),
                    lambda: env.reload_from_model(bad_model),
                ]
                with (
                    mock.patch.object(
                        env, "_close_camera_backend", wraps=env._close_camera_backend
                    ) as close_cameras,
                    mock.patch.object(
                        env,
                        "initialize_randomization",
                        wraps=env.initialize_randomization,
                    ) as initialize,
                ):
                    for attempt in attempts:
                        with self.assertRaisesRegex(
                            ValueError, "goal region must belong"
                        ):
                            attempt()
                        self.assertIs(env.model, model)
                        self.assertIs(env.data, data)
                        self.assertIs(source.model, model)
                        self.assertIs(source._episode, initial_episode)
                        self.assertIs(env._step_evaluator, source)
                        self.assertIs(env._step_task_eval, evaluation)
                        self.assertIs(env._episode_restorer, owner)
                        self.assertIs(env._last_randomization, saved)
                        self.assertEqual(env._scene_xml_string, xml)
                        self.assertEqual(env.task_runtime_debug_state(), history)
                        self.assertEqual(env.np_random.bit_generator.state, rng)
                        for name, values in physical.items():
                            np.testing.assert_array_equal(
                                getattr(env.data, name), values
                            )
                        self.assertEqual(env.data.time, 12)
                        self.assertEqual(env.cur_step, 7)
                    site = model.site("drawer_search_goal_region").id
                    original_owner = int(model.site_bodyid[site])
                    model.site_bodyid[site] = 0
                    with self.assertRaisesRegex(ValueError, "goal region must belong"):
                        owner.apply(model, data, saved)
                    model.site_bodyid[site] = original_owner
                    close_cameras.assert_not_called()
                    initialize.assert_not_called()
                self.assertEqual(env.task_runtime_debug_state(), history)
                self.assertIs(env._episode_restorer, owner)
                self.assertIs(env._step_task_eval, evaluation)
                self.assertIs(source._episode, initial_episode)
                self.assertEqual(env.np_random.bit_generator.state, rng)
                for name, values in physical.items():
                    np.testing.assert_array_equal(getattr(env.data, name), values)
                self.assertEqual(env.data.time, 12)
                self.assertEqual(env.cur_step, 7)
                env.reset(randomize=False)
                self.assertEqual(env.task_runtime_debug_state()["target_count"], 2)

    def test_configured_goal_names_are_authoritative(self):
        root = ET.fromstring(self.rotated_xml)
        root.find(".//*[@name='drawer_search_goal_bin_joint']").set(
            "name", "custom_bin_joint"
        )
        root.find(".//site[@name='drawer_search_goal_region']").set(
            "name", "custom_goal"
        )
        ET.SubElement(
            root.find("worldbody"),
            "site",
            name="drawer_search_goal_region",
            type="box",
            size=".05 .05 .05",
        )
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        kwargs = dict(
            spec=SimTaskSpec(
                name="custom_goal_preview",
                env_task="multi_drawer_search",
                prompt="find the target",
            ),
            goal_bin_joint_name="custom_bin_joint",
            goal_site_name="custom_goal",
        )
        source = MultiDrawerSearchTargetInBinEvaluator(model=model, **kwargs)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        result = source.evaluate_qpos_batch(data.qpos[None])
        np.testing.assert_allclose(
            result.metrics["goal_bin_center"][0], data.site("custom_goal").xpos
        )
        candidate = copy.deepcopy(model)
        candidate.site_bodyid[candidate.site("custom_goal").id] = 0
        with self.assertRaisesRegex(ValueError, "goal region must belong"):
            MultiDrawerSearchTargetInBinEvaluator(model=candidate, **kwargs)
        episode = source._episode
        for attempt in (
            lambda: source.prepare_randomization(candidate, None),
            lambda: source.configure_episode(candidate, episode),
        ):
            with self.assertRaisesRegex(ValueError, "goal region must belong"):
                attempt()
            self.assertIs(source.model, model)
            self.assertIs(source._episode, episode)

    def test_rejected_task_rebinding_preserves_running_task_then_accepts_valid_spec(
        self,
    ):
        root = ET.fromstring(self.identity_xml)
        ET.SubElement(
            root.find("worldbody"),
            "site",
            name="unattached_goal",
            type="box",
            size=".05 .05 .05",
        )
        env = self.make_env(TASKS[0], ET.tostring(root, encoding="unicode"))
        _, info = env.reset(seed=6, options={"randomization": {"target_count": 2}})
        first = info["randomization"].metadata["target_sequence"][0]
        self.position_target_and_check_site_oracle(env, first, (0, 0, 0), True)
        env.advance_physics(1)
        self.assertEqual(env.task_runtime_debug_state()["completed_target_count"], 1)
        names = (
            "_task_request",
            "_task",
            "_task_spec",
            "_task_reset_request",
            "_task_runtime",
            "_task_evaluator",
            "_step_evaluator",
            "_step_task_eval",
            "_episode_restorer",
            "_last_randomization",
            "_physics_observer",
        )
        before = {name: getattr(env, name) for name in names}
        history = env.task_runtime_debug_state()
        prompt = env.prompt
        rng = copy.deepcopy(env.np_random.bit_generator.state)
        env.data.qvel[:] = 0.3
        env.data.ctrl[:] = 0.2
        env.data.time = 12
        env.cur_step = 7
        physical = {
            name: getattr(env.data, name).copy() for name in ("qpos", "qvel", "ctrl")
        }
        bad = SimTaskSpec(
            name="wrong_goal",
            env_task="multi_drawer_search",
            prompt="find a different target",
            evaluator_name="multi_drawer_target_in_bin",
            evaluator_options=(("goal_site_name", "unattached_goal"),),
        )
        with self.assertRaisesRegex(ValueError, "goal region must belong"):
            env.set_task(bad)
        for name, value in before.items():
            self.assertIs(getattr(env, name), value, name)
        for name, value in physical.items():
            np.testing.assert_array_equal(getattr(env.data, name), value)
        self.assertEqual(env.data.time, 12)
        self.assertEqual(env.cur_step, 7)
        self.assertEqual(env.np_random.bit_generator.state, rng)
        self.assertEqual(env.task_runtime_debug_state(), history)
        self.assertEqual(env.prompt, prompt)
        env.set_task(TASKS[2])
        self.assertIs(env._task_spec, TASKS[2])
        self.assertIsNone(env._task_evaluator)
        self.assertIsNot(env._task_runtime, before["_task_runtime"])
        self.assertIsNot(env._step_evaluator, before["_step_evaluator"])
        self.assertIsNone(env._step_task_eval)
        self.assertEqual(env.prompt, TASKS[2].prompt)
        env.reset(seed=6, options={"randomization": {"target_count": 2}})
        self.assertEqual(env.task_runtime_debug_state()["target_count"], 2)
        self.assertEqual(env.task_runtime_debug_state()["completed_target_count"], 0)


if __name__ == "__main__":
    unittest.main()
