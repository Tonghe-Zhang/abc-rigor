"""Physical initial-scene replay through the native drawer randomizer."""

import copy
import gc
import json
import unittest
import xml.etree.ElementTree as ET
from unittest import mock

import mujoco
import numpy as np

import abc_sim
from abc_sim.config import RobotSystemConfig
from abc_sim.env import MuJoCoYAMEnv
from abc_sim.randomization.core import RandomizationState
from abc_sim.randomization.tasks import multi_drawer_search as drawer
from abc_sim.task_specs import SimTaskSpec


def misplaced_goal(xml):
    """Move the drawer goal from its bin body into the world frame."""
    root = ET.fromstring(xml)
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "site" and child.get("name") == "drawer_search_goal_region":
                parent.remove(child)
                root.find("worldbody").append(child)
                return ET.tostring(root, encoding="unicode")
    raise AssertionError("The drawer fixture must contain its goal region.")


def _model():
    stacks = []
    for stack in drawer._MULTI_DRAWER_SEARCH_STACK_IDS:
        children = "".join(
            f'<body name="{drawer._multi_drawer_search_drawer_body_name(stack, level)}">'
            f'<joint name="{drawer._multi_drawer_search_drawer_joint_name(stack, level)}" '
            'type="slide" range="0 .216"/><geom size=".01" mass=".1"/></body>'
            for level in drawer._MULTI_DRAWER_SEARCH_DRAWER_LEVELS
        )
        stacks.append(
            f'<body name="{drawer._multi_drawer_search_stack_body_name(stack)}">'
            f"{children}</body>"
        )
    objects = "".join(
        f'<body name="{item["body"]}" pos="{" ".join(map(str, item["pos"]))}">'
        f'<freejoint name="{item["joint"]}"/>'
        f'<geom name="{item["body"]}_collision_0" size=".005" mass=".05"/></body>'
        for item in drawer._multi_drawer_search_full_scene_selections()
    )
    return mujoco.MjModel.from_xml_string(
        '<mujoco><option gravity="0 0 0"/><worldbody>'
        f"{''.join(stacks)}{objects}"
        '<body name="drawer_search_goal_bin"><freejoint name="drawer_search_goal_bin_joint"/>'
        '<geom size=".01" mass=".1" contype="0" conaffinity="0"/>'
        '<site name="drawer_search_goal_region" type="box" size=".05 .05 .05"/>'
        "</body></worldbody></mujoco>"
    )


def _env(task="multi_drawer_search_2_shaped"):
    with mock.patch.object(
        MuJoCoYAMEnv, "setup_model", lambda env: env._bind_model(_model())
    ):
        env = MuJoCoYAMEnv(RobotSystemConfig(), render_cameras=False)
    env.get_init_q = lambda: np.zeros(0)
    env.set_task(task)
    randomizer = drawer.MultiDrawerSearchRandomizer()
    randomizer.bind_env(env)
    env._task_randomizer = randomizer
    return env


def _physical(env):
    return {
        "body_pos": env.model.body_pos.copy(),
        "body_quat": env.model.body_quat.copy(),
        "contype": env.model.geom_contype.copy(),
        "conaffinity": env.model.geom_conaffinity.copy(),
        "qpos": env.data.qpos.copy(),
        "qvel": env.data.qvel.copy(),
        "ctrl": env.data.ctrl.copy(),
    }


class DrawerSceneTest(unittest.TestCase):
    def assert_physical_equal(self, env, expected):
        for key, actual in _physical(env).items():
            np.testing.assert_array_equal(actual, expected[key], err_msg=key)

    def assert_fresh_episode(self, env, expected, record):
        self.assert_physical_equal(env, expected)
        self.assertEqual(env.data.time, 0)
        self.assertEqual(env.cur_step, 0)
        self.assertEqual(
            env.prompt, f"find the {record['metadata']['target_sequence'][0]['label']}"
        )
        self.assertEqual(env.task_runtime_debug_state()["completed_target_count"], 0)
        self.assertEqual(env._step_task_eval.reward[0], 0)
        if env._task_spec.name.endswith("_shaped"):
            self.assertEqual(env._step_task_eval.metrics["opened_drawer_count"][0], 0)
            self.assertEqual(env._step_task_eval.metrics["drawer_opening_bonus"][0], 0)
        self.assertEqual(
            json.loads(json.dumps(env._last_randomization.to_dict())), record
        )

    def test_serialized_cross_episode_and_fresh_replay(self):
        for task in ("multi_drawer_search", "multi_drawer_search_2_shaped"):
            with self.subTest(task=task):
                env = _env(task)
                _, info = env.reset(
                    seed=6, options={"randomization": {"target_count": 2}}
                )
                saved = RandomizationState.from_dict(
                    json.loads(json.dumps(info["randomization"].to_dict()))
                )
                original_record = copy.deepcopy(saved.to_dict())
                self.assertEqual(saved.metadata["object_set"], "office")
                initial = _physical(env)
                prompt = env.prompt
                env.step(np.zeros(0), render_obs=False)
                first_action = _physical(env)
                env.reset(seed=0, options={"randomization": {"target_count": 2}})
                self.assertFalse(
                    np.array_equal(env.model.body_pos, initial["body_pos"])
                )
                self.assertFalse(
                    np.array_equal(env.model.geom_contype, initial["contype"])
                )
                env._task_randomizer.apply(env.model, env.data, saved)
                self.assert_physical_equal(env, initial)
                self.assertEqual(env.prompt, prompt)
                self.assertEqual(env.cur_step, 0)
                env.step(np.zeros(0), render_obs=False)
                self.assert_physical_equal(env, first_action)
                fresh = _env(task)
                fresh._task_randomizer.apply(fresh.model, fresh.data, saved)
                self.assert_physical_equal(fresh, initial)
                fresh.step(np.zeros(0), render_obs=False)
                self.assert_physical_equal(fresh, first_action)
                self.assertEqual(saved.to_dict(), original_record)

    def test_retained_reset_restores_parked_default_targets_and_history(self):
        for task in ("multi_drawer_search", "multi_drawer_search_2_shaped"):
            with self.subTest(task=task):
                env = _env(task)
                _, info = env.reset(
                    seed=6, options={"randomization": {"target_count": 2}}
                )
                record = copy.deepcopy(info["randomization"].to_dict())
                initial = _physical(env)
                target = record["metadata"]["target_sequence"][0]
                address = env.model.joint(target["joint"]).qposadr[0]
                self.assertEqual(env.model.qpos0[address + 2], -0.95)
                self.assertGreater(env.data.qpos[address + 2], 0)
                goal = env.model.joint("drawer_search_goal_bin_joint").qposadr[0]
                env.data.qpos[address : address + 3] = env.data.qpos[goal : goal + 3]
                joint = record["metadata"]["drawer_joints"][0]
                env.data.qpos[env.model.joint(joint).qposadr[0]] = 0.15
                env.advance_physics(1)
                self.assertEqual(
                    env.task_runtime_debug_state()["completed_target_count"], 1
                )
                for _ in range(2):
                    _, info = env.reset(randomize=False)
                    self.assertNotIn("randomization", info)
                    self.assert_physical_equal(env, initial)
                    self.assertEqual(env._last_randomization.to_dict(), record)
                    self.assertEqual(
                        env.task_runtime_debug_state()["completed_target_count"], 0
                    )
                    self.assertEqual(env.data.time, 0)
                    self.assertEqual(env.cur_step, 0)
                    self.assertEqual(env.prompt, f"find the {target['label']}")
                    if task.endswith("_shaped"):
                        self.assertEqual(
                            env._step_task_eval.metrics["drawer_opening_bonus"][0], 0
                        )

    def test_template_and_full_xml_episode_lifecycle_with_optional_sampler(self):
        for task, profile in (
            ("multi_drawer_search", None),
            ("multi_drawer_search_2_shaped", "round2_low_friction_v1"),
        ):
            with self.subTest(task=task):
                producer = abc_sim.make_env(
                    task, render_cameras=False, physics_profile=profile
                )
                try:
                    _, info = producer.reset(
                        seed=6, options={"randomization": {"target_count": 2}}
                    )
                    saved = RandomizationState.from_dict(
                        json.loads(json.dumps(info["randomization"].to_dict()))
                    )
                    initial = _physical(producer)
                    record = copy.deepcopy(saved.to_dict())
                    scene_xml = producer._scene_xml_string
                    self.assertIsNotNone(scene_xml)
                    _, next_info = producer.reset(
                        seed=0, options={"randomization": {"target_count": 2}}
                    )
                    next_initial = _physical(producer)
                    next_record = json.loads(
                        json.dumps(next_info["randomization"].to_dict())
                    )
                finally:
                    producer.close()
                    del producer
                    gc.collect()

                for scene_kind, enabled, entry in (
                    ("template", False, "randomized"),
                    ("template", True, "randomized"),
                    ("assembled", False, "randomized"),
                    ("assembled", True, "randomized"),
                    ("assembled", False, "replay"),
                    ("assembled", True, "replay"),
                ):
                    with self.subTest(scene=scene_kind, sampler=enabled, entry=entry):
                        env = abc_sim.make_env(
                            task,
                            scene_xml_string=scene_xml
                            if scene_kind == "assembled"
                            else None,
                            enable_task_randomizer=enabled,
                            render_cameras=False,
                            physics_profile=profile,
                        )
                        restorer = drawer.MultiDrawerSearchRandomizer()
                        model = rejected = None
                        try:
                            if entry == "randomized":
                                env.reset(
                                    seed=6,
                                    options={"randomization": {"target_count": 2}},
                                )
                                self.assert_fresh_episode(env, initial, record)
                            restorer.bind_env(env)
                            restorer.apply(env.model, env.data, saved)
                            model = env.model
                            self.assert_fresh_episode(env, initial, record)
                            has_sampler = enabled or entry == "randomized"
                            self.assertEqual(
                                hasattr(env, "_task_randomizer"), has_sampler
                            )
                            for operation, keep in (
                                ("replay", True),
                                ("retained", True),
                                ("replay", False),
                                ("retained", False),
                            ):
                                with self.subTest(
                                    operation=operation, keep_arm_state=keep
                                ):
                                    target = saved.metadata["target_sequence"][0]
                                    address = env.model.joint(target["joint"]).qposadr[
                                        0
                                    ]
                                    env.data.qpos[address : address + 3] = (
                                        env._step_task_eval.metrics["goal_bin_center"][
                                            0
                                        ]
                                    )
                                    joint = saved.metadata["drawer_joints"][0]
                                    env.data.qpos[env.model.joint(joint).qposadr[0]] = (
                                        0.15
                                    )
                                    env.advance_physics(1)
                                    self.assertEqual(
                                        env.task_runtime_debug_state()[
                                            "completed_target_count"
                                        ],
                                        1,
                                    )
                                    if task.endswith("_shaped"):
                                        self.assertEqual(
                                            env._step_task_eval.metrics[
                                                "opened_drawer_count"
                                            ][0],
                                            1,
                                        )
                                    robot_joints = [
                                        env.model.joint(index)
                                        for index in range(env.model.njnt)
                                        if env.model.joint(index).name.startswith(
                                            ("left_", "right_")
                                        )
                                    ]
                                    self.assertEqual(len(robot_joints), 16)
                                    finger_names = {
                                        f"{arm}_{side}_finger"
                                        for arm in ("left", "right")
                                        for side in ("left", "right")
                                    }
                                    self.assertTrue(
                                        finger_names
                                        <= {joint.name for joint in robot_joints}
                                    )
                                    moved = {}
                                    for robot_joint in robot_joints:
                                        self.assertIn(
                                            int(robot_joint.type[0]),
                                            (
                                                mujoco.mjtJoint.mjJNT_HINGE,
                                                mujoco.mjtJoint.mjJNT_SLIDE,
                                            ),
                                        )
                                        value = 0.25
                                        if robot_joint.name in finger_names:
                                            value = (
                                                -0.017
                                                if robot_joint.name.endswith(
                                                    "right_finger"
                                                )
                                                else 0.017
                                            )
                                        address = int(robot_joint.qposadr[0])
                                        moved[address] = value
                                        self.assertNotEqual(
                                            initial["qpos"][address], value
                                        )
                                        env.data.qpos[address] = value
                                    env.data.qvel[:] = 0.3
                                    env.data.ctrl[:] = 0.2
                                    env.data.time = 12
                                    env.cur_step = 7
                                    if operation == "replay":
                                        restorer.apply(
                                            env.model,
                                            env.data,
                                            saved,
                                            keep_arm_state=keep,
                                        )
                                    else:
                                        _, info = env.reset(
                                            randomize=False,
                                            options={"keep_arm_state": keep},
                                        )
                                        self.assertNotIn("randomization", info)
                                    expected = {
                                        key: value.copy()
                                        for key, value in initial.items()
                                    }
                                    if keep:
                                        for address, value in moved.items():
                                            expected["qpos"][address] = value
                                    self.assert_fresh_episode(env, expected, record)
                                    self.assertIs(env._episode_restorer, restorer)
                                    self.assertIs(env.model, model)
                                    self.assertEqual(
                                        hasattr(env, "_task_randomizer"), has_sampler
                                    )

                            rejected = drawer.MultiDrawerSearchRandomizer()
                            rejected.bind_env(env)
                            bad = copy.deepcopy(saved)
                            bad.metadata["target_sequence"].pop()
                            with self.assertRaises(ValueError):
                                rejected.apply(env.model, env.data, bad)
                            self.assertIs(env._episode_restorer, restorer)
                            env.reset(
                                seed=0, options={"randomization": {"target_count": 2}}
                            )
                            self.assert_fresh_episode(env, next_initial, next_record)
                            self.assertIs(env._episode_restorer, env._task_randomizer)
                            env.data.qvel[:] = 0.3
                            env.data.time = 12
                            env.reset(randomize=False)
                            self.assert_fresh_episode(env, next_initial, next_record)
                            env.reset(
                                seed=6, options={"randomization": {"target_count": 2}}
                            )
                            self.assert_fresh_episode(env, initial, record)
                            self.assertIs(env.model, model)
                            self.assertEqual(saved.to_dict(), record)
                        finally:
                            env.close()
                            del env, restorer, rejected, model
                            gc.collect()

    def test_template_and_assembled_candidate_reject_before_live_replacement(self):
        template = drawer._MULTI_DRAWER_SEARCH_BASE_SCENE_XML.read_text()
        builder = drawer._multi_drawer_search_build_xml
        for task in (
            "multi_drawer_search",
            "multi_drawer_search_2_shaped",
            SimTaskSpec(
                name="drawer_preview",
                env_task="multi_drawer_search",
                prompt="find the target",
            ),
        ):
            for invalid_stage in ("template", "assembled_candidate"):
                with self.subTest(task=task, invalid_stage=invalid_stage):
                    if invalid_stage == "template":
                        for enabled in (False, True):
                            with (
                                self.subTest(initial_sampler=enabled),
                                mock.patch.object(
                                    drawer, "_multi_drawer_search_build_xml"
                                ) as assemble,
                                mock.patch.object(
                                    MuJoCoYAMEnv, "initialize_randomization"
                                ) as initialize,
                            ):
                                with self.assertRaisesRegex(
                                    ValueError, "goal region must belong"
                                ):
                                    env = abc_sim.make_env(
                                        task,
                                        scene_xml_string=misplaced_goal(template),
                                        enable_task_randomizer=enabled,
                                        render_cameras=False,
                                    )
                                assemble.assert_not_called()
                                initialize.assert_not_called()
                            gc.collect()
                        continue
                    env = abc_sim.make_env(
                        task,
                        scene_xml_string=template,
                        enable_task_randomizer=False,
                        render_cameras=False,
                    )
                    try:
                        env.data.qvel[:] = 0.3
                        env.data.ctrl[:] = 0.2
                        env.data.time = 12
                        env.cur_step = 7
                        physical = _physical(env)
                        model, data = env.model, env.data
                        evaluator = env._task_evaluator
                        runtime = env._task_runtime
                        history = env.task_runtime_debug_state()
                        prompt = env.prompt
                        rng = copy.deepcopy(env.np_random.bit_generator.state)
                        xml_before = env._scene_xml_string
                        owner = env._episode_restorer
                        record = env._last_randomization
                        arm_initialized = env._arm_state_initialized

                        def build(*args, **kwargs):
                            return misplaced_goal(builder(*args, **kwargs))

                        with (
                            mock.patch.object(
                                drawer,
                                "_multi_drawer_search_build_xml",
                                side_effect=build,
                            ) as assemble,
                            mock.patch.object(
                                env, "_compile_scene_xml", wraps=env._compile_scene_xml
                            ) as compile_scene,
                            mock.patch.object(
                                env,
                                "_close_camera_backend",
                                wraps=env._close_camera_backend,
                            ) as close_cameras,
                        ):
                            with self.assertRaisesRegex(
                                ValueError, "goal region must belong"
                            ):
                                env.reset(seed=19)
                            close_cameras.assert_not_called()
                            assemble.assert_called_once()
                            compile_scene.assert_called_once()
                        self.assertIs(env.model, model)
                        self.assertIs(env.data, data)
                        self.assertIs(env._task_evaluator, evaluator)
                        self.assertIs(env._task_runtime, runtime)
                        self.assertIs(env._last_randomization, record)
                        self.assertIs(env._episode_restorer, owner)
                        self.assertEqual(env._scene_xml_string, xml_before)
                        self.assertEqual(env._arm_state_initialized, arm_initialized)
                        self.assert_physical_equal(env, physical)
                        self.assertEqual(env.data.time, 12)
                        self.assertEqual(env.cur_step, 7)
                        self.assertEqual(env.prompt, prompt)
                        self.assertEqual(env.task_runtime_debug_state(), history)
                        self.assertEqual(env.np_random.bit_generator.state, rng)
                        self.assertFalse(hasattr(env, "_task_randomizer"))
                    finally:
                        env.close()
                        del env, model, data, evaluator, runtime
                        gc.collect()

    def test_invalid_bound_scene_rejects_reset_before_episode_mutation(self):
        joint_names = tuple(drawer._MULTI_DRAWER_SEARCH_DRAWER_JOINT_BY_DRAWER.values())
        for variant in ("hinge", "unlimited", "nonzero_lower"):
            for has_sampler in (False, True):
                for joint_name in joint_names:
                    with self.subTest(
                        variant=variant, sampler=has_sampler, joint=joint_name
                    ):
                        env = _env()
                        env.reset(seed=6)
                        state = env._last_randomization
                        target = state.metadata["target_sequence"][0]
                        address = env.model.joint(target["joint"]).qposadr[0]
                        env.data.qpos[address : address + 3] = (
                            env._step_task_eval.metrics["goal_bin_center"][0]
                        )
                        opening = env.model.joint(
                            state.metadata["drawer_joints"][0]
                        ).qposadr[0]
                        env.data.qpos[opening] = 0.15
                        env.advance_physics(1)
                        joint = env.model.joint(joint_name).id
                        if variant == "hinge":
                            env.model.jnt_type[joint] = mujoco.mjtJoint.mjJNT_HINGE
                        elif variant == "unlimited":
                            env.model.jnt_limited[joint] = 0
                        else:
                            env.model.jnt_range[joint, 0] = -0.01
                        env.data.qvel[:] = 0.3
                        env.data.ctrl[:] = 0.2
                        env.data.time = 12
                        env.cur_step = 7
                        sampler = env._task_randomizer
                        if not has_sampler:
                            del env._task_randomizer
                        physical = _physical(env)
                        model, data = env.model, env.data
                        runtime = env.task_runtime_debug_state()
                        retained = env._task_runtime.retained_randomization().to_dict()
                        prompt = env.prompt
                        evaluation = env._step_task_eval
                        evaluator = env._task_evaluator
                        opened = evaluator._opened_drawers.copy()
                        credited = evaluator._credited_openings.copy()
                        completed = evaluator._completed_target_count.copy()
                        owner = env._episode_restorer
                        rng = copy.deepcopy(env.np_random.bit_generator.state)
                        message = (
                            "slide joint"
                            if variant == "hinge"
                            else "zero-based travel limit"
                        )
                        with self.assertRaisesRegex(ValueError, message):
                            env.reset(seed=0)
                        self.assertIs(env.model, model)
                        self.assertIs(env.data, data)
                        self.assert_physical_equal(env, physical)
                        self.assertEqual(env.data.time, 12)
                        self.assertEqual(env.cur_step, 7)
                        self.assertEqual(env.prompt, prompt)
                        self.assertEqual(env.task_runtime_debug_state(), runtime)
                        self.assertEqual(
                            env._task_runtime.retained_randomization().to_dict(),
                            retained,
                        )
                        self.assertEqual(env.np_random.bit_generator.state, rng)
                        self.assertIs(env._step_task_eval, evaluation)
                        self.assertIs(env._task_evaluator, evaluator)
                        np.testing.assert_array_equal(evaluator._opened_drawers, opened)
                        np.testing.assert_array_equal(
                            evaluator._credited_openings, credited
                        )
                        np.testing.assert_array_equal(
                            evaluator._completed_target_count, completed
                        )
                        self.assertIs(env._last_randomization, state)
                        self.assertIs(env._episode_restorer, owner)
                        if has_sampler:
                            self.assertIs(env._task_randomizer, sampler)
                        else:
                            self.assertFalse(hasattr(env, "_task_randomizer"))
                        env.model.jnt_type[joint] = mujoco.mjtJoint.mjJNT_SLIDE
                        env.model.jnt_limited[joint] = 1
                        env.model.jnt_range[joint, 0] = 0
                        env.reset(seed=0)
                        self.assertIs(env.model, model)
                        self.assertEqual(
                            env.task_runtime_debug_state()["completed_target_count"], 0
                        )
                        self.assertIs(env._episode_restorer, env._task_randomizer)

    def test_native_scene_preserves_custom_slide_limits(self):
        for variant in ("unlimited", "nonzero_lower"):
            with self.subTest(variant=variant):
                env = _env("multi_drawer_search")
                for name in drawer._MULTI_DRAWER_SEARCH_DRAWER_JOINT_BY_DRAWER.values():
                    joint = env.model.joint(name).id
                    if variant == "unlimited":
                        env.model.jnt_limited[joint] = 0
                    else:
                        env.model.jnt_range[joint, 0] = -0.01
                env.reset(seed=6)
                saved = copy.deepcopy(env._last_randomization)
                env._task_randomizer.apply(env.model, env.data, saved)
                env.reset(randomize=False)
                self.assertEqual(env._last_randomization.to_dict(), saved.to_dict())

    def test_invalid_effective_sequence_rejects_before_any_episode_mutation(self):
        env = _env()
        _, info = env.reset(seed=6)
        saved = copy.deepcopy(info["randomization"])
        joint = saved.metadata["drawer_joints"][0]
        env.data.qpos[env.model.joint(joint).qposadr[0]] = 0.15
        env.advance_physics(1)
        env.data.time = 12
        env.data.qvel[:] = 0.3
        env.cur_step = 7
        before = _physical(env)
        runtime = env.task_runtime_debug_state()
        retained = env._task_runtime.retained_randomization().to_dict()
        opened = env._task_evaluator._opened_drawers.copy()
        credited = env._task_evaluator._credited_openings.copy()
        evaluation = env._step_task_eval
        for variant in (
            "short",
            "long",
            "duplicate",
            "missing_label",
            "bad_drawer",
            "bad_layout",
            "contradictory_label",
        ):
            with self.subTest(variant=variant):
                bad = copy.deepcopy(saved)
                sequence = bad.metadata["target_sequence"]
                if variant == "short":
                    sequence.pop()
                elif variant == "long":
                    sequence.append(copy.deepcopy(sequence[0]))
                elif variant == "duplicate":
                    sequence[1] = copy.deepcopy(sequence[0])
                elif variant == "missing_label":
                    del sequence[0]["label"]
                elif variant == "bad_drawer":
                    sequence[0]["drawer"] = "missing"
                elif variant == "contradictory_label":
                    sequence[0]["label"] = "a different physical object"
                else:
                    bad.metadata["stack_poses"] = {}
                bad_record = copy.deepcopy(bad.to_dict())
                with (
                    mock.patch.object(
                        env,
                        "initialize_randomization",
                        wraps=env.initialize_randomization,
                    ) as initialize,
                    mock.patch.object(
                        env._task_randomizer,
                        "_restore_scene",
                        wraps=env._task_randomizer._restore_scene,
                    ) as restore,
                ):
                    with self.assertRaises(ValueError):
                        env._task_randomizer.apply(env.model, env.data, bad)
                    initialize.assert_not_called()
                    restore.assert_not_called()
                self.assert_physical_equal(env, before)
                self.assertEqual(env.data.time, 12)
                self.assertEqual(env.cur_step, 7)
                self.assertEqual(env.task_runtime_debug_state(), runtime)
                self.assertEqual(
                    env._task_runtime.retained_randomization().to_dict(), retained
                )
                self.assertIs(env._step_task_eval, evaluation)
                np.testing.assert_array_equal(
                    env._task_evaluator._opened_drawers, opened
                )
                np.testing.assert_array_equal(
                    env._task_evaluator._credited_openings, credited
                )
                self.assertEqual(env._task_randomizer._current_scale_states, {})
                self.assertEqual(bad.to_dict(), bad_record)

    def test_all_explicit_counts_and_unchanged_native_seed_sequence(self):
        randomizer = drawer.MultiDrawerSearchRandomizer()
        for count in (1, 2, 3, 4):
            for seed in range(64):
                with self.subTest(count=count, seed=seed):
                    objects, target = randomizer._sample_selections(
                        np.random.default_rng(seed), target_count=count
                    )
                    sequence = target["target_sequence"]
                    self.assertEqual(len(sequence), count)
                    self.assertEqual(len({item["body"] for item in sequence}), count)
                    self.assertTrue(
                        {item["body"] for item in sequence}
                        <= {item["body"] for item in objects}
                    )
                    self.assertTrue(
                        all(
                            item["drawer"] in target["active_drawers"]
                            for item in sequence
                        )
                    )
        # Frozen native no-count samples captured before conditioning inventory.
        expected = {
            0: ([5, 6, 4, 2], [5, 6, 4, 2], [1, 2, 3], "pantry", 1633503582),
            6: ([21, 20, 23, 19], [21, 23, 19, 20], [2, 3], "office", 1921547296),
            14: ([17, 10, 13], [17, 10], [3], "kitchen_tools", 2000612103),
            23: ([12, 10], [12, 10], [3], "kitchen_tools", 1828240904),
        }
        for seed, result in expected.items():
            rng = np.random.default_rng(seed)
            objects, target = randomizer._sample_selections(rng)
            index = drawer._MULTI_DRAWER_SEARCH_OBJECT_BODIES.index
            self.assertEqual(
                (
                    [index(item["body"]) for item in objects],
                    [index(item["body"]) for item in target["target_sequence"]],
                    list(target["active_stack_ids"]),
                    target["object_set"],
                    int(rng.integers(0, 2**31)),
                ),
                result,
            )
