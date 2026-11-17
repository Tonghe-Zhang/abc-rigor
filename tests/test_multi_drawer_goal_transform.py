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
from abc_sim.task_specs import SimTaskSpec
from tests.test_multi_drawer_custom_spec import drawer_xml


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


if __name__ == "__main__":
    unittest.main()
