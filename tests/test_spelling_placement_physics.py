"""Native MuJoCo contact and environment tests without scene assets or rendering."""

import string
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

from abc_sim import make_env
from abc_sim.batched_env import BatchedWarpYAMEnv
from abc_sim.randomization.core import RandomizationState, SceneRandomizer
from abc_sim.spelling_goal import SpellingGoal
from abc_sim.spelling_placement import FreePlacementSpellingCfg, SpellingTableSupport
from abc_sim.task_specs import SimTaskSpec


class SpellingSupportTests(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string("""
            <mujoco><option timestep=".002"/><worldbody>
              <geom name="table_plane" type="plane" size="1 1 .1"/>
              <body name="block_A" pos="0 0 .012">
                <freejoint/><geom type="box" size=".01 .01 .01" mass=".05"/>
              </body>
              <body mocap="true" pos="-.1 0 .012">
                <geom name="left_finger" type="box" size=".005 .02 .008"/>
              </body>
              <body mocap="true" pos=".1 0 .012">
                <geom name="right_finger" type="box" size=".005 .02 .008"/>
              </body>
            </worldbody></mujoco>
        """)
        self.data = mujoco.MjData(self.model)
        self.tracker = SpellingTableSupport(
            FreePlacementSpellingCfg(),
            self.model,
            body_names=["block_A"],
            robot_geom_ids={
                self.model.geom(name).id for name in ("left_finger", "right_finger")
            },
        )

    def advance(self, count):
        for _ in range(count):
            mujoco.mj_step(self.model, self.data)
            result = self.tracker.sample(self.data)
        return result

    def test_table_support_requires_settling_and_is_lost_when_lifted(self):
        self.assertEqual(self.advance(20)["eligible"], [False])
        settled = self.advance(500)
        self.assertEqual(settled["eligible"], [True])
        self.assertAlmostEqual(settled["weight_fraction"][0], 1, delta=0.01)
        self.data.qpos[2] += 0.05
        self.assertEqual(self.advance(1)["eligible"], [False])

    def test_gripped_table_supported_block_requires_release_and_resettling(self):
        self.advance(500)
        self.data.mocap_pos[:, 0] = [-0.0145, 0.0145]
        held = self.advance(500)
        self.assertGreater(held["weight_fraction"][0], 0.5)
        self.assertGreater(
            held["robot_normal_load"][0], self.tracker.cfg.robot_force_tolerance
        )
        self.assertEqual(held["eligible"], [False])
        self.data.mocap_pos[:, 0] = [-0.1, 0.1]
        self.assertEqual(self.advance(1)["eligible"], [False])
        self.assertEqual(self.advance(300)["eligible"], [True])

    def test_hovering_and_missing_contact_samples_cannot_establish_support(self):
        self.data.qpos[2] = 0.029
        mujoco.mj_forward(self.model, self.data)
        hovering = self.tracker.sample(self.data)
        self.assertEqual(hovering["weight_fraction"], [0.0])
        self.assertEqual(hovering["eligible"], [False])
        with self.assertRaisesRegex(ValueError, "every native physics step"):
            self.tracker.sample(self.data)
        mujoco.mj_step(self.model, self.data)
        mujoco.mj_step(self.model, self.data)
        with self.assertRaisesRegex(ValueError, "every native physics step"):
            self.tracker.sample(self.data)


class NativeSpellingTests(unittest.TestCase):
    def setUp(self):
        bodies, actuators = [], []
        for arm in ("left", "right"):
            for index in range(7):
                joint = f"{arm}_joint{index + 1}" if index < 6 else f"{arm}_left_finger"
                actuator = joint if index < 6 else f"{arm}_gripper"
                bodies.append(
                    f'<body name="{arm}_link_left_finger_{index}" pos="3 {index} 1">'
                    f'<joint name="{joint}" damping="1"/>'
                    '<geom type="sphere" size=".01" contype="0" conaffinity="0"/></body>'
                )
                actuators.append(f'<motor name="{actuator}" joint="{joint}"/>')
        for index, letter in enumerate(string.ascii_uppercase):
            bodies.append(
                f'<body name="block_{letter}" pos="{0.4 + 0.1 * (index // 9)} {0.4 - 0.085 * (index % 9)} .012">'
                f'<freejoint name="block_{letter}_jnt"/>'
                '<geom type="box" size=".01 .01 .01" mass=".05"/></body>'
            )
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        xml_path = Path(temporary.name) / "spelling.xml"
        xml_path.write_text(
            '<mujoco><worldbody><geom name="table_plane" type="plane" size="1 1 .1"/>'
            + "".join(bodies)
            + "</worldbody><actuator>"
            + "".join(reversed(actuators))
            + "</actuator></mujoco>"
        )
        spec = SimTaskSpec(
            name="custom_free_placement",
            env_task="blocks",
            prompt="spell the requested letter sequence",
            evaluator_name="abc-spelling-free-placement-v1",
            evaluator_options=(("sequence_length", 2),),
            reset_request=(("sequence_length", 2),),
        )
        self.env = make_env(
            spec,
            scene_xml=xml_path,
            render_cameras=False,
            enable_task_randomizer=False,
            camera_height=8,
            camera_width=8,
            chunk_dim=2,
        )
        self.addCleanup(self.env.close)
        self.requests = []
        requests = self.requests

        class FixtureRandomizer(SceneRandomizer):
            def randomize(self, model, data, seed=None, request=None):
                requests.append(request)
                return RandomizationState(
                    seed,
                    {},
                    metadata=SpellingGoal(
                        "AB" if seed == 37 else "XY", seed
                    ).to_metadata(),
                )

        self.env._task_randomizer = FixtureRandomizer()
        self.env._task_randomizer.bind_env(self.env)
        self.env.reset(seed=37)
        self.model, self.data = self.env.model, self.env.data

    def place_word(self, word="AB"):
        for index, letter in enumerate(word):
            address = self.model.joint(f"block_{letter}_jnt").qposadr[0]
            self.data.qpos[address : address + 3] = [0, 0.1 - 0.04 * index, 0.012]
            self.data.qpos[address + 3 : address + 7] = [2**-0.5, 0, 2**-0.5, 0]

    def test_explicit_actions_use_native_controls_and_score_every_seventeen_ticks(self):
        action = np.arange(14, dtype=np.float32) / 14
        self.env.apply_action(action, physics_steps=16)
        expected = action.copy()
        expected[[6, 13]] *= 0.0475
        np.testing.assert_allclose(self.data.ctrl, expected[::-1])
        self.assertEqual(self.env.cur_step, 1)
        with self.assertRaisesRegex(RuntimeError, "first scheduled"):
            self.env.evaluate_task()
        self.env.apply_action(action, physics_steps=1)
        first = self.env.evaluate_task()
        self.assertEqual(first.metrics["verifier_sample_count"], 1)
        self.assertAlmostEqual(self.data.time, 17 * 0.002)
        for _ in range(3):
            self.assertEqual(self.env.evaluate_task().to_info(), first.to_info())
        self.env.advance_physics(34)
        self.assertEqual(self.env.evaluate_task().metrics["verifier_sample_count"], 3)

    def test_complete_row_earns_credit_after_real_contacts_settle(self):
        self.place_word()
        self.env.advance_physics(17)
        early = self.env.evaluate_task()
        self.assertEqual(early.scalar_reward(), 0)
        self.assertEqual(float(early.metrics["original_spell_reward"][0]), 1)
        self.env.advance_physics(493)
        complete = self.env.evaluate_task()
        self.assertEqual(complete.scalar_reward(), 1)
        self.assertTrue(complete.scalar_success())
        self.assertTrue(
            all(complete.metrics["spelling_placement"]["frame"]["supported"][:2])
        )
        complete.reward[0] = -1
        complete.metrics["spelling_placement"]["frame"]["initial_positions"][0][0] = 999
        self.assertEqual(self.env.evaluate_task().scalar_reward(), 1)
        self.env.advance_physics(17)
        self.assertTrue(self.env.evaluate_task().scalar_success())

    def test_reset_binds_new_goal_and_clears_previous_contact_history(self):
        self.place_word()
        self.env.advance_physics(510)
        self.assertTrue(self.env.evaluate_task().scalar_success())
        self.env.reset(seed=38)
        self.assertEqual(self.requests, [{"sequence_length": 2}] * 2)
        with self.assertRaisesRegex(RuntimeError, "first scheduled"):
            self.env.evaluate_task()
        self.env.advance_physics(17)
        result = self.env.evaluate_task()
        self.assertEqual(result.metrics["word"], "XY")
        self.assertEqual(result.metrics["verifier_sample_count"], 1)
        self.assertFalse(result.scalar_success())

    def test_native_step_and_chunk_keep_requested_physics_counts(self):
        self.env._control_decimation = 18
        self.env.step(np.zeros(14), render_obs=False)
        result = self.env.step_chunk(np.zeros((2, 14)))
        self.assertAlmostEqual(self.data.time, 54 * 0.002)
        self.assertEqual(self.env.cur_step, 3)
        self.assertEqual(result[-1]["task_eval"]["verifier_sample_count"], 3)

    def test_low_level_steps_are_explicit_and_high_level_scoring_requires_readiness(
        self,
    ):
        for steps in (0, -1, True, 1.5):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                self.env.apply_action(np.zeros(14), physics_steps=steps)
        self.env._control_decimation = 1
        with self.assertRaisesRegex(ValueError, "control_decimation.*>= 17"):
            self.env.step(np.zeros(14), render_obs=False)
        self.assertEqual(self.data.time, 0)
        self.assertEqual(self.env.cur_step, 0)
        self.env.apply_action(np.zeros(14), physics_steps=17)
        self.assertEqual(self.env.evaluate_task().metrics["verifier_sample_count"], 1)

    def test_qpos_only_and_batched_replay_reject_only_the_contact_observer(self):
        contact_spec = self.env._task_spec
        with self.assertRaisesRegex(NotImplementedError, "solved contacts"):
            self.env._task_evaluator.evaluate_qpos_batch(self.data.qpos[None])
        with self.assertRaisesRegex(NotImplementedError, "solved contacts"):
            BatchedWarpYAMEnv(self.env, num_worlds=1, camera_height=8, camera_width=8)
        self.env._task_randomizer = None
        self.env.set_task("spell_cat")
        observation, info = self.env.reset(seed=31)
        self.assertEqual(observation["prompt"], "spell cat")
        self.assertNotIn("spelling_goal", info["randomization"].metadata)
        self.place_word("CAT")
        result = self.env.evaluate_task()
        self.assertTrue(result.scalar_success())
        self.assertEqual(result.metrics["word"], "CAT")
        BatchedWarpYAMEnv(self.env, num_worlds=1, camera_height=8, camera_width=8)
        self.env.set_task(contact_spec)
        observation, info = self.env.reset(seed=32)
        goal = SpellingGoal.from_metadata(info["randomization"].metadata)
        self.assertEqual(observation["prompt"], goal.prompt)
        with self.assertRaisesRegex(RuntimeError, "first scheduled"):
            self.env.evaluate_task()
        self.env.advance_physics(17)
        result = self.env.evaluate_task()
        self.assertEqual(result.metrics["word"], goal.sequence)
        self.assertEqual(result.metrics["verifier_sample_count"], 1)


if __name__ == "__main__":
    unittest.main()
