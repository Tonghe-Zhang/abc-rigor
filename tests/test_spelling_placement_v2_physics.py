"""Solved MuJoCo contact, dose, reset and public task contracts for spelling v2."""

import string
import unittest

import mujoco
import numpy as np

from abc_sim import make_env
from abc_sim.batched_env import BatchedWarpYAMEnv
from abc_sim.randomization.core import RandomizationState
from abc_sim.spelling_goal import SpellingGoal
from abc_sim.task_eval.registry import make_task_evaluator
from abc_sim.task_specs import get_task_spec


class SpellingV2PhysicsTests(unittest.TestCase):
    def setUp(self):
        blocks = "".join(
            f'<body name="block_{letter}" pos="{1 + 0.1 * i} .4 .011" quat=".70710678 0 .70710678 0">'
            f'<freejoint name="block_{letter}_jnt"/><geom type="box" size=".01 .01 .01" mass=".03"/></body>'
            for i, letter in enumerate(string.ascii_uppercase)
        )
        self.model = mujoco.MjModel.from_xml_string(
            '<mujoco><option timestep=".002"/><worldbody>'
            '<geom name="table_plane" type="plane" size="5 5 .1"/>'
            '<body name="left_arm" mocap="true" pos="0 1 1">'
            '<geom name="left_contact" type="box" size=".005 .009 .008"/></body>'
            '<body name="right_arm" mocap="true" pos="0 -1 1">'
            '<geom name="right_contact" type="box" size=".005 .009 .008"/></body>' + blocks + "</worldbody></mujoco>"
        )
        self.data = mujoco.MjData(self.model)
        self.bind("ZXBI")

    def bind(self, word):
        self.evaluator = make_task_evaluator(self.model, f"spell_random_{len(word)}_appearance")
        self.evaluator.configure_from_randomization(
            self.model, RandomizationState(31, {}, metadata=SpellingGoal(word, 31).to_metadata())
        )
        mujoco.mj_forward(self.model, self.data)
        self.evaluator.after_reset(self.model, self.data)

    def advance(self, count):
        for _ in range(count):
            mujoco.mj_step(self.model, self.data)
            self.evaluator.after_physics_step(self.model, self.data)
        return self.result()

    def result(self):
        return self.evaluator.evaluate_model_data(self.model, self.data)

    def teleport(self, letter, pos):
        joint = self.model.joint(f"block_{letter}_jnt")
        address, dof = joint.qposadr[0], joint.dofadr[0]
        self.data.qpos[address : address + 3] = pos
        self.data.qpos[address + 3 : address + 7] = [2**-0.5, 0, 2**-0.5, 0]
        self.data.qvel[dof : dof + 6] = 0

    def release_arms(self):
        self.data.mocap_pos[:] = [[0, 1, 1], [0, -1, 1]]

    def place_row(self, row):
        for index, letter in enumerate(row):
            self.teleport(letter, [0.8 + 0.08 * index, -0.3, 0.011])
        for slot, letter in enumerate(row):
            pos = [0, 0.12 - 0.04 * slot, 0.011]
            self.teleport(letter, pos)
            self.data.mocap_pos[0] = [-0.0149, pos[1], pos[2]]
            self.advance(1)
            self.assertTrue(self.result().metrics["letter_handled"][0, ord(letter) - 65])
            self.release_arms()
            self.teleport(letter, pos)
        return self.advance(300)

    def test_reset_and_idle_stay_zero_for_every_length(self):
        for word in ("AB", "ABC", "ZXBI"):
            self.bind(word)
            self.assertEqual(self.result().scalar_reward(), 0)
            self.assertEqual(self.advance(300).scalar_reward(), 0)

    def test_observed_wrong_letter_example_and_reset_clear_all_history(self):
        result = self.place_row("ZXBL")
        self.assertEqual(result.scalar_reward(), 0.75)
        self.assertEqual(result.metrics["placement_reward"][0], 0.75)
        self.assertFalse(result.scalar_success())
        result.reward[0] = -10
        self.assertEqual(self.result().scalar_reward(), 0.75)
        self.evaluator.reset()
        self.evaluator.configure_from_randomization(
            self.model, RandomizationState(32, {}, metadata=SpellingGoal("ZXBI", 32).to_metadata())
        )
        self.evaluator.after_reset(self.model, self.data)
        self.assertEqual(self.advance(100).scalar_reward(), 0)
        self.assertFalse(self.result().metrics["letter_handled"].any())
        np.testing.assert_array_equal(self.result().metrics["force_dose_ns"], 0)

    def test_complete_swapped_and_removed_letters(self):
        self.assertEqual(self.place_row("XZBI").scalar_reward(), 0.625)
        self.assertEqual(self.place_row("ZXBI").scalar_reward(), 1)
        self.assertTrue(self.result().scalar_success())
        self.teleport("I", [0.4, 0, 0.011])
        self.assertEqual(self.advance(100).scalar_reward(), 0.75)

    def test_teleport_and_passive_displacement_cannot_claim_manipulation(self):
        for i, letter in enumerate("ZXBI"):
            self.teleport(letter, [0, 0.12 - i * 0.04, 0.011])
        self.assertEqual(self.advance(300).scalar_reward(), 0)
        self.assertFalse(self.result().metrics["letter_handled"].any())

    def test_held_hovering_and_stacked_letters_lose_only_their_own_credit(self):
        self.place_row("ZXBI")
        z = ord("Z") - 65
        self.data.mocap_pos[0] = [-0.0148, 0.12, 0.011]
        held = self.advance(1)
        self.assertTrue(held.metrics["letter_touching_robot"][0, z])
        self.assertEqual(held.scalar_reward(), 0.75)
        self.release_arms()
        self.assertEqual(self.advance(1).scalar_reward(), 0.75)
        self.assertEqual(self.advance(300).scalar_reward(), 1)
        self.teleport("Z", [0, 0.12, 0.05])
        self.assertEqual(self.advance(1).scalar_reward(), 0.75)
        self.teleport("Z", [0, 0.12, 0.031])
        self.teleport("A", [0, 0.12, 0.011])
        self.advance(200)
        self.assertFalse(self.result().metrics["letter_eligible"][0, z])

    def test_opposing_contacts_sum_and_integrate_every_tick_with_per_letter_damage(self):
        self.place_row("ZXBI")
        z = ord("Z") - 65
        pos = [0, 0.12, 0.011]
        self.teleport("Z", pos)
        self.data.mocap_pos[:] = [[-0.013, pos[1], pos[2]], [0.013, pos[1], pos[2]]]
        self.advance(1)
        before = self.result().metrics["force_dose_ns"][0, z]
        peak = self.result().metrics["peak_force_n"][0, z]
        for _ in range(250):
            result = self.advance(1)
            force = np.empty(6)
            expected = 0.0
            sides = set()
            body = self.model.body("block_Z").id
            for c in range(self.data.ncon):
                contact = self.data.contact[c]
                if body in self.model.geom_bodyid[[contact.geom1, contact.geom2]]:
                    mujoco.mj_contactForce(self.model, self.data, c, force)
                    expected += np.linalg.norm(force[:3])
                    sides.update(
                        {int(contact.geom1), int(contact.geom2)}
                        & {self.model.geom("left_contact").id, self.model.geom("right_contact").id}
                    )
            self.assertEqual(len(sides), 2)
            self.assertAlmostEqual(result.metrics["contact_force_n"][0, z], expected)
            dose = before + max(expected - 15, 0) * self.model.opt.timestep
            self.assertAlmostEqual(result.metrics["force_dose_ns"][0, z], dose)
            self.assertGreaterEqual(result.metrics["peak_force_n"][0, z], peak)
            before, peak = dose, result.metrics["peak_force_n"][0, z]
        self.assertGreater(before, 0)
        self.release_arms()
        self.teleport("Z", pos)
        result = self.advance(300)
        self.assertAlmostEqual(result.scalar_reward(), 1 - 0.075 * min(before / 5, 1))
        np.testing.assert_allclose(result.metrics["letter_reward"][0, [ord(c) - 65 for c in "XBI"]], 0.25)
        self.assertAlmostEqual(result.metrics["force_dose_ns"][0, z], before)
        self.assertAlmostEqual(self.advance(200).scalar_reward(), result.scalar_reward())

    def test_duplicate_missing_ticks_and_qpos_replay_fail(self):
        self.advance(1)
        with self.assertRaisesRegex(ValueError, "every physics tick"):
            self.evaluator.after_physics_step(self.model, self.data)
        mujoco.mj_step(self.model, self.data)
        mujoco.mj_step(self.model, self.data)
        with self.assertRaisesRegex(ValueError, "every physics tick"):
            self.evaluator.after_physics_step(self.model, self.data)
        with self.assertRaisesRegex(NotImplementedError, "solved contacts"):
            self.evaluator.evaluate_qpos_batch(self.data.qpos[None])

    def test_table_load_exhausts_only_the_letter_reserve_and_stays_lost(self):
        self.place_row("ZXBI")
        body = self.model.body("block_Z").id
        # Gravity plus this applied load produces approximately 20 N of table contact.
        self.data.xfrc_applied[body, 2] = -(20 - self.model.body_mass[body] * 9.81)
        halfway = self.advance(250)
        self.assertAlmostEqual(halfway.scalar_reward(), 1 - 0.075 / 2, delta=0.002)
        exhausted = self.advance(350)
        self.assertAlmostEqual(exhausted.scalar_reward(), 0.925)
        self.assertTrue(exhausted.scalar_success())
        self.data.xfrc_applied[body] = 0
        self.assertAlmostEqual(self.advance(300).scalar_reward(), 0.925)
        self.assertAlmostEqual(self.place_row("XZBI").scalar_reward(), 0.60625)


class PublicSpellingV2Tests(unittest.TestCase):
    def test_all_public_random_variants_bind_v2_and_idle_zero_across_seeds(self):
        for length in (2, 3, 4):
            spec = get_task_spec(f"spell_random_{length}_appearance")
            self.assertEqual(spec.evaluator_name, "abc-spelling-placement-v2")
            with self.subTest(length=length):
                env = make_env(spec, render_cameras=False)
                try:
                    for seed in (0, 11, 31):
                        observation, info = env.reset(seed=seed)
                        self.assertEqual(observation["prompt"], SpellingGoal.sample(length, seed).prompt)
                        self.assertEqual(env.evaluate_task().scalar_reward(), 0)
                        env.advance_physics(150)
                        self.assertEqual(env.evaluate_task().scalar_reward(), 0)
                        self.assertIsNone(env.max_episode_steps)
                    with self.assertRaisesRegex(NotImplementedError, "solved contacts"):
                        BatchedWarpYAMEnv(env, num_worlds=1, camera_height=8, camera_width=8)
                finally:
                    env.close()


if __name__ == "__main__":
    unittest.main()
