"""Seeded board layouts, physical placement, independent credit and force history."""

from __future__ import annotations

import copy
import json
import unittest
import xml.etree.ElementTree as ET

import abc_sim
import mujoco
import numpy as np

from abc_sim.randomization.assets.shape_matching import (
    FLOOR_HEIGHT,
    PIECE_HALF_HEIGHT,
    PIECE_JOINTS,
    build_board,
    slot_center,
)
from abc_sim.randomization.core import RandomizationState
from abc_sim.task_eval.shape_matching import ShapeMatchingEvaluator
from abc_sim.task_specs import SHAPE_MATCHING_PROMPTS, get_task_spec


class ShapeMatchingSceneTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = abc_sim.make_env(task="shape_matching", render_cameras=False)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_seeded_layouts_are_empty_collision_free_and_diverse(self):
        signatures, palettes, prompts = set(), set(), set()
        for seed in range(16):
            self.env.reset(seed=seed)
            state = self.env._last_randomization
            self.assertEqual(len(state.object_states), 17)
            self.assertEqual(set(state.metadata["order"]), set(range(16)))
            self.assertTrue(self.env._task_randomizer._contacts_ok(self.env.model, self.env.data))
            self.assertEqual(self.env.evaluate_task().scalar_reward(), 0)
            self.assertEqual(get_task_spec(self.env.prompt).name, "shape_matching")
            signatures.add(tuple(state.metadata["order"]))
            palettes.add(tuple(state.metadata["palette"]))
            prompts.add(self.env.prompt)
            for piece in range(16):
                slot = state.metadata["order"].index(piece)
                np.testing.assert_allclose(self.env.model.site(f"shape_target_{piece:02d}").pos[:2], slot_center(slot))
                np.testing.assert_array_equal(
                    self.env.model.geom(f"shape_piece_{piece:02d}_geom").rgba,
                    self.env.model.geom(f"shape_reference_{slot:02d}").rgba,
                )
        self.assertEqual(len(signatures), 16)
        self.assertGreater(len(palettes), 10)
        self.assertEqual(prompts, set(SHAPE_MATCHING_PROMPTS))

    def test_seed_and_json_replay_restore_meshes_poses_colors_and_prompt(self):
        self.env.reset(seed=19)
        saved = json.loads(json.dumps(self.env._last_randomization.to_dict()))
        qpos = self.env.data.qpos.copy()
        meshes = self.env.model.mesh_vert.copy()
        colors = self.env.model.geom_rgba.copy()
        for seed in (19, 7):
            self.env.reset(seed=seed)
            self.env._task_randomizer.apply(self.env.model, self.env.data, RandomizationState.from_dict(saved))
            np.testing.assert_array_equal(self.env.data.qpos, qpos)
            np.testing.assert_array_equal(self.env.model.mesh_vert, meshes)
            np.testing.assert_array_equal(self.env.model.geom_rgba, colors)
            self.assertEqual(self.env._last_randomization.to_dict(), saved)
        other = abc_sim.make_env(task="shape_matching", render_cameras=False)
        try:
            other.reset(seed=19)
            np.testing.assert_array_equal(other.data.qpos, qpos)
            np.testing.assert_array_equal(other.model.mesh_vert, meshes)
            self.assertEqual(other.prompt, self.env.prompt)
        finally:
            other.close()

    def test_invalid_reset_or_replay_does_not_mutate_live_scene(self):
        self.env.reset(seed=4)
        saved = self.env._last_randomization.to_dict()
        model, qpos = self.env.model, self.env.data.qpos.copy()
        for options in ({"palette": ["red"] * 4}, {"order": [0] * 16}, {"wood": -1}, {"unexpected": 1}):
            with self.assertRaises(ValueError):
                self.env.reset(seed=2, options={"randomization": options})
        with self.assertRaises(ValueError):
            self.env.reset(seed=2, randomize=False)
        for key, value in (("order", [0] * 16), ("wood", 12), ("prompt", "wrong")):
            invalid = copy.deepcopy(saved)
            invalid["metadata"][key] = value
            with self.assertRaises(ValueError):
                self.env._task_randomizer.apply(model, self.env.data, RandomizationState.from_dict(invalid))
        self.assertIs(self.env.model, model)
        np.testing.assert_array_equal(self.env.data.qpos, qpos)


class ShapeMatchingRewardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = ET.Element("mujoco")
        ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast")
        asset = ET.SubElement(root, "asset")
        world = ET.SubElement(root, "worldbody")
        ET.SubElement(world, "geom", name="table", type="plane", pos="0 0 0.75", size="2 2 0.1")
        for side in ("left", "right"):
            arm = ET.SubElement(world, "body", name=f"{side}_arm", pos="-2 0 2")
            ET.SubElement(arm, "geom", name=f"{side}_finger", type="box", size="0.01 0.01 0.01")
        assets, bodies = build_board(("red", "blue", "yellow", "green"), tuple(range(16)), 0)
        asset.extend(assets)
        world.extend(bodies)
        cls.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))

    def setUp(self):
        self.data = mujoco.MjData(self.model)
        self.evaluator = ShapeMatchingEvaluator(model=self.model, spec=get_task_spec("shape_matching"))
        mujoco.mj_forward(self.model, self.data)
        self.evaluator.after_reset(self.model, self.data)
        self.step(0.4)

    def step(self, seconds):
        for _ in range(round(seconds / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)
            self.evaluator.after_physics_step(self.model, self.data)
        return self.evaluator.evaluate_model_data(self.model, self.data)

    def place(self, piece, *, slot=None, height=0.0005, yaw=0.0):
        board = self.data.body("shape_board")
        center = slot_center(piece if slot is None else slot)
        position = np.array((*center, FLOOR_HEIGHT + PIECE_HALF_HEIGHT + height))
        qpos = self.data.joint(PIECE_JOINTS[piece]).qpos
        qpos[:3] = board.xpos + board.xmat.reshape(3, 3) @ position
        mujoco.mju_mulQuat(qpos[3:], board.xquat, np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]))
        self.data.joint(PIECE_JOINTS[piece]).qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def test_every_piece_has_equal_independent_credit_and_all_complete(self):
        for piece in range(16):
            self.place(piece)
            result = self.step(0.45)
            self.assertEqual(result.metrics["num_matched_pieces"][0], piece + 1)
            self.assertAlmostEqual(result.scalar_reward(), (piece + 1) / 16, places=5)
        self.assertTrue(result.scalar_success())

    def test_color_swap_same_shape_and_wrong_slot_give_zero(self):
        self.place(0, slot=1)
        self.place(1, slot=0)
        self.place(4, slot=5)
        self.assertEqual(self.step(0.5).scalar_reward(), 0)

    def test_symmetric_rotations_fit_and_misaligned_square_does_not(self):
        for piece, yaw in ((0, np.pi / 2), (4, np.pi / 2), (8, 2 * np.pi / 3), (12, np.pi / 3)):
            self.place(piece, yaw=yaw)
        self.place(5, yaw=np.pi / 4, height=0.02)
        result = self.step(0.6)
        np.testing.assert_array_equal(np.flatnonzero(result.metrics["piece_matched_mask"]), [0, 4, 8, 12])
        self.assertAlmostEqual(result.scalar_reward(), 4 / 16)

    def test_hovering_stacking_and_brief_contact_do_not_qualify(self):
        self.place(0, height=0.03)
        self.assertEqual(self.step(0.01).scalar_reward(), 0)
        self.place(0)
        self.assertEqual(self.step(0.15).scalar_reward(), 0)
        self.place(1, slot=0, height=2 * PIECE_HALF_HEIGHT)
        result = self.step(0.5)
        self.assertAlmostEqual(result.scalar_reward(), 1 / 16)
        self.assertFalse(result.metrics["piece_matched_mask"][0, 1])

    def test_removal_only_loses_that_piece_and_repeated_reads_are_pure(self):
        self.place(0)
        self.place(4)
        self.step(0.5)
        self.place(0, slot=2)
        result = self.step(0.02)
        self.assertAlmostEqual(result.scalar_reward(), 1 / 16)
        for _ in range(10):
            self.assertAlmostEqual(self.evaluator.evaluate_model_data(self.model, self.data).scalar_reward(), 1 / 16)
        self.assertEqual(self.evaluator._settling[0], 0)

    def test_force_dose_is_irreversible_and_does_not_gate_other_pieces(self):
        self.place(0)
        self.place(4)
        result = self.step(0.5)
        self.assertAlmostEqual(result.scalar_reward(), 2 / 16)
        dof = self.model.joint(PIECE_JOINTS[0]).dofadr[0]
        self.data.qfrc_applied[dof + 2] = -60
        loaded = self.step(0.4)
        self.assertGreater(loaded.metrics["force_dose_ns"][0, 0], 5)
        self.data.qfrc_applied[:] = 0
        released = self.step(0.5)
        self.assertGreaterEqual(released.metrics["force_dose_ns"][0, 0], loaded.metrics["force_dose_ns"][0, 0])
        self.assertAlmostEqual(released.metrics["piece_reward"][0, 0], 0.7 / 16, places=5)
        self.assertAlmostEqual(released.metrics["piece_reward"][0, 4], 0.9 / 16, places=5)
        self.assertEqual(released.metrics["num_matched_pieces"][0], 2)

    def test_contacts_and_single_world_history_are_required(self):
        with self.assertRaises(NotImplementedError):
            self.evaluator.evaluate_qpos_batch(self.data.qpos[None])
        with self.assertRaises(NotImplementedError):
            self.evaluator.reset(nworld=16)
        with self.assertRaises(ValueError):
            self.evaluator.after_physics_step(self.model, self.data)

    def test_opposing_finger_loads_add_and_robot_contact_prevents_qualification(self):
        self.place(4)
        position = self.data.body("shape_piece_04").xpos.copy()
        arms = [self.model.body(f"{side}_arm").id for side in ("left", "right")]
        original = self.model.body_pos[arms].copy()
        try:
            for arm, sign in zip(arms, (-1, 1)):
                self.model.body_pos[arm] = position + np.array((sign * 0.0305, 0, 0))
            mujoco.mj_forward(self.model, self.data)
            self.evaluator.after_reset(self.model, self.data)
            result = self.evaluator.evaluate_model_data(self.model, self.data)
            loads, net = [], np.zeros(3)
            touched = set()
            piece_geom = self.model.geom("shape_piece_04_geom").id
            for index in range(self.data.ncon):
                contact = self.data.contact[index]
                if piece_geom not in (contact.geom1, contact.geom2):
                    continue
                force = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, index, force)
                loads.append(np.linalg.norm(force[:3]))
                sign = 1 if contact.geom2 == piece_geom else -1
                net += sign * contact.frame.reshape(3, 3).T @ force[:3]
                other = contact.geom1 if contact.geom2 == piece_geom else contact.geom2
                touched.add(self.model.geom_bodyid[other])
            self.assertTrue(set(arms) <= touched)
            self.assertAlmostEqual(result.metrics["contact_force_n"][0, 4], sum(loads))
            self.assertGreater(sum(loads), 2 * np.linalg.norm(net))
            self.assertTrue(result.metrics["piece_touching_robot"][0, 4])
            self.assertEqual(result.scalar_reward(), 0)
        finally:
            self.model.body_pos[arms] = original
        self.assertAlmostEqual(self.step(0.5).scalar_reward(), 1 / 16)


if __name__ == "__main__":
    unittest.main()
