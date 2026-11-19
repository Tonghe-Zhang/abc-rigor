"""Seeded board layouts, physical placement, independent credit and force history."""

from __future__ import annotations

import copy
import json
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import abc_sim
import mujoco
import numpy as np

from abc_sim.randomization.assets.shape_matching import (
    DEFAULT_SELECTED,
    EMPTY_SLOT_RGBA,
    PIECE_HALF_HEIGHT,
    PIECE_JOINTS,
    WALL_HEIGHT,
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
        signatures, palettes, prompts, selections = set(), set(), set(), set()
        for seed in range(16):
            self.env.reset(seed=seed)
            state = self.env._last_randomization
            self.assertEqual(len(state.object_states), 17)
            self.assertEqual(set(state.metadata["order"]), set(range(16)))
            selected = set(state.metadata["selected"])
            self.assertEqual(len(selected), 8)
            selections.add(tuple(sorted(selected)))
            self.assertTrue(self.env._task_randomizer._contacts_ok(self.env.model, self.env.data))
            self.assertEqual(self.env.evaluate_task().scalar_reward(), 0)
            self.assertEqual(get_task_spec(self.env.prompt).name, "shape_matching")
            signatures.add(tuple(state.metadata["order"]))
            palettes.add(tuple(state.metadata["palette"]))
            prompts.add(self.env.prompt)
            for piece in range(16):
                slot = state.metadata["order"].index(piece)
                np.testing.assert_allclose(self.env.model.site(f"shape_target_{piece:02d}").pos[:2], slot_center(slot))
                expected = (
                    self.env.model.geom(f"shape_piece_{piece:02d}_geom").rgba
                    if piece in selected
                    else np.asarray(EMPTY_SLOT_RGBA, dtype=np.float32)
                )
                np.testing.assert_array_equal(expected, self.env.model.geom(f"shape_reference_{slot:02d}").rgba)
                self.assertEqual(self.env.model.site(f"shape_target_{piece:02d}").user[0], piece in selected)
            np.testing.assert_array_equal(
                self.env.evaluate_task().metrics["piece_requested_mask"][0],
                [piece in selected for piece in range(16)],
            )
        self.assertEqual(len(signatures), 16)
        self.assertGreater(len(palettes), 10)
        self.assertGreater(len(selections), 10)
        self.assertEqual(prompts, set(SHAPE_MATCHING_PROMPTS))

    def test_seed_and_json_replay_restore_meshes_poses_colors_and_prompt(self):
        self.env.reset(seed=19)
        saved = json.loads(json.dumps(self.env._last_randomization.to_dict()))
        qpos = self.env.data.qpos.copy()
        meshes = self.env.model.mesh_vert.copy()
        colors = self.env.model.geom_rgba.copy()
        requested = self.env.model.site_user.copy()
        for seed in (19, 7):
            self.env.reset(seed=seed)
            self.env._task_randomizer.apply(self.env.model, self.env.data, RandomizationState.from_dict(saved))
            np.testing.assert_array_equal(self.env.data.qpos, qpos)
            np.testing.assert_array_equal(self.env.model.mesh_vert, meshes)
            np.testing.assert_array_equal(self.env.model.geom_rgba, colors)
            np.testing.assert_array_equal(self.env.model.site_user, requested)
            self.assertEqual(self.env._last_randomization.to_dict(), saved)
        other = abc_sim.make_env(task="shape_matching", render_cameras=False)
        try:
            other.reset(seed=19)
            np.testing.assert_array_equal(other.data.qpos, qpos)
            np.testing.assert_array_equal(other.model.mesh_vert, meshes)
            np.testing.assert_array_equal(other.model.site_user, requested)
            self.assertEqual(other.prompt, self.env.prompt)
        finally:
            other.close()

    def test_invalid_reset_or_replay_does_not_mutate_live_scene(self):
        self.env.reset(seed=4)
        saved = self.env._last_randomization.to_dict()
        model, qpos = self.env.model, self.env.data.qpos.copy()
        for options in (
            {"palette": ["red"] * 4},
            {"order": [0] * 16},
            {"wood": -1},
            {"selected": [0] * 8},
            {"selected": list(range(7))},
            {"selected": list(range(7)) + [16]},
            {"selected": list(range(7)) + [True]},
            {"unexpected": 1},
        ):
            with self.assertRaises(ValueError):
                self.env.reset(seed=2, options={"randomization": options})
        with self.assertRaises(ValueError):
            self.env.reset(seed=2, randomize=False)
        for key, value in (("order", [0] * 16), ("wood", 12), ("selected", [0] * 8), ("prompt", "wrong")):
            invalid = copy.deepcopy(saved)
            invalid["metadata"][key] = value
            with self.assertRaises(ValueError):
                self.env._task_randomizer.apply(model, self.env.data, RandomizationState.from_dict(invalid))
        self.assertIs(self.env.model, model)
        np.testing.assert_array_equal(self.env.data.qpos, qpos)

    def test_production_inertias_match_geometry_and_preserve_station(self):
        self.env.reset(seed=11)
        models = Path(abc_sim.__file__).parent / "models"
        station = mujoco.MjModel.from_xml_path(str(models / "yam_bimanual_empty.xml"))
        for body in range(1, station.nbody):
            actual = self.env.model.body(station.body(body).name)
            np.testing.assert_allclose(actual.mass, station.body_mass[body], rtol=1e-8)
            np.testing.assert_allclose(actual.inertia, station.body_inertia[body], rtol=1e-8)
        reference = ET.Element("mujoco")
        assets, bodies = build_board(("red", "blue", "yellow", "green"), tuple(range(16)), 0, DEFAULT_SELECTED)
        ET.SubElement(reference, "asset").extend(assets)
        ET.SubElement(reference, "worldbody").extend(bodies)
        geometry = mujoco.MjModel.from_xml_string(ET.tostring(reference, encoding="unicode"))
        for piece in range(16):
            name = f"shape_piece_{piece:02d}"
            actual, expected = self.env.model.body(name), geometry.body(name)
            np.testing.assert_allclose(actual.mass, expected.mass, rtol=1e-8)
            np.testing.assert_allclose(actual.inertia, expected.inertia, rtol=1e-8)
            self.assertLess(actual.inertia.max(), 1e-5)

    def test_production_scene_can_select_every_piece_with_equal_credit(self):
        for selected in (list(range(8)), list(range(8, 16))):
            with self.subTest(selected=selected):
                self.check_production_selection(selected)

    def check_production_selection(self, selected):
        self.env.reset(seed=11, options={"randomization": {"selected": selected}})
        model, data = self.env.model, self.env.data
        self.env.advance_physics(200)
        np.testing.assert_array_equal(self.env.evaluate_task().metrics["force_dose_ns"], 0)
        table = model.geom("table_plane").id
        board_id = model.body("shape_board").id
        support_contacts = []
        for contact in data.contact:
            if table in (contact.geom1, contact.geom2):
                other = contact.geom2 if contact.geom1 == table else contact.geom1
                if model.geom_bodyid[other] == board_id:
                    support_contacts.append(other)
        self.assertGreater(len(support_contacts), 0)
        self.assertLessEqual(len(support_contacts), 32)
        self.assertTrue(all(model.geom(geom).name.startswith("shape_board_border_") for geom in support_contacts))
        board = data.body("shape_board")
        for piece in selected:
            joint = PIECE_JOINTS[piece]
            target = model.site(f"shape_target_{piece:02d}").pos.copy()
            target[2] += PIECE_HALF_HEIGHT + 0.0001
            data.joint(joint).qpos[:3] = board.xpos + board.xmat.reshape(3, 3) @ target
            data.joint(joint).qpos[3:] = board.xquat
            data.joint(joint).qvel[:] = 0
            mujoco.mj_forward(model, data)
            self.env.advance_physics(200)
        result = self.env.evaluate_task()
        self.assertTrue(result.scalar_success())
        expected = np.array([1 / 8 if piece in selected else 0 for piece in range(16)])
        np.testing.assert_allclose(result.metrics["piece_reward"][0], expected, atol=1e-7)


class ShapeMatchingRewardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = cls.make_model()

    @staticmethod
    def make_model(*, fixed_raised_board=False):
        root = ET.Element("mujoco")
        ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast")
        asset = ET.SubElement(root, "asset")
        world = ET.SubElement(root, "worldbody")
        ET.SubElement(
            world,
            "geom",
            name="table_plane",
            type="box",
            pos="0 0 0.745",
            size="2 2 0.005",
            contype="2",
            conaffinity="1",
        )
        for side in ("left", "right"):
            arm = ET.SubElement(world, "body", name=f"{side}_arm", pos="-2 0 2")
            ET.SubElement(arm, "geom", name=f"{side}_finger", type="box", size="0.01 0.01 0.01")
        assets, bodies = build_board(("red", "blue", "yellow", "green"), tuple(range(16)), 0, DEFAULT_SELECTED)
        asset.extend(assets)
        world.extend(bodies)
        if fixed_raised_board:
            board = world.find("./body[@name='shape_board']")
            board.set("pos", "0.59 0 0.9")
            board.remove(board.find("freejoint"))
        return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))

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
        position = np.array((*center, PIECE_HALF_HEIGHT + height))
        qpos = self.data.joint(PIECE_JOINTS[piece]).qpos
        qpos[:3] = board.xpos + board.xmat.reshape(3, 3) @ position
        mujoco.mju_mulQuat(qpos[3:], board.xquat, np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]))
        self.data.joint(PIECE_JOINTS[piece]).qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def test_requested_pieces_have_equal_independent_credit_and_complete(self):
        for count, piece in enumerate(DEFAULT_SELECTED, 1):
            self.place(piece)
            result = self.step(0.45)
            self.assertEqual(result.metrics["num_matched_pieces"][0], count)
            self.assertAlmostEqual(result.scalar_reward(), count / 8, places=5)
        self.assertTrue(result.scalar_success())

    def test_unrequested_pieces_do_not_score_or_gate_requested_credit(self):
        unrequested = sorted(set(range(16)) - set(DEFAULT_SELECTED))
        for piece in unrequested:
            self.place(piece)
        result = self.step(0.5)
        self.assertEqual(result.scalar_reward(), 0)
        self.assertFalse(result.scalar_success())
        np.testing.assert_array_equal(result.metrics["piece_matched_mask"], False)
        for count, piece in enumerate(DEFAULT_SELECTED, 1):
            self.place(piece)
            result = self.step(0.45)
            self.assertAlmostEqual(result.scalar_reward(), count / 8, places=5)
            np.testing.assert_array_equal(result.metrics["piece_reward"][0, unrequested], 0)
        self.assertTrue(result.scalar_success())

    def test_compiled_goal_requires_exactly_eight_binary_selections(self):
        site = self.model.site("shape_target_00")
        original = site.user.copy()
        try:
            for value in (0, 0.5, 2):
                site.user[0] = value
                with self.assertRaisesRegex(ValueError, "exactly eight"):
                    ShapeMatchingEvaluator(model=self.model, spec=get_task_spec("shape_matching"))
        finally:
            site.user[:] = original

    def test_color_swap_same_shape_and_wrong_slot_give_zero(self):
        self.place(0, slot=1)
        self.place(1, slot=0)
        self.place(4, slot=5)
        self.assertEqual(self.step(0.5).scalar_reward(), 0)

    def test_open_slots_pass_every_shape_through_a_raised_board(self):
        self.model = self.make_model(fixed_raised_board=True)
        self.data = mujoco.MjData(self.model)
        self.evaluator = ShapeMatchingEvaluator(model=self.model, spec=get_task_spec("shape_matching"))
        mujoco.mj_forward(self.model, self.data)
        self.evaluator.after_reset(self.model, self.data)
        for piece in (0, 4, 8, 12):
            self.place(piece, height=WALL_HEIGHT + 0.02)
        self.place(5, height=WALL_HEIGHT + 0.02, yaw=np.pi / 4)
        result = self.step(0.7)
        self.assertAlmostEqual(self.data.body("shape_board").xpos[2], 0.9, places=5)
        for piece in (0, 4, 8, 12):
            np.testing.assert_allclose(
                self.data.body(f"shape_piece_{piece:02d}").xpos[2], 0.75 + PIECE_HALF_HEIGHT, rtol=0, atol=0.0002
            )
            self.assertTrue(result.metrics["piece_supported_mask"][0, piece])
            self.assertFalse(result.metrics["piece_inside_mask"][0, piece])
        self.assertGreater(self.data.body("shape_piece_05").xpos[2], 0.91)
        self.assertFalse(result.metrics["piece_supported_mask"][0, 5])
        self.assertEqual(result.scalar_reward(), 0)

    def test_symmetric_rotations_fit_and_misaligned_square_does_not(self):
        for piece, yaw in ((0, np.pi / 2), (4, np.pi / 2), (8, 2 * np.pi / 3), (12, np.pi / 3)):
            self.place(piece, yaw=yaw)
        self.place(5, yaw=np.pi / 4, height=0.02)
        result = self.step(0.6)
        np.testing.assert_array_equal(np.flatnonzero(result.metrics["piece_matched_mask"]), [0, 4, 8, 12])
        self.assertAlmostEqual(result.scalar_reward(), 4 / 8)

    def test_hovering_stacking_and_brief_contact_do_not_qualify(self):
        self.place(0, height=0.03)
        self.assertEqual(self.step(0.01).scalar_reward(), 0)
        self.place(0)
        self.assertEqual(self.step(0.15).scalar_reward(), 0)
        self.place(1, slot=0, height=2 * PIECE_HALF_HEIGHT)
        result = self.step(0.5)
        self.assertAlmostEqual(result.scalar_reward(), 1 / 8)
        self.assertFalse(result.metrics["piece_matched_mask"][0, 1])

    def test_removal_only_loses_that_piece_and_repeated_reads_are_pure(self):
        original = self.data.joint(PIECE_JOINTS[0]).qpos.copy()
        self.place(0)
        self.place(4)
        self.step(0.5)
        self.data.joint(PIECE_JOINTS[0]).qpos[:] = original
        self.data.joint(PIECE_JOINTS[0]).qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        result = self.step(0.02)
        self.assertAlmostEqual(result.scalar_reward(), 1 / 8)
        for _ in range(10):
            self.assertAlmostEqual(self.evaluator.evaluate_model_data(self.model, self.data).scalar_reward(), 1 / 8)
        self.assertEqual(self.evaluator._settling[0], 0)

    def test_force_dose_is_irreversible_and_does_not_gate_other_pieces(self):
        self.place(0)
        self.place(4)
        result = self.step(0.5)
        self.assertAlmostEqual(result.scalar_reward(), 2 / 8)
        dof = self.model.joint(PIECE_JOINTS[0]).dofadr[0]
        self.data.qfrc_applied[dof + 2] = -60
        loaded = self.step(0.4)
        self.assertGreater(loaded.metrics["force_dose_ns"][0, 0], 5)
        self.data.qfrc_applied[:] = 0
        released = self.step(0.5)
        self.assertGreaterEqual(released.metrics["force_dose_ns"][0, 0], loaded.metrics["force_dose_ns"][0, 0])
        self.assertAlmostEqual(released.metrics["piece_reward"][0, 0], 0.8 / 8, places=5)
        self.assertAlmostEqual(released.metrics["piece_reward"][0, 4], 1 / 8, places=5)
        self.assertEqual(released.metrics["num_matched_pieces"][0], 2)
        board_dof = self.model.joint("shape_board_joint").dofadr[0]
        self.data.qfrc_applied[board_dof + 2] = -80
        self.step(0.2)
        self.data.qfrc_applied[:] = 0
        board_loaded = self.step(0.5)
        self.assertGreater(board_loaded.metrics["force_dose_ns"][0, -1], 5)
        self.assertAlmostEqual(board_loaded.metrics["piece_reward"][0, 0], 0.7 / 8, places=5)
        self.assertAlmostEqual(board_loaded.metrics["piece_reward"][0, 4], 0.9 / 8, places=5)

    def test_contacts_and_single_world_history_are_required(self):
        with self.assertRaises(NotImplementedError):
            self.evaluator.evaluate_qpos_batch(self.data.qpos[None])
        with self.assertRaises(NotImplementedError):
            self.evaluator.reset(nworld=16)
        with self.assertRaises(ValueError):
            self.evaluator.after_physics_step(self.model, self.data)

    def test_opposing_finger_loads_add_and_robot_contact_prevents_qualification(self):
        self.evaluator = ShapeMatchingEvaluator(
            model=self.model, spec=get_task_spec("shape_matching"), force_allowance_n=1e9
        )
        self.place(4)
        position = self.data.body("shape_piece_04").xpos.copy()
        arms = [self.model.body(f"{side}_arm").id for side in ("left", "right")]
        original = self.model.body_pos[arms].copy()
        try:
            for arm, sign in zip(arms, (-1, 1)):
                self.model.body_pos[arm] = position + np.array((sign * 0.03095, 0, 0.014))
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
            result = self.step(0.8)
            self.assertTrue(result.metrics["piece_inside_mask"][0, 4])
            self.assertTrue(result.metrics["piece_supported_mask"][0, 4])
            self.assertTrue(result.metrics["piece_touching_robot"][0, 4])
            self.assertLess(np.linalg.norm(self.data.joint(PIECE_JOINTS[4]).qvel), 0.025)
            self.assertEqual(result.scalar_reward(), 0)
        finally:
            self.model.body_pos[arms] = original
        self.assertEqual(self.step(0.15).scalar_reward(), 0)
        self.assertAlmostEqual(self.step(0.5).scalar_reward(), 1 / 8)


if __name__ == "__main__":
    unittest.main()
