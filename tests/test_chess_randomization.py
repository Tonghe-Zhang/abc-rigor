"""CPU regressions for separated chess board and container resets."""

from __future__ import annotations

import unittest
from unittest import mock

import mujoco
import numpy as np

from abc_sim.randomization.assets.chess_tin import _CHESS_BASE_SCENE_XML
from abc_sim.randomization.core import (
    RandomizationSamplingError,
    _quat_from_axis_angle,
    _quat_from_yaw,
    _quat_mul,
)
from abc_sim.randomization.requests import _ChessPlacementFailure
from abc_sim.randomization.tasks.chess import ChessRandomizer

_TIN_HALF_XY = (0.1825, 0.13284832939963015)
_SCENE = """
<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <geom name="table" type="plane" size="1 1 .1" pos="0 0 .75"/>
    <body name="chessboard" pos=".6 0 .75" euler="1.5708 0 0">
      <freejoint name="chessboard"/>
      <geom name="board_collision" type="box" size=".232 .232 .022"
            pos="-.006 .022 -.004" euler="-1.5708 0 0" mass="2"/>
    </body>
    <body name="tin_box" pos=".6 .460 .757">
      <freejoint name="tin_box_joint"/>
      <geom name="tin_box_floor" type="box" size=".1825 .13284832939963015 .006" mass=".06"/>
      <geom name="tin_box_wall_y_neg" type="box" size=".1825 .011 .05"
            pos="0 -.12184832939963015 .056" mass=".015"/>
    </body>
    <body name="black_pawn_1" pos=".6 -.168 .805">
      <freejoint name="black_pawn_1_jnt"/>
      <geom type="sphere" size=".01" mass=".02"/>
    </body>
    <body name="black_pawn_2" pos=".656 -.168 .805">
      <freejoint name="black_pawn_2_jnt"/>
      <geom type="sphere" size=".01" mass=".02"/>
    </body>
    <body name="left_arm" pos="0 0 2">
      <geom type="sphere" size=".025"/>
    </body>
  </worldbody>
</mujoco>
"""
_SEED_20082_BOARD = {
    "pos": [0.5775233469516128, 0.09375952622512057, 0.75],
    "quat": [
        0.7066217438350179,
        0.7066243394063805,
        -0.026151091010665015,
        -0.026150994952520476,
    ],
}
_SEED_20082_TIN = {
    "pos": [0.5718380821475652, 0.4621337241310489, 0.757],
    "quat": [0.9996253248425457, 0.0, 0.0, 0.02737169949847658],
}


def geom_corners(model, data, geom_name):
    geom_id = model.geom(geom_name).id
    signs = np.array(
        [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)],
        dtype=np.float64,
    )
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    return data.geom_xpos[geom_id] + (signs * model.geom_size[geom_id]) @ rotation.T


def tin_board_contacts(model, data):
    board_root = model.body_rootid[model.body("chessboard").id]
    tin_root = model.body_rootid[model.body("tin_box").id]
    return [
        contact
        for contact in data.contact
        if {
            model.body_rootid[model.geom_bodyid[contact.geom1]],
            model.body_rootid[model.geom_bodyid[contact.geom2]],
        }
        == {board_root, tin_root}
    ]


class ChessPlacementTest(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string(_SCENE)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.randomizer = ChessRandomizer()
        self.randomizer._active_tin_outer_half_xy = _TIN_HALF_XY
        self.randomizer._before_sampling(self.model, self.data)
        self.nominals = {
            name: (
                self.data.qpos[self.model.joint(name).qposadr[0] :][:3].copy(),
                self.data.qpos[self.model.joint(name).qposadr[0] + 3 :][:4].copy(),
            )
            for name in ("chessboard", "tin_box_joint")
        }

    def apply(self, board, tin):
        self.randomizer._apply_states(
            self.model, self.data, {"chessboard": board, "tin_box_joint": tin}
        )
        mujoco.mj_forward(self.model, self.data)

    def assert_tin_inside_table(self):
        corners = geom_corners(self.model, self.data, "tin_box_floor")[:, :2]
        x_min, x_max, y_min, y_max = self.randomizer.table_bounds
        self.assertTrue(np.all(corners >= [x_min - 1e-10, y_min - 1e-10]))
        self.assertTrue(np.all(corners <= [x_max + 1e-10, y_max + 1e-10]))

    def test_seed_20082_overlap_is_rejected_even_with_piece_contacts_allowed(self):
        self.apply(_SEED_20082_BOARD, _SEED_20082_TIN)
        contacts = tin_board_contacts(self.model, self.data)
        self.assertTrue(contacts)
        self.assertLess(min(contact.dist for contact in contacts), -0.01)
        for allow_piece_contacts in (False, True):
            self.assertFalse(
                self.randomizer._contacts_ok(
                    self.model, self.data, allow_piece_contacts=allow_piece_contacts
                )
            )

    def test_seed_20082_container_moves_outward_and_fits_on_table(self):
        for seed in range(20):
            with self.subTest(seed=seed):
                tin = self.randomizer._sample_tin_state(
                    self.nominals, np.random.default_rng(seed), _SEED_20082_BOARD
                )
                self.apply(_SEED_20082_BOARD, tin)
                self.assertGreater(tin["pos"][1], _SEED_20082_TIN["pos"][1])
                self.assertFalse(tin_board_contacts(self.model, self.data))
                self.assert_tin_inside_table()

    def test_rotation_and_collision_offset_match_physical_geometry(self):
        roll = _quat_from_axis_angle(np.array([1, 0, 0]), 1.5708)
        for board_yaw in (-0.15, 0, 0.15):
            for tin_yaw in (-0.25, 0, 0.25):
                for tin_y in (0.39, 0.43, 0.47):
                    with self.subTest(
                        board_yaw=board_yaw, tin_yaw=tin_yaw, tin_y=tin_y
                    ):
                        board = {
                            "pos": [0.6, 0, 0.75],
                            "quat": _quat_mul(_quat_from_yaw(board_yaw), roll).tolist(),
                        }
                        tin = {
                            "pos": [0.6, tin_y, 0.757],
                            "quat": _quat_from_yaw(tin_yaw).tolist(),
                        }
                        self.apply(board, tin)
                        footprint = self.randomizer._board_footprint(board)
                        physical = geom_corners(
                            self.model, self.data, "board_collision"
                        )
                        np.testing.assert_allclose(
                            footprint.min(axis=0),
                            physical[:, :2].min(axis=0),
                            atol=1e-10,
                        )
                        np.testing.assert_allclose(
                            footprint.max(axis=0),
                            physical[:, :2].max(axis=0),
                            atol=1e-10,
                        )
                        if tin_board_contacts(self.model, self.data):
                            self.assertTrue(
                                self.randomizer._tin_overlaps_board(tin, board)
                            )
                        if not self.randomizer._tin_overlaps_board(tin, board):
                            self.assertFalse(tin_board_contacts(self.model, self.data))
        board = {"pos": [0.6, 0, 0.75], "quat": roll.tolist()}
        # The old check missed this collision because it ignored the tin's yaw.
        tin = {"pos": [0.6, 0.39, 0.757], "quat": _quat_from_yaw(0.25).tolist()}
        self.apply(board, tin)
        self.assertTrue(tin_board_contacts(self.model, self.data))
        self.assertTrue(self.randomizer._tin_overlaps_board(tin, board))

    def test_impossible_or_exhausted_sampling_raises(self):
        impossible_board = {
            "pos": [0.6, 0.65, 0.75],
            "quat": self.nominals["chessboard"][1],
        }
        with self.assertRaises(_ChessPlacementFailure):
            self.randomizer._sample_tin_state(
                self.nominals, np.random.default_rng(20082), impossible_board
            )
        with (
            mock.patch.object(
                self.randomizer, "_tin_overlaps_board", return_value=True
            ),
            self.assertRaises(_ChessPlacementFailure),
        ):
            self.randomizer._sample_tin_state(
                self.nominals, np.random.default_rng(20082), _SEED_20082_BOARD
            )

    def test_outer_reset_retries_placement_failure_and_reports_exhaustion(self):
        self.randomizer.perturbations = [
            p for p in self.randomizer.perturbations if p.joint_name in self.nominals
        ]
        states = {
            name: {"pos": pos.tolist(), "quat": quat.tolist()}
            for name, (pos, quat) in self.nominals.items()
        }
        metadata = {"scenario": "tin_setup", "tin_active": True}
        with (
            mock.patch.object(
                self.randomizer, "_resolve_tin_variant", return_value="tin_2"
            ),
            mock.patch.object(self.randomizer, "_reload_chess_tin_scene"),
            mock.patch.object(
                self.randomizer,
                "_sample_setup_once",
                side_effect=[_ChessPlacementFailure(), (states, metadata)],
            ) as sample,
        ):
            state = self.randomizer.randomize(self.model, self.data, seed=20082)
            self.assertEqual(state.seed, 20082)
            self.assertEqual(sample.call_count, 2)
        self.randomizer.max_tries = 2
        with (
            mock.patch.object(
                self.randomizer, "_resolve_tin_variant", return_value="tin_2"
            ),
            mock.patch.object(self.randomizer, "_reload_chess_tin_scene"),
            mock.patch.object(
                self.randomizer,
                "_sample_setup_once",
                side_effect=_ChessPlacementFailure,
            ),
            self.assertRaisesRegex(RandomizationSamplingError, "seed=20082"),
        ):
            self.randomizer.randomize(self.model, self.data, seed=20082)

    def test_piece_support_contacts_remain_allowed(self):
        self.randomizer._apply_states(
            self.model,
            self.data,
            {
                "black_pawn_1_jnt": {
                    "pos": [0.594, 0.004, 0.8038],
                    "quat": [1, 0, 0, 0],
                },
                "black_pawn_2_jnt": {"pos": [0.6, 0.46, 0.7728], "quat": [1, 0, 0, 0]},
            },
        )
        mujoco.mj_forward(self.model, self.data)
        self.assertTrue(self.data.ncon)
        self.assertTrue(self.randomizer._contacts_ok(self.model, self.data))

    def test_piece_contacts_allowed_only_for_knocked_setup_and_arms_always_rejected(
        self,
    ):
        for name in ("black_pawn_1_jnt", "black_pawn_2_jnt"):
            address = self.model.joint(name).qposadr[0]
            self.data.qpos[address : address + 3] = [0.6, -0.168, 0.805]
        mujoco.mj_forward(self.model, self.data)
        self.assertFalse(self.randomizer._contacts_ok(self.model, self.data))
        self.assertTrue(
            self.randomizer._contacts_ok(
                self.model, self.data, allow_piece_contacts=True
            )
        )
        address = self.model.joint("black_pawn_1_jnt").qposadr[0]
        self.data.qpos[address : address + 3] = [0, 0, 2]
        mujoco.mj_forward(self.model, self.data)
        self.assertFalse(
            self.randomizer._contacts_ok(
                self.model, self.data, allow_piece_contacts=True
            )
        )


@unittest.skipUnless(
    (
        _CHESS_BASE_SCENE_XML.parent / "assets/task_chess/tin_box/tin_2/body/body.xml"
    ).exists(),
    "Full chess scene assets are not provisioned",
)
class ChessSceneResetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from abc_sim import make_env

        cls.env = make_env(task="set_up_chess_pieces_on_board", render_cameras=False)
        cls.addClassCleanup(cls.env.close)

    def test_seed_20082_stays_separated_during_initial_hold(self):
        self.env.reset(seed=20082)
        state = self.env._last_randomization
        self.assertEqual(state.metadata["scenario"], "tin_setup")
        self.assertEqual(state.metadata["target_count"], 8)
        self.assertFalse(tin_board_contacts(self.env.model, self.env.data))
        initial_board = self.env.data.body("chessboard").xpos.copy()
        initial_tin = self.env.data.body("tin_box").xpos.copy()
        joint_ids = self.env.model.actuator_trnid[:, 0]
        self.env.data.ctrl[:] = self.env.data.qpos[
            self.env.model.jnt_qposadr[joint_ids]
        ]
        for _ in range(100):
            mujoco.mj_step(self.env.model, self.env.data)
            self.assertFalse(tin_board_contacts(self.env.model, self.env.data))
        self.assertTrue(np.isfinite(self.env.data.qvel).all())
        self.assertLess(
            np.linalg.norm(
                self.env.data.body("chessboard").xpos[:2] - initial_board[:2]
            ),
            0.005,
        )
        self.assertLess(
            np.linalg.norm(self.env.data.body("tin_box").xpos[:2] - initial_tin[:2]),
            0.01,
        )

    def test_initial_seed_sweep_has_no_container_board_contacts(self):
        for seed in range(20000, 20100):
            with self.subTest(seed=seed):
                self.env.reset(seed=seed)
                self.assertFalse(tin_board_contacts(self.env.model, self.env.data))
                if self.env._last_randomization.metadata["tin_active"]:
                    distance = mujoco.mj_geomDistance(
                        self.env.model,
                        self.env.data,
                        self.env.model.geom("board_collision").id,
                        self.env.model.geom("tin_box_floor").id,
                        0.2,
                        None,
                    )
                    self.assertGreaterEqual(distance, 0.02 - 1e-8)
                    corners = geom_corners(
                        self.env.model, self.env.data, "tin_box_floor"
                    )[:, :2]
                    self.assertTrue(np.all(corners >= [0.34 - 1e-10, -0.65 - 1e-10]))
                    self.assertTrue(np.all(corners <= [0.84 + 1e-10, 0.65 + 1e-10]))


if __name__ == "__main__":
    unittest.main()
