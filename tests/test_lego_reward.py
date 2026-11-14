"""LEGO sorting: a block in a wrong bin costs as much as a correctly sorted one earns."""

from __future__ import annotations

import unittest

import mujoco

from abc_sim.task_eval.lego_blocks import LegoBlocksSortingEvaluator
from abc_sim.task_specs import get_task_spec

BIN_X = {"red": 0.0, "yellow": 1.0, "blue": 2.0}


def _body(name: str, joint: str, pos: str, extra: str = "") -> str:
    return (
        f'<body name="{name}" pos="{pos}"><freejoint name="{joint}"/>'
        f'<geom type="sphere" size=".01" mass=".1"/>{extra}</body>'
    )


class LegoWrongBinPenaltyTest(unittest.TestCase):
    def setUp(self):
        # One bin and one block per colour; every block starts inside its own bin's goal region.
        self.model = mujoco.MjModel.from_xml_string(
            "<mujoco><worldbody>"
            + "".join(
                _body(
                    f"{color}_bin",
                    f"{color}_lego_sorting_bin_joint",
                    f"{x} 0 .2",
                    f'<site name="{color}_lego_bin_goal_region" type="box" size=".1 .1 .1"/>',
                )
                + _body(f"{color}_lego_1", f"{color}_lego_1_joint", f"{x} 0 .2")
                for color, x in BIN_X.items()
            )
            + "</worldbody></mujoco>"
        )
        self.data = mujoco.MjData(self.model)
        spec = get_task_spec("lego blocks sorting")
        self.strict = LegoBlocksSortingEvaluator(model=self.model, spec=spec)
        self.lenient = LegoBlocksSortingEvaluator(
            model=self.model, spec=spec, penalize_wrong_bin=False
        )

    def scores(self, **into_bin):
        """(reward, success) for strict and lenient scoring, with each named block moved into that colour's bin."""
        for block, bin_color in into_bin.items():
            address = int(
                self.model.jnt_qposadr[self.model.joint(f"{block}_lego_1_joint").id]
            )
            self.data.qpos[address : address + 3] = [BIN_X[bin_color], 0, 0.2]
        qpos = self.data.qpos[None, :]
        return [
            (round(float(result.reward[0]), 4), result.scalar_success())
            for result in (
                self.strict.evaluate_qpos_batch(qpos),
                self.lenient.evaluate_qpos_batch(qpos),
            )
        ]

    def test_every_block_in_its_own_bin(self):
        self.assertEqual(self.scores(), [(1.0, True), (1.0, True)])

    def test_one_block_in_a_wrong_bin(self):
        # 2 correct - 1 wrong; lenient scoring ignores the wrong one. Neither succeeds.
        self.assertEqual(
            self.scores(blue="red"),
            [(round(1 / 3, 4), False), (round(2 / 3, 4), False)],
        )

    def test_reward_never_goes_below_zero(self):
        # 1 correct - 2 wrong, clipped at 0.
        self.assertEqual(
            self.scores(blue="red", yellow="red"),
            [(0.0, False), (round(1 / 3, 4), False)],
        )


if __name__ == "__main__":
    unittest.main()
