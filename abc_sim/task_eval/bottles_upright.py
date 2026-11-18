"""Pose scoring for three bottles standing on their bases on the table."""

from __future__ import annotations

import mujoco
import numpy as np

from abc_sim.task_eval.base import TaskEvalResult
from abc_sim.task_eval.bottles import _resolve_bottle_qpos_addrs
from abc_sim.task_specs import SimTaskSpec


class BottlesUprightEvaluator:
    """Count cap-up bottles within 14 degrees of vertical and 8 mm of the table.

    The bottle free-joint origin is the centre of its base. This pose rubric
    rejects airborne, inverted, sideways, and off-table bottles. It does not
    require gripper release or measure a dwell time.
    """

    def __init__(self, *, model: mujoco.MjModel, spec: SimTaskSpec) -> None:
        self.model = model
        self.spec = spec
        self.bottle_names, self._addrs = _resolve_bottle_qpos_addrs(model)
        if self.bottle_names != ["bottle_1", "bottle_2", "bottle_3"]:
            raise ValueError("Upright bottles requires exactly bottle_1, bottle_2, bottle_3.")
        self.reset()

    def reset(self, *, nworld: int = 1) -> None:
        self._max_upright = np.zeros(nworld, dtype=np.int32)
        self._ever_success = np.zeros(nworld, dtype=bool)

    def evaluate_qpos_batch(self, qpos_batch: np.ndarray) -> TaskEvalResult:
        qpos = np.asarray(qpos_batch)
        if qpos.ndim != 2 or qpos.shape[1] != self.model.nq:
            raise ValueError(f"Expected qpos shape (B, {self.model.nq}), got {qpos.shape}")
        if len(qpos) != len(self._max_upright):
            self.reset(nworld=len(qpos))
        poses = qpos[:, self._addrs[:, None] + np.arange(7)]
        x, y, z = np.moveaxis(poses[:, :, :3], -1, 0)
        quat = poses[:, :, 3:]
        upright_z = 1 - 2 * (quat[:, :, 1] ** 2 + quat[:, :, 2] ** 2)
        upright = (
            (upright_z >= 0.97) & (np.abs(z - 0.75) <= 0.008) & (x >= 0.34) & (x <= 0.86) & (y >= -0.60) & (y <= 0.60)
        )
        count = upright.sum(axis=1)
        success = count == 3
        self._max_upright = np.maximum(self._max_upright, count)
        self._ever_success |= success
        return TaskEvalResult(
            reward=count / 3,
            success=success,
            metrics={
                "num_upright_bottles": count,
                "num_active_bottles": np.full(len(qpos), 3),
                "bottle_upright_mask": upright,
                "bottle_upright_z": upright_z,
                "max_upright_bottles_so_far": self._max_upright.copy(),
                "ever_success": self._ever_success.copy(),
            },
        )
