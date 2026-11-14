"""LEGO block color-sorting task evaluator."""

from __future__ import annotations

import re

import mujoco
import numpy as np

from abc_sim.task_eval.base import TaskEvalResult
from abc_sim.task_eval.bottles import _quat_to_rotmat_batch
from abc_sim.task_specs import SimTaskSpec


_LEGO_COLORS = ("red", "yellow", "blue")


class LegoBlocksSortingEvaluator:
    """Score success when every LEGO block is in its matching color bin."""

    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        spec: SimTaskSpec,
        penalize_wrong_bin: bool = True,
    ) -> None:
        self.model = model
        self.spec = spec
        # Each block in a wrong bin cancels one correctly sorted block in the reward (never below 0); off: only
        # correctly sorted blocks count. Either way a wrong-bin block blocks success.
        self.penalize_wrong_bin = bool(penalize_wrong_bin)
        self._block_qpos_addrs: dict[str, np.ndarray] = {}
        self._bin_qpos_addrs: dict[str, int] = {}
        self._goal_regions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.block_names: dict[str, list[str]] = {}
        for color in _LEGO_COLORS:
            names, addrs = self._resolve_block_qpos_addrs(model, color)
            self.block_names[color] = names
            self._block_qpos_addrs[color] = addrs
            self._bin_qpos_addrs[color] = self._resolve_joint_qpos_addr(
                model,
                f"{color}_lego_sorting_bin_joint",
            )
            self._goal_regions[color] = self._resolve_goal_site(
                model,
                f"{color}_lego_bin_goal_region",
            )
        self.reset(nworld=1)

    def reset(self, *, nworld: int = 1) -> None:
        self._nworld = int(nworld)
        self._ever_success = np.zeros((self._nworld,), dtype=bool)
        self._active = {
            color: np.ones((self._nworld, len(names)), dtype=bool)
            for color, names in self.block_names.items()
        }

    def set_active_object_joints(self, active_joints_by_world: Any) -> None:
        """Score only each world's active blocks; the rest sit parked off the table."""
        if self._nworld == 1 and active_joints_by_world and isinstance(active_joints_by_world[0], str):
            active_joints_by_world = [active_joints_by_world]
        for world, joints in enumerate(active_joints_by_world):
            active = set(joints)
            for color, names in self.block_names.items():
                self._active[color][world] = [f"{name}_joint" in active for name in names]

    @staticmethod
    def _resolve_block_qpos_addrs(
        model: mujoco.MjModel,
        color: str,
    ) -> tuple[list[str], np.ndarray]:
        entries: list[tuple[int, str, int]] = []
        pattern = re.compile(rf"{re.escape(color)}_lego_(\d+)_joint")
        for joint_id in range(model.njnt):
            joint_name = model.jnt(joint_id).name
            if not joint_name:
                continue
            match = pattern.fullmatch(joint_name)
            if match is None:
                continue
            index = int(match.group(1))
            entries.append((index, f"{color}_lego_{index}", int(model.jnt_qposadr[joint_id])))

        if not entries:
            raise ValueError(f"No {color}_lego_*_joint freejoints found in model")

        entries.sort(key=lambda item: item[0])
        names = [name for _, name, _ in entries]
        addrs = np.asarray([addr for _, _, addr in entries], dtype=np.int32)
        return names, addrs

    @staticmethod
    def _resolve_joint_qpos_addr(model: mujoco.MjModel, joint_name: str) -> int:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Joint {joint_name!r} not found in model")
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"Joint {joint_name!r} must be a free joint")
        return int(model.jnt_qposadr[joint_id])

    @staticmethod
    def _resolve_goal_site(
        model: mujoco.MjModel,
        site_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            raise ValueError(f"Site {site_name!r} not found in model")
        return (
            np.asarray(model.site_pos[site_id], dtype=np.float32),
            np.asarray(model.site_size[site_id], dtype=np.float32),
        )

    def _inside_bin(
        self,
        positions: np.ndarray,
        qpos_batch: np.ndarray,
        color: str,
    ) -> np.ndarray:
        bin_qpos_addr = self._bin_qpos_addrs[color]
        goal_local_center, goal_size = self._goal_regions[color]
        bin_pos = qpos_batch[:, bin_qpos_addr: bin_qpos_addr + 3]
        bin_quat = qpos_batch[:, bin_qpos_addr + 3: bin_qpos_addr + 7]
        rel_world = positions - bin_pos[:, None, :]
        rot_world_from_bin = _quat_to_rotmat_batch(bin_quat)
        rel_bin = np.einsum("bij,bnj->bni", np.swapaxes(rot_world_from_bin, 1, 2), rel_world)
        return np.all(
            np.abs(rel_bin - goal_local_center[None, None, :]) <= goal_size[None, None, :],
            axis=-1,
        )

    def evaluate_qpos_batch(self, qpos_batch: np.ndarray) -> TaskEvalResult:
        qpos_batch = np.asarray(qpos_batch, dtype=np.float32)
        if qpos_batch.ndim != 2 or qpos_batch.shape[1] != self.model.nq:
            raise ValueError(
                f"Expected qpos batch shape (B, {self.model.nq}), got {qpos_batch.shape}"
            )
        if qpos_batch.shape[0] != self._nworld:
            self.reset(nworld=qpos_batch.shape[0])

        num_correct = np.zeros((qpos_batch.shape[0],), dtype=np.int32)
        num_wrong_bin = np.zeros((qpos_batch.shape[0],), dtype=np.int32)
        metrics: dict[str, np.ndarray] = {}
        total_blocks = np.zeros((qpos_batch.shape[0],), dtype=np.int32)
        for color in _LEGO_COLORS:
            positions = np.stack(
                [qpos_batch[:, addr : addr + 3] for addr in self._block_qpos_addrs[color]],
                axis=1,
            )
            active = self._active[color]
            in_matching_bin = active & self._inside_bin(positions, qpos_batch, color)
            color_sorted = in_matching_bin.sum(axis=1).astype(np.int32)
            metrics[f"num_{color}_sorted"] = color_sorted
            num_correct += color_sorted
            total_blocks += active.sum(axis=1)

            for other_color in _LEGO_COLORS:
                if other_color == color:
                    continue
                num_wrong_bin += (active & self._inside_bin(positions, qpos_batch, other_color)).sum(
                    axis=1
                ).astype(np.int32)

        success = (num_correct == total_blocks) & (num_wrong_bin == 0)
        self._ever_success |= success
        metrics.update(
            {
                "num_correct": num_correct,
                "num_wrong_bin": num_wrong_bin,
                "ever_success": self._ever_success.copy(),
            }
        )
        return TaskEvalResult(
            reward=(
                np.maximum(num_correct - (num_wrong_bin if self.penalize_wrong_bin else 0), 0)
                / np.maximum(total_blocks, 1)
            ).astype(np.float32),
            success=success,
            metrics=metrics,
        )
