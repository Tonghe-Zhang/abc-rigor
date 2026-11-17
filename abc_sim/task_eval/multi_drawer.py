"""Multi-drawer search task evaluator."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import mujoco
import numpy as np

from abc_sim.multi_drawer_episode import DrawerEpisode
from abc_sim.task_eval.base import EpisodeTaskEvaluator, TaskEvalResult
from abc_sim.task_eval.bottles import _quat_to_rotmat_batch
from abc_sim.task_specs import SimTaskSpec


def resolve_drawer_goal(
    model: mujoco.MjModel, *, goal_bin_joint_name: str, goal_site_name: str
) -> tuple[int, int]:
    """Validate a bin freejoint and its directly attached goal-site frame."""
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, goal_bin_joint_name)
    if joint_id < 0 or model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError(f"Drawer goal joint {goal_bin_joint_name!r} must be a free joint.")
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, goal_site_name)
    if site_id < 0 or model.site_bodyid[site_id] != model.jnt_bodyid[joint_id]:
        raise ValueError("The drawer goal region must belong to the configured goal bin.")
    return joint_id, site_id


@runtime_checkable
class DrawerProgressSource(EpisodeTaskEvaluator, Protocol):
    """Ordered drawer deposits shared by prompt sequencing and optional scores.

    Results identify ``target_count`` and ``completed_target_count`` for each
    world. Preparation and configuration consume ``DrawerEpisode`` definitions.
    Joint validation declares the source's native or shaped travel requirements.
    """

    def validate_drawer_joints(
        self, model: mujoco.MjModel, names: Sequence[str]
    ) -> list[int]:
        """Validate every potentially sampled drawer without changing state."""


class MultiDrawerSearchProgress:
    """Observe ordered deposits and produce native drawer progress diagnostics.

    This source also serves tasks without a reward evaluator. Reward exposure
    is selected by the task specification, independently of prompt progression.
    """

    requires_step_updates = True

    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        spec: SimTaskSpec,
        target_joint_name: str = "drawer_search_target_joint",
        goal_bin_joint_name: str = "drawer_search_goal_bin_joint",
        goal_site_name: str = "drawer_search_goal_region",
    ) -> None:
        self.model = model
        self.spec = spec
        self._initial_target_joint_name = target_joint_name
        self._goal_bin_joint_name = goal_bin_joint_name
        self._goal_site_name = goal_site_name
        self._episode: DrawerEpisode | None = None
        self.validate_model(model)
        self._target_qpos_addrs = [
            self._resolve_free_joint_qpos_addr(model, target_joint_name)
        ]
        self._bin_qpos_addr = self._resolve_free_joint_qpos_addr(
            model,
            goal_bin_joint_name,
        )
        (
            self._goal_local_center,
            self._rot_bin_from_goal,
            self._goal_half_size,
        ) = self._resolve_goal_site(model, goal_site_name)
        self.reset(nworld=1)

    def configure_from_randomization(
        self,
        model: mujoco.MjModel,
        randomization: Any,
    ) -> None:
        self.configure_episode(model, self.prepare_randomization(model, randomization))

    def prepare_randomization(self, model: mujoco.MjModel, randomization: Any) -> DrawerEpisode:
        """Validate the effective sequence before changing episode or physics state."""
        self.validate_model(model)
        if randomization is None:
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, self._initial_target_joint_name
            )
            self._resolve_free_joint_qpos_addr(model, self._initial_target_joint_name)
            metadata = {
                "target_joint": self._initial_target_joint_name,
                "target_body": mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_BODY, model.jnt_bodyid[joint_id]
                ),
            }
        else:
            metadata = randomization.metadata
        episode = DrawerEpisode.from_metadata(metadata)
        self.spec.resolve_reset_request({"target_count": len(episode.targets)})
        for target in episode.targets:
            self._resolve_free_joint_qpos_addr(model, target.joint)
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target.body)
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, target.joint)
            if body_id < 0 or model.jnt_bodyid[joint_id] != body_id:
                raise ValueError("Target body and free joint must identify the same object.")
        return episode

    def validate_model(self, model: mujoco.MjModel) -> None:
        """Require the configured qpos target and directly attached goal frame."""
        self._resolve_free_joint_qpos_addr(model, self._initial_target_joint_name)
        resolve_drawer_goal(
            model,
            goal_bin_joint_name=self._goal_bin_joint_name,
            goal_site_name=self._goal_site_name,
        )

    def validate_drawer_joints(
        self, model: mujoco.MjModel, names: Sequence[str]
    ) -> list[int]:
        """Resolve drawer slides without changing the model or episode history."""
        joint_ids = []
        for name in names:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0 or model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_SLIDE:
                raise ValueError(f"Drawer joint {name!r} must be a slide joint.")
            joint_ids.append(joint_id)
        return joint_ids

    def configure_episode(self, model: mujoco.MjModel, episode: DrawerEpisode) -> None:
        """Install the same validated episode definition used by the runtime."""
        self.validate_model(model)
        target_qpos_addrs = [
            self._resolve_free_joint_qpos_addr(model, target.joint)
            for target in episode.targets
        ]
        if model is not self.model:
            self.model = model
            self._bin_qpos_addr = self._resolve_free_joint_qpos_addr(
                model,
                self._goal_bin_joint_name,
            )
            (
                self._goal_local_center,
                self._rot_bin_from_goal,
                self._goal_half_size,
            ) = self._resolve_goal_site(model, self._goal_site_name)
        self._target_qpos_addrs = target_qpos_addrs
        self._episode = episode
        self.reset(nworld=self._nworld)

    def reset(self, *, nworld: int = 1) -> None:
        self._nworld = int(nworld)
        self._ever_success = np.zeros((self._nworld,), dtype=bool)
        self._completed_target_count = np.zeros((self._nworld,), dtype=np.int32)

    @staticmethod
    def _resolve_free_joint_qpos_addr(model: mujoco.MjModel, joint_name: str) -> int:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Joint {joint_name!r} not found in model")
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"Joint {joint_name!r} must be a free joint")
        return int(model.jnt_qposadr[joint_id])

    @staticmethod
    def _resolve_goal_site(
        model: mujoco.MjModel,
        goal_site_name: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        site_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            goal_site_name,
        )
        if site_id < 0:
            raise ValueError(f"Site {goal_site_name!r} not found in model")
        return (
            np.asarray(model.site_pos[site_id], dtype=np.float32).copy(),
            _quat_to_rotmat_batch(model.site_quat[site_id][None])[0],
            np.asarray(model.site_size[site_id], dtype=np.float32).copy(),
        )

    def evaluate_qpos_batch(self, qpos_batch: np.ndarray) -> TaskEvalResult:
        if self._episode is None:
            self.configure_episode(
                self.model, self.prepare_randomization(self.model, None)
            )
        qpos_batch = np.asarray(qpos_batch, dtype=np.float32)
        if qpos_batch.ndim != 2 or qpos_batch.shape[1] != self.model.nq:
            raise ValueError(
                f"Expected qpos batch shape (B, {self.model.nq}), got {qpos_batch.shape}"
            )
        if qpos_batch.shape[0] != self._nworld:
            self.reset(nworld=qpos_batch.shape[0])

        target_positions = np.stack(
            [
                qpos_batch[:, qpos_addr: qpos_addr + 3]
                for qpos_addr in self._target_qpos_addrs
            ],
            axis=1,
        )
        bin_pos = qpos_batch[:, self._bin_qpos_addr: self._bin_qpos_addr + 3]
        bin_quat = qpos_batch[:, self._bin_qpos_addr + 3: self._bin_qpos_addr + 7]
        rot_world_from_bin = _quat_to_rotmat_batch(bin_quat)
        rel_world = target_positions - bin_pos[:, None, :]
        rel_bin = np.einsum(
            "bij,btj->bti",
            np.swapaxes(rot_world_from_bin, 1, 2),
            rel_world,
        )
        rel_goal = np.einsum(
            "ij,btj->bti",
            self._rot_bin_from_goal.T,
            rel_bin - self._goal_local_center[None, None, :],
        )
        delta = np.abs(rel_goal)
        targets_in_bin = np.all(
            delta <= self._goal_half_size[None, None, :],
            axis=2,
        )

        target_count = len(self._target_qpos_addrs)
        for world_index in range(qpos_batch.shape[0]):
            while (
                self._completed_target_count[world_index] < target_count
                and targets_in_bin[
                    world_index,
                    self._completed_target_count[world_index],
                ]
            ):
                self._completed_target_count[world_index] += 1

        success = self._completed_target_count >= target_count
        self._ever_success |= success

        position_indices = np.minimum(self._completed_target_count, target_count - 1)
        active_target_in_bin = targets_in_bin[
            np.arange(qpos_batch.shape[0]), position_indices
        ]
        target_pos = target_positions[
            np.arange(qpos_batch.shape[0]),
            position_indices,
        ]
        world_goal_center = bin_pos + np.einsum(
            "bij,j->bi",
            rot_world_from_bin,
            self._goal_local_center,
        )
        xy_distance = np.linalg.norm(target_pos[:, :2] - world_goal_center[:, :2], axis=1)
        xy_radius = max(float(np.linalg.norm(self._goal_half_size[:2])), 1e-6)
        target_progress = np.clip(
            1.0 - (xy_distance / xy_radius),
            0.0,
            1.0,
        )
        reward = (
            self._completed_target_count.astype(np.float32)
            + np.where(success, 0.0, target_progress)
        ) / float(target_count)
        reward = np.where(success, 1.0, reward).astype(np.float32)
        return TaskEvalResult(
            reward=reward,
            success=success,
            metrics={
                "target_pos": target_pos.astype(np.float32),
                "goal_bin_pos": bin_pos.astype(np.float32),
                "goal_bin_center": np.broadcast_to(
                    world_goal_center,
                    target_pos.shape,
                ).copy(),
                "goal_bin_half_size": np.broadcast_to(
                    self._goal_half_size,
                    target_pos.shape,
                ).copy(),
                "target_bin_xy_distance_m": xy_distance.astype(np.float32),
                "target_in_bin": active_target_in_bin,
                "target_sequence_in_bin": targets_in_bin,
                "current_target_index": np.minimum(
                    self._completed_target_count,
                    target_count,
                ).copy(),
                "completed_target_count": self._completed_target_count.copy(),
                "target_count": np.full(
                    (qpos_batch.shape[0],),
                    target_count,
                    dtype=np.int32,
                ),
                "ever_success": self._ever_success.copy(),
            },
        )


class MultiDrawerSearchTargetInBinEvaluator(MultiDrawerSearchProgress):
    """Expose the native ordered-retrieval score and success condition."""


class ShapedMultiDrawerSearchEvaluator(MultiDrawerSearchTargetInBinEvaluator):
    """Track unique drawer openings and ordered retrievals on every physics tick.

    A drawer is open at half of its joint's available travel (0.108 m in the
    standard scene). All first openings in one observation use the target and
    opening history from before that observation. Replay must supply the same
    chronological observations; a final qpos snapshot cannot recover history.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._drawer_qpos_addrs: list[int] = []
        self._drawer_open_thresholds = np.empty(0)
        self._target_drawer_indices: list[int] = []
        super().__init__(**kwargs)

    def prepare_randomization(self, model: mujoco.MjModel, randomization: Any) -> DrawerEpisode:
        if randomization is None:
            raise RuntimeError(
                "Shaped drawer evaluation requires reset randomization metadata."
            )
        episode = super().prepare_randomization(model, randomization)
        if not episode.drawers or any(target.drawer not in episode.drawers for target in episode.targets):
            raise ValueError("Each target must identify an active source drawer.")
        self.validate_drawer_joints(model, episode.drawer_joints)
        return episode

    def validate_drawer_joints(
        self, model: mujoco.MjModel, names: Sequence[str]
    ) -> list[int]:
        """Require the travel limits used by chronological opening detection."""
        joint_ids = super().validate_drawer_joints(model, names)
        for name, joint_id in zip(names, joint_ids):
            if not model.jnt_limited[joint_id] or model.jnt_range[joint_id, 0] != 0:
                raise ValueError(f"Drawer joint {name!r} must have a zero-based travel limit.")
        return joint_ids

    def configure_episode(self, model: mujoco.MjModel, episode: DrawerEpisode) -> None:
        super().configure_episode(model, episode)
        joint_ids = self.validate_drawer_joints(model, episode.drawer_joints)
        self._drawer_qpos_addrs = [int(model.jnt_qposadr[j]) for j in joint_ids]
        self._drawer_open_thresholds = 0.5 * model.jnt_range[joint_ids, 1]
        self._target_drawer_indices = [episode.drawers.index(target.drawer) for target in episode.targets]
        self.reset(nworld=self._nworld)

    def reset(self, *, nworld: int = 1) -> None:
        super().reset(nworld=nworld)
        self._opened_drawers = np.zeros((nworld, len(self._drawer_qpos_addrs)), dtype=bool)
        self._credited_openings = np.zeros(nworld, dtype=np.int32)

    def evaluate_qpos_batch(self, qpos_batch: np.ndarray) -> TaskEvalResult:
        if not self._target_drawer_indices:
            raise RuntimeError("Shaped drawer evaluation requires reset randomization metadata.")
        qpos_batch = np.asarray(qpos_batch)
        if qpos_batch.ndim != 2 or qpos_batch.shape[1] != self.model.nq:
            raise ValueError(f"Expected qpos batch shape (B, {self.model.nq}).")
        if qpos_batch.shape[0] != self._nworld:
            self.reset(nworld=qpos_batch.shape[0])
        target_count = len(self._target_drawer_indices)
        active = self._completed_target_count < target_count
        indices = np.minimum(self._completed_target_count, target_count - 1)
        target_drawers = np.asarray(self._target_drawer_indices)[indices]
        eligible = active & ~self._opened_drawers[np.arange(self._nworld), target_drawers]
        opened = qpos_batch[:, self._drawer_qpos_addrs] >= self._drawer_open_thresholds
        first_openings = opened & ~self._opened_drawers
        self._credited_openings = np.minimum(
            4, self._credited_openings + eligible * first_openings.sum(axis=1)
        )
        self._opened_drawers |= opened
        result = super().evaluate_qpos_batch(qpos_batch)
        bonus = 0.05 * self._credited_openings
        reward = np.where(result.success, 1.0, 0.75 * result.reward + bonus)
        return TaskEvalResult(
            reward=reward,
            success=result.success,
            metrics={
                **result.metrics,
                "retrieval_reward": result.reward,
                "drawer_opening_bonus": bonus,
                "credited_drawer_openings": self._credited_openings.copy(),
                "opened_drawer_count": self._opened_drawers.sum(axis=1),
                "opened_drawers": self._opened_drawers.copy(),
            },
        )
