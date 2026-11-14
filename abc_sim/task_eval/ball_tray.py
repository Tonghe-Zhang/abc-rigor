"""Ball-on-tray balancing task evaluator."""

from __future__ import annotations

import math

import mujoco
import numpy as np

from abc_sim.task_eval.base import TaskEvalResult
from abc_sim.task_eval.contacts import gripper_geom_ids, has_contact, subtree_geom_ids
from abc_sim.task_specs import SimTaskSpec


def sample_hold_duration_s(seed: int | np.random.Generator | None) -> float:
    """Sample the 21 half-second requirements using a seed or existing RNG."""
    sampled = float(np.random.default_rng(seed).uniform(5.0, 15.5))
    return math.floor(sampled * 2.0) / 2.0


def format_hold_prompt(duration_s: float) -> str:
    return f"balance the ball in the tray for at least {duration_s:g} seconds"


class BallTrayBalancingEvaluator:
    """Hold the lifted tray continuously, then return it to table contact.

    Invalid samples restart the hold timer. Completion latches after table contact.

    By default the ball must come down with the tray: losing it after the hold
    ends the attempt with reward 0 (``require_ball_on_tray_when_lowered``), and
    a hold with a gripper touching the ball does not count
    (``forbid_gripper_ball_contact``). With both off, the ball no longer affects
    success once the sampled duration is met.
    """

    requires_step_updates = True

    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        spec: SimTaskSpec,
        ball_joint_name: str = "ball_tray_ball_joint",
        tray_site_name: str = "ball_tray_center_site",
        usable_half_extents_m: tuple[float, float] = (0.126, 0.1215),
        ball_radius_m: float = 0.022,
        dropped_local_z_threshold_m: float = 0.008,
        min_lift_m: float = 0.05,
        require_ball_on_tray_when_lowered: bool = True,
        forbid_gripper_ball_contact: bool = True,
    ) -> None:
        self.spec = spec
        self._ball_joint_name = ball_joint_name
        self._tray_site_name = tray_site_name
        self._usable_half_extents = np.asarray(usable_half_extents_m, dtype=np.float32)
        self._ball_radius_m = float(ball_radius_m)
        self._dropped_local_z_threshold_m = float(dropped_local_z_threshold_m)
        self._min_lift_m = float(min_lift_m)
        self.require_ball_on_tray_when_lowered = bool(require_ball_on_tray_when_lowered)
        self.forbid_gripper_ball_contact = bool(forbid_gripper_ball_contact)
        self._configured_hold_duration_s = sample_hold_duration_s(0)
        self._bind_model(model)
        self.reset(nworld=1)

    def _bind_model(self, model: mujoco.MjModel) -> None:
        self.model = model
        self._ball_qpos_addr = self._resolve_free_joint_qpos_addr(
            model,
            self._ball_joint_name,
        )
        self._tray_site_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            self._tray_site_name,
        )
        if self._tray_site_id < 0:
            raise ValueError(f"Site {self._tray_site_name!r} not found in model")
        tray_body = int(model.site_bodyid[self._tray_site_id])
        self._tray_geom_ids = subtree_geom_ids(model, tray_body)
        table_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_plane")
        if table_geom < 0:
            raise ValueError("Table geom 'table_plane' not found")
        self._table_geom_ids = {int(table_geom)}
        ball_body = int(model.jnt_bodyid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, self._ball_joint_name)])
        self._ball_geom_ids = subtree_geom_ids(model, ball_body)
        self._gripper_geom_ids = gripper_geom_ids(model)

    def reset(self, *, nworld: int = 1) -> None:
        """Clear temporal state while retaining the configured requirement."""
        self._nworld = int(nworld)
        self._hold_duration_s = np.full(self._nworld, self._configured_hold_duration_s)
        self._hold_started_at = np.full(self._nworld, np.nan)
        self._last_time_s = np.full(self._nworld, np.nan)
        self._hold_complete = np.zeros(self._nworld, dtype=bool)
        self._hold_completed_at = np.full(self._nworld, np.nan)
        self._completed_hold_s = np.zeros(self._nworld, dtype=np.float64)
        self._last_table_contact = np.zeros(self._nworld, dtype=bool)
        self._completed = np.zeros(self._nworld, dtype=bool)
        self._ball_lost = np.zeros(self._nworld, dtype=bool)
        self._rest_tray_z: np.ndarray | None = None

    def configure_from_randomization(
        self,
        model: mujoco.MjModel,
        randomization: object,
    ) -> None:
        if model is not self.model:
            self._bind_model(model)
        seed = getattr(randomization, "seed", None)
        metadata = getattr(randomization, "metadata", None)
        duration = (
            float(metadata["hold_duration_s"])
            if isinstance(metadata, dict) and "hold_duration_s" in metadata
            else sample_hold_duration_s(seed)
        )
        if not np.isfinite(duration) or not 5.0 <= duration <= 15.0:
            raise ValueError("hold_duration_s must be between 5 and 15 seconds")
        self._configured_hold_duration_s = duration
        self._hold_duration_s[:] = duration
        if not isinstance(metadata, dict):
            return
        usable_half_extents = metadata.get("usable_half_extents_m")
        if usable_half_extents is not None:
            extents = np.asarray(usable_half_extents, dtype=np.float32)
            if extents.shape != (2,) or np.any(extents <= 0.0):
                raise ValueError(
                    "ball tray usable_half_extents_m must contain two positive values"
                )
            self._usable_half_extents = extents
        ball_radius_m = metadata.get("ball_radius_m")
        if ball_radius_m is not None:
            radius = float(ball_radius_m)
            if not np.isfinite(radius) or radius <= 0.0:
                raise ValueError("ball tray ball_radius_m must be positive and finite")
            self._ball_radius_m = radius

    @staticmethod
    def _resolve_free_joint_qpos_addr(model: mujoco.MjModel, joint_name: str) -> int:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Joint {joint_name!r} not found in model")
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"Joint {joint_name!r} must be a free joint")
        return int(model.jnt_qposadr[joint_id])

    def evaluate_model_data(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> TaskEvalResult:
        if model is not self.model:
            self._bind_model(model)
            self.reset(nworld=1)
        metrics = self._sample_metrics_from_data(data)
        return self._result_from_metrics(metrics)

    def evaluate_qpos_batch(
        self, qpos_batch: np.ndarray, *, time_s: float | np.ndarray | None = None
    ) -> TaskEvalResult:
        if time_s is None:
            raise ValueError(
                "Timed ball-tray scoring requires time_s or evaluate_model_data(model, data)"
            )
        qpos_batch = np.asarray(qpos_batch, dtype=np.float32)
        if qpos_batch.ndim != 2 or qpos_batch.shape[1] != self.model.nq:
            raise ValueError(
                f"Expected qpos batch shape (B, {self.model.nq}), got {qpos_batch.shape}"
            )
        if qpos_batch.shape[0] != self._nworld:
            self.reset(nworld=qpos_batch.shape[0])
        times = np.broadcast_to(np.asarray(time_s, dtype=np.float64), (self._nworld,))

        data = mujoco.MjData(self.model)
        samples = []
        for qpos, timestamp in zip(qpos_batch, times):
            data.qpos[:] = qpos
            data.time = float(timestamp)
            mujoco.mj_forward(self.model, data)
            samples.append(self._sample_metrics_from_data(data))
        metrics = {
            key: np.stack([sample[key] for sample in samples], axis=0)
            for key in samples[0]
        }
        return self._result_from_metrics(metrics)

    def _sample_metrics_from_data(self, data: mujoco.MjData) -> dict[str, np.ndarray]:
        ball_pos = np.asarray(
            data.qpos[self._ball_qpos_addr : self._ball_qpos_addr + 3],
            dtype=np.float32,
        ).copy()
        tray_center = np.asarray(
            data.site_xpos[self._tray_site_id], dtype=np.float32
        ).copy()
        rot_world_from_tray = np.asarray(
            data.site_xmat[self._tray_site_id],
            dtype=np.float32,
        ).reshape(3, 3)
        local_pos = rot_world_from_tray.T @ (ball_pos - tray_center)
        center_error = np.linalg.norm(local_pos[:2]).astype(np.float32)
        normalized_center_error = np.max(
            np.abs(local_pos[:2]) / self._usable_half_extents
        ).astype(np.float32)
        inside_tray = np.array(
            np.all(np.abs(local_pos[:2]) <= self._usable_half_extents),
            dtype=bool,
        )
        ball_dropped = np.array(
            local_pos[2] < self._dropped_local_z_threshold_m,
            dtype=bool,
        )
        return {
            "ball_pos": ball_pos,
            "tray_center": tray_center,
            "ball_tray_local_pos": local_pos.astype(np.float32),
            "center_error_m": np.array(center_error, dtype=np.float32),
            "normalized_center_error": np.array(
                normalized_center_error, dtype=np.float32
            ),
            "inside_tray": inside_tray,
            "ball_dropped": ball_dropped,
            "time_s": np.asarray(data.time, dtype=np.float64),
            "ball_gripper_contact": np.asarray(
                has_contact(data, self._ball_geom_ids, self._gripper_geom_ids), dtype=bool
            ),
            "tray_table_contact": np.asarray(
                has_contact(data, self._tray_geom_ids, self._table_geom_ids), dtype=bool
            ),
        }

    def _result_from_metrics(self, metrics: dict[str, np.ndarray]) -> TaskEvalResult:
        current_balanced = np.logical_and(
            np.asarray(metrics["inside_tray"], dtype=bool),
            np.logical_not(np.asarray(metrics["ball_dropped"], dtype=bool)),
        )
        if current_balanced.ndim == 0:
            current_balanced = current_balanced[None]
            metrics = {
                key: value[None]
                if np.asarray(value).ndim > 0
                else np.asarray(value)[None]
                for key, value in metrics.items()
            }
        if current_balanced.shape[0] != self._nworld:
            self.reset(nworld=current_balanced.shape[0])

        tray_z = np.asarray(metrics["tray_center"], dtype=np.float32)[:, 2]
        if self._rest_tray_z is None:
            self._rest_tray_z = tray_z.copy()
        tray_lift = (tray_z - self._rest_tray_z).astype(np.float32)
        lifted = tray_lift >= self._min_lift_m

        now = np.asarray(metrics["time_s"], dtype=np.float64)
        if np.any(now < self._last_time_s):
            raise ValueError(
                "Simulation time moved backwards without an evaluator reset"
            )
        self._last_time_s = now.copy()
        ready_to_lower = self._hold_complete.copy()
        valid_hold = current_balanced & lifted
        if self.forbid_gripper_ball_contact:
            valid_hold &= ~np.asarray(metrics["ball_gripper_contact"], dtype=bool)
        self._hold_started_at[~valid_hold & ~self._hold_complete] = np.nan
        start = valid_hold & ~self._hold_complete & np.isnan(self._hold_started_at)
        self._hold_started_at[start] = now[start]
        elapsed = np.where(
            np.isnan(self._hold_started_at), 0.0, now - self._hold_started_at
        )
        self._hold_complete |= valid_hold & (elapsed >= self._hold_duration_s)
        just_held = self._hold_complete & ~ready_to_lower
        self._completed_hold_s[just_held] = elapsed[just_held]
        self._hold_completed_at[just_held] = now[just_held]
        elapsed = np.where(self._hold_complete, self._completed_hold_s, elapsed)
        if self.require_ball_on_tray_when_lowered:
            self._ball_lost |= ready_to_lower & ~self._completed & ~current_balanced
        table_contact = np.asarray(metrics["tray_table_contact"], dtype=bool)
        new_contact = table_contact & ~self._last_table_contact
        self._completed |= (
            ready_to_lower & (now > self._hold_completed_at) & new_contact & ~self._ball_lost
        )
        self._last_table_contact = table_contact.copy()
        progress = np.clip(elapsed / self._hold_duration_s, 0.0, 1.0)
        reward = np.where(
            self._completed,
            1.0,
            np.where(self._ball_lost, 0.0, np.where(self._hold_complete, 0.9, 0.9 * progress)),
        )

        return TaskEvalResult(
            reward=reward,
            success=self._completed.copy(),
            metrics={
                **metrics,
                "current_balanced": current_balanced.copy(),
                "tray_lift_m": tray_lift,
                "lifted": lifted.copy(),
                "continuous_hold_s": elapsed.copy(),
                "hold_duration_s": self._hold_duration_s.copy(),
                "hold_complete": self._hold_complete.copy(),
                "ball_lost_after_hold": self._ball_lost.copy(),
                "ever_success": self._completed.copy(),
                "usable_half_extents_m": np.broadcast_to(
                    self._usable_half_extents,
                    (self._nworld, 2),
                ).copy(),
                "ball_radius_m": np.full(
                    (self._nworld,),
                    self._ball_radius_m,
                    dtype=np.float32,
                ),
            },
        )
