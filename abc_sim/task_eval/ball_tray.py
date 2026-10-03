"""Ball-on-tray balancing task evaluator."""

from __future__ import annotations

import mujoco
import numpy as np

from abc_sim.task_eval.base import TaskEvalResult
from abc_sim.task_specs import SimTaskSpec


class BallTrayBalancingEvaluator:
    """Score whether the tray is held up with the ball still on it.

    A world succeeds while the ball has never left the tray *and* the tray centre sits at least
    ``min_lift_m`` above the height it had at the first evaluation after reset. Without the lift
    requirement a policy that never touches the resting tray scores a perfect balance.
    """

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
    ) -> None:
        self.spec = spec
        self._ball_joint_name = ball_joint_name
        self._tray_site_name = tray_site_name
        self._usable_half_extents = np.asarray(usable_half_extents_m, dtype=np.float32)
        self._ball_radius_m = float(ball_radius_m)
        self._dropped_local_z_threshold_m = float(dropped_local_z_threshold_m)
        self._min_lift_m = float(min_lift_m)
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

    def reset(self, *, nworld: int = 1) -> None:
        self._nworld = int(nworld)
        self._alive = np.ones((self._nworld,), dtype=bool)
        self._ever_failed = np.zeros((self._nworld,), dtype=bool)
        self._ever_held = np.zeros((self._nworld,), dtype=bool)
        self._rest_tray_z: np.ndarray | None = None

    def configure_from_randomization(
        self,
        model: mujoco.MjModel,
        randomization: object,
    ) -> None:
        if model is not self.model:
            self._bind_model(model)
        metadata = getattr(randomization, "metadata", None)
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

    def evaluate_qpos_batch(self, qpos_batch: np.ndarray) -> TaskEvalResult:
        qpos_batch = np.asarray(qpos_batch, dtype=np.float32)
        if qpos_batch.ndim != 2 or qpos_batch.shape[1] != self.model.nq:
            raise ValueError(
                f"Expected qpos batch shape (B, {self.model.nq}), got {qpos_batch.shape}"
            )
        if qpos_batch.shape[0] != self._nworld:
            self.reset(nworld=qpos_batch.shape[0])

        data = mujoco.MjData(self.model)
        samples = []
        for qpos in qpos_batch:
            data.qpos[:] = qpos
            mujoco.mj_forward(self.model, data)
            samples.append(self._sample_metrics_from_data(data))
        metrics = {
            key: np.stack([sample[key] for sample in samples], axis=0)
            for key in samples[0]
        }
        return self._result_from_metrics(metrics)

    def _sample_metrics_from_data(self, data: mujoco.MjData) -> dict[str, np.ndarray]:
        ball_pos = np.asarray(
            data.qpos[self._ball_qpos_addr: self._ball_qpos_addr + 3],
            dtype=np.float32,
        ).copy()
        tray_center = np.asarray(data.site_xpos[self._tray_site_id], dtype=np.float32).copy()
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
            "normalized_center_error": np.array(normalized_center_error, dtype=np.float32),
            "inside_tray": inside_tray,
            "ball_dropped": ball_dropped,
        }

    def _result_from_metrics(self, metrics: dict[str, np.ndarray]) -> TaskEvalResult:
        current_balanced = np.logical_and(
            np.asarray(metrics["inside_tray"], dtype=bool),
            np.logical_not(np.asarray(metrics["ball_dropped"], dtype=bool)),
        )
        if current_balanced.ndim == 0:
            current_balanced = current_balanced[None]
            metrics = {
                key: value[None] if np.asarray(value).ndim > 0 else np.asarray(value)[None]
                for key, value in metrics.items()
            }
        if current_balanced.shape[0] != self._nworld:
            self.reset(nworld=current_balanced.shape[0])

        tray_z = np.asarray(metrics["tray_center"], dtype=np.float32)[:, 2]
        if self._rest_tray_z is None:
            self._rest_tray_z = tray_z.copy()
        tray_lift = (tray_z - self._rest_tray_z).astype(np.float32)
        lifted = tray_lift >= self._min_lift_m

        self._alive &= current_balanced
        self._ever_failed |= np.logical_not(current_balanced)
        held = np.logical_and(self._alive, lifted)
        self._ever_held |= held
        center_error = np.asarray(metrics["center_error_m"], dtype=np.float32)
        tray_radius = max(float(np.linalg.norm(self._usable_half_extents)), 1e-6)
        reward = np.clip(1.0 - center_error / tray_radius, 0.0, 1.0).astype(np.float32)
        reward = np.where(held, reward, 0.0).astype(np.float32)

        return TaskEvalResult(
            reward=reward,
            success=held.copy(),
            metrics={
                **metrics,
                "current_balanced": current_balanced.copy(),
                "tray_lift_m": tray_lift,
                "lifted": lifted.copy(),
                "alive": self._alive.copy(),
                "ever_failed": self._ever_failed.copy(),
                # Balancing is a maintenance task: a world that has lifted the tray keeps
                # ever_success until it drops the ball, and the rollout loops stop on
                # ever_failed rather than ever_success. A trial that ends with the tray back
                # down fails on final success.
                "ever_success": np.logical_and(self._ever_held, np.logical_not(self._ever_failed)),
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
