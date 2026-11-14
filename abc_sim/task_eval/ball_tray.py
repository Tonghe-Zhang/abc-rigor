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


HOLD_WINDOW_S = (10.0, 15.0)
TRAY_HANDLE_GEOMS = (
    "ball_tray_left_handle_collision",
    "ball_tray_right_handle_collision",
)
_HANDLE_GRACE_S = 0.1
_HELD_REWARD = 0.9
_OVER_HOLD_FLOOR = 0.5
_EARLY_PLACEMENT_SHARE = (
    0.5  # of the hold credit kept by a placement before the minimum
)
_LATE_SUCCESS_REWARD = (
    0.8  # success reward at max_hold_s; 1.0 up to the prompt's upper bound
)
_IDLE, _HOLDING, _SETTLING, _DONE = 0, 1, 2, 3


def format_window_prompt(window_s: tuple[float, float]) -> str:
    low, high = window_s
    return (
        f"keep the ball balanced on the tray for {low:g} to {high:g} seconds, "
        "then gently place the tray on the table"
    )


class BallTrayBalancingEvaluator:
    """Hold the lifted tray continuously, then return it to table contact.

    Invalid samples restart the hold timer. Completion latches after table contact.

    By default the ball must come down with the tray: losing it after the hold
    ends the attempt with reward 0 (``require_ball_on_tray_when_lowered``), and
    a hold with a gripper touching the ball does not count
    (``forbid_gripper_ball_contact``). With both off, the ball no longer affects
    success once the sampled duration is met.

    By default the prompt asks for a 10-15 s hold (``hold_window_s``), then a
    gentle placement. Air time runs from the start of the hold until the tray is
    back below the lift height, dips included; touchdown must follow within
    ``max_lowering_s``. A gentle placement touches down with both handles
    gripped (``require_handles_at_touchdown``), and the ball then stays on the
    tray, untouched, for ``settle_s``.

    While up, the reward is 0.9 x air time / 10 s, 0.9 through 15 s, then falls
    to 0.5 over ``over_hold_decay_s``. Placing ends the attempt. Success needs
    10 s to ``max_hold_s`` of air time and scores 1.0 up to 15 s, then down to
    0.8 at ``max_hold_s``: the demonstrations hold 13.7 s at the median and
    17.6 s at p95, and touch down within 5.3 s of the final drop at p95, so the
    defaults (18 s, 5 s) pass about 95% of them at each end while favouring the
    asked 10-15 s. A gentle placement before 10 s keeps half the reward reached,
    and a later one (or a slow touchdown) keeps all of it; neither succeeds. A
    placement without both handles keeps at most 0.5. Touching or losing the
    ball before 10 s restarts the hold; after 10 s, or while settling, it
    scores 0.
    ``hold_window_s=None`` restores the sampled "at least X seconds" hold.
    """

    requires_step_updates = True
    evaluate_at_control_rate = True

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
        hold_window_s: tuple[float, float] | None = HOLD_WINDOW_S,
        max_hold_s: float = 18.0,
        max_lowering_s: float = 5.0,
        require_handles_at_touchdown: bool = True,
        settle_s: float = 1.0,
        over_hold_decay_s: float = 8.0,
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
        self.hold_window_s = (
            None if hold_window_s is None else tuple(map(float, hold_window_s))
        )
        if (
            self.hold_window_s is not None
            and not 0.0 < self.hold_window_s[0] <= self.hold_window_s[1]
        ):
            raise ValueError("hold_window_s must be (low, high) with 0 < low <= high")
        self.max_hold_s = float(max_hold_s)
        self.max_lowering_s = float(max_lowering_s)
        if self.hold_window_s is not None and self.max_hold_s < self.hold_window_s[0]:
            raise ValueError("max_hold_s must be at least the low end of hold_window_s")
        self.require_handles_at_touchdown = bool(require_handles_at_touchdown)
        self.settle_s = float(settle_s)
        self.over_hold_decay_s = float(over_hold_decay_s)
        self._configured_hold_duration_s = (
            self.hold_window_s[0]
            if self.hold_window_s is not None
            else sample_hold_duration_s(0)
        )
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
        ball_body = int(
            model.jnt_bodyid[
                mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_JOINT, self._ball_joint_name
                )
            ]
        )
        self._ball_geom_ids = subtree_geom_ids(model, ball_body)
        self._gripper_geom_ids = gripper_geom_ids(model)
        self._handle_geom_ids = [
            {geom}
            for name in TRAY_HANDLE_GEOMS
            if (geom := mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)) >= 0
        ]

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
        self._phase = np.full(self._nworld, _IDLE, dtype=np.int8)
        self._air_ended_at = np.full(self._nworld, np.nan)
        self._handles_seen_at = np.full(self._nworld, np.nan)
        self._settle_until = np.full(self._nworld, np.nan)
        self._touch_value = np.zeros(self._nworld, dtype=np.float64)
        self._touch_in_window = np.zeros(self._nworld, dtype=bool)
        self._touch_air_s = np.zeros(self._nworld, dtype=np.float64)
        self._final_reward = np.zeros(self._nworld, dtype=np.float64)
        self._early_placement = np.zeros(self._nworld, dtype=bool)
        self._late_placement = np.zeros(self._nworld, dtype=bool)
        self._rough_placement = np.zeros(self._nworld, dtype=bool)
        self._needs_handles = False

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
        if self.hold_window_s is not None:
            duration = self.hold_window_s[0]
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
                has_contact(data, self._ball_geom_ids, self._gripper_geom_ids),
                dtype=bool,
            ),
            "tray_table_contact": np.asarray(
                has_contact(data, self._tray_geom_ids, self._table_geom_ids), dtype=bool
            ),
            # Only touchdown uses the handles: scan only while a hold is running.
            "both_handles_gripped": np.asarray(
                self._needs_handles
                and len(self._handle_geom_ids) == 2
                and all(
                    has_contact(data, handle, self._gripper_geom_ids)
                    for handle in self._handle_geom_ids
                ),
                dtype=bool,
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
        if self.hold_window_s is not None:
            return self._window_result(
                metrics, current_balanced, lifted, tray_lift, now
            )
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
            ready_to_lower
            & (now > self._hold_completed_at)
            & new_contact
            & ~self._ball_lost
        )
        self._last_table_contact = table_contact.copy()
        progress = np.clip(elapsed / self._hold_duration_s, 0.0, 1.0)
        reward = np.where(
            self._completed,
            1.0,
            np.where(
                self._ball_lost, 0.0, np.where(self._hold_complete, 0.9, 0.9 * progress)
            ),
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
                "hold_window_s": np.full((self._nworld, 2), np.nan),
                "hold_complete": self._hold_complete.copy(),
                "ball_lost_after_hold": self._ball_lost.copy(),
                "ever_success": self._completed.copy(),
                **self._shape_metrics(),
            },
        )

    def _shape_metrics(self) -> dict[str, np.ndarray]:
        return {
            "usable_half_extents_m": np.broadcast_to(
                self._usable_half_extents,
                (self._nworld, 2),
            ).copy(),
            "ball_radius_m": np.full(
                (self._nworld,),
                self._ball_radius_m,
                dtype=np.float32,
            ),
        }

    def _window_result(
        self,
        metrics: dict[str, np.ndarray],
        current_balanced: np.ndarray,
        lifted: np.ndarray,
        tray_lift: np.ndarray,
        now: np.ndarray,
    ) -> TaskEvalResult:
        """Score a hold that ends with a gentle placement, best inside the window."""
        low, target, high = (
            self.hold_window_s[0],
            self.hold_window_s[1],
            self.max_hold_s,
        )
        ball_ok = current_balanced.copy()
        if self.forbid_gripper_ball_contact:
            ball_ok &= ~np.asarray(metrics["ball_gripper_contact"], dtype=bool)
        phase = self._phase

        start = (phase == _IDLE) & ball_ok & lifted
        self._hold_started_at[start] = now[start]
        self._air_ended_at[start] = np.nan
        phase[start] = _HOLDING
        holding = phase == _HOLDING
        # Air time stops while the tray is below the lift height and resumes if it goes back up.
        self._air_ended_at[holding & lifted] = np.nan
        dipped = holding & ~lifted & np.isnan(self._air_ended_at)
        self._air_ended_at[dipped] = now[dipped]
        since_start = np.where(holding, now - self._hold_started_at, 0.0)
        air_s = np.where(
            holding,
            np.where(np.isnan(self._air_ended_at), now, self._air_ended_at)
            - self._hold_started_at,
            0.0,
        )
        self._hold_complete = holding & (air_s >= low)

        # The ball: before 10 s a problem restarts the hold, after it ends the attempt at 0.
        bad = holding & ~ball_ok
        restart = bad & ~self._hold_complete
        lost = bad & self._hold_complete & self.require_ball_on_tray_when_lowered
        phase[restart] = _IDLE
        self._ball_lost |= lost
        phase[lost] = _DONE
        holding = phase == _HOLDING

        # Past the prompt's upper bound, staying up always pays less.
        over = np.minimum(
            np.maximum(since_start - target, 0.0) / max(self.over_hold_decay_s, 1e-9),
            1.0,
        )
        hold_value = np.where(
            air_s < low,
            _HELD_REWARD * air_s / low,
            _HELD_REWARD - (_HELD_REWARD - _OVER_HOLD_FLOOR) * over,
        )

        gripped = np.asarray(metrics["both_handles_gripped"], dtype=bool)
        self._handles_seen_at[gripped] = now[gripped]
        handles_held = now - self._handles_seen_at <= _HANDLE_GRACE_S
        touchdown = (
            holding & ~lifted & np.asarray(metrics["tray_table_contact"], dtype=bool)
        )
        gentle = (
            handles_held
            if self.require_handles_at_touchdown
            else np.ones_like(touchdown)
        )
        # Hovering just above the table is still holding: touchdown must follow the air time promptly.
        in_window = (
            (air_s >= low)
            & (air_s <= high)
            & (since_start - air_s <= self.max_lowering_s)
        )
        # Placing before the minimum keeps only part of the credit, so it is clearly worse than finishing.
        kept = np.where(air_s < low, _EARLY_PLACEMENT_SHARE * hold_value, hold_value)
        settle = touchdown & gentle
        phase[settle] = _SETTLING
        self._settle_until[settle] = now[settle] + self.settle_s
        self._touch_value[settle] = kept[settle]
        self._touch_in_window[settle] = in_window[settle]
        self._touch_air_s[settle] = air_s[settle]
        rough = touchdown & ~gentle
        self._rough_placement |= rough
        self._final_reward[rough] = np.minimum(kept[rough], _OVER_HOLD_FLOOR)
        phase[rough] = _DONE

        settling = phase == _SETTLING
        if self.require_ball_on_tray_when_lowered:
            spilled = settling & ~ball_ok
            self._ball_lost |= spilled
            phase[spilled] = _DONE
            settling &= ~spilled
        settled = settling & (now >= self._settle_until)
        succeeded = settled & self._touch_in_window
        self._completed |= succeeded
        # Full reward inside the prompt's window, then down to 0.8 at max_hold_s.
        late_share = np.clip(
            (self._touch_air_s - target) / max(high - target, 1e-9), 0.0, 1.0
        )
        self._final_reward[succeeded] = (
            1.0 - (1.0 - _LATE_SUCCESS_REWARD) * late_share[succeeded]
        )
        outside = settled & ~self._touch_in_window
        self._early_placement |= outside & (self._touch_air_s < low)
        self._late_placement |= outside & (self._touch_air_s >= low)
        self._final_reward[outside] = self._touch_value[outside]
        phase[settled] = _DONE
        # Scored once per control step, a fast placement can go from lifted to touchdown in one sample.
        self._needs_handles = bool(np.any(phase == _HOLDING))

        # A lost ball leaves the final reward at 0; success sets it to 0.8-1.0.
        reward = np.where(
            phase == _DONE,
            self._final_reward,
            np.where(
                phase == _SETTLING,
                self._touch_value,
                np.where(phase == _HOLDING, hold_value, 0.0),
            ),
        )
        return TaskEvalResult(
            reward=reward,
            success=self._completed.copy(),
            metrics={
                **metrics,
                "current_balanced": current_balanced.copy(),
                "tray_lift_m": tray_lift,
                "lifted": lifted.copy(),
                "continuous_hold_s": air_s,
                "hold_duration_s": self._hold_duration_s.copy(),
                "hold_window_s": np.broadcast_to(
                    np.asarray(self.hold_window_s), (self._nworld, 2)
                ).copy(),
                "hold_complete": self._hold_complete.copy(),
                "ball_lost_after_hold": self._ball_lost.copy(),
                "handles_held": handles_held,
                "settling": phase == _SETTLING,
                "early_placement": self._early_placement.copy(),
                "late_placement": self._late_placement.copy(),
                "rough_placement": self._rough_placement.copy(),
                "ever_success": self._completed.copy(),
                **self._shape_metrics(),
            },
        )
