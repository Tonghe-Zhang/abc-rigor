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
_LIFT_REWARD = 0.1
_HOLD_REWARD = 0.8
_PLACEMENT_REWARD = 0.1
_OVERTIME_PENALTY_PER_S = 0.02
_OVERTIME_PENALTY_CAP = 0.2


def _touching(data: mujoco.MjData, first: set[int], second: set[int]) -> bool:
    """Require geometric contact, excluding separated margin-only contacts."""
    return any(
        contact.dist <= 0.0
        and (
            (int(contact.geom1) in first and int(contact.geom2) in second)
            or (int(contact.geom2) in first and int(contact.geom1) in second)
        )
        for contact in data.contact[: data.ncon]
    )


def format_window_prompt(window_s: tuple[float, float]) -> str:
    low, high = window_s
    return (
        f"keep the ball balanced on the tray for {low:g} to {high:g} seconds, "
        "then gently place the tray on the table"
    )


class BallTrayBalancingEvaluator:
    """Bank lift and longest uninterrupted hold credit, then score safe placement.

    A valid hold requires tray support of the ball, distinct hands on both
    handles, no direct ball grasp, and a tray lifted clear of the table. Lift
    earns 0.1 once; the longest valid hold earns up to 0.8. Interruptions reset
    the current hold without erasing either credit.

    Placement earns 0.1 and succeeds only after the immediately preceding hold
    meets its target, lowering takes at most ``max_lowering_s``, and the ball
    stays supported through gentle touchdown and ``settle_s`` of continuous
    table contact. An early safe placement keeps the banked credit and permits
    another hold without starting an overtime clock before qualification.

    Window mode deducts 0.02 per second beyond the upper bound, capped at 0.2
    and the earned credit. The placement clock starts when the first continuous
    hold reaches its target. The remaining window (upper minus lower bound) is
    free of deductions; an unqualified partial hold never starts this clock.
    Qualification never restarts after a dip, loss or relift. Tentative safe
    touchdown pauses the clock; interrupted settling resumes from the same
    anchor, including the failed settling interval. Finalized placement excludes
    its proven settling time.
    ``hold_window_s=None`` uses the sampled target with the same credit and
    placement rules, without overtime because its prompt has no upper bound.
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
        max_lowering_s: float = 5.0,
        require_handles_at_touchdown: bool = True,
        settle_s: float = 1.0,
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
        self.max_lowering_s = float(max_lowering_s)
        self.require_handles_at_touchdown = bool(require_handles_at_touchdown)
        self.settle_s = float(settle_s)
        if not np.isfinite(self.max_lowering_s) or self.max_lowering_s <= 0.0:
            raise ValueError("max_lowering_s must be positive and finite")
        if not np.isfinite(self.settle_s) or self.settle_s < 0.0:
            raise ValueError("settle_s must be nonnegative and finite")
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
        self._robot_geom_ids = self._gripper_geom_ids.copy()
        for name in ("left_arm", "right_arm"):
            root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if root >= 0:
                self._robot_geom_ids.update(subtree_geom_ids(model, int(root)))
        self._hand_geom_ids = [
            {
                geom
                for geom in self._gripper_geom_ids
                if (model.body(int(model.geom_bodyid[geom])).name or "").startswith(
                    side
                )
            }
            for side in ("left_", "right_")
        ]
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
        self._first_lift_at = np.full(self._nworld, np.nan)
        self._qualified_hold_at = np.full(self._nworld, np.nan)
        self._current_hold_s = np.zeros(self._nworld, dtype=np.float64)
        self._best_hold_s = np.zeros(self._nworld, dtype=np.float64)
        self._was_valid_hold = np.zeros(self._nworld, dtype=bool)
        self._lowering_started_at = np.full(self._nworld, np.nan)
        self._lowering_hold_s = np.zeros(self._nworld, dtype=np.float64)
        self._completed = np.zeros(self._nworld, dtype=bool)
        self._ball_lost = np.zeros(self._nworld, dtype=bool)
        self._rest_tray_z: np.ndarray | None = None
        self._handles_seen_at = np.full(self._nworld, np.nan)
        self._settle_started_at = np.full(self._nworld, np.nan)
        self._settle_qualified = np.zeros(self._nworld, dtype=bool)
        self._gentle_touch = np.zeros(self._nworld, dtype=bool)
        self._finalized = np.zeros(self._nworld, dtype=bool)
        self._final_touchdown_at = np.full(self._nworld, np.nan)
        self._early_placement = np.zeros(self._nworld, dtype=bool)
        self._late_placement = np.zeros(self._nworld, dtype=bool)
        self._rough_placement = np.zeros(self._nworld, dtype=bool)

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
            "ball_robot_contact": np.asarray(
                has_contact(data, self._ball_geom_ids, self._robot_geom_ids),
                dtype=bool,
            ),
            "tray_table_contact": np.asarray(
                _touching(data, self._tray_geom_ids, self._table_geom_ids), dtype=bool
            ),
            "ball_tray_contact": np.asarray(
                _touching(data, self._ball_geom_ids, self._tray_geom_ids), dtype=bool
            ),
            "ball_table_contact": np.asarray(
                _touching(data, self._ball_geom_ids, self._table_geom_ids), dtype=bool
            ),
            "both_handles_gripped": np.asarray(
                len(self._handle_geom_ids) == 2
                and any(
                    _touching(data, self._handle_geom_ids[left], self._hand_geom_ids[0])
                    and _touching(
                        data, self._handle_geom_ids[1 - left], self._hand_geom_ids[1]
                    )
                    for left in (0, 1)
                ),
                dtype=bool,
            ),
        }

    def _result_from_metrics(self, metrics: dict[str, np.ndarray]) -> TaskEvalResult:
        current_balanced = np.logical_and(
            np.asarray(metrics["inside_tray"], dtype=bool),
            np.logical_not(np.asarray(metrics["ball_dropped"], dtype=bool)),
        )
        current_balanced &= np.asarray(metrics["ball_tray_contact"], dtype=bool)
        current_balanced &= ~np.asarray(metrics["ball_table_contact"], dtype=bool)
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
        if np.any(~np.isfinite(now)) or np.any(now < self._last_time_s):
            raise ValueError(
                "Simulation time must be finite and cannot move backwards without reset"
            )
        self._last_time_s = now.copy()
        return self._earned_credit_result(
            metrics, current_balanced, lifted, tray_lift, now
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

    def _earned_credit_result(
        self,
        metrics: dict[str, np.ndarray],
        current_balanced: np.ndarray,
        lifted: np.ndarray,
        tray_lift: np.ndarray,
        now: np.ndarray,
    ) -> TaskEvalResult:
        """Keep earned credit separate from the current placement qualification."""
        active = ~self._finalized
        ball_ok = current_balanced.copy()
        if self.forbid_gripper_ball_contact:
            ball_ok &= ~np.asarray(metrics["ball_robot_contact"], dtype=bool)
        gripped = np.asarray(metrics["both_handles_gripped"], dtype=bool)
        self._handles_seen_at[gripped & active] = now[gripped & active]
        handles_held = now - self._handles_seen_at <= _HANDLE_GRACE_S
        table_contact = np.asarray(metrics["tray_table_contact"], dtype=bool)
        valid_hold = active & ball_ok & lifted & ~table_contact & handles_held

        first_lift = valid_hold & np.isnan(self._first_lift_at)
        self._first_lift_at[first_lift] = now[first_lift]
        start = valid_hold & ~self._was_valid_hold
        self._hold_started_at[start] = now[start]
        # A hold can transition into lowering, but not through grip or ball loss.
        self._gentle_touch[active & (lifted | ~ball_ok)] = False
        lowering = active & ~lifted & ball_ok & (handles_held | self._gentle_touch)
        begin_lowering = self._was_valid_hold & lowering
        self._lowering_started_at[begin_lowering] = now[begin_lowering]
        self._lowering_hold_s[begin_lowering] = self._current_hold_s[begin_lowering]
        interrupted = active & ~lowering
        self._lowering_started_at[interrupted] = np.nan
        self._lowering_hold_s[interrupted] = 0.0
        self._hold_started_at[active & ~valid_hold] = np.nan
        self._current_hold_s[active] = np.where(
            valid_hold, now - self._hold_started_at, 0.0
        )[active]
        self._best_hold_s = np.maximum(self._best_hold_s, self._current_hold_s)
        self._was_valid_hold = valid_hold
        current_complete = valid_hold & (self._current_hold_s >= self._hold_duration_s)
        first_complete = current_complete & np.isnan(self._qualified_hold_at)
        self._qualified_hold_at[first_complete] = (
            self._hold_started_at + self._hold_duration_s
        )[first_complete]
        lowering_qualified = (
            lowering
            & (self._lowering_hold_s >= self._hold_duration_s)
            & (now <= self._lowering_started_at + self.max_lowering_s)
        )
        self._ball_lost |= (
            active & ~ball_ok & (self._best_hold_s >= self._hold_duration_s)
        )

        placed = ~lifted & table_contact
        placement_ball_ok = (
            ball_ok if self.require_ball_on_tray_when_lowered else np.ones_like(ball_ok)
        )
        safe_support = placed & placement_ball_ok
        settling = active & ~np.isnan(self._settle_started_at)
        interrupted_settle = settling & ~safe_support
        self._settle_started_at[interrupted_settle] = np.nan
        self._settle_qualified[interrupted_settle] = False
        gentle = (
            handles_held | self._gentle_touch
            if self.require_handles_at_touchdown
            else np.ones_like(handles_held)
        )
        touchdown = (
            active
            & safe_support
            & np.isnan(self._settle_started_at)
            & ~np.isnan(self._first_lift_at)
        )
        self._rough_placement |= touchdown & ~gentle
        begin_settle = touchdown & gentle
        self._settle_started_at[begin_settle] = now[begin_settle]
        self._settle_qualified[begin_settle] = lowering_qualified[begin_settle]
        self._gentle_touch[begin_settle] = True
        settling = active & ~np.isnan(self._settle_started_at)
        settled = settling & (now >= self._settle_started_at + self.settle_s)
        self._completed[settled] = self._settle_qualified[settled]
        self._early_placement |= (
            settled
            & ~self._settle_qualified
            & (self._best_hold_s < self._hold_duration_s)
        )
        self._finalized |= self._completed
        self._final_touchdown_at[self._completed] = self._settle_started_at[
            self._completed
        ]

        # Failed settling resumes the original clock, including time spent on the table.
        clock_end = np.where(
            self._finalized,
            self._final_touchdown_at,
            np.where(settling, self._settle_started_at, now),
        )
        elapsed_after_qualification = np.where(
            np.isnan(self._qualified_hold_at),
            0.0,
            clock_end - self._qualified_hold_at,
        )
        overtime = (
            np.maximum(
                elapsed_after_qualification
                - (self.hold_window_s[1] - self.hold_window_s[0]),
                0.0,
            )
            if self.hold_window_s is not None
            else np.zeros(self._nworld)
        )
        self._late_placement |= settled & (overtime > 0.0)
        lift_credit = _LIFT_REWARD * ~np.isnan(self._first_lift_at)
        hold_credit = _HOLD_REWARD * np.minimum(
            self._best_hold_s / self._hold_duration_s, 1.0
        )
        placement_credit = _PLACEMENT_REWARD * self._completed
        earned = lift_credit + hold_credit + placement_credit
        # Saturation at zero is part of the declared normalized reward objective.
        penalty = np.minimum(
            earned,
            np.minimum(_OVERTIME_PENALTY_CAP, _OVERTIME_PENALTY_PER_S * overtime),
        )
        hold_complete = (
            current_complete
            | lowering_qualified
            | (settling & self._settle_qualified)
            | self._completed
        )
        return TaskEvalResult(
            reward=earned - penalty,
            success=self._completed.copy(),
            metrics={
                **metrics,
                "current_balanced": current_balanced.copy(),
                "tray_lift_m": tray_lift,
                "lifted": lifted.copy(),
                "valid_hold": valid_hold,
                "continuous_hold_s": self._current_hold_s.copy(),
                "best_continuous_hold_s": self._best_hold_s.copy(),
                "hold_duration_s": self._hold_duration_s.copy(),
                "hold_window_s": (
                    None
                    if self.hold_window_s is None
                    else np.broadcast_to(
                        np.asarray(self.hold_window_s), (self._nworld, 2)
                    ).copy()
                ),
                "hold_complete": hold_complete,
                "perturbations_allowed": valid_hold & ~current_complete,
                "ball_lost_after_hold": self._ball_lost.copy(),
                "handles_held": handles_held,
                "settling": settling & ~self._finalized,
                "placement_finalized": self._finalized.copy(),
                "early_placement": self._early_placement.copy(),
                "late_placement": self._late_placement.copy(),
                "rough_placement": self._rough_placement.copy(),
                "lift_credit": lift_credit,
                "hold_credit": hold_credit,
                "placement_credit": placement_credit,
                "overtime_s": overtime,
                "overtime_penalty": penalty,
                "ever_success": self._completed.copy(),
                **self._shape_metrics(),
            },
        )
