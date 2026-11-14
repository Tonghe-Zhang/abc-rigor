"""Runtime countdown logic for the ball-on-tray balancing task."""

from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np

from abc_sim.task_eval.ball_tray import format_hold_prompt
from abc_sim.task_eval.base import TaskEvalResult

_DEFAULT_DURATION_S = 12
_HOLD_TO_START_S = 0.0
_HANDLE_CONTACT_GRACE_S = 0.1
_PERTURBATION_INTERVAL_S = 0.5
_PERTURBATION_PULSE_DURATION_S = 0.1
_PERTURBATION_FORCE_RANGE_N = (0.024, 0.034)
_PERTURBATION_DIRECTION_JITTER_RANGE_RAD = (-0.30, 0.30)
_PERTURBATION_DIRECTION_REDIRECT_RANGE_RAD = (0.80, 1.80)
_PERTURBATION_DIRECTION_RUN_LENGTH_PULSES = (3, 5)
_BALL_BODY = "ball_tray_ball"
_TRAY_CENTER_SITE = "ball_tray_center_site"
_LEFT_HANDLE_GEOM = "ball_tray_left_handle_collision"
_RIGHT_HANDLE_GEOM = "ball_tray_right_handle_collision"
_GRIPPER_ROOT_BODIES = {
    "left": ("left_link_left_finger", "left_link_right_finger"),
    "right": ("right_link_left_finger", "right_link_right_finger"),
}


class BallTrayBalancingRuntime:
    """Keep the duration prompt fixed while displaying hold-and-lower progress."""

    def __init__(self) -> None:
        self._gripper_geom_ids: dict[str, set[int]] = {"left": set(), "right": set()}
        self._handle_geom_ids: dict[str, int] = {}
        self._ball_body_id: int | None = None
        self._tray_center_site_id: int | None = None
        self._duration_s = _DEFAULT_DURATION_S
        self._started_at_s: float | None = None
        self._deadline_s: float | None = None
        self._grip_detected_at_s: float | None = None
        self._last_both_handles_contact_at_s: float | None = None
        self._remaining_s: float | None = None
        self._display_prompt = format_hold_prompt(self._duration_s)
        self._state = "waiting_for_lift"
        self._contacted_handles: dict[str, tuple[str, ...]] = {"left": (), "right": ()}
        self._perturbation_rng = np.random.default_rng(0)
        self._next_perturbation_at_s: float | None = None
        self._active_perturbation_until_s: float | None = None
        self._perturbation_direction_rad = 0.0
        self._perturbation_run_pulses_remaining = 0
        self._active_local_force_n = np.zeros(3, dtype=np.float64)
        self._applied_world_force_n = np.zeros(3, dtype=np.float64)
        self._perturbation_count = 0
        self._last_perturbation_force_n = 0.0
        self._balance_result: TaskEvalResult | None = None

    def bind(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Refresh cached model indices after a model swap."""

        self._gripper_geom_ids = {
            side: self._geom_ids_for_body_subtrees(model, roots)
            for side, roots in _GRIPPER_ROOT_BODIES.items()
        }
        self._handle_geom_ids = {
            "left": self._optional_geom_id(model, _LEFT_HANDLE_GEOM),
            "right": self._optional_geom_id(model, _RIGHT_HANDLE_GEOM),
        }
        self._handle_geom_ids = {
            name: geom_id
            for name, geom_id in self._handle_geom_ids.items()
            if geom_id >= 0
        }
        ball_body_id = self._optional_body_id(model, _BALL_BODY)
        self._ball_body_id = ball_body_id if ball_body_id >= 0 else None
        tray_site_id = self._optional_site_id(model, _TRAY_CENTER_SITE)
        self._tray_center_site_id = tray_site_id if tray_site_id >= 0 else None
        self._applied_world_force_n[:] = 0.0
        self._balance_result = None

    def after_reset(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        randomization: Any = None,
    ) -> None:
        """Reset countdown and perturbations before receiving the initial score."""

        self._balance_result = None
        self._duration_s = _DEFAULT_DURATION_S
        self._started_at_s = None
        self._deadline_s = None
        self._grip_detected_at_s = None
        self._last_both_handles_contact_at_s = None
        self._remaining_s = float(self._duration_s)
        self._display_prompt = format_hold_prompt(self._duration_s)
        self._state = "waiting_for_lift"
        self._contacted_handles = {"left": (), "right": ()}
        seed = int(getattr(randomization, "seed", 0) or 0)
        self._perturbation_rng = np.random.default_rng(seed)
        self._next_perturbation_at_s = None
        self._active_perturbation_until_s = None
        self._perturbation_direction_rad = float(
            self._perturbation_rng.uniform(0.0, 2.0 * np.pi)
        )
        self._perturbation_run_pulses_remaining = int(
            self._perturbation_rng.integers(
                _PERTURBATION_DIRECTION_RUN_LENGTH_PULSES[0],
                _PERTURBATION_DIRECTION_RUN_LENGTH_PULSES[1] + 1,
            )
        )
        self._active_local_force_n[:] = 0.0
        self._applied_world_force_n[:] = 0.0
        if self._ball_body_id is not None:
            data.xfrc_applied[self._ball_body_id, :3] = 0.0
        self._perturbation_count = 0
        self._last_perturbation_force_n = 0.0

    def on_task_evaluation(self, result: TaskEvalResult) -> None:
        """Consume the environment's authoritative hold-and-lower score."""
        self._balance_result = result
        self._duration_s = float(result.metrics["hold_duration_s"][0])
        self._update_display()

    def observation_prompt(self) -> str:
        """Use the episode's required duration throughout the policy rollout."""
        return format_hold_prompt(self._duration_s)

    def before_step(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Apply an active perturbation as an additive external force."""

        self._apply_active_perturbation_force(data)

    def after_step(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Update perturbations after receiving this physics step's task score."""

        now_s = float(data.time)
        assert self._balance_result is not None
        self._contacted_handles = self._detect_gripper_handle_contacts(model, data)
        both_posts_gripped = self._both_posts_gripped(self._contacted_handles)
        if both_posts_gripped:
            self._last_both_handles_contact_at_s = now_s

        if self._started_at_s is None:
            if both_posts_gripped:
                if self._grip_detected_at_s is None:
                    self._grip_detected_at_s = now_s
                hold_s = now_s - self._grip_detected_at_s
                if hold_s >= _HOLD_TO_START_S:
                    self._started_at_s = now_s
                    self._deadline_s = now_s + float(self._duration_s)
                    self._next_perturbation_at_s = now_s + _PERTURBATION_INTERVAL_S
            else:
                self._grip_detected_at_s = None

        handles_held = (
            self._last_both_handles_contact_at_s is not None
            and now_s - self._last_both_handles_contact_at_s <= _HANDLE_CONTACT_GRACE_S
        )
        self._maybe_schedule_perturbation(now_s, handles_held)

    def debug_state(self) -> dict[str, Any]:
        """Return countdown and contact state for the VR streamer/debug UI."""

        return {
            "state": self._state,
            "display_prompt": self._display_prompt,
            "duration_s": float(self._duration_s),
            "remaining_s": None
            if self._remaining_s is None
            else float(self._remaining_s),
            "started": self._started_at_s is not None,
            "complete": self._state == "complete",
            "both_posts_gripped": self._both_posts_gripped(self._contacted_handles),
            "left_contacted_handles": self._contacted_handles["left"],
            "right_contacted_handles": self._contacted_handles["right"],
            "perturbation_interval_s": _PERTURBATION_INTERVAL_S,
            "perturbation_pulse_duration_s": _PERTURBATION_PULSE_DURATION_S,
            "perturbation_count": self._perturbation_count,
            "last_perturbation_force_n": self._last_perturbation_force_n,
            "perturbation_direction_rad": self._perturbation_direction_rad,
        }

    @staticmethod
    def _optional_geom_id(model: mujoco.MjModel, name: str) -> int:
        return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name))

    @staticmethod
    def _optional_body_id(model: mujoco.MjModel, name: str) -> int:
        return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))

    @staticmethod
    def _optional_site_id(model: mujoco.MjModel, name: str) -> int:
        return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name))

    def _geom_ids_for_body_subtrees(
        self,
        model: mujoco.MjModel,
        root_body_names: tuple[str, ...],
    ) -> set[int]:
        root_ids = {
            body_id
            for body_name in root_body_names
            if (body_id := self._optional_body_id(model, body_name)) >= 0
        }
        if not root_ids:
            return set()

        body_ids: set[int] = set()
        for body_id in range(model.nbody):
            current = body_id
            while current > 0:
                if current in root_ids:
                    body_ids.add(body_id)
                    break
                current = int(model.body_parentid[current])

        return {
            geom_id
            for geom_id in range(model.ngeom)
            if int(model.geom_bodyid[geom_id]) in body_ids
        }

    def _detect_gripper_handle_contacts(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> dict[str, tuple[str, ...]]:
        contacts: dict[str, set[str]] = {"left": set(), "right": set()}
        handle_by_geom_id = {
            geom_id: handle_name
            for handle_name, geom_id in self._handle_geom_ids.items()
        }
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            for side, gripper_geoms in self._gripper_geom_ids.items():
                if geom1 in gripper_geoms and geom2 in handle_by_geom_id:
                    contacts[side].add(handle_by_geom_id[geom2])
                elif geom2 in gripper_geoms and geom1 in handle_by_geom_id:
                    contacts[side].add(handle_by_geom_id[geom1])
        return {
            side: tuple(sorted(handle_names)) for side, handle_names in contacts.items()
        }

    def _maybe_schedule_perturbation(
        self,
        now_s: float,
        handles_held: bool,
    ) -> None:
        if (
            self._started_at_s is None
            or self._deadline_s is None
            or self._ball_body_id is None
            or self._tray_center_site_id is None
        ):
            return
        if self._balance_result is not None and bool(
            self._balance_result.metrics["hold_complete"][0]
        ):
            self._active_perturbation_until_s = None
            return
        if not handles_held:
            self._next_perturbation_at_s = now_s + _PERTURBATION_INTERVAL_S
            self._active_perturbation_until_s = None
            return
        if self._next_perturbation_at_s is None:
            self._next_perturbation_at_s = now_s + _PERTURBATION_INTERVAL_S
            return
        if now_s + 1e-9 < self._next_perturbation_at_s:
            return

        if self._perturbation_count > 0:
            if self._perturbation_run_pulses_remaining <= 0:
                redirect = float(
                    self._perturbation_rng.uniform(
                        *_PERTURBATION_DIRECTION_REDIRECT_RANGE_RAD,
                    )
                )
                redirect *= -1.0 if self._perturbation_rng.random() < 0.5 else 1.0
                self._perturbation_direction_rad += redirect
                self._perturbation_run_pulses_remaining = int(
                    self._perturbation_rng.integers(
                        _PERTURBATION_DIRECTION_RUN_LENGTH_PULSES[0],
                        _PERTURBATION_DIRECTION_RUN_LENGTH_PULSES[1] + 1,
                    )
                )
            else:
                self._perturbation_direction_rad += float(
                    self._perturbation_rng.uniform(
                        *_PERTURBATION_DIRECTION_JITTER_RANGE_RAD,
                    )
                )
        force_n = float(self._perturbation_rng.uniform(*_PERTURBATION_FORCE_RANGE_N))
        self._active_local_force_n[:] = force_n * np.array(
            [
                np.cos(self._perturbation_direction_rad),
                np.sin(self._perturbation_direction_rad),
                0.0,
            ],
            dtype=np.float64,
        )
        self._active_perturbation_until_s = now_s + _PERTURBATION_PULSE_DURATION_S
        self._perturbation_count += 1
        self._perturbation_run_pulses_remaining -= 1
        self._last_perturbation_force_n = force_n
        self._next_perturbation_at_s = now_s + _PERTURBATION_INTERVAL_S

    def _apply_active_perturbation_force(self, data: mujoco.MjData) -> None:
        if self._ball_body_id is None:
            return

        data.xfrc_applied[self._ball_body_id, :3] -= self._applied_world_force_n
        self._applied_world_force_n[:] = 0.0
        if (
            self._active_perturbation_until_s is None
            or float(data.time) >= self._active_perturbation_until_s
            or self._tray_center_site_id is None
        ):
            return

        rotation_world_from_tray = np.asarray(
            data.site_xmat[self._tray_center_site_id],
            dtype=np.float64,
        ).reshape(3, 3)
        self._applied_world_force_n[:] = (
            rotation_world_from_tray @ self._active_local_force_n
        )
        data.xfrc_applied[self._ball_body_id, :3] += self._applied_world_force_n

    @staticmethod
    def _both_posts_gripped(contacted_handles: dict[str, tuple[str, ...]]) -> bool:
        left_handles = contacted_handles.get("left", ())
        right_handles = contacted_handles.get("right", ())
        return any(
            left_handle != right_handle
            for left_handle in left_handles
            for right_handle in right_handles
        )

    def _update_display(self) -> None:
        if self._balance_result is None:
            return
        metrics = self._balance_result.metrics
        self._remaining_s = max(
            0.0, self._duration_s - float(metrics["continuous_hold_s"][0])
        )
        if self._balance_result.scalar_success():
            self._state = "complete"
            self._display_prompt = "Task complete"
        elif bool(metrics["hold_complete"][0]):
            self._state = "lowering"
            self._display_prompt = "Lower the tray onto the table"
        elif bool(metrics["lifted"][0]) and bool(metrics["current_balanced"][0]):
            self._state = "counting_down"
            self._display_prompt = f"Balance: {math.ceil(self._remaining_s)}s"
        else:
            self._state = "waiting_for_lift"
            self._display_prompt = self.observation_prompt()
