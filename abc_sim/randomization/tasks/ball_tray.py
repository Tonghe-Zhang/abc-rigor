from __future__ import annotations

from typing import Any, ClassVar

import mujoco
import numpy as np

from abc_sim.task_eval.ball_tray import format_hold_prompt, sample_hold_duration_s

from ..core import PerturbRange, RandomizationState, SceneRandomizer

_BALL_TRAY_BALL_JOINT = "ball_tray_ball_joint"
_BALL_TRAY_TRAY_JOINT = "ball_tray_joint"
_BALL_TRAY_BALL_BODY = "ball_tray_ball"
_BALL_TRAY_BODY = "ball_tray"
_BALL_TRAY_CENTER_SITE = "ball_tray_center_site"
_BALL_TRAY_BALL_RADIUS_M = 0.022
_BALL_TRAY_USABLE_HALF_EXTENTS_M = (0.126, 0.1215)
_BALL_TRAY_SCALE_RANGE = (0.95, 1.05)
_BALL_TRAY_POSITION_X_RANGE_M = (-0.025, 0.025)
_BALL_TRAY_POSITION_Y_RANGE_M = (-0.025, 0.025)
_BALL_TRAY_FLOOR_HALF_HEIGHT_M = 0.004
_BALL_TRAY_SPAWN_X_RANGE_M = (0.0, 0.0)
_BALL_TRAY_SPAWN_Y_RANGE_M = (0.0, 0.0)
_BALL_TRAY_LINEAR_VELOCITY_RANGE_M_S = (0.0, 0.0)
_BALL_TRAY_ANGULAR_VELOCITY_RANGE_RAD_S = (0.0, 0.0)
_BALL_TRAY_COUNTDOWN_DURATION_RANGE_S = (5.0, 15.0)


class BallTrayBalancingRandomizer(SceneRandomizer):
    """Fast reset sampler for ball-on-tray balancing."""

    perturbations: ClassVar[list[PerturbRange]] = []

    def __init__(self) -> None:
        super().__init__()
        self._scale_model: Any | None = None
        self._scale_baseline: dict[str, Any] | None = None

    def randomize(
        self,
        model: Any,
        data: Any,
        seed: int | None = None,
        request: Any | None = None,
    ) -> RandomizationState:
        rng = np.random.default_rng(seed)
        tray_scale = float(rng.uniform(*_BALL_TRAY_SCALE_RANGE))
        ball_scale = float(rng.uniform(*_BALL_TRAY_SCALE_RANGE))
        self._apply_runtime_scales(model, data, tray_scale, ball_scale)
        site_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            _BALL_TRAY_CENTER_SITE,
        )
        if site_id < 0:
            raise ValueError(f"Site {_BALL_TRAY_CENTER_SITE!r} not found in model")

        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            _BALL_TRAY_BALL_JOINT,
        )
        if joint_id < 0:
            raise ValueError(f"Joint {_BALL_TRAY_BALL_JOINT!r} not found in model")
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"Joint {_BALL_TRAY_BALL_JOINT!r} must be a free joint")
        tray_joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            _BALL_TRAY_TRAY_JOINT,
        )
        if tray_joint_id < 0:
            raise ValueError(f"Joint {_BALL_TRAY_TRAY_JOINT!r} not found in model")
        if model.jnt_type[tray_joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"Joint {_BALL_TRAY_TRAY_JOINT!r} must be a free joint")

        tray_qpos_addr = int(model.jnt_qposadr[tray_joint_id])
        tray_pos = np.asarray(
            data.qpos[tray_qpos_addr : tray_qpos_addr + 3],
            dtype=np.float64,
        ).copy()
        tray_pos[0] += float(rng.uniform(*_BALL_TRAY_POSITION_X_RANGE_M))
        tray_pos[1] += float(rng.uniform(*_BALL_TRAY_POSITION_Y_RANGE_M))
        tray_pos[2] += _BALL_TRAY_FLOOR_HALF_HEIGHT_M * (tray_scale - 1.0)
        tray_quat = np.asarray(
            data.qpos[tray_qpos_addr + 3 : tray_qpos_addr + 7],
            dtype=np.float64,
        ).copy()
        rot_world_from_tray = np.empty((3, 3), dtype=np.float64)
        mujoco.mju_quat2Mat(rot_world_from_tray.ravel(), tray_quat)
        tray_center = tray_pos + rot_world_from_tray @ np.asarray(
            model.site_pos[site_id],
            dtype=np.float64,
        )
        local_x = 0.0
        local_y = 0.0
        ball_radius_m = _BALL_TRAY_BALL_RADIUS_M * ball_scale
        local_pos = np.array(
            [local_x, local_y, ball_radius_m + 0.001],
            dtype=np.float64,
        )
        ball_pos = tray_center + rot_world_from_tray @ local_pos
        ball_quat = [1.0, 0.0, 0.0, 0.0]
        linear_velocity = np.zeros(3, dtype=np.float64)
        angular_velocity = np.zeros(3, dtype=np.float64)
        balance_duration_s = sample_hold_duration_s(rng)
        states = {
            _BALL_TRAY_TRAY_JOINT: {
                "pos": tray_pos.tolist(),
                "quat": tray_quat.tolist(),
            },
            _BALL_TRAY_BALL_JOINT: {
                "pos": ball_pos.tolist(),
                "quat": ball_quat,
            },
        }
        self._apply_states(model, data, states)
        self._apply_free_joint_velocity(
            model,
            data,
            _BALL_TRAY_TRAY_JOINT,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
        )
        self._apply_free_joint_velocity(
            model,
            data,
            _BALL_TRAY_BALL_JOINT,
            linear_velocity,
            angular_velocity,
        )
        mujoco.mj_forward(model, data)

        return RandomizationState(
            seed=seed or 0,
            object_states=states,
            scale_states={
                _BALL_TRAY_TRAY_JOINT: tray_scale,
                _BALL_TRAY_BALL_JOINT: ball_scale,
            },
            metadata={
                "prompt": format_hold_prompt(balance_duration_s),
                "prompt_type": "ball_tray_balancing",
                "ball_body": _BALL_TRAY_BALL_BODY,
                "ball_joint": _BALL_TRAY_BALL_JOINT,
                "tray_body": _BALL_TRAY_BODY,
                "tray_joint": _BALL_TRAY_TRAY_JOINT,
                "tray_center_site": _BALL_TRAY_CENTER_SITE,
                "ball_radius_m": ball_radius_m,
                "usable_half_extents_m": [
                    extent * tray_scale for extent in _BALL_TRAY_USABLE_HALF_EXTENTS_M
                ],
                "tray_scale": tray_scale,
                "ball_scale": ball_scale,
                "scale_range": list(_BALL_TRAY_SCALE_RANGE),
                "tray_position": tray_pos.tolist(),
                "tray_position_x_range_m": list(_BALL_TRAY_POSITION_X_RANGE_M),
                "tray_position_y_range_m": list(_BALL_TRAY_POSITION_Y_RANGE_M),
                "spawn_local_pos": local_pos.tolist(),
                "ball_linear_velocity": linear_velocity.tolist(),
                "ball_angular_velocity": angular_velocity.tolist(),
                "balance_duration_s": balance_duration_s,
                "hold_duration_s": balance_duration_s,
                "hold_duration_range_s": list(_BALL_TRAY_COUNTDOWN_DURATION_RANGE_S),
                "balance_duration_range_s": list(_BALL_TRAY_COUNTDOWN_DURATION_RANGE_S),
            },
        )

    def apply(self, model: Any, data: Any, state: RandomizationState) -> None:
        self._apply_runtime_scales(
            model,
            data,
            float(state.scale_states.get(_BALL_TRAY_TRAY_JOINT, 1.0)),
            float(state.scale_states.get(_BALL_TRAY_BALL_JOINT, 1.0)),
        )
        self._apply_states(model, data, state.object_states)
        linear_velocity = np.asarray(
            state.metadata.get("ball_linear_velocity", [0.0, 0.0, 0.0]),
            dtype=np.float64,
        )
        angular_velocity = np.asarray(
            state.metadata.get("ball_angular_velocity", [0.0, 0.0, 0.0]),
            dtype=np.float64,
        )
        self._apply_free_joint_velocity(
            model,
            data,
            _BALL_TRAY_TRAY_JOINT,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
        )
        self._apply_free_joint_velocity(
            model,
            data,
            _BALL_TRAY_BALL_JOINT,
            linear_velocity,
            angular_velocity,
        )
        mujoco.mj_forward(model, data)

    def _apply_runtime_scales(
        self,
        model: Any,
        data: Any,
        tray_scale: float,
        ball_scale: float,
    ) -> None:
        baseline = self._scale_baseline_for(model)
        self._scale_body_from_baseline(model, baseline["tray"], tray_scale)
        self._scale_body_from_baseline(model, baseline["ball"], ball_scale)
        mujoco.mj_setConst(model, data)

    def _scale_baseline_for(self, model: Any) -> dict[str, Any]:
        if self._scale_model is model and self._scale_baseline is not None:
            return self._scale_baseline

        self._scale_model = model
        self._scale_baseline = {
            "tray": self._capture_body_scale_baseline(model, _BALL_TRAY_BODY),
            "ball": self._capture_body_scale_baseline(model, _BALL_TRAY_BALL_BODY),
        }
        return self._scale_baseline

    @staticmethod
    def _capture_body_scale_baseline(model: Any, body_name: str) -> dict[str, Any]:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"Body {body_name!r} not found in model")
        geom_ids = np.flatnonzero(np.asarray(model.geom_bodyid) == body_id)
        site_ids = np.flatnonzero(np.asarray(model.site_bodyid) == body_id)
        return {
            "body_id": body_id,
            "geom_ids": geom_ids,
            "site_ids": site_ids,
            "geom_pos": np.asarray(model.geom_pos[geom_ids], dtype=np.float64).copy(),
            "geom_size": np.asarray(model.geom_size[geom_ids], dtype=np.float64).copy(),
            "geom_rbound": np.asarray(
                model.geom_rbound[geom_ids], dtype=np.float64
            ).copy(),
            "geom_aabb": np.asarray(model.geom_aabb[geom_ids], dtype=np.float64).copy(),
            "site_pos": np.asarray(model.site_pos[site_ids], dtype=np.float64).copy(),
            "site_size": np.asarray(model.site_size[site_ids], dtype=np.float64).copy(),
            "body_ipos": np.asarray(model.body_ipos[body_id], dtype=np.float64).copy(),
            "body_mass": float(model.body_mass[body_id]),
            "body_inertia": np.asarray(
                model.body_inertia[body_id],
                dtype=np.float64,
            ).copy(),
        }

    @staticmethod
    def _scale_body_from_baseline(
        model: Any,
        baseline: dict[str, Any],
        scale: float,
    ) -> None:
        body_id = int(baseline["body_id"])
        geom_ids = baseline["geom_ids"]
        site_ids = baseline["site_ids"]
        model.geom_pos[geom_ids] = baseline["geom_pos"] * scale
        model.geom_size[geom_ids] = baseline["geom_size"] * scale
        model.geom_rbound[geom_ids] = baseline["geom_rbound"] * scale
        model.geom_aabb[geom_ids] = baseline["geom_aabb"] * scale
        model.site_pos[site_ids] = baseline["site_pos"] * scale
        model.site_size[site_ids] = baseline["site_size"] * scale
        model.body_ipos[body_id] = baseline["body_ipos"] * scale
        model.body_mass[body_id] = baseline["body_mass"] * scale**3
        model.body_inertia[body_id] = baseline["body_inertia"] * scale**5

    @staticmethod
    def _apply_free_joint_velocity(
        model: Any,
        data: Any,
        joint_name: str,
        linear_velocity: np.ndarray,
        angular_velocity: np.ndarray,
    ) -> None:
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )
        if joint_id < 0:
            return
        dof_adr = int(model.jnt_dofadr[joint_id])
        data.qvel[dof_adr : dof_adr + 3] = linear_velocity
        data.qvel[dof_adr + 3 : dof_adr + 6] = angular_velocity
