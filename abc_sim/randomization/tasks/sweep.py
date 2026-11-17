from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from ..core import (
    PerturbRange,
    RandomizationState,
    ScalePerturbRange,
    SceneRandomizer,
    _body_subtree_xy_keepout_discs,
    _quat_mul,
    _sample_orientation_delta,
    _yaw_from_quat,
)
from ..requests import SweepResetRequest, _SweepPlacementFailure

class SweepRandomizer(SceneRandomizer):
    max_tries = 400
    min_clearance_m = 0.015
    min_trash_count = 2
    default_trash_count = 3
    max_trash_count = 4
    _all_trash_joints = [f"trash_{i}_jnt" for i in range(1, 8)]
    _tool_joints = ("brush_jnt", "bin_joint", "dustpan_jnt")
    _trash_cluster_shift_x = (-0.03, 0.07)
    _trash_cluster_shift_y = (-0.18, 0.18)
    _cluster_shift_sample_tries = 24
    _tool_sample_tries = 32
    _trash_sample_tries = 48
    _trash_clearance_m = 0.04
    _trash_tool_clearance_m = 0.065
    _brush_trash_clearance_m = 0.09
    _tool_tool_clearance_m = 0.08
    _dustpan_keepout_margin_m = 0.006
    _brush_keepout_margin_m = 0.01

    def __init__(self) -> None:
        super().__init__()
        self._tool_keepout_discs: dict[str, list[tuple[np.ndarray, float]]] = {}
        self._tool_nominal_yaws: dict[str, float] = {}
        self._trash_joints: list[str] = []
        self._set_active_trash_count(self.default_trash_count)

    def prepare_env(self) -> None:
        self._set_active_trash_count(self.default_trash_count)

    def randomize(
        self,
        model: Any,
        data: Any,
        seed: int | None = None,
        request: Any | None = None,
    ) -> RandomizationState:
        reset_request = SweepResetRequest.from_value(request)
        rng = np.random.default_rng(seed)
        trash_count = self._resolve_trash_count(rng, reset_request)
        self._set_active_trash_count(trash_count)
        should_randomize_scales = (
            True if reset_request.randomize_scales is None else bool(reset_request.randomize_scales)
        )
        scale_states = self._sample_scale_states(rng) if should_randomize_scales else {}
        self._current_scale_states = dict(scale_states)
        if scale_states:
            self._reload_scene_for_scale_states(scale_states)
            if self._env_ref is not None:
                model = self._env_ref.model
                data = self._env_ref.data

        self._before_sampling(model, data)
        nominals = self._read_nominals(model, data)
        for _ in range(self.max_tries):
            try:
                states = self._sample_once(nominals, rng)
            except _SweepPlacementFailure:
                continue
            if not self._bounds_ok(states):
                continue
            if not self._pairwise_ok(states):
                continue
            self._apply_states(model, data, states)
            mujoco.mj_forward(model, data)
            if not self._contacts_ok(model, data):
                continue
            return self._finalize_randomization_state(
                model,
                data,
                seed=seed or 0,
                object_states=states,
                scale_states=scale_states,
                trash_count=trash_count,
            )

        self._raise_sampling_failure(seed)

    def _restore_scene(self, model: Any, data: Any, state: RandomizationState) -> RandomizationState:
        raw_trash_count = state.metadata.get("trash_count")
        if raw_trash_count is None:
            trash_count = sum(
                1
                for joint_name in self._all_trash_joints
                if joint_name in state.object_states
                and float(state.object_states[joint_name]["pos"][0]) >= 0.0
                and float(state.object_states[joint_name]["pos"][2]) >= 0.0
            )
            trash_count = min(max(trash_count, self.min_trash_count), self.max_trash_count)
        else:
            trash_count = int(raw_trash_count)
        self._set_active_trash_count(trash_count)

        object_states = dict(state.object_states)
        # Normalize saved below-plane parking positions when replaying old resets.
        object_states.update(self._inactive_trash_states())
        full_state = RandomizationState(
            seed=state.seed,
            object_states=object_states,
            scale_states=dict(state.scale_states),
            metadata={
                **state.metadata,
                "trash_count": trash_count,
                "trash_joints": list(self._trash_joints),
            },
        )
        return super()._restore_scene(model, data, full_state)

    def _resolve_trash_count(
        self,
        rng: np.random.Generator,
        request: SweepResetRequest,
    ) -> int:
        if request.trash_count is None:
            trash_count = int(rng.integers(self.min_trash_count, self.max_trash_count + 1))
        else:
            trash_count = int(request.trash_count)
        if not self.min_trash_count <= trash_count <= self.max_trash_count:
            raise ValueError(
                f"Sweep trash_count must be in [{self.min_trash_count}, {self.max_trash_count}], "
                f"got {trash_count}"
            )
        return trash_count

    def _set_active_trash_count(self, trash_count: int) -> None:
        if not self.min_trash_count <= trash_count <= self.max_trash_count:
            raise ValueError(
                f"Sweep trash_count must be in [{self.min_trash_count}, {self.max_trash_count}], "
                f"got {trash_count}"
            )

        self._trash_joints = list(self._all_trash_joints[:trash_count])
        self.perturbations = [
            PerturbRange("brush_jnt", delta_x=(-0.08, 0.10), delta_y=(-0.14, 0.12), delta_yaw=(-0.6, 0.6)),
            PerturbRange("bin_joint", delta_x=(-0.08, 0.05), delta_y=(-0.20, 0.20), delta_yaw=(-0.5, 0.5)),
            PerturbRange("dustpan_jnt", delta_x=(-0.08, 0.04), delta_y=(-0.12, 0.16), delta_yaw=(-0.6, 0.6)),
            *[
                PerturbRange(
                    joint_name,
                    delta_x=(-0.015, 0.015),
                    delta_y=(-0.015, 0.015),
                    delta_roll=(-np.pi, np.pi),
                    delta_pitch=(-np.pi, np.pi),
                    delta_yaw=(-np.pi, np.pi),
                )
                for joint_name in self._trash_joints
            ],
        ]
        self.size_perturbations = [
            ScalePerturbRange("brush_jnt"),
            ScalePerturbRange("bin_joint"),
            ScalePerturbRange("dustpan_jnt"),
            *[
                ScalePerturbRange(joint_name, scale_factor=(0.85, 1.15))
                for joint_name in self._trash_joints
            ],
        ]

    def _inactive_trash_joints(self) -> list[str]:
        active = set(self._trash_joints)
        return [joint_name for joint_name in self._all_trash_joints if joint_name not in active]

    def _inactive_trash_states(self) -> dict[str, dict[str, list[float]]]:
        states: dict[str, dict[str, list[float]]] = {}
        for index, joint_name in enumerate(self._inactive_trash_joints()):
            states[joint_name] = {
                # The table collision plane extends beyond its visible edges.
                # Inactive balls settle on it outside the workspace.
                "pos": [-1.5 - 0.1 * index, 0.0, 0.80],
                "quat": [1.0, 0.0, 0.0, 0.0],
            }
        return states

    def _finalize_randomization_state(
        self,
        model: Any,
        data: Any,
        *,
        seed: int,
        object_states: dict[str, dict[str, list[float]]],
        scale_states: dict[str, float],
        trash_count: int,
    ) -> RandomizationState:
        state = RandomizationState(
            seed=seed,
            object_states=dict(object_states),
            scale_states=dict(scale_states),
            metadata={
                "trash_count": trash_count,
                "trash_joints": list(self._trash_joints),
            },
        )
        inactive_states = self._inactive_trash_states()
        if inactive_states:
            state.object_states.update(inactive_states)
            self._apply_states(model, data, inactive_states)
            mujoco.mj_forward(model, data)
        return state

    def _before_sampling(self, model: Any, data: Any) -> None:
        self._refresh_collision_metadata(model, data)

    def _refresh_collision_metadata(self, model: Any, data: Any) -> None:
        tool_keepout_discs: dict[str, list[tuple[np.ndarray, float]]] = {}
        tool_nominal_yaws: dict[str, float] = {}
        for joint_name in self._tool_joints:
            jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if jnt_id < 0:
                continue
            body_id = int(model.jnt_bodyid[jnt_id])
            tool_keepout_discs[joint_name] = _body_subtree_xy_keepout_discs(model, data, body_id)
            tool_nominal_yaws[joint_name] = _yaw_from_quat(np.asarray(data.xquat[body_id], dtype=np.float64))
        self._tool_keepout_discs = tool_keepout_discs
        self._tool_nominal_yaws = tool_nominal_yaws

    def _clearance_for_pair(self, name_a: str, name_b: str) -> float:
        scale_a = float(self._current_scale_states.get(name_a, 1.0))
        scale_b = float(self._current_scale_states.get(name_b, 1.0))
        scale = max(scale_a, scale_b)
        if name_a in self._trash_joints and name_b in self._trash_joints:
            return self._trash_clearance_m * scale
        if name_a in self._tool_joints and name_b in self._tool_joints:
            return self._tool_tool_clearance_m * scale
        if "brush_jnt" in (name_a, name_b):
            return self._brush_trash_clearance_m * scale
        return self._trash_tool_clearance_m * scale

    @staticmethod
    def _rotate_xy(vec: np.ndarray, yaw: float) -> np.ndarray:
        cos_yaw = float(np.cos(yaw))
        sin_yaw = float(np.sin(yaw))
        return np.array(
            [
                cos_yaw * vec[0] - sin_yaw * vec[1],
                sin_yaw * vec[0] + cos_yaw * vec[1],
            ],
            dtype=np.float64,
        )

    def _tool_keepout_ok(
        self,
        candidate_xy: np.ndarray,
        *,
        states: dict[str, dict[str, list[float]]],
    ) -> bool:
        for tool_name in ("brush_jnt", "dustpan_jnt"):
            tool_state = states.get(tool_name)
            nominal_yaw = self._tool_nominal_yaws.get(tool_name)
            if tool_state is None or nominal_yaw is None:
                continue

            candidate_tool_xy = np.asarray(tool_state["pos"][:2], dtype=np.float64)
            yaw_delta = _yaw_from_quat(np.asarray(tool_state["quat"], dtype=np.float64)) - nominal_yaw
            margin = (
                self._brush_keepout_margin_m
                if tool_name == "brush_jnt"
                else self._dustpan_keepout_margin_m
            )
            for offset_xy, radius in self._tool_keepout_discs.get(tool_name, []):
                geom_xy = candidate_tool_xy + self._rotate_xy(offset_xy, yaw_delta)
                if np.linalg.norm(candidate_xy - geom_xy) < radius + margin:
                    return False
        return True

    def _sample_cluster_shift(
        self,
        *,
        states: dict[str, dict[str, list[float]]],
        nominals: dict[str, tuple[np.ndarray, np.ndarray]],
        rng: np.random.Generator,
    ) -> tuple[float, float]:
        x_min, x_max, y_min, y_max = self.table_bounds
        last_shift = (0.0, 0.0)
        for _ in range(self._cluster_shift_sample_tries):
            cluster_dx = float(rng.uniform(*self._trash_cluster_shift_x))
            cluster_dy = float(rng.uniform(*self._trash_cluster_shift_y))
            last_shift = (cluster_dx, cluster_dy)
            placed_xy = {
                name: np.asarray(state["pos"][:2], dtype=np.float64)
                for name, state in states.items()
            }
            ok = True
            for joint_name in self._trash_joints:
                nominal = nominals.get(joint_name)
                if nominal is None:
                    continue
                candidate_xy = nominal[0][:2] + np.array([cluster_dx, cluster_dy], dtype=np.float64)
                if not (x_min <= candidate_xy[0] <= x_max and y_min <= candidate_xy[1] <= y_max):
                    ok = False
                    break
                if not self._tool_keepout_ok(candidate_xy, states=states):
                    ok = False
                    break
                if any(
                    np.linalg.norm(candidate_xy - other_xy) < self._clearance_for_pair(joint_name, other_name)
                    for other_name, other_xy in placed_xy.items()
                ):
                    ok = False
                    break
                placed_xy[joint_name] = candidate_xy
            if ok:
                return cluster_dx, cluster_dy
        return last_shift

    def _sample_once(
        self,
        nominals: dict[str, tuple[np.ndarray, np.ndarray]],
        rng: np.random.Generator,
    ) -> dict[str, dict[str, list[float]]]:
        x_min, x_max, y_min, y_max = self.table_bounds
        states: dict[str, dict[str, list[float]]] = {}

        def sample_from_range(
            perturbation: PerturbRange,
            *,
            extra_xy: tuple[float, float] | None = None,
        ) -> dict[str, list[float]]:
            nom_pos, nom_quat = nominals[perturbation.joint_name]
            eff_dx = (
                max(perturbation.delta_x[0], x_min - nom_pos[0]),
                min(perturbation.delta_x[1], x_max - nom_pos[0]),
            )
            eff_dy = (
                max(perturbation.delta_y[0], y_min - nom_pos[1]),
                min(perturbation.delta_y[1], y_max - nom_pos[1]),
            )
            if eff_dx[0] > eff_dx[1]:
                eff_dx = (0.0, 0.0)
            if eff_dy[0] > eff_dy[1]:
                eff_dy = (0.0, 0.0)
            dx = rng.uniform(*eff_dx)
            dy = rng.uniform(*eff_dy)
            if extra_xy is not None:
                dx += extra_xy[0]
                dy += extra_xy[1]
            new_pos = nom_pos + np.array([dx, dy, rng.uniform(*perturbation.delta_z)])
            new_pos[0] = np.clip(new_pos[0], x_min, x_max)
            new_pos[1] = np.clip(new_pos[1], y_min, y_max)
            new_quat = _quat_mul(_sample_orientation_delta(perturbation, rng), nom_quat)
            return {"pos": new_pos.tolist(), "quat": new_quat.tolist()}

        perturb_by_name = {p.joint_name: p for p in self.perturbations}

        placed_xy: dict[str, np.ndarray] = {}

        def sample_clear_placement(
            joint_name: str,
            *,
            extra_xy: tuple[float, float] | None = None,
            tries: int,
        ) -> dict[str, list[float]]:
            perturbation = perturb_by_name[joint_name]
            for _ in range(tries):
                candidate = sample_from_range(perturbation, extra_xy=extra_xy)
                candidate_xy = np.asarray(candidate["pos"][:2], dtype=np.float64)
                if joint_name in self._trash_joints and not self._tool_keepout_ok(
                    candidate_xy,
                    states=states,
                ):
                    continue
                if all(
                    np.linalg.norm(candidate_xy - other_xy) >= self._clearance_for_pair(joint_name, other_name)
                    for other_name, other_xy in placed_xy.items()
                ):
                    return candidate
            raise _SweepPlacementFailure(f"could not place {joint_name}")

        for joint_name in self._tool_joints:
            if joint_name not in nominals:
                continue
            state = sample_clear_placement(joint_name, tries=self._tool_sample_tries)
            states[joint_name] = state
            placed_xy[joint_name] = np.asarray(state["pos"][:2], dtype=np.float64)

        cluster_dx, cluster_dy = self._sample_cluster_shift(
            states=states,
            nominals=nominals,
            rng=rng,
        )
        for joint_name in self._trash_joints:
            if joint_name not in nominals:
                continue
            state = sample_clear_placement(
                extra_xy=(cluster_dx, cluster_dy),
                joint_name=joint_name,
                tries=self._trash_sample_tries,
            )
            states[joint_name] = state
            placed_xy[joint_name] = np.asarray(state["pos"][:2], dtype=np.float64)

        return states

    def _pairwise_ok(self, states: dict[str, dict[str, list[float]]]) -> bool:
        if len(states) < 2:
            return True

        names = list(states.keys())
        positions = {name: np.asarray(state["pos"][:2], dtype=np.float64) for name, state in states.items()}
        for i, name_a in enumerate(names):
            for name_b in names[i + 1:]:
                clearance = self._clearance_for_pair(name_a, name_b)
                if np.linalg.norm(positions[name_a] - positions[name_b]) < clearance:
                    return False

        for trash_name in self._trash_joints:
            trash_state = states.get(trash_name)
            if trash_state is None:
                continue
            if not self._tool_keepout_ok(
                np.asarray(trash_state["pos"][:2], dtype=np.float64),
                states=states,
            ):
                return False
        return True




__all__ = ["SweepRandomizer"]
