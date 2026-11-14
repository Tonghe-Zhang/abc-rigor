"""Put-relative task evaluator."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import mujoco
import numpy as np

from abc_sim.task_eval.base import TaskEvalResult
from abc_sim.task_specs import SimTaskSpec


class PutRelativeEvaluator:
    """Score put_relative against the episode's sampled directive.

    The task has no fixed goal geometry: every reset the randomizer samples a
    mover object, a reference object, and a direction, sets the env prompt to
    e.g. "put the mug to the left of the apple", and records all three (plus
    the offset vector it aimed the layout at) in the randomization metadata.
    The evaluator configures itself from that metadata each reset.

    Success is directional, measured where the reference *currently* is:
    decompose the mover's xy offset from the live reference along the
    directive axis. The mover must sit ``along_min_m``..``along_max_m`` up
    that axis, within ``perp_max_m`` of it, and low enough to be placed
    rather than carried (within ``placed_z_tolerance_m`` above its own spawn
    height -- an object set down on its side rests lower than it spawned,
    never higher).

    By default (``require_past_reference_edge``) the mover's centre must also be
    past the reference's edge in the asked direction (from the reference's
    current pose and collision geometry), and no more sideways than its
    distance from the reference centre (a 45-degree cone). Short of that edge,
    including the wrong side, the reward is 0; past it but outside the zone,
    partial reward is at most ``partial_reward_max`` and fades to 0 over
    ``partial_reward_range_m`` of violation. With the flag off, the zone starts
    at ``along_min_m`` with a fixed ``perp_max_m`` and the reward decays with
    the distance to it on any side.

    The tolerance forms a directional cone rather than requiring the mover to
    land on one exact generated goal point.

    Until the first reset hands over metadata the evaluator has no directive
    to score, so it reports failure alongside a ``goal_configured`` flag
    instead of guessing one -- a wiring problem then shows up as a 0% summary
    with the flag down rather than as a plausible-looking number.
    """

    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        spec: SimTaskSpec,
        along_min_m: float = 0.0,
        along_max_m: float = 0.30,
        perp_max_m: float = 0.08,
        placed_z_tolerance_m: float = 0.03,
        require_past_reference_edge: bool = True,
        partial_reward_max: float = 0.5,
        partial_reward_range_m: float = 0.15,
    ) -> None:
        self.spec = spec
        self.along_min_m = float(along_min_m)
        self.along_max_m = float(along_max_m)
        self.perp_max_m = float(perp_max_m)
        self.placed_z_tolerance_m = float(placed_z_tolerance_m)
        self.require_past_reference_edge = bool(require_past_reference_edge)
        self.partial_reward_max = float(partial_reward_max)
        self.partial_reward_range_m = float(partial_reward_range_m)
        self._goal: dict[str, Any] | None = None
        self._bind_model(model)
        self.reset(nworld=1)

    def _bind_model(self, model: mujoco.MjModel) -> None:
        self.model = model
        self._goal = None

    def reset(self, *, nworld: int = 1) -> None:
        self._nworld = int(nworld)
        self._ever_success = np.zeros((self._nworld,), dtype=bool)
        self._max_reward = np.zeros((self._nworld,), dtype=np.float32)

    def configure_from_randomization(
        self,
        model: mujoco.MjModel,
        randomization: object,
    ) -> None:
        """Adopt the episode's sampled directive (and any reloaded model)."""

        if model is not self.model:
            self._bind_model(model)
            self.reset(nworld=self._nworld)

        metadata = getattr(randomization, "metadata", None)
        if metadata is None and isinstance(randomization, Mapping):
            metadata = randomization.get("metadata")
        if not isinstance(metadata, Mapping):
            return
        required = ("mover_object", "reference_object", "direction_offset_xy")
        if any(key not in metadata for key in required):
            return

        mover = str(metadata["mover_object"])
        reference = str(metadata["reference_object"])
        offset = np.asarray(metadata["direction_offset_xy"], dtype=np.float32)
        norm = float(np.linalg.norm(offset))
        if norm <= 0.0:
            raise ValueError(f"direction_offset_xy must be non-zero, got {offset}")

        object_states = getattr(randomization, "object_states", None)
        if object_states is None and isinstance(randomization, Mapping):
            object_states = randomization.get("object_states")
        spawn_z = None
        if isinstance(object_states, Mapping):
            state = object_states.get(f"{mover}_joint")
            if isinstance(state, Mapping) and "pos" in state:
                spawn_z = float(state["pos"][2])

        reference_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{reference}_joint")
        self._goal = {
            "reference_extent": self._body_extent_points(
                self.model, int(self.model.jnt_bodyid[reference_joint])
            ),
            "mover_adr": self._resolve_free_joint_qpos_adr(
                self.model, f"{mover}_joint"
            ),
            "reference_adr": self._resolve_free_joint_qpos_adr(
                self.model, f"{reference}_joint"
            ),
            "axis": offset / norm,
            "direction": str(metadata.get("direction", "")),
            "mover": mover,
            "reference": reference,
            "spawn_z": spawn_z,
        }

    def debug_spec(self) -> dict[str, object] | None:
        return None

    @staticmethod
    def _body_extent_points(model: mujoco.MjModel, body: int) -> np.ndarray:
        """Corners of the body's collision-geom bounding boxes, in the body frame (N, 3)."""
        geoms = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == body]
        colliding = [g for g in geoms if model.geom_contype[g] or model.geom_conaffinity[g]]
        signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64)
        points = []
        for g in colliding or geoms:
            center, half = model.geom_aabb[g, :3], model.geom_aabb[g, 3:]
            rot = np.empty(9)
            mujoco.mju_quat2Mat(rot, model.geom_quat[g])
            points.append(model.geom_pos[g] + (center + signs * half) @ rot.reshape(3, 3).T)
        return np.concatenate(points) if points else np.zeros((1, 3))

    @staticmethod
    def _rotate(quat: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Rotate body-frame points (N, 3) by a batch of wxyz quaternions (B, 4) -> (B, N, 3)."""
        w, x, y, z = (quat[:, i, None, None] for i in range(4))
        rot = np.concatenate(
            [
                np.concatenate([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=2),
                np.concatenate([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=2),
                np.concatenate([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=2),
            ],
            axis=1,
        )
        return np.einsum("bij,nj->bni", rot, points)

    @staticmethod
    def _resolve_free_joint_qpos_adr(model: mujoco.MjModel, joint_name: str) -> int:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Joint {joint_name!r} not found in model")
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"Joint {joint_name!r} must be a free joint")
        return int(model.jnt_qposadr[joint_id])

    def evaluate_qpos_batch(self, qpos_batch: np.ndarray) -> TaskEvalResult:
        qpos_batch = np.asarray(qpos_batch, dtype=np.float32)
        if qpos_batch.ndim != 2 or qpos_batch.shape[1] != self.model.nq:
            raise ValueError(
                f"Expected qpos batch shape (B, {self.model.nq}), got {qpos_batch.shape}"
            )
        if qpos_batch.shape[0] != self._nworld:
            self.reset(nworld=qpos_batch.shape[0])

        batch = qpos_batch.shape[0]
        if self._goal is None:
            zeros = np.zeros((batch,), dtype=np.float32)
            return TaskEvalResult(
                reward=zeros,
                success=np.zeros((batch,), dtype=bool),
                metrics={
                    "goal_configured": False,
                    "ever_success": self._ever_success.copy(),
                },
            )

        goal = self._goal
        mover_pos = qpos_batch[:, goal["mover_adr"] : goal["mover_adr"] + 3]
        reference_pos = qpos_batch[:, goal["reference_adr"] : goal["reference_adr"] + 3]
        rel = mover_pos[:, :2] - reference_pos[:, :2]
        axis = goal["axis"]
        along = rel @ axis
        perp = np.abs(rel @ np.asarray([-axis[1], axis[0]], dtype=np.float32))
        if self.require_past_reference_edge:
            reference_quat = qpos_batch[:, goal["reference_adr"] + 3 : goal["reference_adr"] + 7].astype(np.float64)
            extent = self._rotate(reference_quat, goal["reference_extent"])[..., :2] @ axis
            edge = extent.max(axis=1).astype(np.float32)
            side_limit = np.minimum(self.perp_max_m, along)
            in_cone = (
                (along >= np.maximum(edge, self.along_min_m))
                & (along <= self.along_max_m)
                & (perp <= side_limit)
            )
        else:
            edge = np.zeros_like(along)
            in_cone = (
                (along >= self.along_min_m)
                & (along <= self.along_max_m)
                & (perp <= self.perp_max_m)
            )

        if goal["spawn_z"] is None:
            placed = np.ones((batch,), dtype=bool)
        else:
            placed = mover_pos[:, 2] <= goal["spawn_z"] + self.placed_z_tolerance_m

        success = in_cone & placed
        self._ever_success |= success
        if self.require_past_reference_edge:
            # Nothing short of the reference's edge; past it, partial credit for the summed violations.
            gap = np.maximum(0.0, along - self.along_max_m) + np.maximum(0.0, perp - side_limit)
            partial = self.partial_reward_max * np.clip(1.0 - gap / self.partial_reward_range_m, 0.0, 1.0)
            reward = np.where(along > edge, partial, 0.0)
        else:
            # Progress: how far the mover has come toward the cone's near edge,
            # measured as the distance to the closest satisfying point.
            target_along = np.clip(along, self.along_min_m, self.along_max_m)
            target_perp = np.minimum(perp, self.perp_max_m)
            gap = np.hypot(along - target_along, perp - target_perp)
            reward = np.minimum(np.clip(1.0 - gap / 0.25, 0.0, 1.0), 0.99)
        reward = np.where(success, 1.0, reward).astype(np.float32)
        self._max_reward = np.maximum(self._max_reward, reward)

        return TaskEvalResult(
            reward=reward,
            success=success,
            metrics={
                "goal_configured": True,
                "direction": goal["direction"],
                "mover_object": goal["mover"],
                "reference_object": goal["reference"],
                "mover_along_m": along.astype(np.float32),
                "mover_perp_m": perp.astype(np.float32),
                "reference_edge_m": edge.astype(np.float32),
                "mover_in_cone": in_cone,
                "mover_placed": placed,
                "max_reward_so_far": self._max_reward.copy(),
                "ever_success": self._ever_success.copy(),
            },
        )
