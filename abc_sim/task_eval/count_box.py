"""Count-into-opaque-box task evaluator."""

from __future__ import annotations

from typing import Any, Mapping

import mujoco
import numpy as np

from abc_sim.task_eval.base import TaskEvalResult
from abc_sim.task_specs import SimTaskSpec


class CountIntoOpaqueBoxEvaluator:
    """Score current box contents against the reset-defined directive.

    Success requires exactly ``target_count`` eligible objects and no
    ineligible objects in the geometry-derived interior. Success is not
    latched because a rollout may later overfill the box; ``ever_correct``
    records whether the count was ever right.

    Reward counts up in equal steps, ``k / target_count`` for ``k`` eligible
    objects in the box, and is 0 once the box holds more than
    ``target_count`` eligible objects or any ineligible one: the box is
    lidded, so neither can be taken back out, and over-counting must never
    score like an honest partial count. Reward is 1 exactly on success.
    """

    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        spec: SimTaskSpec,
        box_body_name: str = "opaque_count_box",
        object_joint_suffix: str = "_joint",
        floor_tolerance_m: float = 0.01,
        partial_overfill_penalty: bool = False,
    ) -> None:
        self.spec = spec
        self._box_body_name = box_body_name
        self._object_joint_suffix = object_joint_suffix
        self.floor_tolerance_m = float(floor_tolerance_m)
        self.partial_overfill_penalty = bool(partial_overfill_penalty)
        self._directive: dict[str, Any] | None = None
        self._bind_model(model)
        self.reset(nworld=1)

    def _bind_model(self, model: mujoco.MjModel) -> None:
        self.model = model
        self._directive = None
        (
            self._box_x_bounds,
            self._box_y_bounds,
            self._box_floor_z,
            self._box_ceiling_z,
        ) = self._resolve_box_interior(model, self._box_body_name)

    def reset(self, *, nworld: int = 1) -> None:
        self._nworld = int(nworld)
        self._ever_correct = np.zeros((self._nworld,), dtype=bool)

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
        if "target_count" not in metadata or "eligible_objects" not in metadata:
            return

        objects = metadata.get("objects") or []
        names = [str(entry["name"]) for entry in objects]
        if not names:
            return
        addrs = {
            name: self._resolve_free_joint_qpos_adr(
                self.model, name + self._object_joint_suffix
            )
            for name in names
        }
        eligible = {str(name) for name in metadata["eligible_objects"]}
        unknown = eligible - set(names)
        if unknown:
            raise ValueError(
                f"eligible_objects name objects missing from the scene: {sorted(unknown)}"
            )
        self._directive = {
            "names": names,
            "addrs": np.asarray([addrs[name] for name in names], dtype=np.int32),
            "eligible_mask": np.asarray(
                [name in eligible for name in names], dtype=bool
            ),
            "target_count": int(metadata["target_count"]),
            "prompt_type": str(metadata.get("prompt_type", "")),
            "prompt": str(metadata.get("prompt", "")),
        }

    def debug_spec(self) -> dict[str, object] | None:
        return None

    @staticmethod
    def _resolve_free_joint_qpos_adr(model: mujoco.MjModel, joint_name: str) -> int:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Joint {joint_name!r} not found in model")
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"Joint {joint_name!r} must be a free joint")
        return int(model.jnt_qposadr[joint_id])

    @staticmethod
    def _resolve_box_interior(
        model: mujoco.MjModel,
        box_body_name: str,
    ) -> tuple[tuple[float, float], tuple[float, float], float, float]:
        """Measure the lidded box's interior as (x bounds, y bounds, floor, ceiling).

        The box body must be welded to the world through joint-free bodies so
        its geoms have a fixed world pose. Its box geoms split by shape: thin-z
        plates are the floor (the lowest one) and the lid panels (everything
        plate-shaped above it); the rest are walls, each bounding the axis its
        centre is displaced along by its inner face. The ceiling is the lowest
        lid underside, so nothing resting on the lid or wedged in the drop
        slot can be inside.
        """

        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, box_body_name)
        if body_id < 0:
            raise ValueError(f"Body {box_body_name!r} not found in model")

        offset = np.zeros(3, dtype=np.float64)
        current = body_id
        while current != 0:
            if int(model.body_jntnum[current]) != 0:
                raise ValueError(
                    f"Body {box_body_name!r} must be welded to the world, but "
                    f"{model.body(current).name!r} has a joint"
                )
            if not np.allclose(model.body_quat[current], (1.0, 0.0, 0.0, 0.0)):
                raise ValueError(
                    f"Body {model.body(current).name!r} on the {box_body_name!r} "
                    "chain is rotated; the axis-aligned interior test assumes "
                    "identity body orientation"
                )
            offset += np.asarray(model.body_pos[current], dtype=np.float64)
            current = int(model.body_parentid[current])

        plates: list[tuple[np.ndarray, np.ndarray]] = []
        walls: list[tuple[np.ndarray, np.ndarray]] = []
        for geom_id in range(model.ngeom):
            if int(model.geom_bodyid[geom_id]) != body_id:
                continue
            if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_BOX):
                continue
            if not np.allclose(model.geom_quat[geom_id], (1.0, 0.0, 0.0, 0.0)):
                raise ValueError(
                    f"Geom {model.geom(geom_id).name!r} of {box_body_name!r} is "
                    "rotated; the axis-aligned interior test assumes identity "
                    "geom orientation"
                )
            pos = np.asarray(model.geom_pos[geom_id], dtype=np.float64) + offset
            size = np.asarray(model.geom_size[geom_id], dtype=np.float64)
            if size[2] < min(size[0], size[1]):
                plates.append((pos, size))
            else:
                walls.append((pos, size))

        if not plates or len(walls) < 4:
            raise ValueError(
                f"Body {box_body_name!r} has {len(plates)} plate and {len(walls)} "
                "wall geoms; expected a floor plate and four walls"
            )

        floor_pos, floor_size = min(plates, key=lambda plate: plate[0][2] + plate[1][2])
        floor_top_z = float(floor_pos[2] + floor_size[2])
        lid_bottoms = [
            float(pos[2] - size[2])
            for pos, size in plates
            if pos is not floor_pos and pos[2] - size[2] > floor_top_z
        ]

        x_faces: list[float] = []
        y_faces: list[float] = []
        wall_top_z = floor_top_z
        for pos, size in walls:
            wall_top_z = max(wall_top_z, float(pos[2] + size[2]))
            dx = pos[0] - floor_pos[0]
            dy = pos[1] - floor_pos[1]
            if abs(dx) >= abs(dy):
                x_faces.append(float(abs(dx) - size[0]))
            else:
                y_faces.append(float(abs(dy) - size[1]))

        if not x_faces or not y_faces or wall_top_z <= floor_top_z:
            raise ValueError(
                f"Body {box_body_name!r} walls do not enclose the floor plate"
            )

        x_half = min(x_faces)
        y_half = min(y_faces)
        if x_half <= 0.0 or y_half <= 0.0:
            raise ValueError(
                f"Body {box_body_name!r} measured a non-positive interior extent"
            )
        ceiling_z = min(lid_bottoms) if lid_bottoms else wall_top_z
        return (
            (float(floor_pos[0] - x_half), float(floor_pos[0] + x_half)),
            (float(floor_pos[1] - y_half), float(floor_pos[1] + y_half)),
            floor_top_z,
            float(ceiling_z),
        )

    def evaluate_qpos_batch(self, qpos_batch: np.ndarray) -> TaskEvalResult:
        qpos_batch = np.asarray(qpos_batch, dtype=np.float32)
        if qpos_batch.ndim != 2 or qpos_batch.shape[1] != self.model.nq:
            raise ValueError(
                f"Expected qpos batch shape (B, {self.model.nq}), got {qpos_batch.shape}"
            )
        if qpos_batch.shape[0] != self._nworld:
            self.reset(nworld=qpos_batch.shape[0])

        batch = qpos_batch.shape[0]
        if self._directive is None:
            zeros = np.zeros((batch,), dtype=np.float32)
            return TaskEvalResult(
                reward=zeros,
                success=np.zeros((batch,), dtype=bool),
                metrics={
                    "goal_configured": False,
                    "termination_mode": "maintenance",
                    "ever_failed": np.zeros((batch,), dtype=bool),
                    "ever_success": np.zeros((batch,), dtype=bool),
                },
            )

        directive = self._directive
        positions = np.stack(
            [qpos_batch[:, adr : adr + 3] for adr in directive["addrs"]],
            axis=1,
        )
        x_lo, x_hi = self._box_x_bounds
        y_lo, y_hi = self._box_y_bounds
        inside = (
            (positions[..., 0] >= x_lo)
            & (positions[..., 0] <= x_hi)
            & (positions[..., 1] >= y_lo)
            & (positions[..., 1] <= y_hi)
            & (positions[..., 2] >= self._box_floor_z - self.floor_tolerance_m)
            & (positions[..., 2] <= self._box_ceiling_z)
        )

        eligible = directive["eligible_mask"]
        target = directive["target_count"]
        num_eligible_in = (inside & eligible[None, :]).sum(axis=1).astype(np.int32)
        num_ineligible_in = (inside & ~eligible[None, :]).sum(axis=1).astype(np.int32)
        success = (num_eligible_in == target) & (num_ineligible_in == 0)
        self._ever_correct |= success

        violated = (num_eligible_in > target) | (num_ineligible_in > 0)
        errors = np.maximum(num_eligible_in - target, 0) + num_ineligible_in
        progress = np.minimum(num_eligible_in.astype(np.float32) / float(target), 1.0)
        if self.partial_overfill_penalty:
            multiplier = np.array([1.0, 0.4, 0.2, 0.1, 0.0])[np.minimum(errors, 4)]
            reward = (progress * multiplier).astype(np.float32)
        else:
            reward = np.where(violated, 0.0, progress).astype(np.float32)

        in_box = [
            [name for name, is_in in zip(directive["names"], world) if bool(is_in)]
            for world in inside
        ]
        return TaskEvalResult(
            reward=reward,
            success=success,
            metrics={
                "goal_configured": True,
                "prompt_type": directive["prompt_type"],
                "target_count": target,
                "num_eligible_in_box": num_eligible_in,
                "num_ineligible_in_box": num_ineligible_in,
                "directive_violated": violated,
                "extra_or_wrong_count": errors,
                "objects_in_box": in_box,
                # Final-state semantics: never stop a rollout early on a count
                # that may still change, and let the last frame decide.
                "termination_mode": "maintenance",
                "ever_failed": np.zeros((batch,), dtype=bool),
                "ever_success": success.copy(),
                "ever_correct": self._ever_correct.copy(),
                "box_x_bounds": list(self._box_x_bounds),
                "box_y_bounds": list(self._box_y_bounds),
                "box_floor_z": self._box_floor_z,
                "box_ceiling_z": self._box_ceiling_z,
            },
        )
