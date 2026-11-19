"""Conveyor-pick task evaluator."""

from __future__ import annotations

import re

import mujoco
import numpy as np

from abc_sim.task_eval.base import TaskEvalResult
from abc_sim.task_runtime.conveyor_pick import REMOVED_OBJECT_MIN_X
from abc_sim.task_specs import SimTaskSpec


class ConveyorPickObjectsInBinEvaluator:
    """Score success when every conveyor object is inside the static bin.

    The bin interior comes from its collision geometry. Peak object count and
    ``ever_success`` remain latched for rollout progress and early stopping.
    An object the task runtime removed for going off the track (parked past
    ``REMOVED_OBJECT_MIN_X``) never counts. ``ever_failed`` latches once no
    object is left that could still raise the reward: every object is either
    removed or in the bin, and not all are in the bin. Until then the policy can
    still earn partial credit, so rollouts keep going after the first removal.
    """

    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        spec: SimTaskSpec,
        bin_body_name: str = "conveyor_bin",
        object_joint_regex: str = r"conveyor_object_(\d+)_joint",
        floor_tolerance_m: float = 0.01,
    ) -> None:
        self.spec = spec
        self._bin_body_name = bin_body_name
        self._object_joint_regex = object_joint_regex
        self.floor_tolerance_m = float(floor_tolerance_m)
        self._bind_model(model)
        self.reset(nworld=1)

    def _bind_model(self, model: mujoco.MjModel) -> None:
        self.model = model
        self.object_names, self._object_qpos_addrs = self._resolve_object_qpos_addrs(
            model,
            self._object_joint_regex,
        )
        (
            self._bin_x_bounds,
            self._bin_y_bounds,
            self._bin_floor_z,
            self._bin_rim_z,
        ) = self._resolve_bin_interior(model, self._bin_body_name)

    def reset(self, *, nworld: int = 1) -> None:
        self._nworld = int(nworld)
        self._max_objects_in_bin = np.zeros((self._nworld,), dtype=np.int32)
        self._ever_success = np.zeros((self._nworld,), dtype=bool)
        self._ever_failed = np.zeros((self._nworld,), dtype=bool)

    def configure_from_randomization(
        self,
        model: mujoco.MjModel,
        randomization: object,
    ) -> None:
        """Rebind to a model the object randomizer swapped in under us."""

        if model is not self.model:
            self._bind_model(model)
            self.reset(nworld=self._nworld)

    def debug_spec(self) -> dict[str, object] | None:
        return None

    @staticmethod
    def _resolve_object_qpos_addrs(
        model: mujoco.MjModel,
        object_joint_regex: str,
    ) -> tuple[list[str], np.ndarray]:
        pattern = re.compile(object_joint_regex)
        entries: list[tuple[int, str, int]] = []
        for joint_id in range(model.njnt):
            name = model.jnt(joint_id).name
            if not name:
                continue
            match = pattern.fullmatch(name)
            if match is None:
                continue
            if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
                raise ValueError(f"Joint {name!r} must be a free joint")
            entries.append(
                (
                    int(match.group(1)),
                    name.removesuffix("_joint"),
                    int(model.jnt_qposadr[joint_id]),
                )
            )

        if not entries:
            raise ValueError(
                f"No joints matching {object_joint_regex!r} found in model"
            )

        entries.sort(key=lambda item: item[0])
        names = [name for _, name, _ in entries]
        addrs = np.asarray([addr for _, _, addr in entries], dtype=np.int32)
        return names, addrs

    @staticmethod
    def _resolve_bin_interior(
        model: mujoco.MjModel,
        bin_body_name: str,
    ) -> tuple[tuple[float, float], tuple[float, float], float, float]:
        """Measure the static bin's interior as (x bounds, y bounds, floor, rim).

        The bin body must be welded to the world through joint-free bodies so
        its geoms have a fixed world pose the evaluator can bake in. Its box
        geoms split into the floor plate (the one whose top face sits lowest)
        and four walls; each wall bounds the axis its centre is displaced
        along, by its inner face.
        """

        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bin_body_name)
        if body_id < 0:
            raise ValueError(f"Body {bin_body_name!r} not found in model")

        # Accumulate the fixed world offset. Any joint on the chain would make
        # the bin's pose state-dependent, which this world-frame test cannot
        # represent, so fail loudly rather than silently score a stale pose.
        offset = np.zeros(3, dtype=np.float64)
        current = body_id
        while current != 0:
            if int(model.body_jntnum[current]) != 0:
                raise ValueError(
                    f"Body {bin_body_name!r} must be welded to the world, but "
                    f"{model.body(current).name!r} has a joint"
                )
            if not np.allclose(model.body_quat[current], (1.0, 0.0, 0.0, 0.0)):
                raise ValueError(
                    f"Body {model.body(current).name!r} on the {bin_body_name!r} "
                    "chain is rotated; the axis-aligned interior test assumes "
                    "identity body orientation"
                )
            offset += np.asarray(model.body_pos[current], dtype=np.float64)
            current = int(model.body_parentid[current])

        boxes: list[tuple[np.ndarray, np.ndarray]] = []
        for geom_id in range(model.ngeom):
            if int(model.geom_bodyid[geom_id]) != body_id:
                continue
            if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_BOX):
                continue
            if not np.allclose(model.geom_quat[geom_id], (1.0, 0.0, 0.0, 0.0)):
                raise ValueError(
                    f"Geom {model.geom(geom_id).name!r} of {bin_body_name!r} is "
                    "rotated; the axis-aligned interior test assumes identity "
                    "geom orientation"
                )
            boxes.append(
                (
                    np.asarray(model.geom_pos[geom_id], dtype=np.float64) + offset,
                    np.asarray(model.geom_size[geom_id], dtype=np.float64),
                )
            )

        if len(boxes) < 5:
            raise ValueError(
                f"Body {bin_body_name!r} has {len(boxes)} box geoms; expected a "
                "floor plate and four walls"
            )

        floor_pos, floor_size = min(boxes, key=lambda box: box[0][2] + box[1][2])
        floor_top_z = float(floor_pos[2] + floor_size[2])

        x_faces: list[float] = []
        y_faces: list[float] = []
        rim_z = floor_top_z
        for pos, size in boxes:
            if pos is floor_pos:
                continue
            rim_z = max(rim_z, float(pos[2] + size[2]))
            dx = pos[0] - floor_pos[0]
            dy = pos[1] - floor_pos[1]
            if abs(dx) >= abs(dy):
                x_faces.append(float(abs(dx) - size[0]))
            else:
                y_faces.append(float(abs(dy) - size[1]))

        if not x_faces or not y_faces or rim_z <= floor_top_z:
            raise ValueError(
                f"Body {bin_body_name!r} walls do not enclose the floor plate"
            )

        x_half = min(x_faces)
        y_half = min(y_faces)
        if x_half <= 0.0 or y_half <= 0.0:
            raise ValueError(
                f"Body {bin_body_name!r} measured a non-positive interior extent"
            )
        return (
            (float(floor_pos[0] - x_half), float(floor_pos[0] + x_half)),
            (float(floor_pos[1] - y_half), float(floor_pos[1] + y_half)),
            floor_top_z,
            rim_z,
        )

    def evaluate_qpos_batch(self, qpos_batch: np.ndarray) -> TaskEvalResult:
        qpos_batch = np.asarray(qpos_batch, dtype=np.float32)
        if qpos_batch.ndim != 2 or qpos_batch.shape[1] != self.model.nq:
            raise ValueError(
                f"Expected qpos batch shape (B, {self.model.nq}), got {qpos_batch.shape}"
            )
        if qpos_batch.shape[0] != self._nworld:
            self.reset(nworld=qpos_batch.shape[0])

        object_pos = np.stack(
            [qpos_batch[:, adr : adr + 3] for adr in self._object_qpos_addrs],
            axis=1,
        )

        x_lo, x_hi = self._bin_x_bounds
        y_lo, y_hi = self._bin_y_bounds
        z_lo = self._bin_floor_z - self.floor_tolerance_m
        in_bin_mask = (
            (object_pos[..., 0] >= x_lo)
            & (object_pos[..., 0] <= x_hi)
            & (object_pos[..., 1] >= y_lo)
            & (object_pos[..., 1] <= y_hi)
            & (object_pos[..., 2] >= z_lo)
            & (object_pos[..., 2] <= self._bin_rim_z)
        )

        removed_mask = object_pos[..., 0] >= REMOVED_OBJECT_MIN_X
        in_bin_mask &= ~removed_mask

        num_objects = in_bin_mask.sum(axis=1).astype(np.int32)
        num_removed = removed_mask.sum(axis=1).astype(np.int32)
        total = len(self.object_names)
        self._max_objects_in_bin = np.maximum(self._max_objects_in_bin, num_objects)
        success = num_objects >= total
        self._ever_success |= success
        # Nothing left on the track: the reward can no longer rise, so rollouts can stop.
        self._ever_failed |= ~success & np.all(removed_mask | in_bin_mask, axis=1)

        objects_in_bin = [
            [name for name, inside in zip(self.object_names, world_mask) if bool(inside)]
            for world_mask in in_bin_mask
        ]
        return TaskEvalResult(
            reward=num_objects.astype(np.float32) / float(max(total, 1)),
            success=success,
            metrics={
                "num_objects_in_bin": num_objects,
                "num_active_objects": np.full_like(num_objects, total),
                "max_objects_in_bin_so_far": self._max_objects_in_bin.copy(),
                "ever_success": self._ever_success.copy(),
                "num_removed_objects": num_removed,
                "ever_failed": self._ever_failed.copy(),
                "object_in_bin_mask": in_bin_mask,
                "objects_in_bin": objects_in_bin,
                "removed_objects": [
                    [name for name, removed in zip(self.object_names, world_mask) if bool(removed)]
                    for world_mask in removed_mask
                ],
                "object_names": list(self.object_names),
                "bin_x_bounds": list(self._bin_x_bounds),
                "bin_y_bounds": list(self._bin_y_bounds),
                "bin_floor_z": self._bin_floor_z,
                "bin_rim_z": self._bin_rim_z,
            },
        )
