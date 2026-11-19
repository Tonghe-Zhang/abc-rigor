"""Pouring task evaluator."""

from __future__ import annotations

import re
from typing import Any

import mujoco
import numpy as np

from abc_sim.task_eval.base import TaskEvalResult
from abc_sim.task_eval.bottles import (
    _body_subtree_ids,
    _quat_to_rotmat_batch,
    _subtree_body_transforms,
)
from abc_sim.task_specs import SimTaskSpec

# Azimuth bins used to measure the receiving container's wall radius. A mug handle
# spans a minority of the bins, so the median across them tracks the body wall and
# ignores the handle -- see _resolve_container_bounds.
_AZIMUTH_BINS = 24


class PourBeadsInContainerEvaluator:
    """Score pouring by counting beads inside the receiving container.

    Membership uses a geometry-derived cylinder in the container's live frame.
    """

    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        spec: SimTaskSpec,
        target_joint_name: str = "cup_1_jnt",
        bead_joint_regex: str = r"bead_(\d+)_jnt",
        success_count: int | None = None,
        allowed_escapes: int = 1,
        target_upright_z_min: float = 0.5,
        radius_tolerance_frac: float = 0.05,
    ) -> None:
        self.spec = spec
        self._target_joint_name = target_joint_name
        self._bead_joint_regex = bead_joint_regex
        self.radius_tolerance_frac = float(radius_tolerance_frac)
        self.allowed_escapes = int(allowed_escapes)
        if self.allowed_escapes < 0:
            raise ValueError(
                f"allowed_escapes must be >= 0, got {self.allowed_escapes}"
            )
        self._success_count_override = (
            None if success_count is None else int(success_count)
        )
        # A container knocked onto its side keeps a local frame, so beads strewn
        # around it can still land inside the cylinder. Requiring the container's
        # local +Z to stay within 60 degrees of world up keeps a toppled cup from
        # scoring; a cup tilted less than that still holds its beads.
        self.target_upright_z_min = float(target_upright_z_min)
        self._bind_model(model)
        self.reset(nworld=1)

    def _bind_model(self, model: mujoco.MjModel) -> None:
        self.model = model
        self.bead_names, self._bead_qpos_addrs = self._resolve_bead_qpos_addrs(
            model,
            self._bead_joint_regex,
        )
        self._target_qpos_adr = self._resolve_free_joint_qpos_adr(
            model,
            self._target_joint_name,
        )
        (
            self._wall_radius,
            self._container_floor_z,
            self._container_rim_z,
        ) = self._resolve_container_bounds(model, self._target_joint_name)
        self._container_radius = self._wall_radius * (1.0 + self.radius_tolerance_frac)
        if self._success_count_override is None:
            # Allow one settling-related miss by default.
            self.success_count = max(1, len(self.bead_names) - self.allowed_escapes)
        else:
            self.success_count = self._success_count_override
        if self.success_count < 1:
            raise ValueError(f"success_count must be >= 1, got {self.success_count}")

    def reset(self, *, nworld: int = 1) -> None:
        self._nworld = int(nworld)
        self._max_beads_in_container = np.zeros((self._nworld,), dtype=np.int32)
        self._ever_success = np.zeros((self._nworld,), dtype=bool)

    def configure_from_randomization(
        self,
        model: mujoco.MjModel,
        randomization: object,
    ) -> None:
        """Rebind to a model the scale randomization swapped in under us."""

        if model is not self.model:
            self._bind_model(model)
            self.reset(nworld=self._nworld)

    @staticmethod
    def _resolve_free_joint_qpos_adr(model: mujoco.MjModel, joint_name: str) -> int:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Joint {joint_name!r} not found in model")
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"Joint {joint_name!r} must be a free joint")
        return int(model.jnt_qposadr[joint_id])

    @staticmethod
    def _resolve_bead_qpos_addrs(
        model: mujoco.MjModel,
        bead_joint_regex: str,
    ) -> tuple[list[str], np.ndarray]:
        pattern = re.compile(bead_joint_regex)
        entries: list[tuple[int, str, int]] = []
        for joint_id in range(model.njnt):
            name = model.jnt(joint_id).name
            if not name:
                continue
            match = pattern.fullmatch(name)
            if match is None:
                continue
            entries.append(
                (
                    int(match.group(1)),
                    name.removesuffix("_jnt"),
                    int(model.jnt_qposadr[joint_id]),
                )
            )

        if not entries:
            raise ValueError(f"No joints matching {bead_joint_regex!r} found in model")

        entries.sort(key=lambda item: item[0])
        bead_names = [name for _, name, _ in entries]
        addrs = np.asarray([adr for _, _, adr in entries], dtype=np.int32)
        return bead_names, addrs

    @staticmethod
    def _resolve_container_bounds(
        model: mujoco.MjModel,
        target_joint_name: str,
    ) -> tuple[float, float, float]:
        """Measure the receiving container as (wall radius, floor z, rim z).

        The wall radius is the bare geometry; the caller widens it by
        ``radius_tolerance_frac`` to get the containment radius.

        All three come from the container's group-3 collision meshes expressed in
        its body frame, so a rescaled asset measures itself. The radius is the
        median over azimuth bins of the farthest vertex in each bin: the body
        wall occupies every bin, the handle only a few, so the median lands on
        the wall.
        """

        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, target_joint_name)
        if joint_id < 0:
            raise ValueError(f"Joint {target_joint_name!r} not found in model")
        root_body_id = int(model.jnt_bodyid[joint_id])
        body_ids = _body_subtree_ids(model, root_body_id)
        transforms = _subtree_body_transforms(model, root_body_id)

        vertices: list[np.ndarray] = []
        for geom_id in range(model.ngeom):
            if int(model.geom_bodyid[geom_id]) not in body_ids:
                continue
            if int(model.geom_group[geom_id]) != 3:
                continue
            mesh_id = int(model.geom_dataid[geom_id])
            if mesh_id < 0:
                continue
            mesh_start = int(model.mesh_vertadr[mesh_id])
            mesh_count = int(model.mesh_vertnum[mesh_id])
            mesh_vertices = np.asarray(
                model.mesh_vert[mesh_start : mesh_start + mesh_count],
                dtype=np.float32,
            )
            geom_rot = _quat_to_rotmat_batch(
                np.asarray(model.geom_quat[geom_id], dtype=np.float32)[None, :]
            )[0]
            geom_pos = np.asarray(model.geom_pos[geom_id], dtype=np.float32)
            body_pos, body_rot = transforms[int(model.geom_bodyid[geom_id])]
            geom_vertices = mesh_vertices @ geom_rot.T + geom_pos
            vertices.append(geom_vertices @ body_rot.T + body_pos)

        if not vertices:
            raise ValueError(
                f"No collision mesh vertices found for the body of {target_joint_name!r}"
            )

        all_vertices = np.concatenate(vertices, axis=0)
        radial = np.linalg.norm(all_vertices[:, :2], axis=1)
        azimuth = np.arctan2(all_vertices[:, 1], all_vertices[:, 0])
        bin_index = np.clip(
            ((azimuth + np.pi) / (2.0 * np.pi) * _AZIMUTH_BINS).astype(np.int32),
            0,
            _AZIMUTH_BINS - 1,
        )
        bin_maxima = [
            float(radial[bin_index == index].max())
            for index in range(_AZIMUTH_BINS)
            if np.any(bin_index == index)
        ]
        radius = float(np.median(bin_maxima))
        if radius <= 0.0:
            raise ValueError(
                f"Container {target_joint_name!r} measured a non-positive wall radius"
            )
        return radius, float(all_vertices[:, 2].min()), float(all_vertices[:, 2].max())

    def evaluate_qpos_batch(self, qpos_batch: np.ndarray) -> TaskEvalResult:
        qpos_batch = np.asarray(qpos_batch, dtype=np.float32)
        if qpos_batch.ndim != 2 or qpos_batch.shape[1] != self.model.nq:
            raise ValueError(
                f"Expected qpos batch shape (B, {self.model.nq}), got {qpos_batch.shape}"
            )
        if qpos_batch.shape[0] != self._nworld:
            self.reset(nworld=qpos_batch.shape[0])

        adr = self._target_qpos_adr
        target_pos = qpos_batch[:, adr : adr + 3]
        target_quat = qpos_batch[:, adr + 3 : adr + 7]
        bead_pos = np.stack(
            [qpos_batch[:, bead_adr : bead_adr + 3] for bead_adr in self._bead_qpos_addrs],
            axis=1,
        )

        rot_world_from_target = _quat_to_rotmat_batch(target_quat)
        rel_world = bead_pos - target_pos[:, None, :]
        rel_local = np.einsum(
            "bij,bnj->bni",
            np.swapaxes(rot_world_from_target, 1, 2),
            rel_world,
        )

        radial_xy = np.linalg.norm(rel_local[..., :2], axis=-1)
        local_z = rel_local[..., 2]
        radial_margin = self._container_radius - radial_xy
        lower_height_margin = local_z - self._container_floor_z
        upper_height_margin = self._container_rim_z - local_z
        # Third row of the rotation: the container's local +Z read in world axes,
        # so element [2] is how much of it points at world up.
        target_upright_z = rot_world_from_target[:, 2, 2]
        target_upright = target_upright_z >= self.target_upright_z_min
        in_container_mask = (
            (radial_margin >= 0.0)
            & (lower_height_margin >= 0.0)
            & (upper_height_margin >= 0.0)
            & target_upright[:, None]
        )

        num_beads = in_container_mask.sum(axis=1).astype(np.int32)
        active_count = np.full(
            (qpos_batch.shape[0],),
            len(self.bead_names),
            dtype=np.int32,
        )
        self._max_beads_in_container = np.maximum(
            self._max_beads_in_container,
            num_beads,
        )
        reward = np.clip(
            num_beads.astype(np.float32) / float(self.success_count),
            0.0,
            1.0,
        )
        success = num_beads >= self.success_count
        self._ever_success |= success
        beads_in_container = [
            [name for name, inside in zip(self.bead_names, world_mask) if bool(inside)]
            for world_mask in in_container_mask
        ]

        metrics: dict[str, Any] = {
            "num_beads_in_container": num_beads,
            "num_active_beads": active_count,
            "max_beads_in_container_so_far": self._max_beads_in_container.copy(),
            "ever_success": self._ever_success.copy(),
            "bead_fraction_in_container": (
                num_beads.astype(np.float32) / active_count.astype(np.float32)
            ),
            "closest_radial_margin": radial_margin.max(axis=1).astype(np.float32),
            "closest_height_margin": lower_height_margin.max(axis=1).astype(np.float32),
            "closest_upper_height_margin": upper_height_margin.max(axis=1).astype(
                np.float32
            ),
            "bead_in_container_mask": in_container_mask,
            "beads_in_container": beads_in_container,
            "bead_names": list(self.bead_names),
            "target_upright_z": target_upright_z.astype(np.float32),
            "success_count": self.success_count,
            "container_wall_radius": self._wall_radius,
            "container_radius": self._container_radius,
            "container_floor_z": self._container_floor_z,
            "container_rim_z": self._container_rim_z,
        }
        return TaskEvalResult(reward=reward, success=success, metrics=metrics)
