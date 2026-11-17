"""Runtime controller for the conveyor pick task.

An object that leaves the belt for the table, floor or a wall is off the track: it is removed from the scene (parked
on the floor behind the back wall, out of every camera's view), so objects that ride off the belt's end cannot pile
up within reach and be put in the bin afterwards.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np


_OBJECT_COUNT = 3
_OBJECT_JOINTS = tuple(f"conveyor_object_{idx}_joint" for idx in range(1, _OBJECT_COUNT + 1))
_OBJECT_BODIES = tuple(f"conveyor_object_{idx}" for idx in range(1, _OBJECT_COUNT + 1))
_ROLLER_ACTUATORS = (
    "conveyor_infeed_roller_vel",
    "conveyor_outfeed_roller_vel",
)
_BELT_GEOM = "conveyor_belt_surface"
_ROLLER_SPEED = 6.0
_DEFAULT_BELT_SPEED_Y = -0.20
_BELT_CONTACT_MAX_DISTANCE_M = 0.0
_BELT_CONTACT_ANGULAR_DAMPING = 0.0
# Scene geoms an object only touches once it has left the belt (rails, rollers, the bin and the robot do not count).
_OFF_TRACK_GEOMS = (
    "table_collision",
    "floor_collision",
    "back_wall_collision",
    "left_wall_collision",
    "right_wall_collision",
)
# Removed objects rest on the floor at x = REMOVED_OBJECT_X, one per y slot; the evaluator treats any object past
# REMOVED_OBJECT_MIN_X as removed.
REMOVED_OBJECT_X = 3.0
REMOVED_OBJECT_MIN_X = 2.0
_REMOVED_OBJECT_SPACING_Y = 0.5
_REMOVED_OBJECT_DROP_Z = 0.2


class ConveyorPickRuntime:
    """Drive conveyor rollers and kinematically carry objects on the belt."""

    def __init__(self) -> None:
        self._object_controls: tuple[tuple[int, int, tuple[int, ...]], ...] | None = None
        self._roller_actuators: tuple[int, ...] | None = None
        self._belt_geom_id: int | None = None
        self._off_track_geom_ids: frozenset[int] | None = None
        self._last_active_objects: tuple[int, ...] = ()
        self._removed_objects: tuple[int, ...] = ()
        self._belt_speed_y = _DEFAULT_BELT_SPEED_Y

    def bind(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Reset cached model indices after a model/data change."""

        self._object_controls = None
        self._roller_actuators = None
        self._belt_geom_id = None
        self._off_track_geom_ids = None
        self._last_active_objects = ()
        self._removed_objects = ()

    def after_reset(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        randomization: Any = None,
    ) -> None:
        """Clear runtime bookkeeping after reset."""

        self._last_active_objects = ()
        self._removed_objects = ()
        self._belt_speed_y = self._speed_from_randomization(randomization)

    def initialize_robot_pose(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Open idle grippers when the environment requests its default reset pose."""
        self._open_idle_grippers(model, data)
        mujoco.mj_forward(model, data)

    def before_step(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Apply conveyor controls before each MuJoCo physics step."""

        roller_speed = self._roller_speed()
        for actuator_id in self._roller_actuators_for(model):
            data.ctrl[actuator_id] = roller_speed

        active_objects: list[int] = []
        needs_forward = False
        for idx, (qpos_adr, qvel_adr, geom_ids) in enumerate(
            self._object_controls_for(model),
            start=1,
        ):
            if self._geoms_contact_belt(model, data, geom_ids):
                self._carry_object_on_belt(model, data, qpos_adr, qvel_adr)
                self._damp_angular_velocity(data, qvel_adr)
                active_objects.append(idx)
                needs_forward = True
            else:
                if idx in self._last_active_objects:
                    data.qvel[qvel_adr + 1] = 0.0
        if needs_forward:
            mujoco.mj_forward(model, data)
        self._last_active_objects = tuple(active_objects)

    def after_step(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Remove objects that went off the track; keep belt-carried objects from accumulating rotation."""

        active_objects: list[int] = []
        removed = list(self._removed_objects)
        for idx, (qpos_adr, qvel_adr, geom_ids) in enumerate(
            self._object_controls_for(model),
            start=1,
        ):
            if idx in removed:
                continue
            if self._geoms_contact_any(data, geom_ids, self._off_track_geoms_for(model)):
                self._remove_object(data, idx, qpos_adr, qvel_adr)
                removed.append(idx)
            elif self._geoms_contact_belt(model, data, geom_ids):
                self._damp_angular_velocity(data, qvel_adr)
                active_objects.append(idx)
        if len(removed) != len(self._removed_objects):
            self._removed_objects = tuple(removed)
            mujoco.mj_forward(model, data)
        self._last_active_objects = tuple(active_objects)

    def debug_state(self) -> dict[str, Any]:
        """Return task runtime diagnostics for dashboards/debuggers."""

        return {
            "active_objects": self._last_active_objects,
            "removed_objects": self._removed_objects,
            "belt_speed_y": self._belt_speed_y,
            "belt_speed_m_s": abs(self._belt_speed_y),
            "belt_direction_y": float(np.sign(self._belt_speed_y)),
            "roller_speed": self._roller_speed(),
        }

    def _speed_from_randomization(self, randomization: Any = None) -> float:
        if randomization is None:
            return _DEFAULT_BELT_SPEED_Y

        metadata = getattr(randomization, "metadata", None)
        if not isinstance(metadata, dict) or "belt_speed_y" not in metadata:
            return _DEFAULT_BELT_SPEED_Y

        speed_y = float(metadata["belt_speed_y"])
        if not np.isfinite(speed_y):
            raise RuntimeError("conveyor_pick randomization provided a non-finite belt_speed_y")
        if abs(speed_y) < 1e-9:
            raise RuntimeError("conveyor_pick randomization provided a zero belt_speed_y")
        return speed_y

    def _roller_speed(self) -> float:
        return _ROLLER_SPEED if self._belt_speed_y < 0.0 else -_ROLLER_SPEED

    def _carry_object_on_belt(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        qpos_adr: int,
        qvel_adr: int,
    ) -> None:
        data.qpos[qpos_adr + 1] += self._belt_speed_y * float(model.opt.timestep)
        data.qvel[qvel_adr + 1] = 0.0

    def _open_idle_grippers(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        for prefix in ("left", "right"):
            left_joint = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                f"{prefix}_left_finger",
            )
            right_joint = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                f"{prefix}_right_finger",
            )
            gripper_actuator = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                f"{prefix}_gripper",
            )
            if left_joint < 0 or right_joint < 0 or gripper_actuator < 0:
                raise RuntimeError(f"conveyor_pick scene is missing {prefix} gripper controls")

            data.qpos[model.jnt_qposadr[left_joint]] = float(model.jnt_range[left_joint][1])
            data.qpos[model.jnt_qposadr[right_joint]] = float(model.jnt_range[right_joint][0])
            data.ctrl[gripper_actuator] = float(model.actuator_ctrlrange[gripper_actuator][1])

    @staticmethod
    def _remove_object(data: mujoco.MjData, idx: int, qpos_adr: int, qvel_adr: int) -> None:
        y = (idx - (_OBJECT_COUNT + 1) / 2) * _REMOVED_OBJECT_SPACING_Y
        data.qpos[qpos_adr : qpos_adr + 7] = (REMOVED_OBJECT_X, y, _REMOVED_OBJECT_DROP_Z, 1.0, 0.0, 0.0, 0.0)
        data.qvel[qvel_adr : qvel_adr + 6] = 0.0

    def _damp_angular_velocity(self, data: mujoco.MjData, qvel_adr: int) -> None:
        data.qvel[qvel_adr + 3 : qvel_adr + 6] *= _BELT_CONTACT_ANGULAR_DAMPING

    def _belt_geom_for(self, model: mujoco.MjModel) -> int:
        if self._belt_geom_id is not None:
            return self._belt_geom_id

        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, _BELT_GEOM)
        if geom_id < 0:
            raise RuntimeError(f"conveyor_pick scene is missing belt geom: {_BELT_GEOM}")

        self._belt_geom_id = int(geom_id)
        return self._belt_geom_id

    def _off_track_geoms_for(self, model: mujoco.MjModel) -> frozenset[int]:
        if self._off_track_geom_ids is not None:
            return self._off_track_geom_ids

        geom_ids = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in _OFF_TRACK_GEOMS}
        missing = [name for name, geom_id in geom_ids.items() if geom_id < 0]
        if missing:
            raise RuntimeError("conveyor_pick scene is missing off-track geoms: " + ", ".join(missing))

        self._off_track_geom_ids = frozenset(int(geom_id) for geom_id in geom_ids.values())
        return self._off_track_geom_ids

    def _roller_actuators_for(self, model: mujoco.MjModel) -> tuple[int, ...]:
        if self._roller_actuators is not None:
            return self._roller_actuators

        actuator_ids: list[int] = []
        missing: list[str] = []
        for actuator_name in _ROLLER_ACTUATORS:
            actuator_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                actuator_name,
            )
            if actuator_id < 0:
                missing.append(actuator_name)
                continue
            actuator_ids.append(int(actuator_id))

        if missing:
            raise RuntimeError(
                "conveyor_pick scene is missing conveyor roller actuators: "
                + ", ".join(missing)
            )

        self._roller_actuators = tuple(actuator_ids)
        return self._roller_actuators

    def _object_controls_for(self, model: mujoco.MjModel) -> tuple[tuple[int, int, tuple[int, ...]], ...]:
        if self._object_controls is not None:
            return self._object_controls

        controls: list[tuple[int, int, tuple[int, ...]]] = []
        missing: list[str] = []
        for joint_name, body_name in zip(
            _OBJECT_JOINTS,
            _OBJECT_BODIES,
        ):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if joint_id < 0:
                missing.append(joint_name)
            if body_id < 0:
                missing.append(body_name)
            if joint_id >= 0 and body_id >= 0:
                geom_ids = tuple(
                    int(geom_id)
                    for geom_id in range(model.ngeom)
                    if int(model.geom_bodyid[geom_id]) == int(body_id)
                    and (
                        int(model.geom_contype[geom_id]) != 0
                        or int(model.geom_conaffinity[geom_id]) != 0
                    )
                )
                if not geom_ids:
                    missing.append(f"{body_name} collision geoms")
                    continue
                controls.append(
                    (
                        int(model.jnt_qposadr[joint_id]),
                        int(model.jnt_dofadr[joint_id]),
                        geom_ids,
                    )
                )

        if missing:
            raise RuntimeError(
                "conveyor_pick scene is missing conveyor object joints/bodies: "
                + ", ".join(missing)
            )

        self._object_controls = tuple(controls)
        return self._object_controls

    @staticmethod
    def _geoms_contact_any(
        data: mujoco.MjData,
        geom_ids: tuple[int, ...],
        others: frozenset[int],
    ) -> bool:
        object_geom_ids = set(geom_ids)
        for contact_idx in range(data.ncon):
            contact = data.contact[contact_idx]
            if float(contact.dist) > _BELT_CONTACT_MAX_DISTANCE_M:
                continue
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if (geom1 in object_geom_ids and geom2 in others) or (geom2 in object_geom_ids and geom1 in others):
                return True
        return False

    def _geoms_contact_belt(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        geom_ids: tuple[int, ...],
    ) -> bool:
        belt_geom_id = self._belt_geom_for(model)
        object_geom_ids = set(geom_ids)
        for contact_idx in range(data.ncon):
            contact = data.contact[contact_idx]
            if float(contact.dist) > _BELT_CONTACT_MAX_DISTANCE_M:
                continue
            if (
                contact.geom1 == belt_geom_id and int(contact.geom2) in object_geom_ids
            ) or (
                contact.geom2 == belt_geom_id and int(contact.geom1) in object_geom_ids
            ):
                return True
        return False
