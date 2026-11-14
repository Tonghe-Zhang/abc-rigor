"""In-hand transfer task evaluator."""

from __future__ import annotations

from collections.abc import Mapping

import mujoco
import numpy as np

from abc_sim.task_eval.base import TaskEvalResult
from abc_sim.task_eval.contacts import gripper_geom_ids, has_contact, subtree_geom_ids
from abc_sim.task_specs import SimTaskSpec


class InhandTransferOtherSideEvaluator:
    """Score the item resting on the table opposite its sampled spawn side.

    The item must cross the configured y band, remain within the tabletop,
    contact the table without touching either gripper. The start side comes
    from reset metadata.
    """

    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        spec: SimTaskSpec,
        object_joint_name: str = "task_object_joint",
        table_geom_name: str = "table_plane",
        other_side_min_y_m: float = 0.10,
        max_resting_height_m: float = 0.10,
    ) -> None:
        self.spec = spec
        self._object_joint_name = object_joint_name
        self._table_geom_name = table_geom_name
        self.other_side_min_y_m = float(other_side_min_y_m)
        self.max_resting_height_m = float(max_resting_height_m)
        self._start_sign: float | None = None
        self._spawn_y: float | None = None
        self._bind_model(model)
        self.reset(nworld=1)

    def _bind_model(self, model: mujoco.MjModel) -> None:
        self.model = model
        self._start_sign = None
        self._spawn_y = None
        self._object_adr = self._resolve_free_joint_qpos_adr(
            model, self._object_joint_name
        )
        (
            self._table_x_bounds,
            self._table_y_bounds,
            self._table_z,
        ) = self._resolve_table(model, self._table_geom_name)
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, self._object_joint_name
        )
        self._object_geom_ids = subtree_geom_ids(model, int(model.jnt_bodyid[joint_id]))
        self._table_geom_ids = {
            int(
                mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_GEOM, self._table_geom_name
                )
            )
        }
        self._gripper_geom_ids = gripper_geom_ids(model)

    def reset(self, *, nworld: int = 1) -> None:
        self._nworld = int(nworld)
        self._ever_success = np.zeros((self._nworld,), dtype=bool)

    def configure_from_randomization(
        self,
        model: mujoco.MjModel,
        randomization: object,
    ) -> None:
        """Adopt the episode's start side (and any reloaded model)."""

        if model is not self.model:
            self._bind_model(model)
            self.reset(nworld=self._nworld)

        metadata = getattr(randomization, "metadata", None)
        if metadata is None and isinstance(randomization, Mapping):
            metadata = randomization.get("metadata")
        if not isinstance(metadata, Mapping):
            return
        side = metadata.get("side")
        if side not in ("left", "right"):
            return
        # The randomizer's "left" is +y (the left arm's side of the table).
        self._start_sign = 1.0 if side == "left" else -1.0

        object_states = getattr(randomization, "object_states", None)
        if object_states is None and isinstance(randomization, Mapping):
            object_states = randomization.get("object_states")
        self._spawn_y = None
        if isinstance(object_states, Mapping):
            state = object_states.get(self._object_joint_name)
            if isinstance(state, Mapping) and "pos" in state:
                self._spawn_y = float(state["pos"][1])

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
    def _resolve_table(
        model: mujoco.MjModel,
        table_geom_name: str,
    ) -> tuple[tuple[float, float], tuple[float, float], float]:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, table_geom_name)
        if geom_id < 0:
            raise ValueError(f"Geom {table_geom_name!r} not found in model")
        if int(model.geom_bodyid[geom_id]) != 0:
            raise ValueError(
                f"Geom {table_geom_name!r} must live on the world body; the "
                "world-frame side test assumes a static table"
            )
        pos = np.asarray(model.geom_pos[geom_id], dtype=np.float64)
        size = np.asarray(model.geom_size[geom_id], dtype=np.float64)
        return (
            (float(pos[0] - size[0]), float(pos[0] + size[0])),
            (float(pos[1] - size[1]), float(pos[1] + size[1])),
            float(pos[2]),
        )

    def evaluate_model_data(
        self, model: mujoco.MjModel, data: mujoco.MjData
    ) -> TaskEvalResult:
        if model is not self.model:
            self._bind_model(model)
            self.reset(nworld=1)
        return self._evaluate(
            np.asarray(data.qpos, dtype=np.float32)[None, :],
            np.asarray(
                [has_contact(data, self._object_geom_ids, self._table_geom_ids)]
            ),
            np.asarray(
                [has_contact(data, self._object_geom_ids, self._gripper_geom_ids)]
            ),
        )

    def evaluate_qpos_batch(self, qpos_batch: np.ndarray) -> TaskEvalResult:
        qpos_batch = np.asarray(qpos_batch, dtype=np.float32)
        if qpos_batch.ndim != 2 or qpos_batch.shape[1] != self.model.nq:
            raise ValueError(
                f"Expected qpos batch shape (B, {self.model.nq}), got {qpos_batch.shape}"
            )
        if qpos_batch.shape[0] != self._nworld:
            self.reset(nworld=qpos_batch.shape[0])
        data = mujoco.MjData(self.model)
        table_contacts, gripper_contacts = [], []
        for qpos in qpos_batch:
            data.qpos[:] = qpos
            mujoco.mj_forward(self.model, data)
            table_contacts.append(
                has_contact(data, self._object_geom_ids, self._table_geom_ids)
            )
            gripper_contacts.append(
                has_contact(data, self._object_geom_ids, self._gripper_geom_ids)
            )
        return self._evaluate(
            qpos_batch, np.asarray(table_contacts), np.asarray(gripper_contacts)
        )

    def _evaluate(
        self,
        qpos_batch: np.ndarray,
        table_contact: np.ndarray,
        gripper_contact: np.ndarray,
    ) -> TaskEvalResult:
        batch = qpos_batch.shape[0]
        if batch != self._nworld:
            self.reset(nworld=batch)
        if self._start_sign is None:
            zeros = np.zeros((batch,), dtype=np.float32)
            return TaskEvalResult(
                reward=zeros,
                success=np.zeros((batch,), dtype=bool),
                metrics={
                    "goal_configured": False,
                    "ever_success": self._ever_success.copy(),
                },
            )

        pos = qpos_batch[:, self._object_adr : self._object_adr + 3]
        sign = self._start_sign
        crossed = (pos[:, 1] * sign) <= -self.other_side_min_y_m
        placed = table_contact & ~gripper_contact
        x_lo, x_hi = self._table_x_bounds
        y_lo, y_hi = self._table_y_bounds
        on_table = (
            (pos[:, 0] >= x_lo)
            & (pos[:, 0] <= x_hi)
            & (pos[:, 1] >= y_lo)
            & (pos[:, 1] <= y_hi)
        )
        success = crossed & placed & on_table
        self._ever_success |= success

        # Progress: the object's signed travel from its spawn y to the far
        # band's near edge; latched success keeps a later bump from erasing it.
        spawn_y = self._spawn_y if self._spawn_y is not None else sign * 0.25
        span = max(spawn_y * sign + self.other_side_min_y_m, 1e-6)
        travel = (spawn_y - pos[:, 1]) * sign
        reward = np.clip(travel / span, 0.0, 1.0).astype(np.float32)
        reward = np.where(success | self._ever_success, 1.0, np.minimum(reward, 0.99))
        reward = reward.astype(np.float32)

        return TaskEvalResult(
            reward=reward,
            success=success,
            metrics={
                "goal_configured": True,
                "start_side": "left" if sign > 0 else "right",
                "object_y": pos[:, 1].astype(np.float32),
                "object_z": pos[:, 2].astype(np.float32),
                "crossed_to_other_side": crossed,
                "object_placed": placed,
                "object_table_contact": table_contact,
                "object_gripper_contact": gripper_contact,
                "object_on_table": on_table,
                "ever_success": self._ever_success.copy(),
            },
        )
