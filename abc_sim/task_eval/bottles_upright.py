"""Released, supported and settled upright-bottle scoring."""

from __future__ import annotations

from copy import deepcopy

import mujoco
import numpy as np

from abc_sim.task_eval.base import TaskEvalResult
from abc_sim.task_eval.bottles import _resolve_bottle_qpos_addrs
from abc_sim.task_eval.contacts import subtree_geom_ids
from abc_sim.task_specs import SimTaskSpec


class BottlesUprightEvaluator:
    """Credit bottles standing independently on the tabletop for 0.5 seconds.

    Every solved native physics tick must satisfy orientation, base height,
    tabletop load, release, and velocity checks. Any interruption clears that
    bottle's dwell and current credit. Reward is the current qualified fraction;
    success requires all three simultaneously. Evaluation reads are idempotent.
    Qpos-only and MJWarp replay scoring cannot supply the required contact history.
    """

    evaluation_interval_steps = 1
    settling_seconds = 0.5

    def __init__(self, *, model: mujoco.MjModel, spec: SimTaskSpec) -> None:
        self.model = model
        self.spec = spec
        self.bottle_names, self._addrs = _resolve_bottle_qpos_addrs(model)
        if self.bottle_names != ["bottle_1", "bottle_2", "bottle_3"]:
            raise ValueError("Upright bottles requires exactly bottle_1, bottle_2, bottle_3.")
        bodies = np.array([model.body(name).id for name in self.bottle_names])
        joints = model.body_jntadr[bodies]
        self._dofs = model.jnt_dofadr[joints]
        self._owner = {geom: index for index, body in enumerate(bodies) for geom in subtree_geom_ids(model, int(body))}
        self._table = model.geom("table_plane").id
        self._weights = model.body_subtreemass[bodies] * -model.opt.gravity[2]
        if not np.isfinite(self._weights).all() or np.any(self._weights <= 0):
            raise ValueError("Upright bottles require positive weight under downward gravity.")
        self.reset()

    def reset(self, *, nworld: int = 1) -> None:
        if nworld != 1:
            raise NotImplementedError("Upright-bottle scoring requires one native physics observer per world.")
        self._data = None
        self._previous_time = None
        self._sustained = np.zeros(3)
        self._previous_eligible = np.zeros(3, dtype=bool)
        self._max_upright = 0
        self._ever_success = False
        self._result = None

    def after_reset(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if model is not self.model:
            raise ValueError("Upright-bottle observations must use the evaluator's model.")
        self._data = data
        self._previous_time = float(data.time)
        self._observe(data, elapsed=0.0)

    def after_physics_step(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if model is not self.model or data is not self._data:
            raise RuntimeError("Reset upright-bottle tracking before observing physics.")
        elapsed = float(data.time) - self._previous_time
        if not np.isclose(elapsed, model.opt.timestep, rtol=1e-7, atol=1e-12):
            raise ValueError("Upright-bottle tracking must observe every physics step exactly once.")
        self._previous_time = float(data.time)
        self._observe(data, elapsed=elapsed)

    def evaluate_model_data(self, model: mujoco.MjModel, data: mujoco.MjData) -> TaskEvalResult:
        if model is not self.model or data is not self._data or self._result is None:
            raise RuntimeError("Reset upright-bottle tracking before reading its score.")
        return deepcopy(self._result)

    def evaluate_qpos_batch(self, qpos_batch: np.ndarray) -> TaskEvalResult:
        raise NotImplementedError("Upright-bottle scoring requires solved contacts, velocities and every physics tick.")

    def _observe(self, data: mujoco.MjData, *, elapsed: float) -> None:
        poses = data.qpos[self._addrs[:, None] + np.arange(7)]
        velocities = data.qvel[self._dofs[:, None] + np.arange(6)]
        if not np.isfinite(poses).all() or not np.isfinite(velocities).all():
            raise ValueError("Upright-bottle poses and velocities must be finite.")
        x, y, z = poses[:, :3].T
        quat = poses[:, 3:]
        upright_z = 1 - 2 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
        pose_ok = (
            (upright_z >= 0.97) & (np.abs(z - 0.75) <= 0.008) & (x >= 0.34) & (x <= 0.84) & (y >= -0.53) & (y <= 0.53)
        )
        table_load = np.zeros(3)
        touching_other = np.zeros(3, dtype=bool)
        force = np.zeros(6)
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            if int(contact.geom1) not in self._owner and int(contact.geom2) not in self._owner:
                continue
            mujoco.mj_contactForce(self.model, data, contact_index, force)
            if not np.isfinite(force).all():
                raise ValueError("Upright-bottle contact forces must be finite.")
            if contact.dist > 0 and force[0] == 0:
                continue
            for mine, other, sign in (
                (int(contact.geom1), int(contact.geom2), -1),
                (int(contact.geom2), int(contact.geom1), 1),
            ):
                if mine not in self._owner:
                    continue
                index = self._owner[mine]
                if other == self._table:
                    # MuJoCo's contact frame maps geom1 -> geom2. Only upward
                    # support from the tabletop at the bottle's base counts.
                    normal_up = sign * contact.frame[2]
                    if normal_up >= 0.9 and abs(contact.pos[2] - 0.75) <= 0.008:
                        table_load[index] += force[0] * normal_up
                elif self._owner.get(other) != index:
                    touching_other[index] = True
        linear_speed = np.linalg.norm(velocities[:, :3], axis=1)
        angular_speed = np.linalg.norm(velocities[:, 3:], axis=1)
        weight_fraction = table_load / self._weights
        eligible = (
            pose_ok & (weight_fraction >= 0.8) & ~touching_other & (linear_speed <= 0.03) & (angular_speed <= 0.25)
        )
        self._sustained = np.where(eligible & self._previous_eligible, self._sustained + elapsed, 0.0)
        self._previous_eligible = eligible
        upright = eligible & (self._sustained >= self.settling_seconds)
        count = int(upright.sum())
        success = count == 3
        self._max_upright = max(self._max_upright, count)
        self._ever_success |= success
        self._result = TaskEvalResult(
            reward=np.array([count / 3]),
            success=np.array([success]),
            metrics={
                "num_upright_bottles": np.array([count]),
                "num_active_bottles": np.array([3]),
                "bottle_upright_mask": upright[None, :],
                "bottle_upright_z": upright_z[None, :],
                "bottle_pose_ok": pose_ok[None, :],
                "bottle_table_weight_fraction": weight_fraction[None, :],
                "bottle_touching_other": touching_other[None, :],
                "bottle_linear_speed": linear_speed[None, :],
                "bottle_angular_speed": angular_speed[None, :],
                "bottle_sustained_seconds": self._sustained[None, :].copy(),
                "max_upright_bottles_so_far": np.array([self._max_upright]),
                "ever_success": np.array([self._ever_success]),
            },
        )
