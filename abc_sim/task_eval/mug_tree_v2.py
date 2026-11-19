"""Contact-qualified mug hanging with full handle threading and release dwell."""

from __future__ import annotations

from copy import deepcopy

import mujoco
import numpy as np

from abc_sim.task_eval.base import TaskEvalResult
from abc_sim.task_eval.contacts import subtree_geom_ids
from abc_sim.task_eval.mug_hanging_geometry import (
    measure_handle_aperture,
    measure_rack_pegs,
)
from abc_sim.task_eval.mug_tree import MugOnRackEvaluator


class MugOnRackV2Evaluator:
    """Score each mug after its threaded, released hang stays stable for 0.5 s."""

    evaluation_interval_steps = 1
    revision = "abc-mug-hanging-v2"
    settling_seconds = 0.5
    max_linear_speed = 0.02
    max_angular_speed = 0.2
    minimum_weight_fraction = 0.5
    max_penetration_m = 0.002
    robot_force_tolerance_n = 1e-6

    def __init__(self, *, model, spec):
        self.model, self.spec = model, spec
        self.mug_names, _ = MugOnRackEvaluator._resolve_mug_qpos_addrs(model, r"mug_(\d+)_jnt")
        self._bodies = np.array([model.body(name).id for name in self.mug_names])
        self._tree = model.body("mug_tree").id
        self._tree_geoms = subtree_geom_ids(model, self._tree)
        self._owner = {
            geom: index for index, body in enumerate(self._bodies) for geom in subtree_geom_ids(model, int(body))
        }
        self._robot = set().union(*(subtree_geom_ids(model, model.body(name).id) for name in ("left_arm", "right_arm")))
        gravity = np.asarray(model.opt.gravity)
        gravity_magnitude = np.linalg.norm(gravity)
        self._weights = model.body_subtreemass[self._bodies] * gravity_magnitude
        if not np.isfinite(self._weights).all() or np.any(self._weights <= 0):
            raise ValueError("Mug hanging requires gravity and positive mug weights.")
        self._up = -gravity / gravity_magnitude
        geometry_data = mujoco.MjData(model)
        mujoco.mj_forward(model, geometry_data)
        self._pegs = measure_rack_pegs(model, geometry_data, self._tree)
        if not self._pegs or len({peg.geom_id for peg in self._pegs}) != len(self._pegs):
            raise ValueError("Mug hanging requires distinct rack peg collision geometries.")
        if any(peg.geom_id not in self._tree_geoms for peg in self._pegs):
            raise ValueError("Every measured peg must belong to the rack.")
        self._peg_indices = {peg.geom_id: index for index, peg in enumerate(self._pegs)}
        self._apertures = [measure_handle_aperture(model, geometry_data, int(body)) for body in self._bodies]
        self.reset()

    def reset(self, *, nworld=1):
        if nworld != 1:
            raise NotImplementedError("Mug hanging v2 requires a native physics observer per world.")
        self._data = self._previous_time = self._result = None
        self._settling = np.zeros((len(self._bodies), len(self._pegs)))
        self._previous_eligible = np.zeros_like(self._settling, dtype=bool)
        self._max_hanging = 0
        self._ever_success = False

    def configure_from_randomization(self, model, randomization):
        if model is not self.model:
            raise ValueError("Mug hanging evaluator must bind the realized model.")

    def after_reset(self, model, data):
        if model is not self.model:
            raise ValueError("Mug hanging observer model mismatch.")
        self._data, self._previous_time = data, float(data.time)
        self._observe(data, 0.0)

    def after_physics_step(self, model, data):
        if model is not self.model or data is not self._data:
            raise RuntimeError("Reset mug hanging tracking before observing physics.")
        elapsed = float(data.time) - self._previous_time
        if not np.isclose(elapsed, model.opt.timestep, rtol=1e-7, atol=1e-12):
            raise ValueError("Mug hanging v2 must observe every physics tick exactly once.")
        self._previous_time = float(data.time)
        self._observe(data, elapsed)

    def evaluate_model_data(self, model, data):
        if model is not self.model or data is not self._data or self._result is None:
            raise RuntimeError("Reset mug hanging tracking before reading its score.")
        return deepcopy(self._result)

    def evaluate_qpos_batch(self, qpos_batch):
        raise NotImplementedError("Mug hanging v2 requires solved contacts, release and continuous settling.")

    def _observe(self, data, elapsed):
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise ValueError("Mug hanging poses and velocities must be finite.")
        tree_rotation = data.xmat[self._tree].reshape(3, 3)
        world_pegs = [peg.endpoints @ tree_rotation.T + data.xpos[self._tree] for peg in self._pegs]
        world_vertices = [peg.vertices @ tree_rotation.T + data.xpos[self._tree] for peg in self._pegs]
        threaded = np.zeros_like(self._previous_eligible)
        for index, (body, aperture) in enumerate(zip(self._bodies, self._apertures, strict=True)):
            rotation = data.xmat[body].reshape(3, 3)
            for peg_index, (peg, endpoints) in enumerate(zip(self._pegs, world_pegs, strict=True)):
                threaded[index, peg_index] = aperture.threaded_by(
                    (endpoints - data.xpos[body]) @ rotation,
                    peg.radius,
                    vertices=(world_vertices[peg_index] - data.xpos[body]) @ rotation,
                    edges=peg.edges,
                )

        touching = np.zeros(len(self._bodies), dtype=bool)
        tree_touching = False
        support = np.zeros_like(self._settling)
        penetration = np.zeros(len(self._bodies))
        force = np.empty(6)
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if not (g1 in self._owner or g2 in self._owner or g1 in self._tree_geoms or g2 in self._tree_geoms):
                continue
            mujoco.mj_contactForce(self.model, data, contact_index, force)
            if not np.isfinite(force).all():
                raise ValueError("Mug hanging contact loads must be finite.")
            physical_contact = contact.dist <= 0 or np.linalg.norm(force[:3]) > self.robot_force_tolerance_n
            for mine, other, sign in ((g1, g2, -1), (g2, g1, 1)):
                if mine in self._tree_geoms and other in self._robot:
                    tree_touching |= physical_contact
                if mine not in self._owner:
                    continue
                owner = self._owner[mine]
                touching[owner] |= other in self._robot and physical_contact
                if other in self._tree_geoms:
                    penetration[owner] = max(penetration[owner], -float(contact.dist))
                if other in self._peg_indices:
                    support[owner, self._peg_indices[other]] += sign * np.dot(
                        force[:3], contact.frame.reshape(3, 3) @ self._up
                    )

        velocity = np.empty(6)
        stable = np.zeros(len(self._bodies) + 1, dtype=bool)
        for index, body in enumerate([*self._bodies, self._tree]):
            mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_BODY, int(body), velocity, 0)
            stable[index] = (
                np.linalg.norm(velocity[:3]) <= self.max_angular_speed
                and np.linalg.norm(velocity[3:]) <= self.max_linear_speed
            )
        released = ~touching & (not tree_touching)
        eligible = (
            threaded
            & (support >= self.minimum_weight_fraction * self._weights[:, None])
            & released[:, None]
            & stable[:-1, None]
            & stable[-1]
            & (penetration[:, None] <= self.max_penetration_m)
        )
        self._settling = np.where(eligible & self._previous_eligible, self._settling + elapsed, 0.0)
        self._previous_eligible = eligible
        qualified_pegs = eligible & (self._settling >= self.settling_seconds)
        hanging = qualified_pegs.any(axis=1)
        num_hanging = int(hanging.sum())
        success = num_hanging == len(self._bodies)
        self._max_hanging = max(self._max_hanging, num_hanging)
        self._ever_success |= success
        self._result = TaskEvalResult(
            reward=np.array([num_hanging / len(self._bodies)]),
            success=np.array([success]),
            metrics={
                "evaluator_revision": self.revision,
                "num_mugs_hanging": np.array([num_hanging]),
                "num_active_mugs": np.array([len(self._bodies)]),
                "max_mugs_hanging_so_far": np.array([self._max_hanging]),
                "mugs_hanging": [[name for name, qualified in zip(self.mug_names, hanging, strict=True) if qualified]],
                "mug_names": self.mug_names.copy(),
                "mug_threaded": threaded.any(axis=1)[None, :],
                "mug_threaded_pegs": threaded[None, :, :],
                "mug_qualified_pegs": qualified_pegs[None, :, :],
                "mug_touching_robot": touching[None, :],
                "rack_touching_robot": np.array([tree_touching]),
                "mug_released": released[None, :],
                "mug_stable": stable[None, :-1],
                "rack_stable": stable[None, -1],
                "mug_peg_support_n": support[None, :, :],
                "mug_rack_penetration_m": penetration[None, :],
                "mug_settling_seconds": self._settling.max(axis=1)[None, :],
                "mug_peg_settling_seconds": self._settling[None, :, :].copy(),
                "ever_success": np.array([self._ever_success]),
            },
        )
