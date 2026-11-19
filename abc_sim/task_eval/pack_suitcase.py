"""Physical containment, release, settling, and lid-closure scoring."""

from __future__ import annotations

from copy import deepcopy
from itertools import product

import mujoco
import numpy as np

from abc_sim.randomization.assets.suitcase import LID_LIMIT_RAD
from abc_sim.task_eval.base import TaskEvalResult
from abc_sim.task_eval.contacts import subtree_geom_ids
from abc_sim.task_specs import SimTaskSpec


class PackSuitcaseEvaluator:
    """Give 0.3 for packing, 0.4 for closure, and 0.3 for preserving case pose.

    Full object collision bounds must lie inside the cavity, with a solved
    support path to its floor, no robot contact, and low velocity for 0.5 s.
    Once packed, an object keeps its credit while its full bounds remain
    inside, including during case motion. Leaving the cavity clears that
    qualification. Closure requires a lid angle within 3 degrees, release
    of the whole case and contents, and another 0.5 s of settled physics.
    Case pose credit has a 5 cm / 5 degree deadband, then falls linearly to
    zero at 20 cm or 45 degrees from the reset pose. The worse fraction wins.
    Pose credit is available only while the lid is within 3 degrees of closed.
    Thirty percent of packing and closure reflects gentle handling. Each entity's
    cumulative external contact load above its allowance irreversibly spends
    that credit over a 5 N*s dose budget. History resets only with the episode.
    """

    evaluation_interval_steps = 1
    settling_seconds = 0.5
    packing_reward = 0.3
    closure_reward = 0.4
    staying_reward = 0.3
    position_tolerance_m = 0.05
    position_limit_m = 0.2
    orientation_tolerance_deg = 5.0
    orientation_limit_deg = 45.0
    lid_closed_tolerance_deg = 3.0
    gentle_fraction = 0.3

    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        spec: SimTaskSpec,
        object_force_allowance_n: float = 15.0,
        case_extra_force_allowance_n: float = 15.0,
        force_dose_budget_ns: float = 5.0,
    ):
        self.model, self.spec = model, spec
        if not (
            np.isfinite([object_force_allowance_n, case_extra_force_allowance_n, force_dose_budget_ns]).all()
            and object_force_allowance_n >= 0
            and case_extra_force_allowance_n >= 0
            and force_dose_budget_ns > 0
        ):
            raise ValueError("Force allowances must be finite and nonnegative; the dose budget must be positive.")
        self.force_dose_budget_ns = float(force_dose_budget_ns)
        self.force_entity_names = ("packing_object_1", "packing_object_2", "packing_object_3", "suitcase")
        self._case = model.body("suitcase").id
        self._case_geoms = subtree_geom_ids(model, self._case)
        self._floor = model.geom("suitcase_floor").id
        front, back, right, left = [model.geom(f"suitcase_{name}").id for name in ("front", "back", "right", "left")]
        self._cavity_lower = np.array(
            [
                model.geom_pos[front, 0] + model.geom_size[front, 0],
                model.geom_pos[right, 1] + model.geom_size[right, 1],
                model.geom_pos[self._floor, 2] + model.geom_size[self._floor, 2],
            ]
        )
        self._cavity_upper = np.array(
            [
                model.geom_pos[back, 0] - model.geom_size[back, 0],
                model.geom_pos[left, 1] - model.geom_size[left, 1],
                min(model.geom_pos[g, 2] + model.geom_size[g, 2] for g in (front, back, right, left)),
            ]
        )
        self._table = model.geom("table_plane").id
        self._hinge = model.joint("suitcase_hinge").id
        self._case_dof = int(model.jnt_dofadr[model.joint("suitcase_joint").id])
        self._bodies = [model.body(f"packing_object_{i}").id for i in range(1, 4)]
        self._dofs = np.array([model.jnt_dofadr[model.body_jntadr[body]] for body in self._bodies])
        self._geoms = [subtree_geom_ids(model, body) for body in self._bodies]
        self._owner = {geom: i for i, geoms in enumerate(self._geoms) for geom in geoms}
        self._force_owner = {**self._owner, **dict.fromkeys(self._case_geoms, 3)}
        weights = model.body_subtreemass[[*self._bodies, self._case]] * -model.opt.gravity[2]
        if not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError("Suitcase handling requires positive masses and downward gravity.")
        # Contents press down on the case and increase the table reaction.
        # Budget for all three even before packing, keeping thresholds fixed
        # throughout the episode and independent of reward qualification.
        self.case_gravity_allowance_n = float(weights[3] + 2 * weights[:3].sum())
        self.force_threshold_n = np.array(
            [object_force_allowance_n] * 3 + [self.case_gravity_allowance_n + case_extra_force_allowance_n]
        )
        self._gentle_credit = self.gentle_fraction * np.array([self.packing_reward / 3] * 3 + [self.closure_reward])
        self._robot = set().union(*(subtree_geom_ids(model, model.body(name).id) for name in ("left_arm", "right_arm")))
        self._bounds_geoms = []
        for geoms in self._geoms:
            collision = [g for g in sorted(geoms) if model.geom_contype[g] or model.geom_conaffinity[g]]
            if not collision or any(model.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX for g in collision):
                raise ValueError("Packing objects require enclosing box collision proxies.")
            self._bounds_geoms.append(collision)
        self._corners = np.array(list(product((-1, 1), repeat=3)))
        self.reset()

    def reset(self, *, nworld=1):
        if nworld != 1:
            raise NotImplementedError("Suitcase scoring requires one native physics observer per world.")
        self._data = None
        self._previous_time = None
        self._initial_case_pos = None
        self._initial_case_quat = None
        self._packed_time = np.zeros(3)
        self._packed = np.zeros(3, dtype=bool)
        self._previous_eligible = np.zeros(3, dtype=bool)
        self._closed_time = 0.0
        self._previous_closed = False
        self._ever_success = False
        self._force_dose_ns = np.zeros(4)
        self._peak_force_n = np.zeros(4)
        self._result = None

    def after_reset(self, model, data):
        if model is not self.model:
            raise ValueError("Suitcase observer model mismatch.")
        self._data, self._previous_time = data, float(data.time)
        self._initial_case_pos = data.xpos[self._case].copy()
        self._initial_case_quat = data.xquat[self._case].copy()
        self._observe(data, elapsed=0)

    def after_physics_step(self, model, data):
        if model is not self.model or data is not self._data:
            raise RuntimeError("Reset suitcase tracking before observing physics.")
        elapsed = float(data.time) - self._previous_time
        if not np.isclose(elapsed, model.opt.timestep, rtol=1e-7, atol=1e-12):
            raise ValueError("Suitcase tracking must observe every physics step exactly once.")
        self._previous_time = float(data.time)
        self._observe(data, elapsed=elapsed)

    def evaluate_model_data(self, model, data):
        if model is not self.model or data is not self._data or self._result is None:
            raise RuntimeError("Reset suitcase tracking before reading its score.")
        return deepcopy(self._result)

    def evaluate_qpos_batch(self, qpos_batch):
        raise NotImplementedError("Suitcase scoring requires solved contacts, velocities and every physics tick.")

    def _observe(self, data, *, elapsed):
        model = self.model
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise ValueError("Suitcase scene poses and velocities must be finite.")
        case_pos = data.xpos[self._case]
        case_rot = data.xmat[self._case].reshape(3, 3)
        translation_m = float(np.linalg.norm(case_pos - self._initial_case_pos))
        rotation_vector = np.empty(3)
        mujoco.mju_subQuat(rotation_vector, data.xquat[self._case], self._initial_case_quat)
        rotation_deg = float(np.rad2deg(np.linalg.norm(rotation_vector)))
        translation_fraction = float(
            np.clip(
                (self.position_limit_m - translation_m) / (self.position_limit_m - self.position_tolerance_m),
                0,
                1,
            )
        )
        rotation_fraction = float(
            np.clip(
                (self.orientation_limit_deg - rotation_deg)
                / (self.orientation_limit_deg - self.orientation_tolerance_deg),
                0,
                1,
            )
        )
        staying_fraction = min(translation_fraction, rotation_fraction)
        inside = np.zeros(3, dtype=bool)
        for i, geoms in enumerate(self._bounds_geoms):
            points = np.concatenate(
                [
                    (
                        (self._corners * model.geom_size[g]) @ data.geom_xmat[g].reshape(3, 3).T
                        + data.geom_xpos[g]
                        - case_pos
                    )
                    @ case_rot
                    for g in geoms
                ]
            )
            lower, upper = points.min(axis=0), points.max(axis=0)
            inside[i] = (
                np.all(lower[:2] >= self._cavity_lower[:2] + 0.001)
                and np.all(upper[:2] <= self._cavity_upper[:2] - 0.001)
                and lower[2] >= self._cavity_lower[2] - 0.002
                and upper[2] <= self._cavity_upper[2] - 0.002
            )
        touching_robot = np.zeros(3, dtype=bool)
        supported = np.zeros(3, dtype=bool)
        supports = np.zeros((3, 3), dtype=bool)
        case_touching_robot = False
        case_table_load = 0.0
        contact_force_n = np.zeros(4)
        force = np.zeros(6)
        for c in range(data.ncon):
            contact = data.contact[c]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if (
                g1 not in self._owner
                and g2 not in self._owner
                and g1 not in self._case_geoms
                and g2 not in self._case_geoms
            ):
                continue
            mujoco.mj_contactForce(model, data, c, force)
            if not np.isfinite(force).all():
                raise ValueError("Suitcase contact loads must be finite.")
            magnitude = np.linalg.norm(force[:3])
            if contact.dist > 0 and magnitude == 0:
                continue
            for mine, other, sign in ((g1, g2, -1), (g2, g1, 1)):
                if mine in self._force_owner and self._force_owner.get(other) != self._force_owner[mine]:
                    contact_force_n[self._force_owner[mine]] += magnitude
                upward = sign * contact.frame[2]
                if mine in self._case_geoms:
                    case_touching_robot |= other in self._robot
                    if mine == self._floor and other == self._table and upward > 0.9:
                        case_table_load += force[0] * upward
                if mine in self._owner:
                    i = self._owner[mine]
                    touching_robot[i] |= other in self._robot
                    if force[0] > 0.001 and upward > 0.5:
                        supported[i] |= other == self._floor
                        if other in self._owner and self._owner[other] != i:
                            supports[i, self._owner[other]] = True
        speeds = data.qvel[self._dofs[:, None] + np.arange(6)]
        settled = (np.linalg.norm(speeds[:, :3], axis=1) <= 0.03) & (np.linalg.norm(speeds[:, 3:], axis=1) <= 0.25)
        candidates = inside & settled & ~touching_robot
        supported &= candidates
        for _ in range(3):
            supported |= candidates & np.any(supports & supported[None, :], axis=1)
        case_speed = data.qvel[self._case_dof : self._case_dof + 6]
        case_ok = (
            case_rot[2, 2] >= 0.98
            and abs(case_pos[2] - 0.75) <= 0.008
            and case_table_load >= 0.8 * model.body_subtreemass[self._case] * -model.opt.gravity[2]
            and np.linalg.norm(case_speed[:3]) <= 0.03
            and np.linalg.norm(case_speed[3:]) <= 0.25
        )
        eligible = candidates & supported & case_ok
        self._packed_time = np.where(eligible & self._previous_eligible, self._packed_time + elapsed, 0)
        self._previous_eligible = eligible
        self._packed = inside & (self._packed | (eligible & (self._packed_time >= self.settling_seconds)))
        packed = self._packed
        angle = float(data.qpos[model.jnt_qposadr[self._hinge]])
        hinge_speed = float(data.qvel[model.jnt_dofadr[self._hinge]])
        lid_is_closed = abs(angle) <= np.deg2rad(self.lid_closed_tolerance_deg)
        staying_reward = lid_is_closed * self.staying_reward * staying_fraction
        closed = bool(
            packed.all() and eligible.all() and not case_touching_robot and lid_is_closed and abs(hinge_speed) <= 0.1
        )
        self._closed_time = self._closed_time + elapsed if closed and self._previous_closed else 0.0
        self._previous_closed = closed
        closure_qualified = closed and self._closed_time >= self.settling_seconds
        success = closure_qualified and np.isclose(staying_fraction, 1.0, rtol=0, atol=1e-12)
        self._ever_success |= success
        count = int(packed.sum())
        excess_force_n = np.maximum(contact_force_n - self.force_threshold_n, 0)
        self._force_dose_ns += excess_force_n * elapsed
        self._peak_force_n = np.maximum(self._peak_force_n, contact_force_n)
        penalty = self._gentle_credit * np.minimum(self._force_dose_ns / self.force_dose_budget_ns, 1)
        object_reward = packed * (self.packing_reward / 3 - penalty[:3])
        lid_reward = closure_qualified * (self.closure_reward - penalty[3])
        self._result = TaskEvalResult(
            reward=np.array([object_reward.sum() + lid_reward + staying_reward]),
            success=np.array([success]),
            metrics={
                "num_packed_objects": np.array([count]),
                "object_reward": object_reward[None, :],
                "packing_reward": np.array([object_reward.sum()]),
                "lid_reward": np.array([lid_reward]),
                "suitcase_staying_reward": np.array([staying_reward]),
                "suitcase_staying_fraction": np.array([staying_fraction]),
                "suitcase_translation_m": np.array([translation_m]),
                "suitcase_rotation_deg": np.array([rotation_deg]),
                "suitcase_translation_fraction": np.array([translation_fraction]),
                "suitcase_rotation_fraction": np.array([rotation_fraction]),
                "lid_closed_and_settled": np.array([closure_qualified]),
                "lid_is_closed": np.array([lid_is_closed]),
                "contact_force_n": contact_force_n[None, :],
                "peak_force_n": self._peak_force_n[None, :].copy(),
                "force_threshold_n": self.force_threshold_n[None, :].copy(),
                "excess_force_n": excess_force_n[None, :],
                "force_dose_ns": self._force_dose_ns[None, :].copy(),
                "force_penalty": penalty[None, :],
                "object_contained_mask": inside[None, :],
                "object_supported_mask": supported[None, :],
                "object_touching_robot": touching_robot[None, :],
                "object_packed_mask": packed[None, :],
                "object_settling_seconds": self._packed_time[None, :].copy(),
                "suitcase_supported_and_settled": np.array([case_ok]),
                "suitcase_touching_robot": np.array([case_touching_robot]),
                "lid_angle_deg": np.array([np.rad2deg(angle)]),
                "lid_closure_fraction": np.array([np.clip(1 - angle / LID_LIMIT_RAD, 0, 1)]),
                "closed_settling_seconds": np.array([self._closed_time]),
                "ever_success": np.array([self._ever_success]),
            },
        )
