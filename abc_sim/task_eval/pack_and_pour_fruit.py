"""Ordered packing, rim-held transport, pouring events, and settled catching."""

from __future__ import annotations

from copy import deepcopy

import mujoco
import numpy as np

from abc_sim.randomization.assets.fruit_bowls import BOWL_SEGMENTS
from abc_sim.task_eval.base import TaskEvalResult
from abc_sim.task_eval.contacts import subtree_geom_ids


class PackAndPourFruitEvaluator:
    """Observe every native physics tick to distinguish pouring from hand transfer."""

    evaluation_interval_steps = 1
    settling_seconds = 0.5
    transport_seconds = 0.2
    rim_grasp_seconds = 0.05
    rim_contact_gap_seconds = 0.02
    containment_tolerance_m = 0.001
    packing_reward = 0.3
    transport_reward = 0.3
    emptying_reward = 0.2
    catching_reward = 0.2
    gentle_fraction = 0.3

    def __init__(self, *, model, spec, force_allowance_n=15.0, force_dose_budget_ns=5.0):
        if (
            not np.isfinite([force_allowance_n, force_dose_budget_ns]).all()
            or force_allowance_n < 0
            or force_dose_budget_ns <= 0
        ):
            raise ValueError("Force allowance must be nonnegative and dose budget positive.")
        self.model, self.spec = model, spec
        self.force_dose_budget_ns = float(force_dose_budget_ns)
        self.force_entity_names = ("fruit_1", "fruit_2", "fruit_3", "source_bowl", "target_bowl")
        self._bodies = np.array([model.body(name).id for name in self.force_entity_names])
        self._dofs = np.array([model.jnt_dofadr[model.joint(f"{name}_joint").id] for name in self.force_entity_names])
        self._geoms = [subtree_geom_ids(model, body) for body in self._bodies]
        self._force_owner = {g: i for i, geoms in enumerate(self._geoms) for g in geoms}
        self._fruit_geoms = np.array([model.geom(f"fruit_{i}_collision").id for i in range(1, 4)])
        if np.any(model.geom_type[self._fruit_geoms] != mujoco.mjtGeom.mjGEOM_SPHERE):
            raise ValueError("Fruit-pouring observers require spherical fruit collision proxies.")
        self._radii = model.geom_size[self._fruit_geoms, 0].copy()
        self._table = model.geom("table_plane").id
        self._floors = [model.geom(f"{name}_floor").id for name in self.force_entity_names[3:]]
        self._bounds = []
        for name in self.force_entity_names[3:]:
            floor, rim = (model.site(f"{name}_cavity_{part}").id for part in ("floor", "rim"))
            self._bounds.append(
                (
                    float(model.site_pos[floor, 2]),
                    float(model.site_pos[rim, 2]),
                    float(model.site_size[floor, 0]),
                    float(model.site_size[rim, 0]),
                )
            )
        angles = np.arange(BOWL_SEGMENTS) * 2 * np.pi / BOWL_SEGMENTS
        self._normals_xy = np.column_stack((np.cos(angles), np.sin(angles)))
        self._robot = set().union(
            *(subtree_geom_ids(model, model.body(f"{side}_arm").id) for side in ("left", "right"))
        )
        self._fingers = {}
        for hand, side in enumerate(("left", "right")):
            for finger, suffix in enumerate(("left", "right")):
                for geom in subtree_geom_ids(model, model.body(f"{side}_link_{suffix}_finger").id):
                    self._fingers[geom] = (hand, finger)
        weights = model.body_subtreemass[self._bodies] * -model.opt.gravity[2]
        if not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError("Fruit pouring requires positive masses and downward gravity.")
        self._weights = weights
        self.force_threshold_n = np.array(
            [force_allowance_n] * 3 + list(force_allowance_n + weights[3:] + 2 * weights[:3].sum())
        )
        fruit_credit = (
            self.gentle_fraction * (self.packing_reward + (self.emptying_reward + self.catching_reward) / 2) / 3
        )
        self._gentle_credit = np.array(
            [fruit_credit] * 3
            + [
                self.gentle_fraction * (self.transport_reward + self.emptying_reward / 2),
                self.gentle_fraction * self.catching_reward / 2,
            ]
        )
        self.reset()

    def reset(self, *, nworld=1):
        if nworld != 1:
            raise NotImplementedError("Fruit pouring requires one native physics observer per world.")
        self._data = self._previous_time = self._result = None
        self._source_time = np.zeros(3)
        self._target_time = np.zeros(3)
        self._previous_source = np.zeros(3, dtype=bool)
        self._previous_target = np.zeros(3, dtype=bool)
        self._packed_stage = self._transport_complete = self._ever_success = False
        self._carried_filled = False
        self._transport_time = 0.0
        self._previous_transport = False
        self._grasp_times = np.zeros(2)
        self._previous_grasps = np.zeros(2, dtype=bool)
        self._rim_holds = np.zeros(2, dtype=bool)
        self._rim_gaps = np.zeros(2)
        self._armed = np.zeros(3, dtype=bool)
        self._emptied = np.zeros(3, dtype=bool)
        self._pour_valid = np.zeros(3, dtype=bool)
        self._previous_local = None
        self._force_dose_ns = np.zeros(5)
        self._peak_force_n = np.zeros(5)

    def after_reset(self, model, data):
        if model is not self.model:
            raise ValueError("Fruit-pouring observer model mismatch.")
        self._data, self._previous_time = data, float(data.time)
        self._initial_source_z = float(data.xpos[self._bodies[3], 2])
        self._observe(data, elapsed=0)

    def after_physics_step(self, model, data):
        if model is not self.model or data is not self._data:
            raise RuntimeError("Reset fruit-pouring tracking before observing physics.")
        elapsed = float(data.time) - self._previous_time
        if not np.isclose(elapsed, model.opt.timestep, rtol=1e-7, atol=1e-12):
            raise ValueError("Fruit pouring must observe every physics step exactly once.")
        self._previous_time = float(data.time)
        self._observe(data, elapsed=elapsed)

    def evaluate_model_data(self, model, data):
        if model is not self.model or data is not self._data or self._result is None:
            raise RuntimeError("Reset fruit-pouring tracking before reading its reward.")
        return deepcopy(self._result)

    def evaluate_qpos_batch(self, qpos_batch):
        raise NotImplementedError("Fruit pouring requires solved contacts and chronological physics.")

    def _containment(self, local, bowl):
        floor, rim, bottom_radius, rim_radius = self._bounds[bowl]
        slope = (rim_radius - bottom_radius) / (rim - floor)
        plane_distance = local[:, :2] @ self._normals_xy.T - slope * (local[:, 2:3] - floor)
        sphere_extent = self._radii[:, None] * np.sqrt(1 + slope * slope)
        return (
            np.all(plane_distance + sphere_extent <= bottom_radius + self.containment_tolerance_m, axis=1)
            & (local[:, 2] - self._radii >= floor - 0.002)
            & (local[:, 2] + self._radii <= rim + self.containment_tolerance_m)
        )

    def _observe(self, data, *, elapsed):
        model = self.model
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise ValueError("Fruit-pouring poses and velocities must be finite.")
        centers = data.geom_xpos[self._fruit_geoms]
        bowl_pos = data.xpos[self._bodies[3:]]
        bowl_rot = data.xmat[self._bodies[3:]].reshape(2, 3, 3)
        local = [(centers - bowl_pos[i]) @ bowl_rot[i] for i in range(2)]
        inside = np.array([self._containment(local[i], i) for i in range(2)])
        loads = np.zeros(5)
        touching_robot = np.zeros(5, dtype=bool)
        fruit_table = np.zeros(3, dtype=bool)
        bowl_table_load = np.zeros(2)
        nested_source_load = 0.0
        supported = np.zeros((2, 3), dtype=bool)
        supports = np.zeros((3, 3), dtype=bool)
        fingers = np.zeros((2, 2), dtype=bool)
        force = np.zeros(6)
        for c in range(data.ncon):
            contact = data.contact[c]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if g1 not in self._force_owner and g2 not in self._force_owner:
                continue
            mujoco.mj_contactForce(model, data, c, force)
            if not np.isfinite(force).all():
                raise ValueError("Fruit-pouring contact forces must be finite.")
            magnitude = float(np.linalg.norm(force[:3]))
            if contact.dist > 0 and magnitude == 0:
                continue
            world_force = contact.frame.reshape(3, 3).T @ force[:3]
            for mine, other, sign in ((g1, g2, -1), (g2, g1, 1)):
                owner = self._force_owner.get(mine)
                if owner is None:
                    continue
                other_owner = self._force_owner.get(other)
                if other_owner != owner:
                    loads[owner] += magnitude
                touching_robot[owner] |= other in self._robot
                if owner < 3:
                    fruit_table[owner] |= other == self._table and force[0] > 0.001
                    if sign * world_force[2] > 0.001:
                        if other_owner in (3, 4):
                            supported[other_owner - 3, owner] = True
                        elif other_owner is not None and other_owner < 3 and other_owner != owner:
                            supports[owner, other_owner] = True
                elif mine == self._floors[owner - 3] and other == self._table:
                    bowl_table_load[owner - 3] += sign * world_force[2]
                if owner == 3 and mine == self._floors[0] and other_owner == 4:
                    nested_source_load += sign * world_force[2]
                if owner == 3 and other in self._fingers and force[0] > 0.02:
                    point = (contact.pos - bowl_pos[0]) @ bowl_rot[0]
                    _, rim, _, rim_radius = self._bounds[0]
                    if abs(point[2] - rim) <= 0.015 and np.linalg.norm(point[:2]) >= rim_radius - 0.018:
                        fingers[self._fingers[other]] = True
        speeds = data.qvel[self._dofs[:, None] + np.arange(6)]
        linear_speed = np.linalg.norm(speeds[:, :3], axis=1)
        angular_speed = np.linalg.norm(speeds[:, 3:], axis=1)
        settled = (linear_speed <= 0.03) & (angular_speed <= 0.25)
        # Surface speed gives the same settling tolerance to every fruit size.
        settled[:3] = linear_speed[:3] + self._radii * angular_speed[:3] <= 0.03
        candidates = inside & settled[None, :3] & ~touching_robot[None, :3]
        supported &= candidates
        for _ in range(3):
            supported |= candidates & np.any(supports[None, :, :] & supported[:, None, :], axis=2)
        bowl_ok = (bowl_rot[:, 2, 2] >= 0.98) & settled[3:] & (bowl_table_load >= 0.6 * self._weights[3:])
        source_eligible = candidates[0] & supported[0] & bowl_ok[0]
        self._source_time = np.where(source_eligible & self._previous_source, self._source_time + elapsed, 0)
        self._previous_source = source_eligible
        source_qualified = source_eligible & (self._source_time >= self.settling_seconds)
        self._packed_stage |= bool(source_qualified.all())
        paired_contacts = fingers.all(axis=1)
        self._grasp_times = np.where(paired_contacts & self._previous_grasps, self._grasp_times + elapsed, 0)
        self._previous_grasps = paired_contacts
        self._rim_holds |= paired_contacts & (self._grasp_times >= self.rim_grasp_seconds)
        self._rim_gaps = np.where(fingers.any(axis=1), 0, self._rim_gaps + elapsed)
        # Once acquired, a grip can transfer load between opposing fingertips.
        # A short gap tolerates contact-manifold changes while the bowl rotates.
        self._rim_holds &= self._rim_gaps <= self.rim_contact_gap_seconds
        held = bool(self._rim_holds.any())
        source_lift = float(bowl_pos[0, 2] - self._initial_source_z)
        self._carried_filled |= bool(self._packed_stage and inside[0].all() and held and source_lift >= 0.05)
        target_rim_world = bowl_pos[1] + bowl_rot[1] @ np.array([0, 0, self._bounds[1][1]])
        overhead_distance = float(np.linalg.norm(bowl_pos[0, :2] - bowl_pos[1, :2]))
        above = (
            overhead_distance <= self._bounds[1][3] - self._bounds[0][3] - 0.005
            and bowl_pos[0, 2] >= target_rim_world[2] + 0.025
        )
        source_nested = (
            overhead_distance <= self._bounds[1][3] - self._bounds[0][3] - 0.005
            and bowl_pos[0, 2] >= bowl_pos[1, 2] + self._bounds[1][0] - 0.002
            and nested_source_load >= 0.6 * self._weights[3]
            and bowl_rot[0, 2, 2] >= 0.98
            and settled[3]
            and bowl_ok[1]
        )
        transport = (
            self._carried_filled
            and inside[0].all()
            and (
                source_nested
                or (
                    held
                    and above
                    and bowl_rot[0, 2, 2] >= 0.9
                    and np.linalg.norm(speeds[3, :3]) <= 0.08
                    and np.linalg.norm(speeds[3, 3:]) <= 0.5
                )
            )
        )
        self._transport_time = self._transport_time + elapsed if transport and self._previous_transport else 0.0
        self._previous_transport = bool(transport)
        self._transport_complete |= bool(transport and self._transport_time >= self.transport_seconds)
        self._armed |= source_qualified & self._packed_stage
        self._armed &= ~touching_robot[:3] & ~fruit_table
        self._emptied &= ~inside[0]
        self._pour_valid &= ~inside[0] & ~touching_robot[:3] & ~fruit_table
        tilt_deg = float(np.rad2deg(np.arccos(np.clip(bowl_rot[0, 2, 2], -1, 1))))
        bottom = local[0][:, 2] - self._radii
        rim = self._bounds[0][1]
        if self._previous_local is not None:
            crossed = (self._previous_local[:, 2] - self._radii < rim) & (bottom >= rim)
            opening = np.linalg.norm(local[0][:, :2], axis=1) <= self._bounds[0][3] + self._radii + 0.005
            departure = (
                crossed
                & opening
                & self._armed
                & ~touching_robot[:3]
                & self._transport_complete
                & held
                & (source_lift >= 0.05)
                & (tilt_deg >= 45)
            )
            self._emptied |= departure
            self._pour_valid |= departure & ~fruit_table
            self._armed &= ~crossed
        self._armed &= (np.linalg.norm(local[0][:, :2], axis=1) <= self._bounds[0][3] + self._radii + 0.01) & (
            local[0][:, 2] + self._radii >= self._bounds[0][0] - 0.002
        )
        self._previous_local = local[0].copy()
        target_eligible = candidates[1] & supported[1] & bowl_ok[1] & ~touching_robot[4] & self._pour_valid
        self._target_time = np.where(target_eligible & self._previous_target, self._target_time + elapsed, 0)
        self._previous_target = target_eligible
        caught = target_eligible & (self._target_time >= self.settling_seconds)
        excess = np.maximum(loads - self.force_threshold_n, 0)
        self._force_dose_ns += excess * elapsed
        self._peak_force_n = np.maximum(self._peak_force_n, loads)
        damage = np.minimum(self._force_dose_ns / self.force_dose_budget_ns, 1)
        packing = (
            (source_qualified | self._packed_stage) * self.packing_reward / 3 * (1 - self.gentle_fraction * damage[:3])
        )
        transporting = self._transport_complete * self.transport_reward * (1 - self.gentle_fraction * damage[3])
        emptying = self._emptied * self.emptying_reward / 3 * (1 - self.gentle_fraction / 2 * (damage[:3] + damage[3]))
        catching = caught * self.catching_reward / 3 * (1 - self.gentle_fraction / 2 * (damage[:3] + damage[4]))
        success = bool(self._transport_complete and caught.all())
        self._ever_success |= success
        self._result = TaskEvalResult(
            reward=np.array([packing.sum() + transporting + emptying.sum() + catching.sum()]),
            success=np.array([success]),
            metrics={
                "packing_reward": np.array([packing.sum()]),
                "transport_reward": np.array([transporting]),
                "emptying_reward": np.array([emptying.sum()]),
                "catching_reward": np.array([catching.sum()]),
                "fruit_packing_reward": packing[None],
                "fruit_emptying_reward": emptying[None],
                "fruit_catching_reward": catching[None],
                "packing_stage_complete": np.array([self._packed_stage]),
                "transport_complete": np.array([self._transport_complete]),
                "filled_bowl_carried": np.array([self._carried_filled]),
                "source_nested_in_target": np.array([source_nested]),
                "num_packed_fruits": np.array([int(source_qualified.sum())]),
                "num_emptied_fruits": np.array([int(self._emptied.sum())]),
                "num_caught_fruits": np.array([int(caught.sum())]),
                "fruit_in_source_mask": inside[0:1],
                "fruit_in_target_mask": inside[1:2],
                "fruit_source_supported_mask": supported[0:1],
                "fruit_target_supported_mask": supported[1:2],
                "fruit_armed_mask": self._armed[None].copy(),
                "fruit_emptied_mask": self._emptied[None].copy(),
                "fruit_valid_pour_mask": self._pour_valid[None].copy(),
                "fruit_caught_mask": caught[None],
                "rim_grasped": np.array([bool(paired_contacts.any())]),
                "rim_held": np.array([held]),
                "rim_grasp_seconds": np.array([self._grasp_times.max()]),
                "transport_settling_seconds": np.array([self._transport_time]),
                "source_lift_m": np.array([source_lift]),
                "source_tilt_deg": np.array([tilt_deg]),
                "source_over_target": np.array([above]),
                "source_target_distance_xy_m": np.array([overhead_distance]),
                "touching_robot": touching_robot[None],
                "contact_force_n": loads[None],
                "peak_force_n": self._peak_force_n[None].copy(),
                "force_threshold_n": self.force_threshold_n[None].copy(),
                "excess_force_n": excess[None],
                "force_dose_ns": self._force_dose_ns[None].copy(),
                "force_penalty": (self._gentle_credit * damage)[None],
                "ever_success": np.array([self._ever_success]),
            },
        )
