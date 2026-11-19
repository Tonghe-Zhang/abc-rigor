"""Ordered packing, grasped transport, pouring events, and settled catching."""

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
    transport_seconds = 0.1
    pour_tilt_degrees = 30.0
    grasp_seconds = 0.05
    contact_gap_seconds = 0.02
    fruit_rearm_seconds = 0.05
    containment_tolerance_m = 0.001
    packing_reward = 0.2
    transport_reward = 0.2
    emptying_reward = 0.25
    catching_reward = 0.25
    return_reward = 0.1
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
        if model.geom_type[self._floors[0]] != mujoco.mjtGeom.mjGEOM_CYLINDER:
            raise ValueError("Fruit pouring requires a cylindrical source-bowl floor.")
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
                self.gentle_fraction * (self.transport_reward + self.emptying_reward / 2 + self.return_reward),
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
        self._released_support_time = np.zeros(3)
        self._previous_released_support = np.zeros(3, dtype=bool)
        self._fruit_support_gaps = np.full(3, self.contact_gap_seconds + self.model.opt.timestep)
        self._packed_fruits = np.zeros(3, dtype=bool)
        self._carried_fruits = np.zeros(3, dtype=bool)
        self._transported_fruits = np.zeros(3, dtype=bool)
        self._packed_stage = self._transport_complete = self._ever_success = False
        self._carried_filled = False
        self._transport_time = 0.0
        self._previous_transport = False
        self._return_time = 0.0
        self._previous_return = False
        # Rows track bowl grasps and rim grasps; columns track the two hands.
        self._grasp_times = np.zeros((2, 2))
        self._previous_grasps = np.zeros((2, 2), dtype=bool)
        self._holds = np.zeros((2, 2), dtype=bool)
        self._contact_gaps = np.zeros(2)
        self._finger_contact_gaps = np.full((2, 2), self.contact_gap_seconds + self.model.opt.timestep)
        self._receiver_support_gap = self.contact_gap_seconds + self.model.opt.timestep
        self._armed = np.zeros(3, dtype=bool)
        self._emptied = np.zeros(3, dtype=bool)
        self._pour_valid = np.zeros(3, dtype=bool)
        self._previous_local = None
        self._previous_resident = np.zeros(3, dtype=bool)
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

    def _containment(self, local, bowl, *, full_sphere=True):
        floor, rim, bottom_radius, rim_radius = self._bounds[bowl]
        slope = (rim_radius - bottom_radius) / (rim - floor)
        plane_distance = local[:, :2] @ self._normals_xy.T - slope * (local[:, 2:3] - floor)
        radii = self._radii if full_sphere else np.zeros_like(self._radii)
        # Check the portion of each sphere below the finite wall's upper edge.
        # Above the rim, a supported pile may occupy the opening's vertical column.
        wall_norm = np.sqrt(1 + slope * slope)
        lowest = local[:, 2] - radii
        support_z = local[:, 2] - radii * slope / wall_norm
        clipped = (support_z > rim) & (lowest < rim)
        slice_radius = np.sqrt(np.maximum(radii**2 - (rim - local[:, 2]) ** 2, 0))
        sphere_extent = np.where(clipped, slice_radius + slope * (local[:, 2] - rim), radii * wall_norm)
        below_wall = np.all(
            plane_distance + sphere_extent[:, None] <= bottom_radius + self.containment_tolerance_m, axis=1
        )
        return (
            (below_wall | (lowest >= rim))
            & np.all(local[:, :2] @ self._normals_xy.T <= rim_radius + self.containment_tolerance_m, axis=1)
            & (lowest >= floor - 0.002)
            & (local[:, 2] + radii <= floor + 2 * self._radii.sum() + self.containment_tolerance_m)
        )

    def _source_floor_placement(self, data, bowl_pos, bowl_rot):
        source_floor = self._floors[0]
        center = (data.geom_xpos[source_floor] - bowl_pos[1]) @ bowl_rot[1]
        rotation = bowl_rot[1].T @ data.geom_xmat[source_floor].reshape(3, 3)
        radius, half_height = self.model.geom_size[source_floor, :2]
        floor, rim, bottom_radius, rim_radius = self._bounds[1]
        slope = (rim_radius - bottom_radius) / (rim - floor)

        def extent(normals):
            local_normals = normals @ rotation
            return radius * np.linalg.norm(local_normals[..., :2], axis=-1) + half_height * np.abs(
                local_normals[..., 2]
            )

        side_normals = np.column_stack((self._normals_xy, np.full(BOWL_SEGMENTS, -slope)))
        opening_normals = np.column_stack((self._normals_xy, np.zeros(BOWL_SEGMENTS)))
        vertical_extent = extent(np.array([0, 0, 1]))
        # The floor footprint must fit through the mouth for overhead credit.
        above = bool(
            bowl_rot[1, 2, 2] >= 0.98
            and np.all(opening_normals @ center + extent(opening_normals) <= rim_radius + self.containment_tolerance_m)
            and center[2] - vertical_extent >= rim + 0.025
        )
        # Nesting follows the receiver's flared cavity, including its floor height.
        inside = bool(
            np.all(
                side_normals @ center + slope * floor + extent(side_normals)
                <= bottom_radius + self.containment_tolerance_m
            )
            and center[2] - vertical_extent >= floor - 0.002
            and center[2] + vertical_extent <= rim + self.containment_tolerance_m
        )
        return above, inside

    def _source_in_transport_zone(self, bowl_pos, bowl_rot):
        source_rim = self._bounds[0][1]
        mouth_world = bowl_pos[0] + source_rim * bowl_rot[0, :, 2]
        mouth = (mouth_world - bowl_pos[1]) @ bowl_rot[1]
        _, target_rim, _, target_radius = self._bounds[1]
        # Center distance defines the transport zone; the mouth must clear the rim.
        return bool(
            bowl_rot[1, 2, 2] >= 0.98
            and np.linalg.norm(bowl_pos[0, :2] - bowl_pos[1, :2]) <= target_radius
            and mouth[2] >= target_rim + 0.01
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
        source_resident = self._containment(local[0], 0, full_sphere=False)
        loads = np.zeros(5)
        touching_robot = np.zeros(5, dtype=bool)
        fruit_table = np.zeros(3, dtype=bool)
        bowl_table_load = np.zeros(2)
        nested_source_load = 0.0
        supported = np.zeros((2, 3), dtype=bool)
        supports = np.zeros((3, 3), dtype=bool)
        fingers = np.zeros((2, 2), dtype=bool)
        rim_fingers = np.zeros((2, 2), dtype=bool)
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
                if owner == 3 and other_owner == 4:
                    nested_source_load += sign * world_force[2]
                if owner == 3 and other in self._fingers and force[0] > 0.02:
                    fingers[self._fingers[other]] = True
                    point = (contact.pos - bowl_pos[0]) @ bowl_rot[0]
                    _, rim, _, rim_radius = self._bounds[0]
                    if abs(point[2] - rim) <= 0.015 and np.linalg.norm(point[:2]) >= rim_radius - 0.018:
                        rim_fingers[self._fingers[other]] = True
        speeds = data.qvel[self._dofs[:, None] + np.arange(6)]
        linear_speed = np.linalg.norm(speeds[:, :3], axis=1)
        angular_speed = np.linalg.norm(speeds[:, 3:], axis=1)
        settled = (linear_speed <= 0.03) & (angular_speed <= 0.25)
        # Surface speed gives the same settling tolerance to every fruit size.
        settled[:3] = linear_speed[:3] + self._radii * angular_speed[:3] <= 0.03
        candidates = inside & settled[None, :3] & ~touching_robot[None, :3]
        geometric_support = supported & inside
        for _ in range(3):
            geometric_support |= inside & np.any(supports[None, :, :] & geometric_support[:, None, :], axis=2)
        self._fruit_support_gaps = np.where(geometric_support[0], 0, self._fruit_support_gaps + elapsed)
        released_support = inside[0] & ~touching_robot[:3] & (self._fruit_support_gaps <= self.contact_gap_seconds)
        self._released_support_time = np.where(
            released_support & self._previous_released_support, self._released_support_time + elapsed, 0
        )
        self._previous_released_support = released_support
        supported &= candidates
        for _ in range(3):
            supported |= candidates & np.any(supports[None, :, :] & supported[:, None, :], axis=2)
        bowl_ok = (bowl_rot[:, 2, 2] >= 0.98) & settled[3:] & (bowl_table_load >= 0.6 * self._weights[3:])
        source_eligible = candidates[0] & supported[0] & bowl_ok[0]
        self._source_time = np.where(source_eligible & self._previous_source, self._source_time + elapsed, 0)
        self._previous_source = source_eligible
        source_qualified = source_eligible & (self._source_time >= self.settling_seconds)
        # Center containment retains earned credit through rolling and solver penetration.
        # Each fruit must qualify in a supported pile or already have been carried.
        self._packed_fruits = (self._packed_fruits | source_qualified) & source_resident
        packing_completed_now = (
            bool(
                (self._packed_fruits | self._carried_fruits).all()
                and np.all(self._carried_fruits | (inside[0] & geometric_support[0]))
            )
            and not self._packed_stage
        )
        self._packed_stage |= packing_completed_now
        paired_contacts = np.array([fingers.all(axis=1), fingers.all(axis=1) & rim_fingers.any(axis=1)])
        self._finger_contact_gaps = np.where(fingers, 0, self._finger_contact_gaps + elapsed)
        paired_recently = (self._finger_contact_gaps <= self.contact_gap_seconds).all(axis=1)
        acquiring = np.array([paired_recently, paired_recently & rim_fingers.any(axis=1)])
        self._grasp_times = np.where(acquiring & self._previous_grasps, self._grasp_times + elapsed, 0)
        self._previous_grasps = acquiring
        self._holds |= acquiring & (self._grasp_times >= self.grasp_seconds)
        self._contact_gaps = np.where(fingers.any(axis=1), 0, self._contact_gaps + elapsed)
        # Once acquired, a grip can transfer load between opposing fingertips.
        # A short gap tolerates contact-manifold changes while the bowl rotates.
        self._holds &= self._contact_gaps[None] <= self.contact_gap_seconds
        held = bool(self._holds[0].any())
        source_lift = float(bowl_pos[0, 2] - self._initial_source_z)
        lift_load = (
            (self._packed_fruits | self._packed_stage)
            & inside[0]
            & (self._fruit_support_gaps <= self.contact_gap_seconds)
        )
        if held and source_lift >= 0.05 and lift_load.any() and np.array_equal(lift_load, source_resident):
            # Remember every qualified fruit lifted in the bowl. Spills cannot
            # shrink the load against which emptying is measured.
            self._carried_fruits |= lift_load
        self._carried_filled = bool(self._carried_fruits.any())
        overhead_distance = float(np.linalg.norm(bowl_pos[0, :2] - bowl_pos[1, :2]))
        above, floor_inside = self._source_floor_placement(data, bowl_pos, bowl_rot)
        in_transport_zone = self._source_in_transport_zone(bowl_pos, bowl_rot)
        receiver_supported = bool(
            nested_source_load >= 0.3 * self._weights[3]
            and bowl_rot[1, 2, 2] >= 0.98
            and bowl_table_load[1] >= 0.6 * self._weights[4]
        )
        self._receiver_support_gap = 0.0 if receiver_supported else self._receiver_support_gap + elapsed
        source_nested = (
            floor_inside
            and nested_source_load >= 0.6 * self._weights[3]
            and bowl_rot[0, 2, 2] >= 0.98
            and settled[3]
            and bowl_ok[1]
        )
        pending_load = self._carried_fruits & ~self._emptied
        transport = (
            bool(pending_load.any())
            and np.all(~pending_load | source_resident)
            and (source_nested or (in_transport_zone and (held or receiver_supported)))
        )
        self._transport_time = self._transport_time + elapsed if transport and self._previous_transport else 0.0
        self._previous_transport = bool(transport)
        transported_now = bool(transport and self._transport_time >= self.transport_seconds)
        self._transport_complete |= transported_now
        self._transported_fruits |= pending_load & transported_now
        # A brushed fruit can re-arm after being released onto the supported pile.
        source_rearmed = released_support & (self._released_support_time >= self.fruit_rearm_seconds)
        packing_credit = self._packed_fruits | self._carried_fruits | self._packed_stage
        self._armed |= ((source_qualified | source_rearmed) & packing_credit) | (
            self._packed_fruits & packing_completed_now
        )
        self._armed &= ~touching_robot[:3] & ~fruit_table
        self._emptied &= ~source_resident
        self._pour_valid &= ~source_resident & ~touching_robot[:3] & ~fruit_table
        tilt_deg = float(np.rad2deg(np.arccos(np.clip(bowl_rot[0, 2, 2], -1, 1))))
        rim = self._bounds[0][1] + self.containment_tolerance_m
        pouring_support = bool(
            in_transport_zone
            and ((held and source_lift >= 0.05) or self._receiver_support_gap <= self.contact_gap_seconds)
        )
        crossed = np.zeros(3, dtype=bool)
        departure = np.zeros(3, dtype=bool)
        if self._previous_local is not None:
            # A pile can start above the rim. Its fruit leaves through the opening's
            # upper column, crossing either its side boundary or its top.
            crossed = self._previous_resident & ~source_resident & (local[0][:, 2] >= rim - 0.002)
            opening = np.linalg.norm(local[0][:, :2], axis=1) <= self._bounds[0][3] + self._radii + 0.005
            departure = (
                crossed
                & opening
                & self._armed
                & ~touching_robot[:3]
                & self._transported_fruits
                & pouring_support
                & (tilt_deg >= self.pour_tilt_degrees)
            )
            self._emptied |= departure
            self._pour_valid |= departure & ~fruit_table
            self._armed &= ~crossed
        self._armed &= (np.linalg.norm(local[0][:, :2], axis=1) <= self._bounds[0][3] + self._radii + 0.01) & (
            local[0][:, 2] + self._radii >= self._bounds[0][0] - 0.002
        )
        self._previous_local = local[0].copy()
        self._previous_resident = source_resident.copy()
        target_eligible = candidates[1] & supported[1] & bowl_ok[1] & ~touching_robot[4] & self._pour_valid
        self._target_time = np.where(target_eligible & self._previous_target, self._target_time + elapsed, 0)
        self._previous_target = target_eligible
        caught = target_eligible & (self._target_time >= self.settling_seconds)
        source_empty_after_pour = bool(
            self._carried_filled and np.all(~self._carried_fruits | self._emptied) and not source_resident.any()
        )
        return_eligible = bool(
            self._transport_complete and source_empty_after_pour and bowl_ok[0] and not touching_robot[3]
        )
        self._return_time = self._return_time + elapsed if return_eligible and self._previous_return else 0.0
        self._previous_return = return_eligible
        returned = return_eligible and self._return_time >= self.settling_seconds
        excess = np.maximum(loads - self.force_threshold_n, 0)
        self._force_dose_ns += excess * elapsed
        self._peak_force_n = np.maximum(self._peak_force_n, loads)
        damage = np.minimum(self._force_dose_ns / self.force_dose_budget_ns, 1)
        emptying_shares = np.full(3, self.emptying_reward / 3)
        if self._carried_filled:
            emptying_shares = self._carried_fruits * self.emptying_reward / self._carried_fruits.sum()
        self._gentle_credit[:3] = self.gentle_fraction * (
            self.packing_reward / 3 + (emptying_shares + self.catching_reward / 3) / 2
        )
        packing = packing_credit * self.packing_reward / 3 * (1 - self.gentle_fraction * damage[:3])
        transporting = self._transport_complete * self.transport_reward * (1 - self.gentle_fraction * damage[3])
        emptying = self._emptied * emptying_shares * (1 - self.gentle_fraction / 2 * (damage[:3] + damage[3]))
        catching = caught * self.catching_reward / 3 * (1 - self.gentle_fraction / 2 * (damage[:3] + damage[4]))
        returning = returned * self.return_reward * (1 - self.gentle_fraction * damage[3])
        success = bool(returned and caught.all())
        self._ever_success |= success
        self._result = TaskEvalResult(
            reward=np.array([packing.sum() + transporting + emptying.sum() + catching.sum() + returning]),
            success=np.array([success]),
            metrics={
                "packing_reward": np.array([packing.sum()]),
                "transport_reward": np.array([transporting]),
                "emptying_reward": np.array([emptying.sum()]),
                "catching_reward": np.array([catching.sum()]),
                "return_reward": np.array([returning]),
                "source_return_eligible": np.array([return_eligible]),
                "source_empty_after_pour": np.array([source_empty_after_pour]),
                "source_return_settling_seconds": np.array([self._return_time]),
                "source_returned": np.array([returned]),
                "fruit_packing_reward": packing[None],
                "fruit_emptying_reward": emptying[None],
                "fruit_catching_reward": catching[None],
                "packing_stage_complete": np.array([self._packed_stage]),
                "transport_complete": np.array([self._transport_complete]),
                "filled_bowl_carried": np.array([self._carried_filled]),
                "source_nested_in_target": np.array([source_nested]),
                "num_packed_fruits": np.array([int(self._packed_fruits.sum())]),
                "num_carried_fruits": np.array([int(self._carried_fruits.sum())]),
                "num_transported_fruits": np.array([int(self._transported_fruits.sum())]),
                "num_emptied_fruits": np.array([int(self._emptied.sum())]),
                "num_caught_fruits": np.array([int(caught.sum())]),
                "fruit_in_source_mask": inside[0:1],
                "fruit_source_resident_mask": source_resident[None],
                "fruit_in_target_mask": inside[1:2],
                "fruit_packed_mask": self._packed_fruits[None].copy(),
                "fruit_packing_credit_mask": packing_credit[None],
                "fruit_carried_mask": self._carried_fruits[None].copy(),
                "fruit_transported_mask": self._transported_fruits[None].copy(),
                "fruit_emptying_capacity": emptying_shares[None],
                "fruit_source_eligible_mask": source_eligible[None],
                "fruit_source_qualified_mask": source_qualified[None],
                "fruit_source_settling_seconds": self._source_time[None].copy(),
                "fruit_target_settling_seconds": self._target_time[None].copy(),
                "fruit_source_supported_mask": supported[0:1],
                "fruit_source_geometric_support_mask": geometric_support[0:1],
                "fruit_source_rearm_seconds": self._released_support_time[None].copy(),
                "fruit_source_support_gap_seconds": self._fruit_support_gaps[None].copy(),
                "fruit_target_supported_mask": supported[1:2],
                "fruit_armed_mask": self._armed[None].copy(),
                "fruit_emptied_mask": self._emptied[None].copy(),
                "fruit_valid_pour_mask": self._pour_valid[None].copy(),
                "fruit_crossed_mouth_mask": crossed[None],
                "fruit_departure_mask": departure[None],
                "fruit_table_contact_mask": fruit_table[None],
                "fruit_caught_mask": caught[None],
                "bowl_grasped": np.array([bool(paired_contacts[0].any())]),
                "bowl_held": np.array([held]),
                "bowl_grasp_seconds": np.array([self._grasp_times[0].max()]),
                "bowl_finger_contact_mask": fingers[None],
                "bowl_finger_contact_gap_seconds": self._finger_contact_gaps[None].copy(),
                "rim_finger_contact_mask": rim_fingers[None],
                "rim_grasped": np.array([bool(paired_contacts[1].any())]),
                "rim_held": np.array([bool(self._holds[1].any())]),
                "rim_grasp_seconds": np.array([self._grasp_times[1].max()]),
                "transport_eligible": np.array([transport]),
                "transport_settling_seconds": np.array([self._transport_time]),
                "source_lift_m": np.array([source_lift]),
                "source_tilt_deg": np.array([tilt_deg]),
                "source_over_target": np.array([above]),
                "source_in_transport_zone": np.array([in_transport_zone]),
                "transport_radius_m": np.array([self._bounds[1][3]]),
                "source_receiver_supported": np.array([receiver_supported]),
                "source_receiver_support_gap_seconds": np.array([self._receiver_support_gap]),
                "pouring_support_valid": np.array([pouring_support]),
                "source_floor_inside_target": np.array([floor_inside]),
                "source_target_support_n": np.array([nested_source_load]),
                "source_target_distance_xy_m": np.array([overhead_distance]),
                "settled_mask": settled[None],
                "bowl_table_load_n": bowl_table_load[None],
                "bowl_ready_for_packing_mask": bowl_ok[None],
                "touching_robot": touching_robot[None],
                "contact_force_n": loads[None],
                "peak_force_n": self._peak_force_n[None].copy(),
                "force_threshold_n": self.force_threshold_n[None].copy(),
                "excess_force_n": excess[None],
                "force_dose_ns": self._force_dose_ns[None].copy(),
                "force_penalty": (self._gentle_credit * damage)[None],
                "force_gentle_credit": self._gentle_credit[None].copy(),
                "ever_success": np.array([self._ever_success]),
            },
        )
