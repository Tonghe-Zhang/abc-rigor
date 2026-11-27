"""Chronological pickup-triggered clutter and relative-placement objectives."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import mujoco
import numpy as np

from abc_sim.task_eval.base import TaskEvalResult
from abc_sim.task_eval.contacts import subtree_geom_ids
from abc_sim.task_eval.put_relative import PutRelativeEvaluator


class InterruptedPickupEvaluator:
    """Own the one-shot pickup clock and task-local solved-contact evidence."""

    evaluation_interval_steps = 1
    requires_step_updates = True
    requires_position_refresh = False
    kind = ""

    def __init__(self, *, model, spec):
        self.model, self.spec = model, spec
        self._episode = None
        self.reset()

    def validate_model(self, model):
        model.geom("table_plane")
        for arm in ("left", "right"):
            model.body(f"{arm}_arm")
            for finger in ("left", "right"):
                model.joint(f"{arm}_{finger}_finger")

    def prepare_randomization(self, model, state):
        self.validate_model(model)
        if state is None:
            raise ValueError("Interrupted tasks require a recorded episode.")
        episode = copy.deepcopy(state.metadata)
        definition = episode["interruption"]
        if definition["kind"] != self.kind or not 1.0 <= definition["delay_s"] <= 2.0:
            raise ValueError("Invalid interrupted-task definition.")
        names = {obj["name"] for obj in episode["objects"]}
        if definition["first_body"] not in names:
            raise ValueError("The first pickup must name an episode object.")
        if self.kind == "clutter":
            if definition["second_body"] not in names - {definition["first_body"]}:
                raise ValueError("The second pickup must name a distinct episode object.")
        else:
            revised = definition["directive"]
            if (
                revised["mover_object"] != definition["first_body"]
                or revised["reference_object"] not in names
                or revised["reference_object"] in {definition["first_body"], episode["reference_object"]}
            ):
                raise ValueError("The revised directive needs the same mover and a new reference.")
        episode["object_states"] = copy.deepcopy(state.object_states)
        return episode

    def configure_episode(self, model, episode):
        self.model = model
        self._episode = copy.deepcopy(episode)
        definition = episode["interruption"]
        names = [definition["first_body"]]
        if self.kind == "clutter":
            names.append(definition["second_body"])
        self._bodies = [model.body(name).id for name in names]
        self._geoms = [subtree_geom_ids(model, body) for body in self._bodies]
        self._joints = [int(model.body_jntadr[body]) for body in self._bodies]
        self._table = model.geom("table_plane").id
        self._robot = set().union(*(subtree_geom_ids(model, model.body(f"{arm}_arm").id) for arm in ("left", "right")))
        self._fingers = {}
        for hand, arm in enumerate(("left", "right")):
            for side, finger in enumerate(("left", "right")):
                joint = model.joint(f"{arm}_{finger}_finger").id
                for geom in subtree_geom_ids(model, int(model.jnt_bodyid[joint])):
                    self._fingers[geom] = hand, side
        self._points = [PutRelativeEvaluator._body_extent_points(model, body) for body in self._bodies]
        if self.kind == "relative":
            revised = definition["directive"]
            self._relative = PutRelativeEvaluator(model=model, spec=self.spec)
            self._relative.configure_from_randomization(
                model,
                SimpleNamespace(metadata=revised, object_states=episode["object_states"]),
            )
        self.reset()

    def reset(self, *, nworld=1):
        if nworld != 1:
            raise ValueError("Interrupted tasks require one native physics observer per world.")
        self._first_lift_at = self._switch_at = None
        self._grasp_since = np.full(2, np.nan)
        self._second_ready = False
        self._origin = None
        self._previous_time = None
        self._result = TaskEvalResult(
            np.array([0.0]),
            np.array([False]),
            {
                "interruption_switched": False,
                "ever_success": np.array([False]),
            },
        )

    def after_reset(self, model, data):
        self._origin = data.xpos[self._bodies].copy()
        self._previous_time = float(data.time)

    def evaluate_qpos_batch(self, qpos_batch):
        raise NotImplementedError("Interrupted tasks require chronological solved contacts.")

    def evaluate_model_data(self, model, data):
        return self._result

    def _physical_state(self, data):
        count = len(self._bodies)
        fingers = np.zeros((count, 2, 2), dtype=bool)
        robot = np.zeros(count, dtype=bool)
        table = np.zeros(count, dtype=bool)
        other_support = np.zeros(count, dtype=bool)
        penetrating = np.zeros(count, dtype=bool)
        force = np.zeros(6)
        for c in range(data.ncon):
            contact = data.contact[c]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            for index, geoms in enumerate(self._geoms):
                if (g1 in geoms) == (g2 in geoms):
                    continue
                other, sign = (g2, -1) if g1 in geoms else (g1, 1)
                mujoco.mj_contactForce(self.model, data, c, force)
                if contact.dist > 0 and force[0] <= 0:
                    continue
                robot[index] |= other in self._robot
                penetrating[index] |= contact.dist < -0.002
                if other in self._fingers and force[0] > 0.001:
                    hand, side = self._fingers[other]
                    fingers[index, hand, side] = True
                if force[0] > 0.001 and sign * contact.frame[2] > 0.5:
                    table[index] |= other == self._table
                    other_support[index] |= other != self._table and other not in self._robot
        settled = []
        within_table = []
        for body, joint, points in zip(self._bodies, self._joints, self._points):
            dof = int(self.model.jnt_dofadr[joint])
            velocity = data.qvel[dof : dof + 6]
            settled.append(np.linalg.norm(velocity[:3]) <= 0.03 and np.linalg.norm(velocity[3:]) <= 0.25)
            world = points @ data.xmat[body].reshape(3, 3).T + data.xpos[body]
            local = world - data.geom_xpos[self._table]
            within_table.append(np.all(np.abs(local[:, :2]) <= self.model.geom_size[self._table, :2] + 0.002))
        valid_table = table & ~other_support & ~robot & ~penetrating & np.asarray(settled) & np.asarray(within_table)
        lifted = (data.xpos[self._bodies, 2] >= self._origin[:, 2] + 0.05) & ~table & ~other_support & ~penetrating
        return fingers.all(axis=2).any(axis=1) & lifted, valid_table

    def after_physics_step(self, model, data):
        now = float(data.time)
        if not np.isclose(now - self._previous_time, model.opt.timestep, rtol=1e-7, atol=1e-12):
            raise ValueError("Interrupted tasks must observe every physics tick exactly once.")
        self._previous_time = now
        grasp, valid_table = self._physical_state(data)
        starts = self._grasp_since[: len(grasp)]
        starts[~grasp] = np.nan
        starts[grasp & np.isnan(starts)] = now
        held = grasp & (now >= starts + 0.05)
        if self._first_lift_at is None and held[0]:
            self._first_lift_at = now
        definition = self._episode["interruption"]
        if (
            self._first_lift_at is not None
            and self._switch_at is None
            and now >= self._first_lift_at + definition["delay_s"]
        ):
            self._switch_at = now
            self._second_ready = self.kind == "clutter" and not grasp[1]
        if self.kind == "clutter" and self._switch_at is not None and not grasp[1]:
            self._second_ready = True
        self._result = self._score(data, held, valid_table)

    def _metrics(self):
        return {
            "interruption_switched": self._switch_at is not None,
            "first_lift_at_s": self._first_lift_at,
            "interruption_at_s": self._switch_at,
            "interruption_delay_s": self._episode["interruption"]["delay_s"],
        }


class InterruptedClutterEvaluator(InterruptedPickupEvaluator):
    kind = "clutter"

    def _score(self, data, held, valid_table):
        picked = self._first_lift_at is not None
        switched = self._switch_at is not None
        distance = float(np.linalg.norm(data.xpos[self._bodies[0], :2] - self._origin[0, :2]))
        returned = 0.0
        if switched and valid_table[0]:
            returned = 0.70 if distance <= 0.05 else 0.50 if distance <= 0.10 else 0.30
        second = switched and self._second_ready and bool(held[1])
        success = returned == 0.70 and second
        return TaskEvalResult(
            np.array([0.10 * picked + returned + 0.20 * second]),
            np.array([success]),
            {
                **self._metrics(),
                "first_pick_credit": 0.10 * picked,
                "return_credit": returned,
                "second_pick_credit": 0.20 * second,
                "return_distance_m": distance,
                "ever_success": np.array([success]),
            },
        )


class InterruptedRelativeEvaluator(InterruptedPickupEvaluator):
    kind = "relative"

    def _score(self, data, held, valid_table):
        if self._switch_at is None:
            return TaskEvalResult(np.array([0.0]), np.array([False]), self._metrics())
        native = self._relative.evaluate_qpos_batch(data.qpos[None])
        goal = self._relative._goal
        mover = data.xpos[self._bodies[0], :2]
        reference = data.qpos[goal["reference_adr"] : goal["reference_adr"] + 2]
        distance = float(np.linalg.norm(mover - reference))
        # Existing native scales define the full neighborhood and its falloff.
        radius = np.hypot(self._relative.along_max_m, self._relative.perp_max_m)
        approach = float(
            np.clip(
                1.0 - max(distance - radius, 0.0) / self._relative.partial_reward_range_m,
                0.0,
                1.0,
            )
        )
        along = float(native.metrics["mover_along_m"][0])
        perp = float(native.metrics["mover_perp_m"][0])
        edge = max(float(native.metrics["reference_edge_m"][0]), self._relative.along_min_m)
        boundary_error = (
            max(edge - along, 0.0)
            + max(along - self._relative.along_max_m, 0.0)
            + max(perp - min(self._relative.perp_max_m, max(along, 0.0)), 0.0)
        )
        relation = float(np.clip(1.0 - boundary_error / self._relative.partial_reward_range_m, 0.0, 1.0))
        # Wrong-side proximity earns approach only.
        relation *= valid_table[0] and along >= edge
        initial_ref = np.asarray(self._episode["object_states"][f"{goal['reference']}_joint"]["pos"])
        reference_stationary = np.linalg.norm(reference - initial_ref[:2]) <= 0.03
        relation *= reference_stationary
        success = bool(native.success[0] and valid_table[0] and reference_stationary)
        return TaskEvalResult(
            np.array([0.30 * approach + 0.70 * relation]),
            np.array([success]),
            {
                **self._metrics(),
                "approach_credit": 0.30 * approach,
                "relative_placement_credit": 0.70 * relation,
                "reference_stationary": reference_stationary,
                "ever_success": np.array([success]),
            },
        )
