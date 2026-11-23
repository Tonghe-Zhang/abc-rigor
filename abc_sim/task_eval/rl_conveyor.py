"""Native chronological scoring for the RL conveyor variants."""

from __future__ import annotations

import mujoco
import numpy as np

from abc_sim.randomization.tasks.rl_conveyor import validate_episode
from abc_sim.task_eval.base import TaskEvalResult
from abc_sim.task_eval.contacts import subtree_geom_ids
from abc_sim.task_eval.conveyor import ConveyorPickObjectsInBinEvaluator
from abc_sim.task_eval.put_relative import PutRelativeEvaluator
from abc_sim.task_runtime.conveyor_pick import (
    REMOVED_OBJECT_MIN_X,
    _OFF_TRACK_GEOMS,
    _OPTIONAL_OFF_TRACK_GEOMS,
)


class RLConveyorEvaluator:
    """Score one-time target grasps, supported deposits, and normal belt exits.

    Each target earns 15% of its unit for a lifted opposing-finger grasp, 60%
    for a released deposit supported inside the bin for 0.5 s, and 25% for
    gentle handling. The unit is 1 / target_count. From the first robot contact,
    a contact load above 15 N permanently forfeits that object's gentle bonus,
    including after deposit. Passive spawning and belt travel do not spend it.
    Deposit credit requires current released, supported and settled placement;
    every interruption restarts its dwell. Returning an object can never add a
    second unit. Grasp credit survives a subsequent loss.

    Every unresolved or failed distractor costs half a target unit. Only a
    normal downstream belt exit refunds this liability. The task objective
    floors earned credit minus penalties at zero; its normalized maximum is one.
    Ending early cannot avoid pending penalties on earned target credit.
    No reward read advances history. Each native physics tick is observed once.
    """

    evaluation_interval_steps = 1
    requires_step_updates = True
    settling_seconds = 0.5
    grasp_seconds = 0.05
    lift_height_m = 0.05
    force_threshold_n = 15.0

    def __init__(self, *, model, spec):
        self.spec = spec
        self._episode = None
        self._bind(model)
        self.reset()

    def validate_model(self, model):
        names, _ = ConveyorPickObjectsInBinEvaluator._resolve_object_qpos_addrs(
            model, r"conveyor_object_(\d+)_joint"
        )
        if (
            names != [f"conveyor_object_{i}" for i in range(1, len(names) + 1)]
            or not 2 <= len(names) <= 4
        ):
            raise ValueError(
                "RL conveyor requires two to four consecutively numbered objects."
            )
        ConveyorPickObjectsInBinEvaluator._resolve_bin_interior(model, "conveyor_bin")
        for name in ("conveyor_belt_surface", "conveyor_bin_bottom", *_OFF_TRACK_GEOMS):
            model.geom(name)
        for arm in ("left", "right"):
            model.body(f"{arm}_arm")
            for finger in ("left", "right"):
                model.joint(f"{arm}_{finger}_finger")

    def _bind(self, model):
        self.validate_model(model)
        self.model = model
        self.object_names, self._addrs = (
            ConveyorPickObjectsInBinEvaluator._resolve_object_qpos_addrs(
                model, r"conveyor_object_(\d+)_joint"
            )
        )
        self._count = len(self.object_names)
        self._bodies = np.array([model.body(name).id for name in self.object_names])
        self._dofs = model.jnt_dofadr[model.body_jntadr[self._bodies]]
        self._geoms = [
            sorted(
                g
                for g in subtree_geom_ids(model, body)
                if model.geom_contype[g] or model.geom_conaffinity[g]
            )
            for body in self._bodies
        ]
        if any(not geoms for geoms in self._geoms):
            raise ValueError("Every conveyor object needs collision geometry.")
        self._owner = {g: i for i, geoms in enumerate(self._geoms) for g in geoms}
        self._robot = set().union(
            *(
                subtree_geom_ids(model, model.body(f"{arm}_arm").id)
                for arm in ("left", "right")
            )
        )
        self._fingers = {}
        for hand, arm in enumerate(("left", "right")):
            for finger, side in enumerate(("left", "right")):
                joint = model.joint(f"{arm}_{side}_finger").id
                for geom in subtree_geom_ids(model, int(model.jnt_bodyid[joint])):
                    self._fingers[geom] = (hand, finger)
        self._floor = model.geom("conveyor_bin_bottom").id
        self._belt = model.geom("conveyor_belt_surface").id
        if model.geom_type[self._belt] != mujoco.mjtGeom.mjGEOM_BOX:
            raise ValueError("RL conveyor exit gate requires a box belt surface.")
        self._off_track = {model.geom(name).id for name in _OFF_TRACK_GEOMS}
        self._off_track.update(
            g
            for name in _OPTIONAL_OFF_TRACK_GEOMS
            if (g := mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)) >= 0
        )
        x, y, floor, rim = ConveyorPickObjectsInBinEvaluator._resolve_bin_interior(
            model, "conveyor_bin"
        )
        self._bin_lower = np.array([x[0], y[0], floor])
        self._bin_upper = np.array([x[1], y[1], rim])
        self._body_points = [
            PutRelativeEvaluator._body_extent_points(model, body)
            for body in self._bodies
        ]

    def prepare_randomization(self, model, state):
        if state is None:
            raise ValueError(
                "RL conveyor scoring requires a sampled or recorded episode definition."
            )
        return validate_episode(state.metadata)

    def configure_episode(self, model, episode):
        episode = validate_episode(episode)
        if model is not self.model:
            self._bind(model)
        if len(episode["objects"]) != self._count:
            raise ValueError(
                "Conveyor episode object count must match the restored model."
            )
        self._episode = episode
        self.reset()

    def reset(self, *, nworld=1):
        if nworld != 1:
            raise NotImplementedError(
                "RL conveyor requires one native physics observer per world."
            )
        self._data = self._previous_time = None
        self._initial_z = None
        self._grasp_start = np.full(self._count, np.nan)
        self._settle_start = np.full(self._count, np.nan)
        self._grasped = np.zeros(self._count, dtype=bool)
        self._placed = np.zeros(self._count, dtype=bool)
        self._inside = np.zeros(self._count, dtype=bool)
        self._robot_touched = np.zeros(self._count, dtype=bool)
        self._on_belt = np.zeros(self._count, dtype=bool)
        self._normal_exit = np.zeros(self._count, dtype=bool)
        self._failed = np.zeros(self._count, dtype=bool)
        self._peak_force = np.zeros(self._count)
        self._force = np.zeros(self._count)
        self._previous_belt_y = np.zeros(self._count)
        self._ever_success = False

    def after_reset(self, model, data):
        if model is not self.model or self._episode is None:
            raise RuntimeError(
                "Configure the RL conveyor episode before observing physics."
            )
        self._data = data
        self._previous_time = float(data.time)
        self._initial_z = data.xpos[self._bodies, 2].copy()
        self._observe(data, elapsed=0.0)

    def after_physics_step(self, model, data):
        if model is not self.model or data is not self._data:
            raise RuntimeError("Reset RL conveyor tracking before observing physics.")
        elapsed = float(data.time) - self._previous_time
        if not np.isclose(elapsed, model.opt.timestep, rtol=1e-7, atol=1e-12):
            raise ValueError(
                "RL conveyor must observe every physics tick exactly once."
            )
        self._previous_time = float(data.time)
        self._observe(data, elapsed=elapsed)

    def evaluate_qpos_batch(self, qpos_batch):
        raise NotImplementedError(
            "RL conveyor scoring requires chronological solved contacts."
        )

    def evaluate_model_data(self, model, data):
        if (
            model is not self.model
            or self._episode is None
            or (self._data is not None and data is not self._data)
        ):
            raise RuntimeError(
                "Configure RL conveyor tracking before reading its score."
            )
        # The environment reads initial runtime progress before after_reset.
        return self._result()

    def _bounds(self, data):
        lower, upper = [], []
        for body, local in zip(self._bodies, self._body_points):
            points = local @ data.xmat[body].reshape(3, 3).T + data.xpos[body]
            lower.append(points.min(axis=0))
            upper.append(points.max(axis=0))
        return np.array(lower), np.array(upper)

    def _observe(self, data, *, elapsed):
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise ValueError("Conveyor poses and velocities must be finite.")
        model, count = self.model, self._count
        now = float(data.time)
        lower, upper = self._bounds(data)
        # Existing native placement evaluators allow 2 mm of floor penetration.
        inside = (
            np.all(lower >= self._bin_lower - [0, 0, 0.002], axis=1)
            & np.all(upper[:, :2] <= self._bin_upper[:2], axis=1)
            & (data.xpos[self._bodies, 2] <= self._bin_upper[2])
        )
        fingers = np.zeros((count, 2, 2), dtype=bool)
        robot, belt, off_track = (np.zeros(count, dtype=bool) for _ in range(3))
        floor_support = np.zeros(count, dtype=bool)
        supports = np.zeros((count, count), dtype=bool)
        object_contacts = np.zeros((count, count), dtype=bool)
        contact_force = np.zeros(count)
        force = np.zeros(6)
        for c in range(data.ncon):
            contact = data.contact[c]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if g1 not in self._owner and g2 not in self._owner:
                continue
            mujoco.mj_contactForce(model, data, c, force)
            if not np.isfinite(force).all():
                raise ValueError("Conveyor contact forces must be finite.")
            magnitude = float(np.linalg.norm(force[:3]))
            if contact.dist > 0 and magnitude == 0:
                continue
            for mine, other, sign in ((g1, g2, -1), (g2, g1, 1)):
                if (
                    mine not in self._owner
                    or self._owner.get(other) == self._owner[mine]
                ):
                    continue
                i = self._owner[mine]
                contact_force[i] += magnitude
                robot[i] |= other in self._robot
                if other in self._owner:
                    object_contacts[i, self._owner[other]] = True
                belt[i] |= other == self._belt
                off_track[i] |= other in self._off_track
                if other in self._fingers and force[0] > 0.001:
                    hand, finger = self._fingers[other]
                    fingers[i, hand, finger] = True
                if force[0] > 0.001 and sign * contact.frame[2] > 0.5:
                    floor_support[i] |= other == self._floor
                    if other in self._owner:
                        supports[i, self._owner[other]] = True
        active = ~(self._normal_exit | self._failed)
        # A manipulated object cannot be used as a tool to push a distractor
        # through the exit gate. After release, clean belt travel can resume.
        manipulated = robot.copy()
        for _ in range(count):
            manipulated |= active & np.any(
                object_contacts & manipulated[None, :], axis=1
            )
        self._robot_touched |= active & manipulated
        self._force = np.where(active, contact_force, 0.0)
        self._peak_force = np.maximum(
            self._peak_force, np.where(self._robot_touched, self._force, 0.0)
        )
        belt_rot = data.geom_xmat[self._belt].reshape(3, 3)
        if not np.allclose(belt_rot, np.eye(3)):
            raise ValueError("Conveyor transport requires an axis-aligned belt.")
        local = data.xpos[self._bodies] - data.geom_xpos[self._belt]
        direction = float(np.sign(self._episode["belt_speed_y"]))
        signed_y = direction * local[:, 1]
        end = float(model.geom_size[self._belt, 1])
        natural_exit = (
            active
            & self._on_belt
            & ~manipulated
            & (self._previous_belt_y < end)
            & (signed_y >= end)
            & (np.abs(local[:, 0]) <= model.geom_size[self._belt, 0])
        )
        self._normal_exit |= natural_exit
        self._failed |= (
            active
            & ~natural_exit
            & (off_track | (data.xpos[self._bodies, 0] >= REMOVED_OBJECT_MIN_X))
        )
        active = ~(self._normal_exit | self._failed)
        lifted = (
            (data.xpos[self._bodies, 2] >= self._initial_z + self.lift_height_m)
            & ~belt
            & ~off_track
            & ~floor_support
            & ~supports.any(axis=1)
        )
        grasp = active & lifted & fingers.all(axis=2).any(axis=1)
        self._grasp_start[~grasp] = np.nan
        self._grasp_start[grasp & np.isnan(self._grasp_start)] = now
        self._grasped |= grasp & (now >= self._grasp_start + self.grasp_seconds)
        velocity = data.qvel[self._dofs[:, None] + np.arange(6)]
        settled = (np.linalg.norm(velocity[:, :3], axis=1) <= 0.03) & (
            np.linalg.norm(velocity[:, 3:], axis=1) <= 0.25
        )
        candidate = active & inside & settled & ~robot
        supported = floor_support & candidate
        for _ in range(count):
            supported |= candidate & np.any(supports & supported[None, :], axis=1)
        placed_now = supported & self._grasped
        self._settle_start[~placed_now] = np.nan
        self._settle_start[placed_now & np.isnan(self._settle_start)] = now
        self._placed = placed_now & (now >= self._settle_start + self.settling_seconds)
        self._inside = inside & active
        self._on_belt = belt & active
        self._previous_belt_y = signed_y

    def _result(self):
        targets = np.array(
            [obj["target"] for obj in self._episode["objects"]], dtype=bool
        )
        unit = 1.0 / int(targets.sum())
        placed = self._placed & self._inside
        gentle = self._peak_force <= self.force_threshold_n
        grasp_credit = targets * self._grasped * (0.15 * unit)
        placement_credit = targets * placed * (0.60 * unit)
        gentle_credit = targets * placed * gentle * (0.25 * unit)
        distractors = ~targets
        failed = distractors & (self._failed | self._placed)
        pending = distractors & ~(failed | self._normal_exit)
        penalty = distractors * ~self._normal_exit * (0.5 * unit)
        unclipped = float(
            (grasp_credit + placement_credit + gentle_credit - penalty).sum()
        )
        success = bool(
            np.all(placed[targets]) and np.all(self._normal_exit[distractors])
        )
        self._ever_success |= success
        return TaskEvalResult(
            reward=np.array([max(0.0, unclipped)]),
            success=np.array([success]),
            metrics={
                "target_mask": targets[None],
                "target_count": np.array([int(targets.sum())]),
                "object_names": list(self.object_names),
                "variant": self._episode["variant"],
                "target_category": self._episode["target_category"],
                "grasp_credit": grasp_credit[None],
                "placement_credit": placement_credit[None],
                "gentle_credit": gentle_credit[None],
                "distractor_penalty": penalty[None],
                "unclipped_reward": np.array([unclipped]),
                "pending_distractor_count": np.array([int(pending.sum())]),
                "failed_distractor_count": np.array([int(failed.sum())]),
                "normal_exit_mask": self._normal_exit[None].copy(),
                "failed_mask": self._failed[None].copy(),
                "remove_mask": (self._normal_exit | self._failed)[None],
                "grasped_mask": self._grasped[None].copy(),
                "placed_mask": placed[None],
                "peak_contact_force_n": self._peak_force[None].copy(),
                "contact_force_n": self._force[None].copy(),
                "ever_success": np.array([self._ever_success]),
                "ever_failed": np.array(
                    [
                        bool(
                            np.all(self._normal_exit | self._failed | placed)
                            and not success
                        )
                    ]
                ),
            },
        )
