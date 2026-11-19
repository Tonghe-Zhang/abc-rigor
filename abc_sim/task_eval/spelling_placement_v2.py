"""Native spelling reward with per-letter progress and lifetime excess-force dose."""

from __future__ import annotations

import string
from copy import deepcopy
from dataclasses import asdict

import mujoco
import numpy as np

from abc_sim.spelling_goal import SpellingGoal
from abc_sim.spelling_placement_v2 import SpellingPlacementV2Cfg, score_spelling_placement
from abc_sim.task_eval.base import TaskEvalResult
from abc_sim.task_eval.contacts import subtree_geom_ids


class SpellingPlacementV2Evaluator:
    """Observe every physics tick and score released, deliberately placed letters."""

    evaluation_interval_steps = 1

    def __init__(self, *, model, spec, sequence_length):
        if type(sequence_length) is not int or sequence_length not in (2, 3, 4):
            raise ValueError("Spelling sequence length must be 2, 3, or 4.")
        self.model, self.spec, self.sequence_length = model, spec, sequence_length
        self.cfg = SpellingPlacementV2Cfg()
        self.letters = list(string.ascii_uppercase)
        self._bodies = np.array([model.body(f"block_{letter}").id for letter in self.letters])
        joints = np.array([model.joint(f"block_{letter}_jnt").id for letter in self.letters])
        if np.any(model.jnt_type[joints] != mujoco.mjtJoint.mjJNT_FREE):
            raise ValueError("Spelling blocks require free joints.")
        self._addresses, self._dofs = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
        self._owner = {
            geom: index for index, body in enumerate(self._bodies) for geom in subtree_geom_ids(model, int(body))
        }
        self._robot = set().union(
            *(subtree_geom_ids(model, model.body(name).id) for name in ("left_arm", "right_arm"))
        )
        self._table = model.geom("table_plane").id
        gravity = np.asarray(model.opt.gravity)
        self._weights = model.body_subtreemass[self._bodies] * np.linalg.norm(gravity)
        if model.geom_type[self._table] != mujoco.mjtGeom.mjGEOM_PLANE or np.any(self._weights <= 0):
            raise ValueError("Spelling requires a planar table, gravity and positive block weights.")
        self._up = -gravity / np.linalg.norm(gravity)
        self.reset()

    def reset(self, *, nworld=1):
        if nworld != 1:
            raise NotImplementedError("Spelling placement v2 requires a native contact observer per world.")
        self.word = ""
        self._data = self._previous_time = self._result = None
        self._handled = np.zeros(26, dtype=bool)
        self._settling = np.zeros(26)
        self._previous_eligible = np.zeros(26, dtype=bool)
        self._dose, self._peak = np.zeros(26), np.zeros(26)
        self._ever_success = False

    def configure_from_randomization(self, model, randomization):
        if model is not self.model:
            raise ValueError("Spelling evaluator must bind the realized model.")
        goal = SpellingGoal.from_metadata(randomization.metadata)
        if len(goal.sequence) != self.sequence_length:
            raise ValueError("The realized spelling length differs from its task.")
        self.word = goal.sequence

    def _positions(self, data):
        return data.qpos[self._addresses[:, None] + np.arange(3)]

    def after_reset(self, model, data):
        if model is not self.model or not self.word:
            raise RuntimeError("Bind the spelling goal before starting contact tracking.")
        self._data, self._previous_time = data, float(data.time)
        self._initial_positions = self._positions(data).copy()
        self._observe(data, 0.0)

    def after_physics_step(self, model, data):
        if model is not self.model or data is not self._data:
            raise RuntimeError("Reset spelling contact tracking before observing physics.")
        elapsed = float(data.time) - self._previous_time
        if not np.isclose(elapsed, model.opt.timestep, rtol=1e-7, atol=1e-12):
            raise ValueError("Spelling placement v2 must observe every physics tick exactly once.")
        self._previous_time = float(data.time)
        self._observe(data, elapsed)

    def evaluate_model_data(self, model, data):
        if model is not self.model or data is not self._data or self._result is None:
            raise RuntimeError("Reset spelling contact tracking before reading its score.")
        return deepcopy(self._result)

    def evaluate_qpos_batch(self, qpos_batch):
        raise NotImplementedError(
            "Spelling placement v2 requires solved contacts, release, settling and force history."
        )

    def _observe(self, data, elapsed):
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise ValueError("Spelling poses and velocities must be finite.")
        normal = data.geom_xmat[self._table].reshape(3, 3)[:, 2]
        if not np.isclose(abs(normal @ self._up), 1, rtol=0, atol=1e-6):
            raise ValueError("The spelling table must be horizontal relative to gravity.")
        loads, table_loads = np.zeros(26), np.zeros(26)
        touching = np.zeros(26, dtype=bool)
        force = np.empty(6)
        for c in range(data.ncon):
            contact = data.contact[c]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if g1 not in self._owner and g2 not in self._owner:
                continue
            mujoco.mj_contactForce(self.model, data, c, force)
            if not np.isfinite(force).all():
                raise ValueError("Spelling contact loads must be finite.")
            magnitude = np.linalg.norm(force[:3])
            for mine, other, sign in ((g1, g2, -1), (g2, g1, 1)):
                if mine not in self._owner:
                    continue
                owner = self._owner[mine]
                if self._owner.get(other) != owner:
                    loads[owner] += magnitude
                touching[owner] |= other in self._robot and magnitude > self.cfg.robot_force_tolerance
                if other == self._table:
                    table_loads[owner] += sign * np.dot(force[:3], contact.frame.reshape(3, 3) @ self._up)
        # Reset geometry and initial constraint solves never establish manipulation.
        if elapsed > 0:
            self._handled |= touching
        positions = self._positions(data)
        moved = np.linalg.norm(positions[:, :2] - self._initial_positions[:, :2], axis=1) >= self.cfg.displacement_m
        placed = self._handled & moved
        velocities = data.qvel[self._dofs[:, None] + np.arange(6)]
        face_up = np.abs(data.xmat[self._bodies].reshape(-1, 3, 3)[:, :, 0] @ self._up) >= self.cfg.upright_dot_min
        eligible = (
            placed
            & face_up
            & (table_loads >= self.cfg.minimum_weight_fraction * self._weights)
            & ~touching
            & (np.linalg.norm(velocities[:, :3], axis=1) <= self.cfg.max_linear_speed)
            & (np.linalg.norm(velocities[:, 3:], axis=1) <= self.cfg.max_angular_speed)
        )
        self._settling = np.where(eligible & self._previous_eligible, self._settling + elapsed, 0.0)
        self._previous_eligible = eligible
        settled = eligible & (self._settling >= self.cfg.settling_seconds)
        score = score_spelling_placement(
            self.cfg,
            word=self.word,
            letters=self.letters,
            positions=positions.tolist(),
            placed=placed.tolist(),
            eligible=settled.tolist(),
        )
        excess = np.maximum(loads - self.cfg.force_allowance_n, 0)
        self._dose += excess * elapsed
        self._peak = np.maximum(self._peak, loads)
        damage = np.minimum(self._dose / self.cfg.force_dose_budget_ns, 1)
        placement_credit = np.asarray(score["letter_progress"]) / self.sequence_length
        penalty = placement_credit * self.cfg.gentle_fraction * damage
        rewards = placement_credit - penalty
        self._ever_success |= score["success"]
        self._result = TaskEvalResult(
            reward=np.array([rewards.sum()]),
            success=np.array([score["success"]]),
            metrics={
                "word": self.word,
                "evaluator_revision": self.cfg.revision,
                "spelling_placement": {"configuration": asdict(self.cfg), "score": score},
                "placement_reward": np.array([placement_credit.sum()]),
                "letter_names": self.letters.copy(),
                "letter_reward": rewards[None, :],
                "letter_placed": placed[None, :],
                "letter_handled": self._handled[None, :].copy(),
                "letter_eligible": settled[None, :],
                "letter_touching_robot": touching[None, :],
                "letter_table_load_n": table_loads[None, :],
                "letter_settling_seconds": self._settling[None, :].copy(),
                "contact_force_n": loads[None, :],
                "peak_force_n": self._peak[None, :].copy(),
                "excess_force_n": excess[None, :],
                "force_dose_ns": self._dose[None, :].copy(),
                "force_penalty": penalty[None, :],
                "ever_success": np.array([self._ever_success]),
            },
        )
