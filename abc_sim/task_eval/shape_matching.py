"""Independent, contact-qualified shape matching with irreversible force dose."""

from __future__ import annotations

from copy import deepcopy

import mujoco
import numpy as np

from abc_sim.randomization.assets.shape_matching import (
    PIECE_HALF_HEIGHT,
    SHAPES,
    TARGET_COUNT,
    hole_polygon,
    polygon,
)
from abc_sim.task_eval.base import TaskEvalResult
from abc_sim.task_eval.contacts import subtree_geom_ids


class ShapeMatchingEvaluator:
    """Score each correctly seated, released piece after 0.3 seconds of settling.

    A requested piece qualifies while resting on the table in its assigned hole.
    It keeps its qualified credit while fully inside that hole.
    Removing it removes that piece's credit. Twenty percent of each piece's
    share reflects its own excess-force dose, and ten percent reflects the
    board's dose. Unrelated unplaced pieces do not gate achieved progress.
    """

    evaluation_interval_steps = 1
    settling_seconds = 0.3

    def __init__(self, *, model, spec, force_allowance_n=15.0, force_dose_budget_ns=5.0):
        if (
            not np.isfinite([force_allowance_n, force_dose_budget_ns]).all()
            or force_allowance_n < 0
            or force_dose_budget_ns <= 0
        ):
            raise ValueError("Force allowance must be finite and nonnegative; dose budget must be positive.")
        self.model, self.spec = model, spec
        self.force_dose_budget_ns = float(force_dose_budget_ns)
        self.force_entity_names = (*(f"shape_piece_{i:02d}" for i in range(16)), "shape_board")
        self._bodies = np.array([model.body(name).id for name in self.force_entity_names])
        self._board = int(self._bodies[-1])
        self._table = model.geom("table_plane").id
        self._geoms = [subtree_geom_ids(model, int(body)) for body in self._bodies]
        self._owner = {geom: owner for owner, geoms in enumerate(self._geoms) for geom in geoms}
        self._robot = set().union(*(subtree_geom_ids(model, model.body(name).id) for name in ("left_arm", "right_arm")))
        self._targets = np.array([model.site(f"shape_target_{i:02d}").pos.copy() for i in range(16)])
        requested = np.array([model.site(f"shape_target_{i:02d}").user[0] for i in range(16)])
        if not np.isin(requested, (0, 1)).all() or requested.sum() != TARGET_COUNT:
            raise ValueError("Shape matching requires exactly eight requested pieces in the compiled scene.")
        self._requested = requested.astype(bool)
        self._dofs = np.array([model.jnt_dofadr[model.body_jntadr[body]] for body in self._bodies[:16]])
        self._vertices, self._normals, self._offsets = [], [], []
        for i in range(16):
            shape = SHAPES[i // 4]
            outline = polygon(shape)
            self._vertices.append(
                np.concatenate(
                    [
                        np.column_stack((outline, np.full(len(outline), z)))
                        for z in (-PIECE_HALF_HEIGHT, PIECE_HALF_HEIGHT)
                    ]
                )
            )
            hole = hole_polygon(shape)
            edges = np.roll(hole, -1, axis=0) - hole
            normals = np.column_stack((edges[:, 1], -edges[:, 0]))
            normals /= np.linalg.norm(normals, axis=1)[:, None]
            self._normals.append(normals)
            self._offsets.append(np.einsum("ij,ij->i", normals, hole))
        weights = model.body_subtreemass[self._bodies] * -model.opt.gravity[2]
        if np.any(weights <= 0) or not np.isfinite(weights).all():
            raise ValueError("Shape matching requires positive masses and downward gravity.")
        self.force_threshold_n = np.full(17, float(force_allowance_n))
        self.force_threshold_n[-1] += weights[-1]
        self.reset()

    def reset(self, *, nworld=1):
        if nworld != 1:
            raise NotImplementedError("Shape matching requires a native physics observer per world.")
        self._data = self._previous_time = self._result = None
        self._settling = np.zeros(16)
        self._previous_eligible = np.zeros(16, dtype=bool)
        self._matched = np.zeros(16, dtype=bool)
        self._dose = np.zeros(17)
        self._peak = np.zeros(17)
        self._ever_success = False

    def after_reset(self, model, data):
        if model is not self.model:
            raise ValueError("Shape matching observer model mismatch.")
        self._data, self._previous_time = data, float(data.time)
        self._observe(data, 0.0)

    def after_physics_step(self, model, data):
        if model is not self.model or data is not self._data:
            raise RuntimeError("Reset shape matching tracking before observing physics.")
        elapsed = float(data.time) - self._previous_time
        if not np.isclose(elapsed, model.opt.timestep, rtol=1e-7, atol=1e-12):
            raise ValueError("Shape matching must observe every physics tick exactly once.")
        self._previous_time = float(data.time)
        self._observe(data, elapsed)

    def evaluate_model_data(self, model, data):
        if model is not self.model or data is not self._data or self._result is None:
            raise RuntimeError("Reset shape matching tracking before reading its score.")
        return deepcopy(self._result)

    def evaluate_qpos_batch(self, qpos_batch):
        raise NotImplementedError("Shape matching requires solved contacts, release, settling and force history.")

    def _observe(self, data, elapsed):
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise ValueError("Shape matching poses and velocities must be finite.")
        board_rotation = data.xmat[self._board].reshape(3, 3)
        inside = np.zeros(16, dtype=bool)
        for i, body in enumerate(self._bodies[:16]):
            points = (
                self._vertices[i] @ data.xmat[body].reshape(3, 3).T + data.xpos[body] - data.xpos[self._board]
            ) @ board_rotation
            relative = points - self._targets[i]
            inside[i] = (
                np.all(relative[:, :2] @ self._normals[i].T <= self._offsets[i] + 0.0007)
                and abs(points[:, 2].min()) <= 0.0015
                and points[:, 2].max() <= 2 * PIECE_HALF_HEIGHT + 0.0015
            )
        touching = np.zeros(16, dtype=bool)
        supported = np.zeros(16, dtype=bool)
        loads = np.zeros(17)
        force = np.empty(6)
        for c in range(data.ncon):
            contact = data.contact[c]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if g1 not in self._owner and g2 not in self._owner:
                continue
            mujoco.mj_contactForce(self.model, data, c, force)
            if not np.isfinite(force).all():
                raise ValueError("Shape matching contact loads must be finite.")
            magnitude = np.linalg.norm(force[:3])
            if contact.dist > 0 and magnitude == 0:
                continue
            for mine, other, sign in ((g1, g2, -1), (g2, g1, 1)):
                if mine not in self._owner:
                    continue
                owner = self._owner[mine]
                if self._owner.get(other) != owner:
                    loads[owner] += magnitude
                if owner < 16:
                    touching[owner] |= other in self._robot
                    supported[owner] |= (
                        other == self._table
                        and force[0] > 0.001
                        and sign * np.dot(contact.frame[:3], board_rotation[:, 2]) > 0.5
                    )
        velocities = data.qvel[self._dofs[:, None] + np.arange(6)]
        settled = (np.linalg.norm(velocities[:, :3], axis=1) <= 0.025) & (
            np.linalg.norm(velocities[:, 3:], axis=1) <= 0.25
        )
        eligible = self._requested & inside & supported & settled & ~touching
        self._settling = np.where(eligible & self._previous_eligible, self._settling + elapsed, 0.0)
        self._previous_eligible = eligible
        self._matched = (
            self._requested & inside & (self._matched | (eligible & (self._settling >= self.settling_seconds)))
        )
        excess = np.maximum(loads - self.force_threshold_n, 0)
        self._dose += excess * elapsed
        self._peak = np.maximum(self._peak, loads)
        damage = np.minimum(self._dose / self.force_dose_budget_ns, 1.0)
        penalty = (0.2 * damage[:16] + 0.1 * damage[16]) / TARGET_COUNT
        rewards = self._matched * (1 / TARGET_COUNT - penalty)
        success = bool(self._matched[self._requested].all() and eligible[self._requested].all())
        self._ever_success |= success
        self._result = TaskEvalResult(
            reward=np.array([rewards.sum()]),
            success=np.array([success]),
            metrics={
                "num_matched_pieces": np.array([int(self._matched.sum())]),
                "piece_requested_mask": self._requested[None, :].copy(),
                "piece_reward": rewards[None, :],
                "piece_matched_mask": self._matched[None, :].copy(),
                "piece_inside_mask": inside[None, :],
                "piece_supported_mask": supported[None, :],
                "piece_touching_robot": touching[None, :],
                "piece_settling_seconds": self._settling[None, :].copy(),
                "contact_force_n": loads[None, :],
                "peak_force_n": self._peak[None, :].copy(),
                "force_threshold_n": self.force_threshold_n[None, :].copy(),
                "excess_force_n": excess[None, :],
                "force_dose_ns": self._dose[None, :].copy(),
                "force_penalty": np.concatenate(
                    (0.2 * damage[:16] * self._requested / TARGET_COUNT, [0.1 * damage[16]])
                )[None, :],
                "ever_success": np.array([self._ever_success]),
            },
        )
