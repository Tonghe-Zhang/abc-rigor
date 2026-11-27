"""Current released marker containment in a sampled moving drawer."""

from __future__ import annotations

import mujoco
import numpy as np

from abc_sim.task_eval.base import TaskEvalResult
from abc_sim.task_eval.contacts import subtree_geom_ids
from abc_sim.task_eval.put_relative import PutRelativeEvaluator


def drawer_interior(model, body):
    """Derive floor and four inner wall faces from compiled collision bounds."""
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    bounds = []
    for geom in sorted(subtree_geom_ids(model, body)):
        if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
            continue
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, model.geom_quat[geom])
        center, half = model.geom_aabb[geom, :3], model.geom_aabb[geom, 3:]
        points = model.geom_pos[geom] + (center + signs * half) @ rotation.reshape(3, 3).T
        bounds.append((geom, points.min(axis=0), points.max(axis=0)))
    floors = [entry for entry in bounds if np.argmin(entry[2] - entry[1]) == 2]
    if not floors:
        raise ValueError("Marker drawer requires an identifiable horizontal floor.")
    floor = max(floors, key=lambda entry: np.prod((entry[2] - entry[1])[:2]))
    lower, upper = floor[1].copy(), floor[2].copy()
    lower[2] = floor[2][2]
    walls = [entry for entry in bounds if entry[0] != floor[0]]
    inner_walls = []
    for axis in (0, 1):
        # Handles do not span the floor. Only full-width walls bound its cavity.
        other_axis = 1 - axis
        candidates = [
            entry
            for entry in walls
            if np.argmin(entry[2] - entry[1]) == axis
            and entry[2][other_axis] - entry[1][other_axis] >= 0.8 * (floor[2][other_axis] - floor[1][other_axis])
            and entry[2][2] > lower[2]
        ]
        if len(candidates) != 2:
            raise ValueError("Marker drawer requires two opposing walls on each horizontal axis.")
        negative, positive = sorted(candidates, key=lambda entry: entry[1][axis])
        lower[axis], upper[axis] = negative[2][axis], positive[1][axis]
        inner_walls.extend(candidates)
    upper[2] = min(entry[2][2] for entry in inner_walls)
    if np.any(upper <= lower):
        raise ValueError("Marker drawer collision geometry has no positive interior.")
    return lower, upper


class MarkerDrawerEvaluator:
    """Bank opening progress; score current released contents and closure."""

    evaluation_interval_steps = 1
    requires_step_updates = True
    requires_position_refresh = False

    def __init__(self, *, model, spec, closed_tolerance_m=0.005):
        self.model, self.spec = model, spec
        self.closed_tolerance_m = float(closed_tolerance_m)
        self._target = None
        self.reset()

    def reset(self, *, nworld=1):
        if nworld != 1:
            raise ValueError("Marker drawers require native contact evaluation.")
        self._best_open = 0.0
        self._opened = False
        self._result = TaskEvalResult(np.array([0.0]), np.array([False]), {"ever_success": np.array([False])})

    def configure_from_randomization(self, model, randomization):
        self.model = model
        self._target = randomization.metadata["marker_target_drawer"]
        if self._target not in ("top", "middle", "bottom"):
            raise ValueError("Unknown marker target drawer.")
        self._drawer = model.body(f"link_{self._target}_drawer_low").id
        self._joint = model.joint(f"joint_{self._target}_drawer_low").id
        self._travel = float(model.jnt_range[self._joint, 1])
        self._lower, self._upper = drawer_interior(model, self._drawer)
        self._drawer_geoms = subtree_geom_ids(model, self._drawer)
        self._markers = [
            body
            for body in range(model.nbody)
            if (model.body(body).name or "").startswith("marker_")
            and model.body(body).name.removeprefix("marker_").isdigit()
        ]
        if not self._markers:
            raise ValueError("Marker drawer scene contains no required markers.")
        self._marker_geoms = [subtree_geom_ids(model, body) for body in self._markers]
        self._geom_marker = {geom: index for index, geoms in enumerate(self._marker_geoms) for geom in geoms}
        self._points = [PutRelativeEvaluator._body_extent_points(model, body) for body in self._markers]
        self._robot = set().union(*(subtree_geom_ids(model, model.body(f"{arm}_arm").id) for arm in ("left", "right")))
        self.reset()

    def evaluate_qpos_batch(self, qpos_batch):
        raise NotImplementedError("Marker placement requires solved support contacts.")

    def evaluate_model_data(self, model, data):
        return self._result

    def after_reset(self, model, data):
        self.after_physics_step(model, data)

    def after_physics_step(self, model, data):
        if self._target is None:
            raise RuntimeError("Configure the marker drawer before evaluating it.")
        opening = float(data.qpos[model.jnt_qposadr[self._joint]])
        self._best_open = max(self._best_open, min(max(opening / self._travel, 0.0), 1.0))
        self._opened |= opening > self.closed_tolerance_m
        drawer_rotation = data.xmat[self._drawer].reshape(3, 3)
        count = len(self._markers)
        contained = np.zeros(count, dtype=bool)
        touched = np.zeros(count, dtype=bool)
        penetrating = np.zeros(count, dtype=bool)
        direct_support = np.zeros(count, dtype=bool)
        support_edges = np.zeros((count, count), dtype=bool)
        force = np.zeros(6)
        for index, (body, points) in enumerate(zip(self._markers, self._points)):
            world = points @ data.xmat[body].reshape(3, 3).T + data.xpos[body]
            local = (world - data.xpos[self._drawer]) @ drawer_rotation
            contained[index] = np.all(local >= self._lower - [0.0, 0.0, 0.002]) and np.all(local <= self._upper)
        for c in range(data.ncon):
            contact = data.contact[c]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if g1 not in self._geom_marker and g2 not in self._geom_marker:
                continue
            mujoco.mj_contactForce(model, data, c, force)
            if contact.dist > 0 and force[0] <= 0:
                continue
            for geom, other, sign in ((g1, g2, -1), (g2, g1, 1)):
                if geom not in self._geom_marker:
                    continue
                index = self._geom_marker[geom]
                if self._geom_marker.get(other) == index:
                    continue
                touched[index] |= other in self._robot
                penetrating[index] |= contact.dist < -0.002
                if force[0] > 0.001 and sign * contact.frame[2] > 0.5:
                    direct_support[index] |= other in self._drawer_geoms
                    if other in self._geom_marker:
                        support_edges[index, self._geom_marker[other]] = True
        valid = contained & ~touched & ~penetrating
        placed = valid & direct_support
        # A load path must reach the selected drawer through valid markers.
        # At most N propagation rounds cover every simple path in this scene.
        for _ in range(count):
            placed = valid & (direct_support | np.any(support_edges & placed[None, :], axis=1))
        fraction = float(np.mean(placed))
        all_inside = all(placed)
        closure = (
            max(
                0.0,
                1.0 - max(opening - self.closed_tolerance_m, 0.0) / (self._travel - self.closed_tolerance_m),
            )
            if self._opened and all_inside
            else 0.0
        )
        success = all_inside and self._opened and opening <= self.closed_tolerance_m
        self._result = TaskEvalResult(
            np.array([0.20 * self._best_open + 0.70 * fraction + 0.10 * closure]),
            np.array([success]),
            {
                "target_drawer": self._target,
                "marker_count": len(placed),
                "placed_marker_mask": np.array([placed]),
                "contained_marker_mask": contained[None],
                "released_marker_mask": (~touched)[None],
                "nonpenetrating_marker_mask": (~penetrating)[None],
                "direct_drawer_support_mask": direct_support[None],
                "opening_credit": 0.20 * self._best_open,
                "placement_credit": 0.70 * fraction,
                "closure_credit": 0.10 * closure,
                "ever_success": np.array([success]),
            },
        )
