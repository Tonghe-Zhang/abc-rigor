"""Three separated water bottles, lying sideways on the ABC table."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np

from abc_sim.task_specs import LIFT_BOTTLES_UPRIGHT_PROMPTS

from ..assets.water_bottles import (
    _water_bottle_flat_center_from_pose,
    _water_bottle_flat_compiled_metadata,
    _water_bottle_flat_quat,
    _water_bottle_flat_yaw_from_quat,
)
from ..core import PerturbRange, RandomizationState, ScalePerturbRange, SceneRandomizer
from .bottles import WaterBottleRandomizer


class BottlesUprightRandomizer(SceneRandomizer):
    """Sample three fixed-size bottle_17 meshes with full-footprint clearance."""

    table_bounds = (0.34, 0.84, -0.53, 0.53)
    perturbations = [PerturbRange(f"bottle_{index}_joint", delta_x=(0, 0), delta_y=(0, 0)) for index in range(1, 4)]

    def _get_size_perturbations(self) -> list[ScalePerturbRange]:
        return []

    def randomize(self, model: Any, data: Any, seed: int | None = None, request: Any = None) -> RandomizationState:
        state = super().randomize(model, data, seed=seed, request=request)
        # Consecutive inspection seeds cover every instruction, starting with the
        # canonical prompt. Geometry draws remain independent of the wording.
        state.metadata.update(
            bottle_count=3,
            bottle_variants=["bottle_17"] * 3,
            prompt=LIFT_BOTTLES_UPRIGHT_PROMPTS[state.seed % len(LIFT_BOTTLES_UPRIGHT_PROMPTS)],
        )
        return state

    def _sample_once(self, nominals, rng):
        half_length, _, radius = _water_bottle_flat_compiled_metadata("bottle_17")
        xmin, xmax, ymin, ymax = self.table_bounds
        states = {}
        for perturbation in self.perturbations:
            yaw = rng.uniform(-np.pi, np.pi)
            center = np.array([rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)])
            anchor = center - half_length * np.array([np.cos(yaw), np.sin(yaw)])
            states[perturbation.joint_name] = {
                "pos": [*anchor.tolist(), 0.75 + radius + 0.002],
                "quat": _water_bottle_flat_quat(yaw).tolist(),
            }
        return states

    @staticmethod
    def _footprint(state):
        half_length, half_radius, _ = _water_bottle_flat_compiled_metadata("bottle_17")
        return (
            _water_bottle_flat_center_from_pose(**state, variant_name="bottle_17", scale_factor=1.0),
            (half_length, half_radius),
            _water_bottle_flat_yaw_from_quat(np.asarray(state["quat"])),
        )

    def _bounds_ok(self, states):
        xmin, xmax, ymin, ymax = self.table_bounds
        for state in states.values():
            center, half, yaw = self._footprint(state)
            dx, dy = WaterBottleRandomizer._rotated_half_extents(*half, yaw)
            if not (
                xmin <= center[0] - dx and center[0] + dx <= xmax and ymin <= center[1] - dy and center[1] + dy <= ymax
            ):
                return False
        return True

    def _pairwise_ok(self, states):
        footprints = [self._footprint(state) for state in states.values()]
        return all(
            not WaterBottleRandomizer._obb_overlap(*first, *second, margin=0.015)
            for first, second in combinations(footprints, 2)
        )

    def _validate_scene(self, model, state):
        if state.scale_states:
            raise ValueError("Upright bottles use fixed-size meshes.")
        if set(state.object_states) != {p.joint_name for p in self.perturbations}:
            raise ValueError("Upright bottles replay requires all three bottle poses.")
        if state.metadata["prompt"] not in LIFT_BOTTLES_UPRIGHT_PROMPTS:
            raise ValueError("Unknown upright-bottles episode prompt.")
        for pose in state.object_states.values():
            pos, quat = np.asarray(pose["pos"]), np.asarray(pose["quat"])
            if pos.shape != (3,) or not np.isfinite(pos).all():
                raise ValueError("Bottle positions must be finite 3-vectors.")
            if quat.shape != (4,) or not np.isfinite(quat).all() or not np.isclose(np.linalg.norm(quat), 1):
                raise ValueError("Bottle orientations must be unit wxyz quaternions.")
