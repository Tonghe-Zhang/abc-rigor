"""Three side-lying water bottles with seeded household distractors."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import asdict
from itertools import combinations
from typing import Any

import mujoco
import numpy as np

from abc_sim.task_specs import LIFT_BOTTLES_UPRIGHT_PROMPTS

from ..assets.tabletop_distractors import (
    DISTRACTOR_CATEGORIES,
    DistractorSpec,
    build_distractor,
    distractor_half_extents,
    distractor_variants,
)
from ..assets.water_bottles import (
    _water_bottle_build_object_block,
    _water_bottle_flat_center_from_pose,
    _water_bottle_flat_compiled_metadata,
    _water_bottle_flat_quat,
    _water_bottle_flat_yaw_from_quat,
)
from ..core import PerturbRange, RandomizationState, SceneRandomizer, _resolve_scene_xml_paths
from .bottles import WaterBottleRandomizer

# These assets are labelled bottled_water in the published pack.
UPRIGHT_BOTTLE_VARIANTS = tuple(f"bottle_{index}" for index in range(5, 19))
_BOTTLE_JOINTS = tuple(f"bottle_{index}_joint" for index in range(1, 4))


class BottlesUprightRandomizer(SceneRandomizer):
    """Sample three water bottles and 1–3 distractors with full-footprint clearance."""

    table_bounds = (0.34, 0.84, -0.53, 0.53)
    perturbations = [PerturbRange(name, delta_x=(0, 0), delta_y=(0, 0)) for name in _BOTTLE_JOINTS]

    def prepare_reset(self, *, request=None, randomize=True, seed=None, keep_arm_state=False):
        if not randomize:
            raise ValueError(
                "Upright bottles requires a randomized reset or saved-state replay; reuse a seed for a fixed scene."
            )
        self._identities(request, np.random.default_rng(0))

    def _identities(self, request, rng):
        options = {} if request is None else dict(request)
        unknown = set(options) - {"bottle_variants", "distractors"}
        if unknown:
            raise ValueError(f"Unknown upright-bottles reset options: {sorted(unknown)}")
        variants = list(
            options["bottle_variants"] if "bottle_variants" in options else rng.choice(UPRIGHT_BOTTLE_VARIANTS, size=3)
        )
        if len(variants) != 3 or any(v not in UPRIGHT_BOTTLE_VARIANTS for v in variants):
            raise ValueError("bottle_variants must contain three approved water-bottle variants.")
        variants = [str(v) for v in variants]
        if "distractors" in options:
            distractors = [DistractorSpec(**item) for item in options["distractors"]]
        else:
            categories = rng.choice(DISTRACTOR_CATEGORIES, size=int(rng.integers(1, 4)), replace=False)
            distractors = [
                DistractorSpec(str(category), str(rng.choice(distractor_variants(str(category)))))
                for category in categories
            ]
        if not 1 <= len(distractors) <= 3 or len({s.category for s in distractors}) != len(distractors):
            raise ValueError("distractors must contain 1–3 distinct categories.")
        return variants, distractors

    def _reload_scene(self, variants, distractors):
        env = self._env_ref
        if env is None:
            raise RuntimeError("Upright-bottles asset randomization requires a bound environment.")
        root = ET.fromstring(self._base_scene_xml_string)
        asset, world = root.find("asset"), root.find("worldbody")
        for elem in list(asset):
            if elem.get("name", "").startswith(("bottle_", "distractor_")):
                asset.remove(elem)
        for elem in list(world):
            if elem.tag == "body" and elem.get("name", "").startswith(("bottle_", "distractor_")):
                world.remove(elem)
        for index, variant in enumerate(variants):
            assets, body = _water_bottle_build_object_block(
                variant_name=variant,
                instance_index=index,
                scale_factor=1.0,
                object_name=f"bottle_{index + 1}",
                joint_name=_BOTTLE_JOINTS[index],
            )
            asset.extend(assets)
            world.append(body)
        for index, spec in enumerate(distractors, 1):
            assets, body = build_distractor(spec, index)
            asset.extend(assets)
            world.append(body)
        # Default keyframe positions follow the freshly assembled body poses.
        home = root.find("./keyframe/key[@name='home']")
        if home is not None:
            home.attrib.pop("qpos", None)
        xml = _resolve_scene_xml_paths(ET.tostring(root, encoding="unicode"), self._base_scene_xml_dir)
        robot_qpos = env._get_reset_robot_qpos()
        if xml != env._scene_xml_string:
            env.reload_from_xml(xml)
        env._set_robot_qpos(robot_qpos)
        self._variants = dict(zip(_BOTTLE_JOINTS, variants))
        self._distractors = {f"distractor_{i}_joint": spec for i, spec in enumerate(distractors, 1)}
        self.perturbations = [
            PerturbRange(name, delta_x=(0, 0), delta_y=(0, 0)) for name in (*self._variants, *self._distractors)
        ]

    def randomize(self, model: Any, data: Any, seed: int | None = None, request: Any = None) -> RandomizationState:
        rng = np.random.default_rng(seed)
        variants, distractors = self._identities(request, rng)
        self._reload_scene(variants, distractors)
        model, data = self._env_ref.model, self._env_ref.data
        joints = (*self._variants, *self._distractors)
        half_sizes = {
            joint: (
                _water_bottle_flat_compiled_metadata(self._variants[joint])[:2]
                if joint in self._variants
                else distractor_half_extents(self._distractors[joint])
            )
            for joint in joints
        }
        placement_order = sorted(joints, key=lambda joint: np.prod(half_sizes[joint]), reverse=True)
        for _ in range(self.max_tries):
            # Park unplaced objects apart, outside the robot's workspace.
            self._apply_states(
                model,
                data,
                {joint: {"pos": [-5 - index, 0, 2], "quat": [1, 0, 0, 0]} for index, joint in enumerate(joints)},
            )
            states = {}
            for joint in placement_order:
                for _ in range(self.max_tries):
                    candidate = {**states, joint: self._sample_object(joint, rng)}
                    if not self._pairwise_ok(candidate):
                        continue
                    self._apply_states(model, data, candidate)
                    mujoco.mj_forward(model, data)
                    if self._contacts_ok(model, data):
                        states = candidate
                        break
                else:
                    break
            if len(states) == len(joints):
                return RandomizationState(
                    seed=seed or 0,
                    object_states=states,
                    metadata={
                        "bottle_count": 3,
                        "bottle_variants": variants,
                        "distractor_count": len(distractors),
                        "distractors": [asdict(spec) for spec in distractors],
                        "prompt": LIFT_BOTTLES_UPRIGHT_PROMPTS[(seed or 0) % len(LIFT_BOTTLES_UPRIGHT_PROMPTS)],
                    },
                )
        self._raise_sampling_failure(seed)

    def _sample_object(self, joint, rng):
        xmin, xmax, ymin, ymax = self.table_bounds
        yaw = float(rng.uniform(-np.pi, np.pi))
        if joint in self._variants:
            half_length, half_radius, radius = _water_bottle_flat_compiled_metadata(self._variants[joint])
            half = (half_length, half_radius)
            offset = half_length * np.array([np.cos(yaw), np.sin(yaw)])
            z = 0.75 + radius + 0.002
            quat = _water_bottle_flat_quat(yaw)
        else:
            half = distractor_half_extents(self._distractors[joint])
            offset, z = np.zeros(2), 0.752
            quat = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        dx, dy = WaterBottleRandomizer._rotated_half_extents(*half, yaw)
        center = np.array([rng.uniform(xmin + dx, xmax - dx), rng.uniform(ymin + dy, ymax - dy)])
        return {"pos": [*(center - offset).tolist(), z], "quat": quat.tolist()}

    def _footprint(self, joint, state):
        if joint in self._variants:
            variant = self._variants[joint]
            half_length, half_radius, _ = _water_bottle_flat_compiled_metadata(variant)
            return (
                _water_bottle_flat_center_from_pose(**state, variant_name=variant, scale_factor=1.0),
                (half_length, half_radius),
                _water_bottle_flat_yaw_from_quat(np.asarray(state["quat"])),
            )
        return (
            np.asarray(state["pos"][:2]),
            distractor_half_extents(self._distractors[joint]),
            2 * np.arctan2(state["quat"][3], state["quat"][0]),
        )

    def _bounds_ok(self, states):
        xmin, xmax, ymin, ymax = self.table_bounds
        for joint, state in states.items():
            center, half, yaw = self._footprint(joint, state)
            dx, dy = WaterBottleRandomizer._rotated_half_extents(*half, yaw)
            if not (
                xmin <= center[0] - dx and center[0] + dx <= xmax and ymin <= center[1] - dy and center[1] + dy <= ymax
            ):
                return False
        return True

    def _pairwise_ok(self, states):
        footprints = [self._footprint(joint, state) for joint, state in states.items()]
        return all(
            not WaterBottleRandomizer._obb_overlap(*first, *second, margin=0.02)
            for first, second in combinations(footprints, 2)
        )

    def _validate_scene(self, model, state):
        if state.scale_states:
            raise ValueError("Upright bottles use the assets' native sizes.")
        variants, distractors = self._identities(
            {key: state.metadata[key] for key in ("bottle_variants", "distractors")},
            np.random.default_rng(0),
        )
        expected = {*_BOTTLE_JOINTS, *(f"distractor_{i}_joint" for i in range(1, len(distractors) + 1))}
        if set(state.object_states) != expected:
            raise ValueError("Upright bottles replay requires every bottle and distractor pose.")
        if state.metadata["bottle_count"] != len(variants) or state.metadata["distractor_count"] != len(distractors):
            raise ValueError("Replay object counts disagree with asset identities.")
        if state.metadata["prompt"] not in LIFT_BOTTLES_UPRIGHT_PROMPTS:
            raise ValueError("Unknown upright-bottles episode prompt.")
        for pose in state.object_states.values():
            pos, quat = np.asarray(pose["pos"]), np.asarray(pose["quat"])
            if pos.shape != (3,) or not np.isfinite(pos).all():
                raise ValueError("Object positions must be finite 3-vectors.")
            if quat.shape != (4,) or not np.isfinite(quat).all() or not np.isclose(np.linalg.norm(quat), 1):
                raise ValueError("Object orientations must be unit wxyz quaternions.")

    def _restore_scene(self, model, data, state):
        self._reload_scene(
            state.metadata["bottle_variants"], [DistractorSpec(**item) for item in state.metadata["distractors"]]
        )
        model, data = self._env_ref.model, self._env_ref.data
        self._apply_states(model, data, state.object_states)
        mujoco.mj_forward(model, data)
        return state
