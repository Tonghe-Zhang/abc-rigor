"""Seeded suitcase appearance, object identities, and collision-free layouts."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import asdict
from itertools import combinations

import mujoco
import numpy as np

from abc_sim.task_specs import PACK_SUITCASE_PROMPTS

from ..assets.suitcase import (
    PACKING_CATEGORIES,
    SHELL_PALETTES,
    SHELL_PATTERNS,
    SuitcaseTexture,
    build_packing_object,
    build_suitcase,
    packing_object_dimensions,
    packing_variants,
)
from ..assets.tabletop_distractors import DistractorSpec
from ..core import PerturbRange, RandomizationState, SceneRandomizer, _resolve_scene_xml_paths
from .bottles import WaterBottleRandomizer

OBJECT_JOINTS = tuple(f"packing_object_{i}_joint" for i in range(1, 4))


class PackSuitcaseRandomizer(SceneRandomizer):
    """Place an open suitcase and three packable objects on the tabletop."""

    table_bounds = (0.35, 0.83, -0.52, 0.52)
    perturbations = [PerturbRange(name, delta_x=(0, 0), delta_y=(0, 0)) for name in ("suitcase_joint", *OBJECT_JOINTS)]

    def _identities(self, request, rng):
        options = {} if request is None else dict(request)
        unknown = set(options) - {"objects", "texture", "lid_angle_deg"}
        if unknown:
            raise ValueError(f"Unknown suitcase reset options: {sorted(unknown)}")
        if "objects" in options:
            objects = [DistractorSpec(**item) for item in options["objects"]]
        else:
            objects = [
                DistractorSpec(str(category), str(rng.choice(packing_variants(str(category)))))
                for category in rng.choice(PACKING_CATEGORIES, size=3, replace=False)
            ]
        if len(objects) != 3 or any(obj.category not in PACKING_CATEGORIES for obj in objects):
            raise ValueError("Suitcase packing requires exactly three approved household objects.")
        texture = (
            SuitcaseTexture(**options["texture"])
            if "texture" in options
            else SuitcaseTexture(str(rng.choice(tuple(SHELL_PALETTES))), str(rng.choice(SHELL_PATTERNS)))
        )
        angle = float(options["lid_angle_deg"] if "lid_angle_deg" in options else rng.uniform(105, 110))
        if not np.isfinite(angle) or not 105 <= angle <= 110:
            raise ValueError("Initial suitcase lid_angle_deg must be between 105 and 110.")
        return objects, texture, angle

    def _reload_scene(self, objects, texture):
        env = self._env_ref
        if env is None:
            raise RuntimeError("Suitcase randomization requires a bound environment.")
        root = ET.fromstring(self._base_scene_xml_string)
        asset, world = root.find("asset"), root.find("worldbody")
        for parent in (asset, world):
            for elem in list(parent):
                if elem.get("name", "").startswith(("suitcase", "packing_object_")):
                    parent.remove(elem)
        assets, case = build_suitcase(texture)
        asset.extend(assets)
        world.append(case)
        for i, spec in enumerate(objects, 1):
            assets, body = build_packing_object(spec, i)
            asset.extend(assets)
            world.append(body)
        root.find("./keyframe/key[@name='home']").attrib.pop("qpos", None)
        xml = _resolve_scene_xml_paths(ET.tostring(root, encoding="unicode"), self._base_scene_xml_dir)
        robot_qpos = env._get_reset_robot_qpos()
        if xml != env._scene_xml_string:
            env.reload_from_xml(xml)
        env._set_robot_qpos(robot_qpos)
        self._objects = dict(zip(OBJECT_JOINTS, objects))

    @staticmethod
    def _pose(x, y, yaw, z=0.752):
        return {"pos": [float(x), float(y), z], "quat": [float(np.cos(yaw / 2)), 0, 0, float(np.sin(yaw / 2))]}

    def _footprint(self, joint, pose):
        yaw = 2 * np.arctan2(pose["quat"][3], pose["quat"][0])
        center = np.asarray(pose["pos"][:2])
        if joint == "suitcase_joint":
            # Include the front handle and the open lid's swept projection.
            center = center + np.array([np.cos(yaw), np.sin(yaw)]) * 0.04
            half = (0.21, 0.20)
        else:
            half = np.asarray(packing_object_dimensions(self._objects[joint])[1][:2]) / 2
        return center, half, yaw

    def _pairwise_ok(self, states):
        footprints = [self._footprint(joint, pose) for joint, pose in states.items()]
        return all(
            not WaterBottleRandomizer._obb_overlap(*first, *second, margin=0.025)
            for first, second in combinations(footprints, 2)
        )

    def randomize(self, model, data, seed=None, request=None):
        rng = np.random.default_rng(seed)
        objects, texture, angle = self._identities(request, rng)
        self._reload_scene(objects, texture)
        model, data = self._env_ref.model, self._env_ref.data
        for _ in range(self.max_tries):
            self._apply_states(
                model,
                data,
                {joint: {"pos": [-5 - i, 0, 2], "quat": [1, 0, 0, 0]} for i, joint in enumerate(OBJECT_JOINTS)},
            )
            states = {
                "suitcase_joint": self._pose(
                    rng.uniform(0.61, 0.63), rng.uniform(-0.04, 0.04), rng.uniform(-0.1, 0.1), z=0.75
                )
            }
            data.joint("suitcase_hinge").qpos[0] = np.deg2rad(angle)
            for joint in OBJECT_JOINTS:
                half = np.asarray(packing_object_dimensions(self._objects[joint])[1][:2]) / 2
                for _ in range(self.max_tries):
                    yaw = rng.uniform(-np.pi, np.pi)
                    dx, dy = WaterBottleRandomizer._rotated_half_extents(*half, yaw)
                    xmin, xmax, ymin, ymax = self.table_bounds
                    pose = self._pose(rng.uniform(xmin + dx, xmax - dx), rng.uniform(ymin + dy, ymax - dy), yaw)
                    candidate = {**states, joint: pose}
                    if not self._pairwise_ok(candidate):
                        continue
                    self._apply_states(model, data, candidate)
                    mujoco.mj_forward(model, data)
                    if self._contacts_ok(model, data):
                        states = candidate
                        break
                else:
                    break
            if len(states) == 4:
                return RandomizationState(
                    seed=seed or 0,
                    object_states=states,
                    metadata={
                        "objects": [asdict(obj) for obj in objects],
                        "texture": asdict(texture),
                        "lid_angle_deg": angle,
                        "prompt": PACK_SUITCASE_PROMPTS[(seed or 0) % len(PACK_SUITCASE_PROMPTS)],
                    },
                )
        self._raise_sampling_failure(seed)

    def _validate_scene(self, model, state):
        self._identities(
            {key: state.metadata[key] for key in ("objects", "texture", "lid_angle_deg")},
            np.random.default_rng(0),
        )
        if state.scale_states or set(state.object_states) != {"suitcase_joint", *OBJECT_JOINTS}:
            raise ValueError("Suitcase replay requires all four free-body poses and no additional scales.")
        if state.metadata["prompt"] not in PACK_SUITCASE_PROMPTS:
            raise ValueError("Unknown suitcase episode prompt.")
        for pose in state.object_states.values():
            pos, quat = np.asarray(pose["pos"]), np.asarray(pose["quat"])
            if pos.shape != (3,) or not np.isfinite(pos).all():
                raise ValueError("Suitcase scene positions must be finite 3-vectors.")
            if quat.shape != (4,) or not np.isfinite(quat).all() or not np.isclose(np.linalg.norm(quat), 1):
                raise ValueError("Suitcase scene orientations must be unit wxyz quaternions.")

    def _restore_scene(self, model, data, state):
        self._reload_scene(
            [DistractorSpec(**obj) for obj in state.metadata["objects"]],
            SuitcaseTexture(**state.metadata["texture"]),
        )
        model, data = self._env_ref.model, self._env_ref.data
        self._apply_states(model, data, state.object_states)
        data.joint("suitcase_hinge").qpos[0] = np.deg2rad(state.metadata["lid_angle_deg"])
        mujoco.mj_forward(model, data)
        return state
