"""Seeded fruit identities, distinct bowl colors, and clear tabletop layouts."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import asdict
from itertools import combinations

import mujoco
import numpy as np

from abc_sim.task_specs import PACK_AND_POUR_FRUIT_TEMPLATES

from ..assets.fruit_bowls import (
    BOWL_COLORS,
    BOWL_SEGMENTS,
    FRUIT_CATEGORIES,
    FRUIT_RADIUS_RANGE_M,
    FruitSpec,
    build_bowl,
    build_fruit,
    fruit_variants,
    source_bowl_spec,
    target_bowl_spec,
)
from ..core import PerturbRange, RandomizationState, SceneRandomizer, _resolve_scene_xml_paths

JOINTS = ("source_bowl_joint", "target_bowl_joint", *(f"fruit_{i}_joint" for i in range(1, 4)))


class PackAndPourFruitRandomizer(SceneRandomizer):
    table_bounds = (0.35, 0.86, -0.52, 0.52)
    perturbations = [PerturbRange(name, delta_x=(0, 0), delta_y=(0, 0)) for name in JOINTS]

    def prepare_reset(self, *, request=None, randomize=True, seed=None, keep_arm_state=False):
        if not randomize:
            raise ValueError(
                "Fruit pouring requires a randomized reset or saved-state replay; reuse a seed for a fixed scene."
            )
        self._validate_request(request)

    @staticmethod
    def _validate_request(request):
        options = {} if request is None else dict(request)
        unknown = set(options) - {"fruits", "source_color", "target_color", "prompt_index"}
        if unknown:
            raise ValueError(f"Unknown fruit-pouring reset options: {sorted(unknown)}")
        if "fruits" in options:
            options["fruits"] = [FruitSpec(**item) for item in options["fruits"]]
            if len(options["fruits"]) != 3:
                raise ValueError("Fruit pouring requires exactly three fruits.")
        for key in ("source_color", "target_color"):
            if key in options and options[key] not in BOWL_COLORS:
                raise ValueError("Choose two distinct, named bowl colors.")
        if "source_color" in options and options["source_color"] == options.get("target_color"):
            raise ValueError("Choose two distinct, named bowl colors.")
        if "prompt_index" in options:
            index = options["prompt_index"]
            if type(index) is not int or not 0 <= index < len(PACK_AND_POUR_FRUIT_TEMPLATES):
                raise ValueError("Unknown fruit-pouring prompt_index.")
        return options

    def _identities(self, request, rng):
        options = self._validate_request(request)
        fruits = (
            options["fruits"]
            if "fruits" in options
            else [
                FruitSpec(
                    str(category),
                    str(rng.choice(fruit_variants(str(category)))),
                    float(rng.uniform(*FRUIT_RADIUS_RANGE_M)),
                )
                for category in rng.choice(FRUIT_CATEGORIES, size=3, replace=False)
            ]
        )
        source = (
            options["source_color"]
            if "source_color" in options
            else str(rng.choice([c for c in BOWL_COLORS if c != options.get("target_color")]))
        )
        target = (
            options["target_color"]
            if "target_color" in options
            else str(rng.choice([c for c in BOWL_COLORS if c != source]))
        )
        index = (
            options["prompt_index"]
            if "prompt_index" in options
            else int(rng.integers(len(PACK_AND_POUR_FRUIT_TEMPLATES)))
        )
        return fruits, source, target, index

    def _reload_scene(self, fruits, source, target):
        env = self._env_ref
        if env is None:
            raise RuntimeError("Fruit-pouring randomization requires a bound environment.")
        root = ET.fromstring(self._base_scene_xml_string)
        asset, world = root.find("asset"), root.find("worldbody")
        for parent in (asset, world):
            for elem in list(parent):
                if elem.get("name", "").startswith(("source_bowl", "target_bowl", "fruit_")):
                    parent.remove(elem)
        bowls = (("source_bowl", source_bowl_spec(source)), ("target_bowl", target_bowl_spec(target)))
        for name, spec in bowls:
            assets, body = build_bowl(name, spec)
            asset.extend(assets)
            world.append(body)
        for i, spec in enumerate(fruits, 1):
            assets, body = build_fruit(spec, i)
            asset.extend(assets)
            world.append(body)
        root.find("./keyframe/key[@name='home']").attrib.pop("qpos", None)
        xml = _resolve_scene_xml_paths(ET.tostring(root, encoding="unicode"), self._base_scene_xml_dir)
        robot_qpos = env._get_reset_robot_qpos()
        if xml != env._scene_xml_string:
            env.reload_from_xml(xml)
        env._set_robot_qpos(robot_qpos)
        self._radii = {
            **{f"{name}_joint": (spec.rim_radius + spec.wall) / np.cos(np.pi / BOWL_SEGMENTS) for name, spec in bowls},
            **{f"fruit_{i}_joint": fruit.radius for i, fruit in enumerate(fruits, 1)},
        }

    def _pairwise_ok(self, states):
        return all(
            np.linalg.norm(np.asarray(first["pos"][:2]) - second["pos"][:2]) >= self._radii[a] + self._radii[b] + 0.025
            for (a, first), (b, second) in combinations(states.items(), 2)
        )

    @staticmethod
    def _pose(x, y, z, yaw=0):
        return {"pos": [float(x), float(y), float(z)], "quat": [float(np.cos(yaw / 2)), 0, 0, float(np.sin(yaw / 2))]}

    def randomize(self, model, data, seed=None, request=None):
        rng = np.random.default_rng(seed)
        fruits, source, target, index = self._identities(request, rng)
        self._reload_scene(fruits, source, target)
        model, data = self._env_ref.model, self._env_ref.data
        for _ in range(self.max_tries):
            mirror = rng.choice((-1, 1))
            states = {
                "source_bowl_joint": self._pose(rng.uniform(0.61, 0.65), mirror * rng.uniform(0.23, 0.27), 0.75),
                "target_bowl_joint": self._pose(rng.uniform(0.64, 0.67), -mirror * rng.uniform(0.23, 0.27), 0.75),
            }
            for i, fruit in enumerate(fruits, 1):
                for _ in range(self.max_tries):
                    pose = self._pose(
                        rng.uniform(0.38, 0.49),
                        rng.uniform(-0.47, 0.47),
                        0.751 + fruit.radius,
                        rng.uniform(-np.pi, np.pi),
                    )
                    candidate = {**states, f"fruit_{i}_joint": pose}
                    if self._pairwise_ok(candidate):
                        states = candidate
                        break
                else:
                    break
            if len(states) != 5:
                continue
            self._apply_states(model, data, states)
            mujoco.mj_forward(model, data)
            if self._contacts_ok(model, data):
                return RandomizationState(
                    seed=seed or 0,
                    object_states=states,
                    metadata={
                        "fruits": [asdict(f) for f in fruits],
                        "source_color": source,
                        "target_color": target,
                        "prompt_index": index,
                        "prompt": PACK_AND_POUR_FRUIT_TEMPLATES[index].format(source_color=source, target_color=target),
                    },
                )
        self._raise_sampling_failure(seed)

    def _validate_scene(self, model, state):
        fruits, source, target, index = self._identities(
            {k: state.metadata[k] for k in ("fruits", "source_color", "target_color", "prompt_index")},
            np.random.default_rng(0),
        )
        if state.scale_states or set(state.object_states) != set(JOINTS):
            raise ValueError("Fruit-pouring replay requires five free-body poses and no scales.")
        expected = PACK_AND_POUR_FRUIT_TEMPLATES[index].format(source_color=source, target_color=target)
        if state.metadata["prompt"] != expected:
            raise ValueError("The fruit-pouring prompt must match the bowl colors.")
        for pose in state.object_states.values():
            pos, quat = np.asarray(pose["pos"]), np.asarray(pose["quat"])
            if pos.shape != (3,) or not np.isfinite(pos).all():
                raise ValueError("Replay positions must be finite 3-vectors.")
            if quat.shape != (4,) or not np.isfinite(quat).all() or not np.isclose(np.linalg.norm(quat), 1):
                raise ValueError("Replay rotations must be unit wxyz quaternions.")

    def _restore_scene(self, model, data, state):
        self._reload_scene(
            [FruitSpec(**item) for item in state.metadata["fruits"]],
            state.metadata["source_color"],
            state.metadata["target_color"],
        )
        self._apply_states(self._env_ref.model, self._env_ref.data, state.object_states)
        mujoco.mj_forward(self._env_ref.model, self._env_ref.data)
        return state
