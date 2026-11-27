"""Seeded RL conveyor episodes using the existing conveyor assets and scene."""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from abc_sim.randomization.core import RandomizationState
from abc_sim.randomization.tasks.conveyor_pick import (
    _CONVEYOR_OBJECT_CATEGORIES,
    _CONVEYOR_OBJECT_SLOTS,
    _CONVEYOR_X_JITTER_M,
    _CONVEYOR_Y_JITTER_M,
    ConveyorPickObjectRandomizer,
    _conveyor_belt_speed_range,
    _conveyor_pick_apply_scene_transforms,
    _conveyor_pick_build_xml,
    _conveyor_pick_spawn_z,
)

FRUIT = ("apple", "lemon", "lime", "orange", "tangerine")
OTHER_FOOD = (
    "donut",
    "bagel",
    "cupcake",
    "egg",
    "sandwich_bread",
    "hotdog_bun",
    "waffle",
    "pancake",
    "potato",
)
NON_FOOD = ("bar_soap", "sponge")
CATEGORY_POOL = FRUIT + OTHER_FOOD + NON_FOOD
TARGET_CATEGORIES = {
    "fruit": FRUIT,
    "food": FRUIT + OTHER_FOOD,
    "non-fruit": OTHER_FOOD + NON_FOOD,
    "non-food": NON_FOOD,
}


def episode_prompt(variant: str, category: str, count: int = 3) -> str:
    if variant == "all":
        return (
            f"Pick all {count} {'object' if count == 1 else 'objects'} from the conveyor "
            "and gently place them in the bin."
        )
    return (
        f"Pick only the {category} objects from the conveyor and gently place them in the bin. "
        "Leave all other objects on the belt to travel off its end."
    )


def validate_episode(metadata: dict) -> dict:
    """Validate the complete target definition without reading assets or physics."""
    variant, category, objects = (
        metadata["variant"],
        metadata["target_category"],
        metadata["objects"],
    )
    if metadata["task_category"] != "rl" or variant not in ("all", "category"):
        raise ValueError("Unknown RL conveyor episode variant.")
    if variant == "all":
        if category != "all" or not 1 <= len(objects) <= 3:
            raise ValueError(
                "The all-object conveyor variant requires one to three targets."
            )
        pool = _CONVEYOR_OBJECT_CATEGORIES
    else:
        if category not in TARGET_CATEGORIES or not 2 <= len(objects) <= 4:
            raise ValueError(
                "Category conveyor episodes require a known category and two to four objects."
            )
        pool = CATEGORY_POOL
    for index, obj in enumerate(objects, 1):
        expected = variant == "all" or obj["category"] in TARGET_CATEGORIES[category]
        if (
            obj["index"] != index
            or obj["joint"] != f"conveyor_object_{index}_joint"
            or obj["category"] not in pool
            or type(obj["target"]) is not bool
            or obj["target"] != expected
        ):
            raise ValueError(
                "Conveyor object identities and semantic target mask disagree."
            )
    count = sum(obj["target"] for obj in objects)
    if variant == "category" and (count not in (1, 2) or count == len(objects)):
        raise ValueError(
            "Category episodes require one or two targets and at least one distractor."
        )
    valid_prompts = {episode_prompt(variant, category, len(objects))}
    if variant == "all" and len(objects) == 3:
        valid_prompts.add(
            "Pick all three objects from the conveyor and gently place them in the bin."
        )
    if metadata["prompt"] not in valid_prompts:
        raise ValueError("Conveyor prompt must describe its scored target category.")
    speed = float(metadata["belt_speed_y"])
    if not np.isfinite(speed) or not 0.01 <= abs(speed) <= 0.50:
        raise ValueError("Conveyor belt speed is outside its supported range.")
    return copy.deepcopy(metadata)


def build_xml(selections: list[dict], options=None) -> str:
    """Reuse conveyor geometry with a home keyframe matching the sampled count."""
    root = ET.fromstring(_conveyor_pick_build_xml(selections))
    home = root.find("./keyframe/key[@name='home']")
    robot = home.attrib["qpos"].split()[-16:]
    objects = []
    for item in selections:
        objects.extend(
            (
                item["x"],
                item["y"],
                item["z"],
                np.cos(item["yaw"] / 2),
                0,
                0,
                np.sin(item["yaw"] / 2),
            )
        )
    home.set("qpos", " ".join(map(str, [0, 0, *objects, *robot])))
    return _conveyor_pick_apply_scene_transforms(
        ET.tostring(root, encoding="unicode"), options
    )


class RLConveyorRandomizer(ConveyorPickObjectRandomizer):
    """Sample all-object and semantic-category episodes with equal probability."""

    def _sample_episode(self, *, spawn_side_y: float):
        variant = "all" if self._rng.integers(2) == 0 else "category"
        category = "all"
        if variant == "all":
            count = int(self._rng.integers(1, 4))
            categories = list(self._rng.choice(_CONVEYOR_OBJECT_CATEGORIES, count))
        else:
            category = str(self._rng.choice(tuple(TARGET_CATEGORIES)))
            count = int(self._rng.integers(2, 5))
            target_count = int(self._rng.integers(1, min(2, count - 1) + 1))
            targets = TARGET_CATEGORIES[category]
            distractors = tuple(name for name in CATEGORY_POOL if name not in targets)
            categories = [
                *self._rng.choice(targets, target_count),
                *self._rng.choice(distractors, count - target_count),
            ]
            self._rng.shuffle(categories)
        selections = []
        for category_name, (x, y) in zip(categories, _CONVEYOR_OBJECT_SLOTS):
            variants = self._get_variants(str(category_name))
            path = variants[int(self._rng.integers(len(variants)))]
            selections.append(
                {
                    "category": str(category_name),
                    "variant_dir": path,
                    "variant": path.name,
                    "x": float(x + self._rng.uniform(*_CONVEYOR_X_JITTER_M)),
                    "y": float(
                        spawn_side_y * y + self._rng.uniform(*_CONVEYOR_Y_JITTER_M)
                    ),
                    "z": _conveyor_pick_spawn_z(path),
                    "yaw": float(self._rng.uniform(-np.pi, np.pi)),
                }
            )
        return variant, category, selections

    def _reload(self, selections):
        xml = build_xml(selections, self._scene_xml_transform_options)
        if self._env_ref is not None:
            env = self._env_ref
            robot = env._get_reset_robot_qpos()
            env.reload_from_xml(xml)
            mujoco.mj_resetData(env.model, env.data)
            env._set_robot_qpos(robot)
            mujoco.mj_forward(env.model, env.data)

    def randomize(self, model, data, seed=None, request=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        direction = 1.0 if self._rng.random() < 0.5 else -1.0
        variant, category, selections = self._sample_episode(spawn_side_y=-direction)
        speed_range = _conveyor_belt_speed_range(request, defaults=(0.05, 0.10))
        speed = direction * float(self._rng.uniform(*speed_range))
        objects, states = [], {}
        for index, item in enumerate(selections, 1):
            joint = f"conveyor_object_{index}_joint"
            objects.append(
                {
                    "index": index,
                    "joint": joint,
                    "category": item["category"],
                    "variant": item["variant"],
                    "target": variant == "all"
                    or item["category"] in TARGET_CATEGORIES[category],
                }
            )
            states[joint] = {
                "pos": [item["x"], item["y"], item["z"]],
                "quat": [
                    float(np.cos(item["yaw"] / 2)),
                    0.0,
                    0.0,
                    float(np.sin(item["yaw"] / 2)),
                ],
            }
        metadata = {
            "task_category": "rl",
            "variant": variant,
            "target_category": category,
            "objects": objects,
            "prompt": episode_prompt(variant, category, len(objects)),
            "belt_speed_y": speed,
            "belt_speed_m_s": abs(speed),
            "belt_direction_y": direction,
            "mirrored": direction > 0,
            "spawn_side_y": -direction,
            "belt_speed_range_m_s": list(speed_range),
        }
        validate_episode(metadata)
        self._reload(selections)
        if self._env_ref is not None:
            self._apply_states(self._env_ref.model, self._env_ref.data, states)
            mujoco.mj_forward(self._env_ref.model, self._env_ref.data)
        return RandomizationState(
            seed=seed or 0, object_states=states, metadata=metadata
        )

    def _replay_selections(self, state):
        metadata = validate_episode(state.metadata)
        if state.scale_states or set(state.object_states) != {
            obj["joint"] for obj in metadata["objects"]
        }:
            raise ValueError(
                "RL conveyor replay requires every object pose and no additional scales."
            )
        selections = []
        for obj in metadata["objects"]:
            variants = {path.name: path for path in self._get_variants(obj["category"])}
            if obj["variant"] not in variants:
                raise ValueError("Unknown conveyor asset variant.")
            pose = state.object_states[obj["joint"]]
            pos, quat = np.asarray(pose["pos"]), np.asarray(pose["quat"])
            if (
                pos.shape != (3,)
                or not np.isfinite(pos).all()
                or quat.shape != (4,)
                or not np.isfinite(quat).all()
                or not np.isclose(np.linalg.norm(quat), 1)
            ):
                raise ValueError(
                    "Conveyor replay requires finite positions and unit quaternions."
                )
            selections.append(
                {
                    **obj,
                    "variant_dir": variants[obj["variant"]],
                    "x": pos[0],
                    "y": pos[1],
                    "z": pos[2],
                    "yaw": 0.0,
                }
            )
        return selections

    def _validate_scene(self, model, state):
        self._replay_selections(state)

    def _restore_scene(self, model, data, state):
        if self._env_ref is None:
            raise RuntimeError("RL conveyor asset replay requires a bound environment.")
        self._reload(self._replay_selections(state))
        model, data = self._env_ref.model, self._env_ref.data
        self._apply_states(model, data, state.object_states)
        mujoco.mj_forward(model, data)
        return state
