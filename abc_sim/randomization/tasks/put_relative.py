from __future__ import annotations

import re
import xml.etree.ElementTree as _ET
from pathlib import Path as _Path
from typing import Any

import mujoco
import numpy as np

from ..assets.inhand import _inhand_parse_model_xml
from ..assets.paths import _MODELS_DIR
from ..core import (
    RandomizationState,
    SceneRandomizer,
    _parse_float_list,
    _resolve_scene_xml_paths,
)

_PUT_RELATIVE_BASE_SCENE_XML = _MODELS_DIR / "yam_put_relative_scene.xml"
_PUT_RELATIVE_OBJECT_ASSET_ROOT = _MODELS_DIR / "assets" / "task_put_relative" / "objects"
_PUT_RELATIVE_TASK_ASSETS_PLACEHOLDER = "<!-- RELATIVE_PUT_TASK_ASSETS_PLACEHOLDER -->"
_PUT_RELATIVE_OBJECTS_BEGIN = "<!-- RELATIVE_PUT_OBJECTS_BEGIN -->"
_PUT_RELATIVE_OBJECTS_END = "<!-- RELATIVE_PUT_OBJECTS_END -->"
_PUT_RELATIVE_OBJECT_COUNT_RANGE = (4, 6)
_PUT_RELATIVE_OBJECT_CATEGORIES: tuple[str, ...] = (
    "mug",
    "coffee_cup",
    "bowl",
    "jar",
    "cereal",
    "ketchup",
    "mustard",
    "mayonnaise",
    "juice",
    "water_bottle",
    "soap_dispenser",
    "apple",
    "lemon",
    "orange",
    "pear",
    "donut",
    "cupcake",
    "boxed_food",
    "boxed_drink",
    "canned_food",
    "can",
    "yogurt",
    "bar_soap",
    "sponge",
    "tomato",
    "potato",
    "marshmallow",
    "cookie_dough_ball",
)
_PUT_RELATIVE_CATEGORY_LABELS: dict[str, str] = {
    "bar_soap": "bar soap",
    "boxed_drink": "boxed drink",
    "boxed_food": "boxed food",
    "canned_food": "canned food",
    "coffee_cup": "coffee cup",
    "cookie_dough_ball": "cookie dough ball",
    "soap_dispenser": "soap dispenser",
    "water_bottle": "water bottle",
}
_PUT_RELATIVE_TABLE_Z = 0.75
_PUT_RELATIVE_SPAWN_CLEARANCE_M = 0.015
_PUT_RELATIVE_OBJECT_Z_CLEARANCE_M = 0.002
_PUT_RELATIVE_GOAL_CLEARANCE_M = 0.012
_PUT_RELATIVE_GOAL_OFFSET_M = 0.14
_PUT_RELATIVE_GOAL_RADIUS_M = 0.055
_PUT_RELATIVE_SLOT_JITTER_M = 0.003
_PUT_RELATIVE_MAX_LAYOUT_TRIES = 64
_PUT_RELATIVE_MAX_PROMPT_TRIES = 64
_PUT_RELATIVE_TABLE_BOUNDS = (0.39, 0.79, -0.46, 0.46)
_PUT_RELATIVE_SLOTS: tuple[tuple[float, float], ...] = (
    (0.43, 0.36),
    (0.55, 0.36),
    (0.67, 0.36),
    (0.43, 0.18),
    (0.55, 0.18),
    (0.67, 0.18),
    (0.43, -0.18),
    (0.55, -0.18),
    (0.67, -0.18),
    (0.43, -0.36),
    (0.55, -0.36),
    (0.67, -0.36),
)
_PUT_RELATIVE_DIRECTION_OFFSETS: dict[str, tuple[float, float]] = {
    # Table convention: front is toward the robot/open side of the workspace.
    "front": (-_PUT_RELATIVE_GOAL_OFFSET_M, 0.0),
    "behind": (_PUT_RELATIVE_GOAL_OFFSET_M, 0.0),
    "left": (0.0, _PUT_RELATIVE_GOAL_OFFSET_M),
    "right": (0.0, -_PUT_RELATIVE_GOAL_OFFSET_M),
}
_PUT_RELATIVE_DIRECTION_PHRASES: dict[str, str] = {
    "front": "in front of",
    "behind": "behind",
    "left": "to the left of",
    "right": "to the right of",
}
_PUT_RELATIVE_HOME_KEY_RE = re.compile(
    r'(<key name="home"\s+qpos=")(.*?)("\s+ctrl=)',
    re.DOTALL,
)
_PUT_RELATIVE_ARM_QPOS = "0 1.047 1.047 0 0 0 0 0  0 1.047 1.047 0 0 0 0 0"


def _put_relative_apply_scene_transforms(xml: str, options: Any = None) -> str:
    if options is None:
        return xml
    if not (options.clean or options.mocap or options.flexible_gripper):
        return xml

    from abc_sim.scene_xml import transform_scene_xml

    transformed_xml, _ = transform_scene_xml(xml, options=options)
    return transformed_xml


def _put_relative_xy_clearance_ok(
    candidate: dict[str, Any],
    placed: list[dict[str, Any]],
    *,
    clearance: float,
) -> bool:
    candidate_xy = np.array([float(candidate["x"]), float(candidate["y"])], dtype=np.float64)
    candidate_radius = float(candidate["footprint_radius"])
    for other in placed:
        other_xy = np.array([float(other["x"]), float(other["y"])], dtype=np.float64)
        min_distance = candidate_radius + float(other["footprint_radius"]) + clearance
        if float(np.linalg.norm(candidate_xy - other_xy)) < min_distance:
            return False
    return True


def _put_relative_goal_ok(
    goal_xy: np.ndarray,
    *,
    mover_name: str,
    mover_radius: float,
    objects: list[dict[str, Any]],
) -> bool:
    x_min, x_max, y_min, y_max = _PUT_RELATIVE_TABLE_BOUNDS
    if not (
        x_min + mover_radius <= float(goal_xy[0]) <= x_max - mover_radius
        and y_min + mover_radius <= float(goal_xy[1]) <= y_max - mover_radius
    ):
        return False
    for obj in objects:
        if obj["name"] == mover_name:
            continue
        obj_xy = np.array([float(obj["x"]), float(obj["y"])], dtype=np.float64)
        min_distance = (
            float(obj["footprint_radius"]) + mover_radius + _PUT_RELATIVE_GOAL_CLEARANCE_M
        )
        if float(np.linalg.norm(goal_xy - obj_xy)) < min_distance:
            return False
    return True


def _put_relative_category_label(category: str) -> str:
    return _PUT_RELATIVE_CATEGORY_LABELS.get(category, category.replace("_", " "))


def _put_relative_variant_metrics(variant_dir: _Path) -> dict[str, float]:
    root = _ET.parse(str(variant_dir / "model.xml")).getroot()
    bbox = root.find('.//geom[@name="reg_bbox"]')
    if bbox is None:
        raise RuntimeError(f"put_relative asset variant has no reg_bbox: {variant_dir}")
    bbox_size = _parse_float_list(bbox.get("size", ""))
    if len(bbox_size) < 3:
        raise RuntimeError(f"put_relative asset reg_bbox has invalid size: {variant_dir}")

    parsed = _inhand_parse_model_xml(variant_dir)
    box_collision_geoms = [
        geom for geom in parsed["col_geoms"] if geom.get("type", "mesh") == "box"
    ]
    if not box_collision_geoms:
        raise RuntimeError(f"put_relative asset variant has no box collision proxy: {variant_dir}")

    bottom_z_values: list[float] = []
    footprint_radii: list[float] = []
    for geom in box_collision_geoms:
        pos = _parse_float_list(geom.get("pos", "0 0 0"))
        size = _parse_float_list(geom.get("size", ""))
        if len(size) < 3:
            continue
        pos_z = pos[2] if len(pos) >= 3 else 0.0
        bottom_z_values.append(pos_z - size[2])
        footprint_radii.append(float(np.linalg.norm(size[:2])))

    if not bottom_z_values or not footprint_radii:
        raise RuntimeError(f"put_relative asset variant has invalid collision proxy: {variant_dir}")

    return {
        "spawn_z": float(
            _PUT_RELATIVE_TABLE_Z
            + _PUT_RELATIVE_OBJECT_Z_CLEARANCE_M
            - min(bottom_z_values)
        ),
        "footprint_radius": float(max(footprint_radii)),
    }


def _put_relative_get_variants(category: str) -> list[_Path]:
    cat_dir = _PUT_RELATIVE_OBJECT_ASSET_ROOT / category
    if not cat_dir.is_dir():
        raise FileNotFoundError(f"put_relative asset category is missing: {cat_dir}")

    variants: list[_Path] = []
    for variant_dir in sorted(cat_dir.iterdir()):
        if not variant_dir.is_dir() or not (variant_dir / "model.xml").is_file():
            continue
        _put_relative_variant_metrics(variant_dir)
        variants.append(variant_dir)

    if not variants:
        raise RuntimeError(f"put_relative asset category has no variants: {category}")
    return variants


def _put_relative_replace_object_block(base_text: str, object_xml: str) -> str:
    start = base_text.find(_PUT_RELATIVE_OBJECTS_BEGIN)
    end = base_text.find(_PUT_RELATIVE_OBJECTS_END)
    if start < 0 or end < 0 or end <= start:
        raise RuntimeError("put_relative XML is missing object block markers")
    end += len(_PUT_RELATIVE_OBJECTS_END)
    return base_text[:start] + object_xml + base_text[end:]


def _put_relative_replace_home_qpos(base_text: str, selections: list[dict[str, Any]]) -> str:
    qpos_lines = []
    for selection in selections:
        qpos_lines.append(
            f'{float(selection["x"]):.4f} {float(selection["y"]):.4f} {float(selection["z"]):.4f} '
            f'{float(selection["qw"]):.6f} 0 0 {float(selection["qz"]):.6f}'
        )
    qpos_lines.append(_PUT_RELATIVE_ARM_QPOS)
    qpos = "\n            ".join(qpos_lines)
    replaced, count = _PUT_RELATIVE_HOME_KEY_RE.subn(
        lambda match: match.group(1) + qpos + match.group(3),
        base_text,
        count=1,
    )
    if count != 1:
        raise RuntimeError("put_relative XML is missing home key qpos")
    return replaced


def _put_relative_build_xml(selections: list[dict[str, Any]]) -> str:
    base_text = _PUT_RELATIVE_BASE_SCENE_XML.read_text()
    if _PUT_RELATIVE_TASK_ASSETS_PLACEHOLDER not in base_text:
        raise RuntimeError("put_relative XML is missing task asset placeholder")

    lines_asset: list[str] = []
    lines_body: list[str] = [_PUT_RELATIVE_OBJECTS_BEGIN]
    for index, selection in enumerate(selections, start=1):
        name = str(selection["name"])
        category = str(selection["category"])
        variant_dir = _Path(selection["variant_dir"]).resolve()
        parsed = _inhand_parse_model_xml(variant_dir)
        prefix = f"relative_obj_{index:02d}"
        used_mesh_names = {
            visual_geom["mesh"]
            for visual_geom in parsed["vis_geoms"]
            if visual_geom["mesh"]
        }
        used_mesh_names.update(
            collision_geom["mesh"]
            for collision_geom in parsed["col_geoms"]
            if collision_geom.get("mesh")
        )

        lines_asset.append(f"    <!-- Put-relative object {index}: {category}/{variant_dir.name} -->")
        for mesh in parsed["meshes"]:
            if mesh["name"] not in used_mesh_names:
                continue
            extra = {**mesh["extra"]}
            extra_str = "".join(f' {key}="{value}"' for key, value in extra.items())
            lines_asset.append(
                f'    <mesh file="{variant_dir / mesh["file"]}"'
                f' name="{prefix}_{mesh["name"]}"{extra_str}/>'
            )
        for texture in parsed["textures"]:
            lines_asset.append(
                f'    <texture file="{variant_dir / texture["file"]}"'
                f' name="{prefix}_{texture["name"]}" type="{texture["type"]}"/>'
            )
        for material in parsed["materials"]:
            attrs = f'name="{prefix}_{material["name"]}"'
            if material["texture"]:
                attrs += f' texture="{prefix}_{material["texture"]}"'
            if material["rgba"]:
                attrs += f' rgba="{material["rgba"]}"'
            if material["shininess"]:
                attrs += f' shininess="{material["shininess"]}"'
            if material["specular"]:
                attrs += f' specular="{material["specular"]}"'
            lines_asset.append(f"    <material {attrs}/>")

        lines_body.extend(
            [
                f"    <!-- Put-relative object {index}: {category}/{variant_dir.name} -->",
                f'    <body name="{name}" pos="{float(selection["x"]):.4f} {float(selection["y"]):.4f} {float(selection["z"]):.4f}" quat="{float(selection["qw"]):.6f} 0 0 {float(selection["qz"]):.6f}">',
                f'      <joint name="{name}_joint" class="relative_object"/>',
            ]
        )
        for visual_index, visual_geom in enumerate(parsed["vis_geoms"]):
            mesh_attr = (
                f'mesh="{prefix}_{visual_geom["mesh"]}"'
                if visual_geom["mesh"]
                else ""
            )
            mat_attr = (
                f' material="{prefix}_{visual_geom["material"]}"'
                if visual_geom["material"]
                else ""
            )
            lines_body.append(
                f'      <geom name="{name}_visual_{visual_index}" type="mesh"'
                f' {mesh_attr}{mat_attr} contype="0" conaffinity="0" group="2"'
                f' density="0" solimp="0.998 0.998 0.001" solref="0.001 1"/>'
            )

        for collision_index, collision_geom in enumerate(parsed["col_geoms"]):
            attrs: dict[str, str] = {
                "name": f"{name}_geom" if collision_index == 0 else f"{name}_geom_{collision_index}",
            }
            for key, value in collision_geom.items():
                if not value:
                    continue
                if key == "mesh":
                    attrs[key] = f"{prefix}_{value}"
                else:
                    attrs[key] = value
            attrs.setdefault("type", "mesh")
            attrs_str = "".join(f' {key}="{value}"' for key, value in attrs.items())
            lines_body.append(f"      <geom{attrs_str}/>")
        lines_body.append("    </body>")

    lines_body.append(f"    {_PUT_RELATIVE_OBJECTS_END}")
    xml = base_text.replace(_PUT_RELATIVE_TASK_ASSETS_PLACEHOLDER, "\n".join(lines_asset))
    xml = _put_relative_replace_object_block(xml, "\n".join(lines_body))
    xml = _put_relative_replace_home_qpos(xml, selections)
    return _resolve_scene_xml_paths(xml, _PUT_RELATIVE_BASE_SCENE_XML.parent)


def _put_relative_sample_prompt(
    objects: list[dict[str, Any]],
    rng: np.random.Generator,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for mover in objects:
        mover_xy = np.array([float(mover["x"]), float(mover["y"])], dtype=np.float64)
        mover_radius = float(mover["footprint_radius"])
        for reference in objects:
            if reference["name"] == mover["name"]:
                continue
            reference_xy = np.array([float(reference["x"]), float(reference["y"])], dtype=np.float64)
            for direction, offset in _PUT_RELATIVE_DIRECTION_OFFSETS.items():
                goal_xy = reference_xy + np.array(offset, dtype=np.float64)
                if not _put_relative_goal_ok(
                    goal_xy,
                    mover_name=str(mover["name"]),
                    mover_radius=mover_radius,
                    objects=objects,
                ):
                    continue
                if float(np.linalg.norm(mover_xy - goal_xy)) < (
                    _PUT_RELATIVE_GOAL_RADIUS_M + mover_radius
                ):
                    continue
                phrase = _PUT_RELATIVE_DIRECTION_PHRASES[direction]
                mover_label = str(mover["label"])
                reference_label = str(reference["label"])
                candidates.append(
                    {
                        "prompt": (
                            f"put the {mover_label} {phrase} the {reference_label}"
                        ),
                        "prompt_type": "relative_category_object",
                        "mover_object": mover["name"],
                        "reference_object": reference["name"],
                        "mover_category": mover["category"],
                        "reference_category": reference["category"],
                        "mover_label": mover_label,
                        "reference_label": reference_label,
                        "mover_variant": mover["variant"],
                        "reference_variant": reference["variant"],
                        "direction": direction,
                        "goal_position": [
                            float(goal_xy[0]),
                            float(goal_xy[1]),
                            float(mover["z"]),
                        ],
                        "goal_radius": _PUT_RELATIVE_GOAL_RADIUS_M,
                        "direction_offset_xy": [float(offset[0]), float(offset[1])],
                        "axis_convention": {
                            "front": "-x",
                            "behind": "+x",
                            "left": "+y",
                            "right": "-y",
                        },
                    }
                )
    if not candidates:
        raise RuntimeError("put_relative generated no valid prompt candidates")
    return candidates[int(rng.integers(0, len(candidates)))]


class PutRelativeRandomizer(SceneRandomizer):
    """Reload put_relative with sampled object assets and a relative prompt."""

    perturbations: list = []

    def __init__(self) -> None:
        super().__init__()
        self._variants: dict[str, list[_Path]] = {}

    def clone(self) -> "SceneRandomizer":
        return type(self)()

    def prepare_env(self) -> None:
        if self._env_ref is not None:
            self.randomize(self._env_ref.model, self._env_ref.data)

    def _get_variants(self, category: str) -> list[_Path]:
        if category not in self._variants:
            self._variants[category] = _put_relative_get_variants(category)
        return self._variants[category]

    def _sample_selections(self, rng: np.random.Generator) -> list[dict[str, Any]]:
        count_min, count_max = _PUT_RELATIVE_OBJECT_COUNT_RANGE
        object_count = int(rng.integers(count_min, count_max + 1))
        categories = rng.choice(
            _PUT_RELATIVE_OBJECT_CATEGORIES,
            size=object_count,
            replace=False,
        ).tolist()
        available_slots = list(range(len(_PUT_RELATIVE_SLOTS)))
        rng.shuffle(available_slots)

        selections: list[dict[str, Any]] = []
        for index, category in enumerate(categories):
            variants = self._get_variants(str(category))
            variant_dir = variants[int(rng.integers(0, len(variants)))]
            metrics = _put_relative_variant_metrics(variant_dir)
            yaw = float(rng.uniform(-np.pi, np.pi))
            for slot in available_slots:
                x, y = _PUT_RELATIVE_SLOTS[slot]
                for _ in range(_PUT_RELATIVE_MAX_LAYOUT_TRIES):
                    candidate = {
                        "name": f"relative_obj_{index + 1:02d}",
                        "joint": f"relative_obj_{index + 1:02d}_joint",
                        "category": str(category),
                        "label": _put_relative_category_label(str(category)),
                        "variant": variant_dir.name,
                        "variant_dir": str(variant_dir),
                        "x": float(x + rng.uniform(-_PUT_RELATIVE_SLOT_JITTER_M, _PUT_RELATIVE_SLOT_JITTER_M)),
                        "y": float(y + rng.uniform(-_PUT_RELATIVE_SLOT_JITTER_M, _PUT_RELATIVE_SLOT_JITTER_M)),
                        "z": float(metrics["spawn_z"]),
                        "yaw": yaw,
                        "qw": float(np.cos(yaw / 2)),
                        "qz": float(np.sin(yaw / 2)),
                        "footprint_radius": float(metrics["footprint_radius"]),
                    }
                    if _put_relative_xy_clearance_ok(
                        candidate,
                        selections,
                        clearance=_PUT_RELATIVE_SPAWN_CLEARANCE_M,
                    ):
                        selections.append(candidate)
                        break
                else:
                    continue
                available_slots.remove(slot)
                break
            else:
                raise RuntimeError("put_relative could not place non-overlapping objects")
        return selections

    def randomize(
        self,
        model: Any,
        data: Any,
        seed: int | None = None,
        request: Any | None = None,
    ) -> RandomizationState:
        rng = np.random.default_rng(seed)
        for _ in range(_PUT_RELATIVE_MAX_PROMPT_TRIES):
            selections = self._sample_selections(rng)
            try:
                prompt_info = _put_relative_sample_prompt(selections, rng)
                break
            except RuntimeError:
                continue
        else:
            raise RuntimeError("put_relative could not sample a valid relative prompt")

        xml = _put_relative_build_xml(selections)
        xml = _put_relative_apply_scene_transforms(
            xml,
            self._scene_xml_transform_options,
        )

        env = self._env_ref
        if env is not None:
            preserved_robot_qpos = env._get_reset_robot_qpos()
            env.reload_from_xml(xml)
            mujoco.mj_resetData(env.model, env.data)
            env._set_robot_qpos(preserved_robot_qpos)
            env.prompt = str(prompt_info["prompt"])
            mujoco.mj_forward(env.model, env.data)

        object_states: dict[str, dict[str, list[float]]] = {}
        metadata_objects: list[dict[str, Any]] = []
        for selection in selections:
            joint = str(selection["joint"])
            object_states[joint] = {
                "pos": [
                    float(selection["x"]),
                    float(selection["y"]),
                    float(selection["z"]),
                ],
                "quat": [float(selection["qw"]), 0.0, 0.0, float(selection["qz"])],
            }
            metadata_objects.append(
                {
                    "name": selection["name"],
                    "joint": joint,
                    "category": selection["category"],
                    "label": selection["label"],
                    "variant": selection["variant"],
                }
            )

        metadata = {
            "prompt": str(prompt_info["prompt"]),
            "prompt_type": prompt_info["prompt_type"],
            "mover_object": prompt_info["mover_object"],
            "reference_object": prompt_info["reference_object"],
            "mover_category": prompt_info["mover_category"],
            "reference_category": prompt_info["reference_category"],
            "mover_label": prompt_info["mover_label"],
            "reference_label": prompt_info["reference_label"],
            "mover_variant": prompt_info["mover_variant"],
            "reference_variant": prompt_info["reference_variant"],
            "direction": prompt_info["direction"],
            "goal_position": prompt_info["goal_position"],
            "goal_radius": prompt_info["goal_radius"],
            "direction_offset_xy": prompt_info["direction_offset_xy"],
            "axis_convention": prompt_info["axis_convention"],
            "objects": metadata_objects,
        }
        return RandomizationState(
            seed=seed or 0,
            object_states=object_states,
            metadata=metadata,
        )




__all__ = ["PutRelativeRandomizer"]
