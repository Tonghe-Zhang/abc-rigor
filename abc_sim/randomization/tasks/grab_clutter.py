from __future__ import annotations

import re
from pathlib import Path as _Path
from typing import Any

import mujoco
import numpy as np

from ..assets.inhand import _inhand_parse_model_xml
from ..assets.paths import _MODELS_DIR
from ..assets.xml_common import _apply_bin_visual_styles_to_xml
from ..core import (
    RandomizationState,
    SceneRandomizer,
    _apply_object_scales_to_scene_xml,
    _format_float_list,
    _parse_float_list,
    _resolve_scene_xml_paths,
    _scaled_mesh_attr,
)
from .count_box import _COUNT_BOX_ARM_QPOS, _count_box_apply_scene_transforms
from .put_relative import (
    _PUT_RELATIVE_HOME_KEY_RE,
    _PUT_RELATIVE_TABLE_Z,
    _put_relative_category_label,
    _put_relative_variant_metrics,
    _put_relative_xy_clearance_ok,
)

_GRAB_CLUTTER_BASE_SCENE_XML = _MODELS_DIR / "yam_grab_clutter_scene.xml"
_GRAB_CLUTTER_OBJECT_ASSET_ROOT = _MODELS_DIR / "assets" / "task_grab_clutter" / "objects"
_GRAB_CLUTTER_TASK_ASSETS_PLACEHOLDER = "<!-- GRAB_CLUTTER_TASK_ASSETS_PLACEHOLDER -->"
_GRAB_CLUTTER_OBJECTS_BEGIN = "<!-- GRAB_CLUTTER_OBJECTS_BEGIN -->"
_GRAB_CLUTTER_OBJECTS_END = "<!-- GRAB_CLUTTER_OBJECTS_END -->"
_GRAB_CLUTTER_OBJECT_CATEGORIES: tuple[str, ...] = (
    "mug",
    "water_bottle",
    "soap_dispenser",
    "boxed_food",
    "boxed_drink",
    "canned_food",
    "bar_soap",
    "sponge",
    "apple",
    "lemon",
    "donut",
    "yogurt",
    "jar",
    "bowl",
    "coffee_cup",
    "can",
    "cereal",
    "ketchup",
    "mustard",
    "mayonnaise",
    "orange",
    "tomato",
    "potato",
    "cupcake",
)
_GRAB_CLUTTER_OBJECT_COUNT = len(_GRAB_CLUTTER_OBJECT_CATEGORIES)
_GRAB_CLUTTER_OBJECT_COUNT_RANGE = (8, 16)
_GRAB_CLUTTER_SCALE_FACTOR_RANGE = (1.0, 1.0)
_GRAB_CLUTTER_MAX_USED_MESH_BYTES = 8 * 1024 * 1024
_GRAB_CLUTTER_SPAWN_CLEARANCE_M = 0.002
_GRAB_CLUTTER_SLOT_JITTER_M = 0.006
_GRAB_CLUTTER_SLOT_JITTER_ATTEMPTS = 8
_GRAB_CLUTTER_MAX_LAYOUT_TRIES = 32
_GRAB_CLUTTER_TARGET_BOX_BODY = "grab_clutter_target_box"
_GRAB_CLUTTER_TARGET_BOX_JOINT = "grab_clutter_target_box_joint"
_GRAB_CLUTTER_TARGET_BOX_X_RANGE = (0.43, 0.77)
_GRAB_CLUTTER_TARGET_BOX_SIDE_Y = 0.515
_GRAB_CLUTTER_TARGET_BOX_Y_JITTER_M = 0.010
_GRAB_CLUTTER_TARGET_BOX_SCALE_FACTOR_RANGE = (1.0, 1.0)
_GRAB_CLUTTER_TARGET_BOX_YAW_RANGE = (-np.pi, np.pi)
_GRAB_CLUTTER_TARGET_BOX_VISUAL_STYLE = "low_poly_crate"
_GRAB_CLUTTER_ALLOWED_TABLE_GEOMS = {"table_plane", "floor_collision"}
_GRAB_CLUTTER_OBJECT_SLOTS: tuple[tuple[float, float], ...] = tuple(
    (x, y)
    for y in (0.30, 0.15, 0.0, -0.15, -0.30)
    for x in (0.43, 0.515, 0.60, 0.685, 0.77)
)
_GRAB_CLUTTER_HOME_KEY_RE = _PUT_RELATIVE_HOME_KEY_RE
_GRAB_CLUTTER_TARGET_BOX_BODY_RE = re.compile(
    r'(<body name="grab_clutter_target_box"\s+pos=")([^"]*)("\s+quat=")([^"]*)(")'
)


def _grab_clutter_get_variants(category: str) -> list[_Path]:
    cat_dir = _GRAB_CLUTTER_OBJECT_ASSET_ROOT / category
    if not cat_dir.is_dir():
        raise FileNotFoundError(f"grab_clutter asset category is missing: {cat_dir}")

    variants: list[_Path] = []
    for variant_dir in sorted(cat_dir.iterdir()):
        if not variant_dir.is_dir() or not (variant_dir / "model.xml").is_file():
            continue
        _put_relative_variant_metrics(variant_dir)
        if (
            _grab_clutter_variant_used_mesh_bytes(variant_dir)
            <= _GRAB_CLUTTER_MAX_USED_MESH_BYTES
        ):
            variants.append(variant_dir)

    if not variants:
        raise RuntimeError(f"grab_clutter category has no lightweight variants: {category}")
    return variants


def _grab_clutter_variant_used_mesh_bytes(variant_dir: _Path) -> int:
    parsed = _inhand_parse_model_xml(variant_dir)
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
    total_bytes = 0
    for mesh in parsed["meshes"]:
        if mesh["name"] not in used_mesh_names:
            continue
        mesh_file = variant_dir / mesh["file"]
        if mesh_file.is_file():
            total_bytes += mesh_file.stat().st_size
    return total_bytes


def _grab_clutter_scaled_float_attr(value: str, factor: float) -> str:
    return _format_float_list([part * factor for part in _parse_float_list(value)])


def _grab_clutter_variant_metrics(variant_dir: _Path, scale_factor: float) -> dict[str, float]:
    metrics = _put_relative_variant_metrics(variant_dir)
    return {
        "spawn_z": float(
            _PUT_RELATIVE_TABLE_Z
            + (float(metrics["spawn_z"]) - _PUT_RELATIVE_TABLE_Z) * scale_factor
        ),
        "footprint_radius": float(metrics["footprint_radius"]) * scale_factor,
    }


def _grab_clutter_replace_object_block(base_text: str, object_xml: str) -> str:
    start = base_text.find(_GRAB_CLUTTER_OBJECTS_BEGIN)
    end = base_text.find(_GRAB_CLUTTER_OBJECTS_END)
    if start < 0 or end < 0 or end <= start:
        raise RuntimeError("grab_clutter XML is missing object block markers")
    end += len(_GRAB_CLUTTER_OBJECTS_END)
    return base_text[:start] + object_xml + base_text[end:]


def _grab_clutter_replace_target_box_pose(base_text: str, target_box: dict[str, Any]) -> str:
    pos = f'{float(target_box["x"]):.4f} {float(target_box["y"]):.4f} {float(target_box["z"]):.4f}'
    quat = f'{float(target_box["qw"]):.6f} 0 0 {float(target_box["qz"]):.6f}'
    replaced, count = _GRAB_CLUTTER_TARGET_BOX_BODY_RE.subn(
        lambda match: match.group(1) + pos + match.group(3) + quat + match.group(5),
        base_text,
        count=1,
    )
    if count != 1:
        raise RuntimeError("grab_clutter XML is missing target box body pose")
    return replaced


def _grab_clutter_replace_home_qpos(
    base_text: str,
    selections: list[dict[str, Any]],
    target_box: dict[str, Any],
) -> str:
    qpos_lines = [
        f'{float(target_box["x"]):.4f} {float(target_box["y"]):.4f} {float(target_box["z"]):.4f} '
        f'{float(target_box["qw"]):.6f} 0 0 {float(target_box["qz"]):.6f}'
    ]
    for selection in selections:
        qpos_lines.append(
            f'{float(selection["x"]):.4f} {float(selection["y"]):.4f} {float(selection["z"]):.4f} '
            f'{float(selection["qw"]):.6f} 0 0 {float(selection["qz"]):.6f}'
        )
    qpos_lines.append(_COUNT_BOX_ARM_QPOS)
    qpos = "\n            ".join(qpos_lines)
    replaced, count = _GRAB_CLUTTER_HOME_KEY_RE.subn(
        lambda match: match.group(1) + qpos + match.group(3),
        base_text,
        count=1,
    )
    if count != 1:
        raise RuntimeError("grab_clutter XML is missing home key qpos")
    return replaced


def _grab_clutter_build_xml(
    selections: list[dict[str, Any]],
    target_box: dict[str, Any],
) -> str:
    base_text = _GRAB_CLUTTER_BASE_SCENE_XML.read_text()
    if _GRAB_CLUTTER_TASK_ASSETS_PLACEHOLDER not in base_text:
        raise RuntimeError("grab_clutter XML is missing task asset placeholder")
    base_text = _grab_clutter_replace_target_box_pose(base_text, target_box)

    lines_asset: list[str] = []
    lines_body: list[str] = [_GRAB_CLUTTER_OBJECTS_BEGIN]
    for selection in selections:
        name = str(selection["name"])
        joint_name = str(selection["joint"])
        category = str(selection["category"])
        variant_dir = _Path(selection["variant_dir"]).resolve()
        scale_factor = float(selection.get("scale_factor", 1.0))
        parsed = _inhand_parse_model_xml(variant_dir)
        prefix = name
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

        lines_asset.append(f"    <!-- Grab clutter object: {category}/{variant_dir.name} -->")
        for mesh in parsed["meshes"]:
            if mesh["name"] not in used_mesh_names:
                continue
            extra = {**mesh["extra"]}
            if abs(scale_factor - 1.0) > 1e-9:
                extra["scale"] = _scaled_mesh_attr(extra.get("scale"), scale_factor)
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
                f"    <!-- Grab clutter object: {category}/{variant_dir.name} -->",
                f'    <body name="{name}" pos="{float(selection["x"]):.4f} {float(selection["y"]):.4f} {float(selection["z"]):.4f}" quat="{float(selection["qw"]):.6f} 0 0 {float(selection["qz"]):.6f}">',
                f'      <joint name="{joint_name}" class="clutter_object"/>',
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
                elif key in {"pos", "size", "fromto"} and abs(scale_factor - 1.0) > 1e-9:
                    attrs[key] = _grab_clutter_scaled_float_attr(value, scale_factor)
                elif key == "mass" and abs(scale_factor - 1.0) > 1e-9:
                    attrs[key] = f"{float(value) * scale_factor ** 3:.9g}"
                else:
                    attrs[key] = value
            attrs.setdefault("type", "mesh")
            attrs_str = "".join(f' {key}="{value}"' for key, value in attrs.items())
            lines_body.append(f"      <geom{attrs_str}/>")
        lines_body.append("    </body>")

    lines_body.append(f"    {_GRAB_CLUTTER_OBJECTS_END}")
    xml = base_text.replace(_GRAB_CLUTTER_TASK_ASSETS_PLACEHOLDER, "\n".join(lines_asset))
    xml = _grab_clutter_replace_object_block(xml, "\n".join(lines_body))
    xml = _grab_clutter_replace_home_qpos(xml, selections, target_box)
    xml = _apply_bin_visual_styles_to_xml(
        xml,
        {_GRAB_CLUTTER_TARGET_BOX_BODY: str(target_box["visual_style"])},
    )
    if abs(float(target_box["scale_factor"]) - 1.0) > 1e-9:
        xml = _apply_object_scales_to_scene_xml(
            xml,
            {_GRAB_CLUTTER_TARGET_BOX_JOINT: float(target_box["scale_factor"])},
        )
    return _resolve_scene_xml_paths(xml, _GRAB_CLUTTER_BASE_SCENE_XML.parent)


class GrabClutterRandomizer(SceneRandomizer):
    """Reload grab_clutter with one target object and many clutter distractors."""

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
            self._variants[category] = _grab_clutter_get_variants(category)
        return self._variants[category]

    def _sample_selections(self, rng: np.random.Generator) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        categories = list(_GRAB_CLUTTER_OBJECT_CATEGORIES)
        rng.shuffle(categories)
        object_count_min, object_count_max = _GRAB_CLUTTER_OBJECT_COUNT_RANGE
        object_count = int(
            rng.integers(object_count_min, min(object_count_max, len(categories)) + 1)
        )
        categories = categories[:object_count]
        target_category = categories[int(rng.integers(0, len(categories)))]

        specs: list[dict[str, Any]] = []
        distractor_index = 1
        for category in categories:
            variants = self._get_variants(str(category))
            variant_dir = variants[int(rng.integers(0, len(variants)))]
            scale_factor = float(rng.uniform(*_GRAB_CLUTTER_SCALE_FACTOR_RANGE))
            metrics = _grab_clutter_variant_metrics(variant_dir, scale_factor)
            yaw = float(rng.uniform(-np.pi, np.pi))
            is_target = category == target_category
            name = "target_object" if is_target else f"clutter_obj_{distractor_index:02d}"
            joint = "target_object_joint" if is_target else f"clutter_obj_{distractor_index:02d}_joint"
            specs.append(
                {
                    "name": name,
                    "joint": joint,
                    "category": str(category),
                    "label": _put_relative_category_label(str(category)),
                    "variant": variant_dir.name,
                    "variant_dir": str(variant_dir),
                    "scale_factor": scale_factor,
                    "z": float(metrics["spawn_z"]),
                    "yaw": yaw,
                    "qw": float(np.cos(yaw / 2)),
                    "qz": float(np.sin(yaw / 2)),
                    "footprint_radius": float(metrics["footprint_radius"]),
                    "is_target": is_target,
                }
            )
            if not is_target:
                distractor_index += 1

        for _ in range(_GRAB_CLUTTER_MAX_LAYOUT_TRIES):
            selections: list[dict[str, Any]] = []
            target_selection: dict[str, Any] | None = None
            for spec in specs:
                selection: dict[str, Any] | None = None
                for clearance in (_GRAB_CLUTTER_SPAWN_CLEARANCE_M, 0.0):
                    slot_order = list(range(len(_GRAB_CLUTTER_OBJECT_SLOTS)))
                    rng.shuffle(slot_order)
                    for slot_index in slot_order:
                        slot_x, slot_y = _GRAB_CLUTTER_OBJECT_SLOTS[slot_index]
                        for _ in range(_GRAB_CLUTTER_SLOT_JITTER_ATTEMPTS):
                            candidate = {
                                **spec,
                                "x": float(
                                    slot_x
                                    + rng.uniform(
                                        -_GRAB_CLUTTER_SLOT_JITTER_M,
                                        _GRAB_CLUTTER_SLOT_JITTER_M,
                                    )
                                ),
                                "y": float(
                                    slot_y
                                    + rng.uniform(
                                        -_GRAB_CLUTTER_SLOT_JITTER_M,
                                        _GRAB_CLUTTER_SLOT_JITTER_M,
                                    )
                                ),
                            }
                            if _put_relative_xy_clearance_ok(
                                candidate,
                                selections,
                                clearance=clearance,
                            ):
                                selection = candidate
                                break
                        if selection is not None:
                            break
                    if selection is not None:
                        break
                if selection is None:
                    break
                selections.append(selection)
                if bool(selection["is_target"]):
                    target_selection = selection
            else:
                if target_selection is None:
                    raise RuntimeError("grab_clutter failed to sample a target object")
                return selections, target_selection

        raise RuntimeError("grab_clutter could not place non-overlapping objects")

    def _sample_target_box(self, rng: np.random.Generator) -> dict[str, Any]:
        yaw = float(rng.uniform(*_GRAB_CLUTTER_TARGET_BOX_YAW_RANGE))
        y_sign = -1.0 if int(rng.integers(0, 2)) == 0 else 1.0
        perimeter_side = "negative_y_edge" if y_sign < 0.0 else "positive_y_edge"
        return {
            "name": _GRAB_CLUTTER_TARGET_BOX_BODY,
            "joint": _GRAB_CLUTTER_TARGET_BOX_JOINT,
            "x": float(rng.uniform(*_GRAB_CLUTTER_TARGET_BOX_X_RANGE)),
            "y": float(
                y_sign
                * (
                    _GRAB_CLUTTER_TARGET_BOX_SIDE_Y
                    + rng.uniform(
                        -_GRAB_CLUTTER_TARGET_BOX_Y_JITTER_M,
                        _GRAB_CLUTTER_TARGET_BOX_Y_JITTER_M,
                    )
                )
            ),
            "z": 0.75,
            "yaw": yaw,
            "qw": float(np.cos(yaw / 2)),
            "qz": float(np.sin(yaw / 2)),
            "scale_factor": float(rng.uniform(*_GRAB_CLUTTER_TARGET_BOX_SCALE_FACTOR_RANGE)),
            "visual_style": _GRAB_CLUTTER_TARGET_BOX_VISUAL_STYLE,
            "perimeter_side": perimeter_side,
        }

    def _selection_object_body_ids(
        self,
        model: Any,
        selections: list[dict[str, Any]],
    ) -> set[int]:
        body_ids: set[int] = set()
        for selection in selections:
            joint_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                str(selection["joint"]),
            )
            if joint_id >= 0:
                body_ids.add(int(model.jnt_bodyid[joint_id]))
        return body_ids

    def _layout_contacts_ok(
        self,
        model: Any,
        data: Any,
        selections: list[dict[str, Any]],
    ) -> bool:
        object_body_ids = self._selection_object_body_ids(model, selections)
        if not object_body_ids:
            return False

        arm_root_ids = self._arm_root_body_ids(model)
        box_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _GRAB_CLUTTER_TARGET_BOX_BODY)
        box_body_ids = {int(box_body_id)} if box_body_id >= 0 else set()
        table_geom_ids = {
            int(geom_id)
            for geom_name in _GRAB_CLUTTER_ALLOWED_TABLE_GEOMS
            if (geom_id := mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)) >= 0
        }

        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            geom_1 = int(contact.geom1)
            geom_2 = int(contact.geom2)
            body_1 = int(model.geom_bodyid[geom_1])
            body_2 = int(model.geom_bodyid[geom_2])
            root_1 = int(model.body_rootid[body_1])
            root_2 = int(model.body_rootid[body_2])
            body_1_is_object = self._body_in_any_subtree(model, body_1, object_body_ids)
            body_2_is_object = self._body_in_any_subtree(model, body_2, object_body_ids)
            body_1_is_box = self._body_in_any_subtree(model, body_1, box_body_ids)
            body_2_is_box = self._body_in_any_subtree(model, body_2, box_body_ids)

            if body_1_is_object and body_2_is_object:
                if root_1 != root_2:
                    return False
                continue
            if body_1_is_object != body_2_is_object:
                other_body = body_2 if body_1_is_object else body_1
                other_geom = geom_2 if body_1_is_object else geom_1
                if other_geom in table_geom_ids:
                    continue
                if self._body_in_any_subtree(model, other_body, arm_root_ids):
                    return False
                if self._body_in_any_subtree(model, other_body, box_body_ids):
                    return False
                return False
            if body_1_is_box != body_2_is_box:
                other_body = body_2 if body_1_is_box else body_1
                other_geom = geom_2 if body_1_is_box else geom_1
                if other_geom in table_geom_ids:
                    continue
                if self._body_in_any_subtree(model, other_body, arm_root_ids):
                    return False
                return False
        return True

    def randomize(
        self,
        model: Any,
        data: Any,
        seed: int | None = None,
        request: Any | None = None,
    ) -> RandomizationState:
        rng = np.random.default_rng(seed)
        env = self._env_ref
        accepted: tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], str] | None = None
        max_tries = _GRAB_CLUTTER_MAX_LAYOUT_TRIES if env is not None else 1

        for _ in range(max_tries):
            selections, target_selection = self._sample_selections(rng)
            target_box = self._sample_target_box(rng)
            prompt = f'grab the {target_selection["label"]} and place it in the box'
            xml = _grab_clutter_build_xml(selections, target_box)
            xml = _count_box_apply_scene_transforms(
                xml,
                self._scene_xml_transform_options,
            )

            if env is None:
                accepted = (selections, target_selection, target_box, prompt)
                break

            preserved_robot_qpos = env._get_reset_robot_qpos()
            env.reload_from_xml(xml)
            mujoco.mj_resetData(env.model, env.data)
            env._set_robot_qpos(preserved_robot_qpos)
            mujoco.mj_forward(env.model, env.data)
            if not self._layout_contacts_ok(env.model, env.data, selections):
                continue
            env.prompt = prompt
            accepted = (selections, target_selection, target_box, prompt)
            break

        if accepted is None:
            self._raise_sampling_failure(seed)

        selections, target_selection, target_box, prompt = accepted

        object_states: dict[str, dict[str, list[float]]] = {
            _GRAB_CLUTTER_TARGET_BOX_JOINT: {
                "pos": [
                    float(target_box["x"]),
                    float(target_box["y"]),
                    float(target_box["z"]),
                ],
                "quat": [float(target_box["qw"]), 0.0, 0.0, float(target_box["qz"])],
            }
        }
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
                    "scale_factor": selection["scale_factor"],
                    "is_target": selection["is_target"],
                }
            )

        metadata = {
            "prompt": prompt,
            "prompt_type": "target_object_from_clutter_to_box",
            "target_object": target_selection["name"],
            "target_joint": target_selection["joint"],
            "target_category": target_selection["category"],
            "target_label": target_selection["label"],
            "target_variant": target_selection["variant"],
            "target_scale_factor": target_selection["scale_factor"],
            "target_box_body": _GRAB_CLUTTER_TARGET_BOX_BODY,
            "target_box_joint": _GRAB_CLUTTER_TARGET_BOX_JOINT,
            "target_box_goal_site": "grab_clutter_target_box_goal_region",
            "target_box_scale_factor": target_box["scale_factor"],
            "target_box_visual_style": target_box["visual_style"],
            "target_box_perimeter_side": target_box["perimeter_side"],
            "objects": metadata_objects,
        }
        return RandomizationState(
            seed=seed or 0,
            object_states=object_states,
            scale_states={_GRAB_CLUTTER_TARGET_BOX_JOINT: float(target_box["scale_factor"])},
            metadata=metadata,
        )
