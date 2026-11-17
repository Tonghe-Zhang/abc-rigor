from __future__ import annotations

import copy
import xml.etree.ElementTree as _ET
from pathlib import Path as _Path
from typing import Any

import mujoco
import numpy as np

from abc_sim.multi_drawer_episode import DrawerEpisode
from abc_sim.task_eval.multi_drawer import DrawerProgressSource, resolve_drawer_goal

from ..assets.inhand import _inhand_parse_model_xml
from ..assets.paths import _MODELS_DIR
from ..assets.xml_common import _find_xml_body_parent
from ..core import (
    RandomizationState,
    SceneRandomizer,
    _format_float_list,
    _quat_from_yaw,
    _quat_mul,
    _resolve_scene_xml_paths,
    _scaled_mesh_attr,
)
from ..requests import MultiDrawerSearchResetRequest
from .count_box import _COUNT_BOX_ARM_QPOS, _count_box_apply_scene_transforms
from .grab_clutter import _grab_clutter_scaled_float_attr

_MULTI_DRAWER_SEARCH_BASE_SCENE_XML = _MODELS_DIR / "yam_multi_drawer_search_scene.xml"
_MULTI_DRAWER_SEARCH_OBJECT_SETS: dict[str, tuple[dict[str, Any], ...]] = {
    "pantry": (
        {"category": "apple", "variant": "apple_10", "label": "apple", "scale_factor": 0.85},
        {"category": "lemon", "variant": "lemon_10", "label": "lemon", "scale_factor": 0.85},
        {"category": "orange", "variant": "orange_0", "label": "orange", "scale_factor": 0.85},
        {"category": "tomato", "variant": "tomato_1", "label": "tomato", "scale_factor": 0.85},
        {"category": "potato", "variant": "potato_13", "label": "potato", "scale_factor": 0.85},
        {"category": "boxed_food", "variant": "boxed_food_0", "label": "boxed food", "scale_factor": 0.80},
        {"category": "boxed_drink", "variant": "boxed_drink_1", "label": "boxed drink", "scale_factor": 0.80},
        {"category": "canned_food", "variant": "canned_food_11", "label": "canned food", "scale_factor": 0.80},
        {"category": "can", "variant": "can_12", "label": "can", "scale_factor": 0.80},
    ),
    "kitchen_tools": (
        {"category": "pizza_cutter", "variant": "PizzaCutter003", "label": "pizza cutter", "scale_factor": 0.62, "min_thickness_m": 0.024, "mass_kg": 0.07},
        {"category": "peeler", "variant": "Peeler004", "label": "vegetable peeler", "scale_factor": 0.68, "min_thickness_m": 0.024, "mass_kg": 0.055},
        {"category": "whisk", "variant": "Whisk003", "label": "whisk", "scale_factor": 0.52, "mass_kg": 0.08},
        {"category": "tongs", "variant": "Tongs007", "label": "tongs", "scale_factor": 0.52, "min_thickness_m": 0.024, "mass_kg": 0.09},
        {"category": "wooden_spoon", "variant": "WoodenSpoon005", "label": "wooden spoon", "scale_factor": 0.58, "min_thickness_m": 0.024, "mass_kg": 0.06},
        {"category": "dish_brush", "variant": "DishBrush004", "label": "dish brush", "scale_factor": 0.64, "mass_kg": 0.10},
        {"category": "measuring_cup", "variant": "MeasuringCup003", "label": "measuring cup", "scale_factor": 0.95, "mass_kg": 0.07},
        {"category": "reamer", "variant": "Reamer008", "label": "citrus reamer", "scale_factor": 0.55, "mass_kg": 0.12},
        {"category": "rolling_pin", "variant": "rolling_pin_0", "label": "rolling pin", "scale_factor": 0.62, "mass_kg": 0.18},
    ),
    "office": (
        {"category": "scissors", "variant": "scissors_gso", "label": "scissors", "scale_factor": 1.0, "min_thickness_m": 0.024, "mass_kg": 0.06},
        {"category": "tape_roll", "variant": "tape_roll_gso", "label": "roll of tape", "scale_factor": 1.0, "min_thickness_m": 0.024, "mass_kg": 0.04},
        {"category": "stapler", "variant": "red_stapler", "label": "stapler", "scale_factor": 1.0, "mass_kg": 0.12},
        {"category": "eraser", "variant": "worn_eraser", "label": "eraser", "scale_factor": 1.50, "min_thickness_m": 0.024, "mass_kg": 0.025},
        {"category": "calculator", "variant": "desk_calculator", "label": "calculator", "scale_factor": 1.0, "min_thickness_m": 0.024, "mass_kg": 0.10},
        {"category": "notebook", "variant": "spiral_notebook", "label": "notebook", "scale_factor": 1.0, "min_thickness_m": 0.024, "mass_kg": 0.12},
        {"category": "marker", "variant": "office_marker", "label": "marker", "scale_factor": 1.15, "mass_kg": 0.03},
        {"category": "pencil_sharpener", "variant": "table_sharpener", "label": "pencil sharpener", "scale_factor": 1.20, "mass_kg": 0.03},
    ),
}
_MULTI_DRAWER_SEARCH_OBJECT_SET_NAMES = tuple(_MULTI_DRAWER_SEARCH_OBJECT_SETS)
_MULTI_DRAWER_SEARCH_OBJECT_POOL: tuple[dict[str, Any], ...] = tuple(
    {
        **spec,
        "object_set": object_set,
    }
    for object_set, object_specs in _MULTI_DRAWER_SEARCH_OBJECT_SETS.items()
    for spec in object_specs
)
_MULTI_DRAWER_SEARCH_OBJECT_CATEGORIES = tuple(
    str(spec["category"]) for spec in _MULTI_DRAWER_SEARCH_OBJECT_POOL
)
_MULTI_DRAWER_SEARCH_OBJECT_LABELS = tuple(
    str(spec["label"]) for spec in _MULTI_DRAWER_SEARCH_OBJECT_POOL
)
_MULTI_DRAWER_SEARCH_OBJECT_COUNT = len(_MULTI_DRAWER_SEARCH_OBJECT_POOL)
_MULTI_DRAWER_SEARCH_OBJECT_COUNT_RANGE_BY_STACK_COUNT: dict[int, tuple[int, int]] = {
    1: (2, 5),
    2: (3, 7),
    3: (4, max(len(object_specs) for object_specs in _MULTI_DRAWER_SEARCH_OBJECT_SETS.values())),
}
_MULTI_DRAWER_SEARCH_TARGET_COUNT_RANGE = (2, 4)
_MULTI_DRAWER_SEARCH_STACK_IDS = (1, 2, 3)
_MULTI_DRAWER_SEARCH_STACK_COUNT_RANGE = (1, 3)
_MULTI_DRAWER_SEARCH_DRAWER_LEVELS = ("top", "middle", "bottom")
_MULTI_DRAWER_SEARCH_DRAWERS = tuple(
    f"stack_{stack_id}_{level}"
    for stack_id in _MULTI_DRAWER_SEARCH_STACK_IDS
    for level in _MULTI_DRAWER_SEARCH_DRAWER_LEVELS
)
_MULTI_DRAWER_SEARCH_TEMPLATE_DRAWER_JOINTS = (
    "joint_top_drawer_low",
    "joint_middle_drawer_low",
    "joint_bottom_drawer_low",
)
_MULTI_DRAWER_SEARCH_TEMPLATE_DRAWER_BODY = "drawer_body"
_MULTI_DRAWER_SEARCH_STACK_POSES = {
    1: {"pos": [0.731463, -0.47, 0.71770], "quat": [0.65433247, 0.0, 0.0, -0.756207]},
    2: {"pos": [0.731463, -0.11, 0.71770], "quat": [0.65433247, 0.0, 0.0, -0.756207]},
    3: {"pos": [0.731463, 0.25, 0.71770], "quat": [0.65433247, 0.0, 0.0, -0.756207]},
}
_MULTI_DRAWER_SEARCH_STACK_SLOT_CENTERS = (
    (0.731463, -0.47),
    (0.731463, -0.11),
    (0.731463, 0.25),
)
_MULTI_DRAWER_SEARCH_STACK_Y_JITTER_M = 0.005
_MULTI_DRAWER_SEARCH_STACK_YAW_JITTER_RAD = 0.12
_MULTI_DRAWER_SEARCH_BACK_WALL_X_M = 0.9
_MULTI_DRAWER_SEARCH_BACK_WALL_CLEARANCE_M = 0.005
_MULTI_DRAWER_SEARCH_STACK_VISUAL_FOOTPRINT_CORNERS = np.array(
    [
        [x, y, 0.0]
        for x in (-0.12301374, 0.12302820)
        for y in (-0.14158711, 0.14738981)
    ],
    dtype=np.float64,
)
_MULTI_DRAWER_SEARCH_PARK_Z = -0.95
_MULTI_DRAWER_SEARCH_GOAL_BIN_JOINT = "drawer_search_goal_bin_joint"
_MULTI_DRAWER_SEARCH_TARGET_JOINT = "drawer_search_target_joint"
_MULTI_DRAWER_SEARCH_DISTRACTOR_JOINTS = tuple(
    f"drawer_search_distractor_{index:02d}_joint"
    for index in range(1, _MULTI_DRAWER_SEARCH_OBJECT_COUNT)
)
_MULTI_DRAWER_SEARCH_OBJECT_JOINTS = (
    _MULTI_DRAWER_SEARCH_TARGET_JOINT,
    *_MULTI_DRAWER_SEARCH_DISTRACTOR_JOINTS,
)
_MULTI_DRAWER_SEARCH_OBJECT_BODIES = (
    "drawer_search_target",
    *(f"drawer_search_distractor_{index:02d}" for index in range(1, _MULTI_DRAWER_SEARCH_OBJECT_COUNT)),
)
_MULTI_DRAWER_SEARCH_JOINT_BY_BODY = dict(
    zip(_MULTI_DRAWER_SEARCH_OBJECT_BODIES, _MULTI_DRAWER_SEARCH_OBJECT_JOINTS)
)
_MULTI_DRAWER_SEARCH_OBJECT_SPEC_BY_BODY = dict(
    zip(_MULTI_DRAWER_SEARCH_OBJECT_BODIES, _MULTI_DRAWER_SEARCH_OBJECT_POOL)
)
_MULTI_DRAWER_SEARCH_OBJECT_BODIES_BY_SET = {
    object_set: tuple(
        body
        for body, spec in _MULTI_DRAWER_SEARCH_OBJECT_SPEC_BY_BODY.items()
        if spec["object_set"] == object_set
    )
    for object_set in _MULTI_DRAWER_SEARCH_OBJECT_SET_NAMES
}
_MULTI_DRAWER_SEARCH_TEMPLATE_STACK_POS = np.array(
    _MULTI_DRAWER_SEARCH_STACK_POSES[2]["pos"],
    dtype=np.float64,
)
_MULTI_DRAWER_SEARCH_STACK_QUAT = np.array(
    _MULTI_DRAWER_SEARCH_STACK_POSES[2]["quat"],
    dtype=np.float64,
)
_MULTI_DRAWER_SEARCH_DRAWER_Z = {
    "top": 0.2867,
    "middle": 0.2007,
    "bottom": 0.1133,
}
_MULTI_DRAWER_SEARCH_DRAWER_SLOTS = (
    (-0.0528, -0.012),
    (0.0528, -0.012),
)
_MULTI_DRAWER_SEARCH_OBJECT_LOCAL_Z_BY_LEVEL = {
    "top": 0.012,
    "middle": 0.006,
    "bottom": 0.0,
}
_MULTI_DRAWER_SEARCH_OBJECT_LOCAL_JITTER_M = 0.006
_MULTI_DRAWER_SEARCH_GOAL_BIN_POSE = {
    "pos": [0.44, 0.52, 0.75],
    "quat": [1.0, 0.0, 0.0, 0.0],
}
_MULTI_DRAWER_SEARCH_GOAL_BIN_Y_ABS_M = 0.52
_MULTI_DRAWER_SEARCH_ARM_QPOS = _COUNT_BOX_ARM_QPOS


def _multi_drawer_search_stack_body_name(stack_id: int) -> str:
    return f"drawer_stack_{int(stack_id)}"


def _multi_drawer_search_drawer_name(stack_id: int, level: str) -> str:
    return f"stack_{int(stack_id)}_{level}"


def _multi_drawer_search_drawer_joint_name(stack_id: int, level: str) -> str:
    return f"drawer_stack_{int(stack_id)}_joint_{level}_drawer_low"


def _multi_drawer_search_drawer_body_name(stack_id: int, level: str) -> str:
    return f"drawer_stack_{int(stack_id)}_link_{level}_drawer_low"


_MULTI_DRAWER_SEARCH_DRAWER_JOINTS = tuple(
    _multi_drawer_search_drawer_joint_name(stack_id, level)
    for stack_id in _MULTI_DRAWER_SEARCH_STACK_IDS
    for level in _MULTI_DRAWER_SEARCH_DRAWER_LEVELS
)
_MULTI_DRAWER_SEARCH_DRAWER_JOINT_BY_DRAWER = {
    _multi_drawer_search_drawer_name(stack_id, level): _multi_drawer_search_drawer_joint_name(stack_id, level)
    for stack_id in _MULTI_DRAWER_SEARCH_STACK_IDS
    for level in _MULTI_DRAWER_SEARCH_DRAWER_LEVELS
}
_MULTI_DRAWER_SEARCH_DRAWER_BODY_BY_DRAWER = {
    _multi_drawer_search_drawer_name(stack_id, level): _multi_drawer_search_drawer_body_name(stack_id, level)
    for stack_id in _MULTI_DRAWER_SEARCH_STACK_IDS
    for level in _MULTI_DRAWER_SEARCH_DRAWER_LEVELS
}


def _multi_drawer_search_rotation_from_quat(quat: np.ndarray) -> np.ndarray:
    q0, q1, q2, q3 = quat / np.linalg.norm(quat)
    return np.array(
        [
            [1 - 2 * (q2 * q2 + q3 * q3), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
            [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3), 2 * (q2 * q3 - q0 * q1)],
            [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1 * q1 + q2 * q2)],
        ],
        dtype=np.float64,
    )


def _multi_drawer_search_drawer_rotation() -> np.ndarray:
    return _multi_drawer_search_rotation_from_quat(_MULTI_DRAWER_SEARCH_STACK_QUAT)


_MULTI_DRAWER_SEARCH_ROT = _multi_drawer_search_drawer_rotation()
_MULTI_DRAWER_SEARCH_OBJECT_ASSET_ROOT = (
    _MODELS_DIR / "assets" / "task_multi_drawer_search" / "objects"
)


def _multi_drawer_search_world_pos(
    stack_pose: dict[str, list[float]],
    level: str,
    slot: tuple[float, float],
    rng: np.random.Generator | None = None,
) -> list[float]:
    local_x, local_y = slot
    if rng is not None:
        local_x += float(
            rng.uniform(
                -_MULTI_DRAWER_SEARCH_OBJECT_LOCAL_JITTER_M,
                _MULTI_DRAWER_SEARCH_OBJECT_LOCAL_JITTER_M,
            )
        )
        local_y += float(
            rng.uniform(
                -_MULTI_DRAWER_SEARCH_OBJECT_LOCAL_JITTER_M,
                _MULTI_DRAWER_SEARCH_OBJECT_LOCAL_JITTER_M,
            )
        )
    local = np.array(
        [
            local_x,
            local_y,
            _MULTI_DRAWER_SEARCH_DRAWER_Z[level]
            + _MULTI_DRAWER_SEARCH_OBJECT_LOCAL_Z_BY_LEVEL[level],
        ],
        dtype=np.float64,
    )
    stack_pos = np.asarray(stack_pose["pos"], dtype=np.float64)
    stack_quat = np.asarray(stack_pose["quat"], dtype=np.float64)
    stack_rot = _multi_drawer_search_rotation_from_quat(stack_quat)
    return (stack_pos + stack_rot @ local).tolist()


def _multi_drawer_search_item_quat(
    rng: np.random.Generator,
    *,
    object_set: str,
    stack_pose: dict[str, list[float]],
) -> list[float]:
    if object_set in {"kitchen_tools", "office"}:
        direction = np.pi if int(rng.integers(0, 2)) else 0.0
        local_quat = _quat_from_yaw(direction + float(rng.uniform(-0.18, 0.18)))
        quat = _quat_mul(
            np.asarray(stack_pose["quat"], dtype=np.float64),
            local_quat,
        )
    else:
        quat = _quat_from_yaw(float(rng.uniform(-np.pi, np.pi)))
    quat = quat / np.linalg.norm(quat)
    return quat.tolist()


def _multi_drawer_search_variant_dir(spec: dict[str, Any]) -> _Path:
    category = str(spec["category"])
    variant = str(spec["variant"])
    variant_dir = _MULTI_DRAWER_SEARCH_OBJECT_ASSET_ROOT / category / variant
    if not (variant_dir / "model.xml").is_file():
        raise FileNotFoundError(f"Missing multi-drawer object asset: {variant_dir}")
    return variant_dir


def _multi_drawer_search_rewrite_home_qpos(
    root: _ET.Element,
    selections: list[dict[str, Any]],
    active_stack_ids: list[int],
) -> None:
    key = root.find("./keyframe/key[@name='home']")
    if key is None:
        raise RuntimeError("multi_drawer_search XML is missing home key")
    qpos_lines = [_format_float_list([0.0] * (len(active_stack_ids) * len(_MULTI_DRAWER_SEARCH_DRAWER_LEVELS)))]
    bin_pose = _MULTI_DRAWER_SEARCH_GOAL_BIN_POSE
    qpos_lines.append(
        _format_float_list(
            [
                *bin_pose["pos"],
                *bin_pose["quat"],
            ]
        )
    )
    for selection in selections:
        qpos_lines.append(
            _format_float_list(
                [
                    *selection["pos"],
                    *selection["quat"],
                ]
            )
        )
    qpos_lines.append(_MULTI_DRAWER_SEARCH_ARM_QPOS)
    key.set("qpos", "\n            ".join(qpos_lines))


def _multi_drawer_search_remove_base_object_bodies(
    root: _ET.Element,
) -> tuple[_ET.Element, int]:
    insert_parent: _ET.Element | None = None
    insert_index: int | None = None
    for body_name in _MULTI_DRAWER_SEARCH_OBJECT_BODIES:
        try:
            parent, body = _find_xml_body_parent(root, body_name)
        except RuntimeError:
            continue
        if insert_parent is None:
            insert_parent = parent
            insert_index = list(parent).index(body)
        parent.remove(body)
    if insert_parent is None or insert_index is None:
        worldbody = root.find("worldbody")
        if worldbody is None:
            raise RuntimeError("multi_drawer_search XML is missing worldbody")
        return worldbody, len(worldbody)
    return insert_parent, insert_index


def _multi_drawer_search_prefix_subtree_names(elem: _ET.Element, prefix: str) -> None:
    for child in elem.iter():
        if child is elem:
            continue
        name = child.get("name")
        if name:
            child.set("name", f"{prefix}_{name}")


def _multi_drawer_search_configure_stacks(
    root: _ET.Element,
    active_stack_ids: list[int],
) -> None:
    parent, template_body = _find_xml_body_parent(root, _MULTI_DRAWER_SEARCH_TEMPLATE_DRAWER_BODY)
    insert_index = list(parent).index(template_body)
    parent.remove(template_body)
    for offset, stack_id in enumerate(active_stack_ids):
        stack_body = copy.deepcopy(template_body)
        prefix = _multi_drawer_search_stack_body_name(stack_id)
        stack_body.set("name", prefix)
        stack_pose = _MULTI_DRAWER_SEARCH_STACK_POSES[int(stack_id)]
        stack_body.set("pos", _format_float_list(stack_pose["pos"]))
        stack_body.set("quat", _format_float_list(stack_pose["quat"]))
        _multi_drawer_search_prefix_subtree_names(stack_body, prefix)
        parent.insert(insert_index + offset, stack_body)


def _multi_drawer_search_append_object_assets(
    asset_elem: _ET.Element,
    *,
    body_name: str,
    spec: dict[str, Any],
) -> dict[str, dict[str, str]]:
    variant_dir = _multi_drawer_search_variant_dir(spec)
    parsed = _inhand_parse_model_xml(variant_dir)
    scale_factor = float(spec.get("scale_factor", 1.0))
    prefix = body_name
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

    mesh_names: dict[str, str] = {}
    texture_names: dict[str, str] = {}
    material_names: dict[str, str] = {}
    for mesh in parsed["meshes"]:
        if mesh["name"] not in used_mesh_names:
            continue
        name = f'{prefix}_{mesh["name"]}'
        mesh_names[str(mesh["name"])] = name
        extra = {**mesh["extra"]}
        extra["scale"] = _scaled_mesh_attr(extra.get("scale"), scale_factor)
        attrs = {
            "name": name,
            "file": str(variant_dir / mesh["file"]),
            **extra,
        }
        asset_elem.append(_ET.Element("mesh", attrs))
    for texture in parsed["textures"]:
        name = f'{prefix}_{texture["name"]}'
        texture_names[str(texture["name"])] = name
        asset_elem.append(
            _ET.Element(
                "texture",
                {
                    "name": name,
                    "file": str(variant_dir / texture["file"]),
                    "type": str(texture["type"]),
                },
            )
        )
    for material in parsed["materials"]:
        name = f'{prefix}_{material["name"]}'
        material_names[str(material["name"])] = name
        attrs = {"name": name}
        if material["texture"]:
            attrs["texture"] = texture_names.get(str(material["texture"]), f'{prefix}_{material["texture"]}')
        if material["rgba"]:
            attrs["rgba"] = str(material["rgba"])
        if material["shininess"]:
            attrs["shininess"] = str(material["shininess"])
        if material["specular"]:
            attrs["specular"] = str(material["specular"])
        asset_elem.append(_ET.Element("material", attrs))

    return {
        "meshes": mesh_names,
        "materials": material_names,
    }


def _multi_drawer_search_build_object_body(
    selection: dict[str, Any],
    *,
    asset_refs: dict[str, dict[str, str]],
) -> _ET.Element:
    body_name = str(selection["body"])
    joint_name = str(selection["joint"])
    spec = dict(selection["spec"])
    variant_dir = _multi_drawer_search_variant_dir(spec)
    parsed = _inhand_parse_model_xml(variant_dir)
    scale_factor = float(spec.get("scale_factor", 1.0))
    mesh_names = asset_refs["meshes"]
    material_names = asset_refs["materials"]

    body = _ET.Element(
        "body",
        {
            "name": body_name,
            "pos": _format_float_list([float(value) for value in selection["pos"]]),
            "quat": _format_float_list([float(value) for value in selection["quat"]]),
        },
    )
    mass_kg = spec.get("mass_kg")
    if mass_kg is not None:
        if parsed["bbox_pos"] is None or parsed["bbox_size"] is None:
            raise RuntimeError(f"Multi-drawer asset is missing reg_bbox: {variant_dir}")
        inertial_pos = np.asarray(parsed["bbox_pos"], dtype=np.float64) * scale_factor
        half_extents = np.asarray(parsed["bbox_size"], dtype=np.float64) * scale_factor
        mass = float(mass_kg)
        diagonal_inertia = mass / 3.0 * np.array(
            [
                half_extents[1] ** 2 + half_extents[2] ** 2,
                half_extents[0] ** 2 + half_extents[2] ** 2,
                half_extents[0] ** 2 + half_extents[1] ** 2,
            ],
            dtype=np.float64,
        )
        body.append(
            _ET.Element(
                "inertial",
                {
                    "pos": _format_float_list(inertial_pos.tolist()),
                    "mass": f"{mass:.9g}",
                    "diaginertia": _format_float_list(diagonal_inertia.tolist()),
                },
            )
        )
    body.append(_ET.Element("freejoint", {"name": joint_name}))

    for visual_index, visual_geom in enumerate(parsed["vis_geoms"]):
        attrs = {
            "name": f"{body_name}_visual" if visual_index == 0 else f"{body_name}_visual_{visual_index}",
            "type": "mesh",
            "contype": "0",
            "conaffinity": "0",
            "group": "2",
            "density": "0",
            "solimp": "0.998 0.998 0.001",
            "solref": "0.001 1",
        }
        if visual_geom["mesh"]:
            attrs["mesh"] = mesh_names[str(visual_geom["mesh"])]
        if visual_geom["material"]:
            attrs["material"] = material_names.get(
                str(visual_geom["material"]),
                str(visual_geom["material"]),
            )
        body.append(_ET.Element("geom", attrs))

    for collision_index, collision_geom in enumerate(parsed["col_geoms"]):
        attrs: dict[str, str] = {
            "name": f"{body_name}_collision" if collision_index == 0 else f"{body_name}_collision_{collision_index}",
        }
        for key, value in collision_geom.items():
            if not value:
                continue
            if key == "mesh":
                attrs[key] = mesh_names[str(value)]
            elif key in {"pos", "size", "fromto"}:
                attrs[key] = _grab_clutter_scaled_float_attr(str(value), scale_factor)
            elif key == "mass":
                attrs[key] = f"{float(value) * scale_factor ** 3:.9g}"
            else:
                attrs[key] = str(value)
        attrs.setdefault("type", "mesh")
        attrs.setdefault("group", "3")
        attrs.setdefault("density", "0")
        attrs.setdefault("condim", "6")
        attrs.setdefault("friction", "3.0 0.03 0.003")
        attrs.setdefault("solimp", "0.998 0.998 0.001")
        attrs.setdefault("solref", "0.004 1")
        attrs.setdefault("priority", "1")
        body.append(_ET.Element("geom", attrs))

    return body


def _multi_drawer_search_full_scene_selections() -> list[dict[str, Any]]:
    drawer_slots = [
        (stack_id, level, _multi_drawer_search_drawer_name(stack_id, level), slot)
        for stack_id in _MULTI_DRAWER_SEARCH_STACK_IDS
        for level in _MULTI_DRAWER_SEARCH_DRAWER_LEVELS
        for slot in _MULTI_DRAWER_SEARCH_DRAWER_SLOTS
    ]
    selections: list[dict[str, Any]] = []
    for object_index, body in enumerate(_MULTI_DRAWER_SEARCH_OBJECT_BODIES):
        stack_id, level, drawer, slot = drawer_slots[object_index % len(drawer_slots)]
        joint = _MULTI_DRAWER_SEARCH_JOINT_BY_BODY[body]
        spec = dict(_MULTI_DRAWER_SEARCH_OBJECT_SPEC_BY_BODY[body])
        stack_pose = _MULTI_DRAWER_SEARCH_STACK_POSES[int(stack_id)]
        selections.append(
            {
                "body": body,
                "joint": joint,
                "spec": spec,
                "category": spec["category"],
                "label": spec["label"],
                "variant": spec["variant"],
                "object_set": spec["object_set"],
                "scale_factor": spec["scale_factor"],
                "drawer": drawer,
                "drawer_level": level,
                "stack_id": int(stack_id),
                "is_target": object_index == 0,
                "pos": (
                    _multi_drawer_search_world_pos(stack_pose, level, slot, None)
                    if object_index < len(drawer_slots)
                    else _multi_drawer_search_hidden_pose(object_index)["pos"]
                ),
                "quat": [1.0, 0.0, 0.0, 0.0],
            }
        )
    return selections


def _multi_drawer_search_sample_stack_poses(
    active_stack_ids: list[int],
    rng: np.random.Generator,
) -> dict[int, dict[str, list[float]]]:
    slot_indices = list(range(len(_MULTI_DRAWER_SEARCH_STACK_SLOT_CENTERS)))
    rng.shuffle(slot_indices)
    stack_poses: dict[int, dict[str, list[float]]] = {}
    for stack_id, slot_index in zip(active_stack_ids, slot_indices):
        _slot_x, slot_y = _MULTI_DRAWER_SEARCH_STACK_SLOT_CENTERS[slot_index]
        yaw_delta = float(
            rng.uniform(
                -_MULTI_DRAWER_SEARCH_STACK_YAW_JITTER_RAD,
                _MULTI_DRAWER_SEARCH_STACK_YAW_JITTER_RAD,
            )
        )
        quat = _quat_mul(
            _quat_from_yaw(yaw_delta),
            np.asarray(_MULTI_DRAWER_SEARCH_STACK_POSES[int(stack_id)]["quat"], dtype=np.float64),
        )
        quat = quat / np.linalg.norm(quat)
        stack_rotation = _multi_drawer_search_rotation_from_quat(quat)
        back_edge_offset_x = max(
            float((stack_rotation @ corner)[0])
            for corner in _MULTI_DRAWER_SEARCH_STACK_VISUAL_FOOTPRINT_CORNERS
        )
        pos = np.array(
            [
                _MULTI_DRAWER_SEARCH_BACK_WALL_X_M
                - _MULTI_DRAWER_SEARCH_BACK_WALL_CLEARANCE_M
                - back_edge_offset_x,
                slot_y
                + float(
                    rng.uniform(
                        -_MULTI_DRAWER_SEARCH_STACK_Y_JITTER_M,
                        _MULTI_DRAWER_SEARCH_STACK_Y_JITTER_M,
                    )
                ),
                _MULTI_DRAWER_SEARCH_STACK_POSES[int(stack_id)]["pos"][2],
            ],
            dtype=np.float64,
        )
        stack_poses[int(stack_id)] = {"pos": pos.tolist(), "quat": quat.tolist()}
    return stack_poses


def _multi_drawer_search_hidden_pose(index: int) -> dict[str, list[float]]:
    return {
        "pos": [0.10 + 0.015 * (index % 6), -0.60 + 0.025 * (index // 6), _MULTI_DRAWER_SEARCH_PARK_Z],
        "quat": [1.0, 0.0, 0.0, 0.0],
    }


def _multi_drawer_search_apply_stack_poses(
    model: Any,
    stack_poses: dict[int, dict[str, list[float]]],
) -> None:
    for stack_id in _MULTI_DRAWER_SEARCH_STACK_IDS:
        body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            _multi_drawer_search_stack_body_name(stack_id),
        )
        if body_id < 0:
            continue
        pose = stack_poses.get(int(stack_id))
        if pose is None:
            pose = {
                "pos": [_MULTI_DRAWER_SEARCH_STACK_SLOT_CENTERS[int(stack_id) - 1][0], _MULTI_DRAWER_SEARCH_STACK_SLOT_CENTERS[int(stack_id) - 1][1], _MULTI_DRAWER_SEARCH_PARK_Z],
                "quat": _MULTI_DRAWER_SEARCH_STACK_POSES[int(stack_id)]["quat"],
            }
        model.body_pos[body_id] = np.asarray(pose["pos"], dtype=np.float64)
        model.body_quat[body_id] = np.asarray(pose["quat"], dtype=np.float64)


def _multi_drawer_search_set_object_collision_enabled(
    model: Any,
    body_name: str,
    enabled: bool,
) -> None:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        return
    for geom_id in range(model.ngeom):
        if int(model.geom_bodyid[geom_id]) != body_id:
            continue
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if not geom_name.startswith(f"{body_name}_collision"):
            continue
        model.geom_contype[geom_id] = 1 if enabled else 0
        model.geom_conaffinity[geom_id] = 1 if enabled else 0


def _multi_drawer_search_build_xml(
    base_text: str,
    selections: list[dict[str, Any]],
    *,
    active_stack_ids: list[int],
    base_dir: _Path | None,
) -> str:
    root = _ET.fromstring(base_text)
    _multi_drawer_search_configure_stacks(root, active_stack_ids)
    asset_elem = root.find("asset")
    if asset_elem is None:
        raise RuntimeError("multi_drawer_search XML is missing asset section")
    object_parent, insert_index = _multi_drawer_search_remove_base_object_bodies(root)
    for selection in selections:
        spec = dict(selection["spec"])
        asset_refs = _multi_drawer_search_append_object_assets(
            asset_elem,
            body_name=str(selection["body"]),
            spec=spec,
        )
        object_parent.insert(
            insert_index,
            _multi_drawer_search_build_object_body(selection, asset_refs=asset_refs),
        )
        insert_index += 1
    _multi_drawer_search_rewrite_home_qpos(root, selections, active_stack_ids)
    return _resolve_scene_xml_paths(_ET.tostring(root, encoding="unicode"), base_dir)


class MultiDrawerSearchRandomizer(SceneRandomizer):
    """Prepare multi_drawer_search with one coherent object set hidden in drawers."""

    perturbations: list = []

    def bind_env(self, env: Any) -> None:
        """Require the bound task's ordered drawer progress and scene contract."""
        if not isinstance(env._step_evaluator, DrawerProgressSource):
            raise ValueError("Drawer randomization requires an ordered drawer progress source.")
        env._step_evaluator.validate_model(env.model)
        super().bind_env(env)

    def prepare_env(self) -> None:
        if self._env_ref is not None:
            self._ensure_full_scene_loaded()

    def prepare_reset(
        self, *, request: Any | None = None, randomize: bool = True
    ) -> None:
        """Validate the request and prepare the bound scene before reset."""
        MultiDrawerSearchResetRequest.from_value(request)
        if randomize:
            self._ensure_full_scene_loaded()

    def _ensure_full_scene_loaded(self) -> None:
        if self._env_ref is None:
            return
        model = self._env_ref.model
        template_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, _MULTI_DRAWER_SEARCH_TEMPLATE_DRAWER_BODY
        )
        if template_id < 0:
            self._validate_full_scene(model)
            return
        if any(
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY,
                _multi_drawer_search_stack_body_name(stack_id),
            ) >= 0
            for stack_id in _MULTI_DRAWER_SEARCH_STACK_IDS
        ):
            raise ValueError("Drawer scene mixes template and assembled stacks.")
        self._validate_stack(model, _MULTI_DRAWER_SEARCH_TEMPLATE_DRAWER_BODY, "")
        self._validate_goal(model)

        base_text = (
            self._base_scene_xml_string
            if self._base_scene_xml_string is not None
            else _MULTI_DRAWER_SEARCH_BASE_SCENE_XML.read_text()
        )
        base_dir = self._base_scene_xml_dir or _MULTI_DRAWER_SEARCH_BASE_SCENE_XML.parent
        xml = _multi_drawer_search_build_xml(
            base_text,
            _multi_drawer_search_full_scene_selections(),
            active_stack_ids=list(_MULTI_DRAWER_SEARCH_STACK_IDS),
            base_dir=base_dir,
        )
        if (
            self._scene_xml_transform_options is not None
            and not self._base_scene_xml_transformed
        ):
            xml = _count_box_apply_scene_transforms(
                xml,
                self._scene_xml_transform_options,
            )

        preserved_robot_qpos = self._env_ref._get_reset_robot_qpos()
        self._env_ref.reload_from_xml(xml, validate_model=self._validate_full_scene)
        mujoco.mj_resetData(self._env_ref.model, self._env_ref.data)
        self._env_ref._set_robot_qpos(preserved_robot_qpos)
        mujoco.mj_forward(self._env_ref.model, self._env_ref.data)
        self._base_scene_xml_string = xml
        self._base_scene_xml_dir = base_dir
        self._base_scene_xml_transformed = True

    def _validate_stack(self, model: Any, stack: str, prefix: str) -> None:
        """Require native drawer ownership and the selected progress contract."""
        stack_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, stack)
        if stack_body < 0:
            raise ValueError(f"Drawer stack {stack!r} is missing from the model.")
        joints = []
        for level in _MULTI_DRAWER_SEARCH_DRAWER_LEVELS:
            body = f"{prefix}link_{level}_drawer_low"
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
            joint = f"{prefix}joint_{level}_drawer_low"
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            if (
                body_id < 0 or joint_id < 0
                or model.body_parentid[body_id] != stack_body
                or model.jnt_bodyid[joint_id] != body_id
                or model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_SLIDE
            ):
                raise ValueError(f"Drawer {body!r} requires its own slide joint.")
            joints.append(joint)
        if self._env_ref is not None:
            self._env_ref._step_evaluator.validate_drawer_joints(model, joints)

    def _validate_full_scene(self, model: Any) -> None:
        """Require the complete assembled drawer inventory and joint ownership."""
        if mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, _MULTI_DRAWER_SEARCH_TEMPLATE_DRAWER_BODY
        ) >= 0:
            raise ValueError("An assembled drawer scene must not retain the template stack.")
        for stack_id in _MULTI_DRAWER_SEARCH_STACK_IDS:
            stack = _multi_drawer_search_stack_body_name(stack_id)
            self._validate_stack(model, stack, f"{stack}_")
        for body, joint in _MULTI_DRAWER_SEARCH_JOINT_BY_BODY.items():
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            if (
                body_id < 0 or joint_id < 0
                or model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE
                or model.jnt_bodyid[joint_id] != body_id
            ):
                raise ValueError(f"Drawer object {body!r} requires its own free joint.")
        self._validate_goal(model)

    def _validate_goal(self, model: Any) -> None:
        """Require the goal bin's free joint and attached region in either layout."""
        goal_joint, _ = resolve_drawer_goal(
            model,
            goal_bin_joint_name=_MULTI_DRAWER_SEARCH_GOAL_BIN_JOINT,
            goal_site_name="drawer_search_goal_region",
        )
        goal_body = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "drawer_search_goal_bin"
        )
        if goal_body < 0 or model.jnt_bodyid[goal_joint] != goal_body:
            raise ValueError("The drawer goal bin requires its own free joint.")
        if self._env_ref is not None:
            self._env_ref._step_evaluator.validate_model(model)

    def _sample_selections(
        self, rng: np.random.Generator, *, target_count: int | None = None
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        count_min, count_max = _MULTI_DRAWER_SEARCH_STACK_COUNT_RANGE
        stack_count = int(rng.integers(count_min, count_max + 1))
        sampled_stack_ids = {
            int(stack_id)
            for stack_id in rng.choice(
                np.asarray(_MULTI_DRAWER_SEARCH_STACK_IDS, dtype=np.int64),
                size=stack_count,
                replace=False,
            )
        }
        active_stack_ids = [
            stack_id for stack_id in _MULTI_DRAWER_SEARCH_STACK_IDS
            if stack_id in sampled_stack_ids
        ]
        active_drawers = [
            _multi_drawer_search_drawer_name(stack_id, level)
            for stack_id in active_stack_ids
            for level in _MULTI_DRAWER_SEARCH_DRAWER_LEVELS
        ]
        stack_poses = _multi_drawer_search_sample_stack_poses(active_stack_ids, rng)
        drawer_slots = [
            (stack_id, level, _multi_drawer_search_drawer_name(stack_id, level), slot)
            for stack_id in active_stack_ids
            for level in _MULTI_DRAWER_SEARCH_DRAWER_LEVELS
            for slot in _MULTI_DRAWER_SEARCH_DRAWER_SLOTS
        ]
        object_count_min, object_count_max = _MULTI_DRAWER_SEARCH_OBJECT_COUNT_RANGE_BY_STACK_COUNT[stack_count]
        object_count_max = min(
            object_count_max,
            len(drawer_slots),
            _MULTI_DRAWER_SEARCH_OBJECT_COUNT,
        )
        object_count_min = min(object_count_min, object_count_max)
        if target_count is not None:
            object_count_min = max(object_count_min, target_count)
            if object_count_min > object_count_max:
                raise ValueError("Requested target count exceeds drawer capacity.")
        active_object_count = int(rng.integers(object_count_min, object_count_max + 1))

        object_set = str(rng.choice(_MULTI_DRAWER_SEARCH_OBJECT_SET_NAMES))
        body_names = list(_MULTI_DRAWER_SEARCH_OBJECT_BODIES_BY_SET[object_set])
        rng.shuffle(body_names)
        active_bodies = body_names[:active_object_count]
        slot_indices = list(range(len(drawer_slots)))
        rng.shuffle(slot_indices)
        selected_slots = [drawer_slots[index] for index in slot_indices[:active_object_count]]

        selections: list[dict[str, Any]] = []
        for body, (stack_id, level, drawer, slot) in zip(active_bodies, selected_slots):
            joint = _MULTI_DRAWER_SEARCH_JOINT_BY_BODY[body]
            spec = dict(_MULTI_DRAWER_SEARCH_OBJECT_SPEC_BY_BODY[body])
            selection = {
                "body": body,
                "joint": joint,
                "spec": spec,
                "category": str(spec["category"]),
                "label": str(spec["label"]),
                "variant": str(spec["variant"]),
                "object_set": str(spec["object_set"]),
                "scale_factor": float(spec["scale_factor"]),
                "drawer": str(drawer),
                "drawer_level": str(level),
                "stack_id": int(stack_id),
                "is_target": False,
                "pos": _multi_drawer_search_world_pos(stack_poses[int(stack_id)], str(level), slot, rng),
                "quat": _multi_drawer_search_item_quat(
                    rng,
                    object_set=object_set,
                    stack_pose=stack_poses[int(stack_id)],
                ),
            }
            selections.append(selection)

        target_count_min = min(
            _MULTI_DRAWER_SEARCH_TARGET_COUNT_RANGE[0],
            len(selections),
        )
        target_count_max = min(
            _MULTI_DRAWER_SEARCH_TARGET_COUNT_RANGE[1],
            len(selections),
        )
        if target_count is None:
            target_count = int(rng.integers(target_count_min, target_count_max + 1))
        elif target_count > len(selections):
            raise ValueError("Requested target count exceeds available drawer objects.")
        target_indices = rng.choice(
            np.arange(len(selections), dtype=np.int64),
            size=target_count,
            replace=False,
        ).tolist()
        target_sequence = [selections[int(index)] for index in target_indices]
        for sequence_index, selection in enumerate(target_sequence):
            selection["is_sequence_target"] = True
            selection["sequence_index"] = sequence_index
        target = target_sequence[0]
        target["is_target"] = True
        selections = [
            target,
            *(selection for selection in selections if selection is not target),
        ]
        drawer_contents: dict[str, list[str]] = {drawer: [] for drawer in active_drawers}
        drawer_objects: dict[str, list[dict[str, str]]] = {drawer: [] for drawer in active_drawers}
        for selection in selections:
            drawer = str(selection["drawer"])
            drawer_contents[drawer].append(str(selection["label"]))
            drawer_objects[drawer].append(
                {
                    "body": str(selection["body"]),
                    "joint": str(selection["joint"]),
                    "category": str(selection["category"]),
                    "label": str(selection["label"]),
                    "object_set": str(selection["object_set"]),
                }
            )
        selections[0]["active_drawers"] = tuple(active_drawers)
        selections[0]["active_stack_ids"] = tuple(active_stack_ids)
        selections[0]["stack_poses"] = stack_poses
        selections[0]["drawer_contents"] = drawer_contents
        selections[0]["drawer_objects"] = drawer_objects
        selections[0]["empty_drawers"] = tuple(
            drawer for drawer, contents in drawer_contents.items() if not contents
        )
        selections[0]["multi_item_drawers"] = tuple(
            drawer for drawer, contents in drawer_contents.items() if len(contents) > 1
        )
        selections[0]["target_sequence"] = tuple(
            {
                "body": str(selection["body"]),
                "joint": str(selection["joint"]),
                "category": str(selection["category"]),
                "label": str(selection["label"]),
                "variant": str(selection["variant"]),
                "object_set": str(selection["object_set"]),
                "drawer": str(selection["drawer"]),
                "drawer_level": str(selection["drawer_level"]),
                "stack_id": int(selection["stack_id"]),
                "sequence_index": int(selection["sequence_index"]),
            }
            for selection in target_sequence
        )
        return selections, selections[0]

    def randomize(
        self,
        model: Any,
        data: Any,
        seed: int | None = None,
        request: Any | None = None,
    ) -> RandomizationState:
        reset_request = MultiDrawerSearchResetRequest.from_value(request)
        self._ensure_full_scene_loaded()
        if self._env_ref is not None:
            model = self._env_ref.model
            data = self._env_ref.data
        rng = np.random.default_rng(seed)
        selections, target = self._sample_selections(
            rng, target_count=reset_request.target_count
        )
        goal_bin_side = "left" if int(rng.integers(0, 2)) == 0 else "right"
        goal_bin_pose = {
            "pos": [
                float(_MULTI_DRAWER_SEARCH_GOAL_BIN_POSE["pos"][0]),
                _MULTI_DRAWER_SEARCH_GOAL_BIN_Y_ABS_M
                if goal_bin_side == "left"
                else -_MULTI_DRAWER_SEARCH_GOAL_BIN_Y_ABS_M,
                float(_MULTI_DRAWER_SEARCH_GOAL_BIN_POSE["pos"][2]),
            ],
            "quat": list(_MULTI_DRAWER_SEARCH_GOAL_BIN_POSE["quat"]),
        }
        active_drawers = list(target["active_drawers"])
        active_stack_ids = [int(stack_id) for stack_id in target["active_stack_ids"]]
        stack_poses = dict(target["stack_poses"])

        target_sequence = [dict(item) for item in target["target_sequence"]]
        prompt = f'find the {target["label"]}'
        object_states = {
            _MULTI_DRAWER_SEARCH_GOAL_BIN_JOINT: {
                "pos": list(goal_bin_pose["pos"]),
                "quat": list(goal_bin_pose["quat"]),
            }
        }
        objects: list[dict[str, Any]] = []
        active_body_names = {str(selection["body"]) for selection in selections}
        for selection in selections:
            joint = str(selection["joint"])
            object_states[joint] = {
                "pos": [float(value) for value in selection["pos"]],
                "quat": [float(value) for value in selection["quat"]],
            }
            objects.append(
                {
                    "body": selection["body"],
                    "joint": joint,
                    "category": selection["category"],
                    "label": selection["label"],
                    "variant": selection["variant"],
                    "object_set": selection["object_set"],
                    "scale_factor": selection["scale_factor"],
                    "drawer": selection["drawer"],
                    "drawer_level": selection["drawer_level"],
                    "stack_id": selection["stack_id"],
                    "is_target": selection["is_target"],
                    "is_sequence_target": selection.get("is_sequence_target", False),
                    "sequence_index": selection.get("sequence_index"),
                }
            )
        for object_index, body_name in enumerate(_MULTI_DRAWER_SEARCH_OBJECT_BODIES):
            if body_name in active_body_names:
                continue
            joint = _MULTI_DRAWER_SEARCH_JOINT_BY_BODY[body_name]
            object_states[joint] = _multi_drawer_search_hidden_pose(object_index)
        metadata = {
            "prompt": prompt,
            "prompt_type": "multi_drawer_search",
            "prompt_mode": "sequence",
            "object_set": target["object_set"],
            "prompt_sequence": [
                *(f'find the {item["label"]}' for item in target_sequence),
                "finished",
            ],
            "target_count": len(target_sequence),
            "target_sequence": target_sequence,
            "target_labels": [item["label"] for item in target_sequence],
            "target_bodies": [item["body"] for item in target_sequence],
            "target_joints": [item["joint"] for item in target_sequence],
            "target_category": target["category"],
            "target_label": target["label"],
            "target_variant": target["variant"],
            "target_scale_factor": target["scale_factor"],
            "target_drawer": target["drawer"],
            "target_drawer_level": target["drawer_level"],
            "target_stack_id": target["stack_id"],
            "target_body": target["body"],
            "target_joint": target["joint"],
            "goal_bin_body": "drawer_search_goal_bin",
            "goal_bin_joint": _MULTI_DRAWER_SEARCH_GOAL_BIN_JOINT,
            "goal_site": "drawer_search_goal_region",
            "goal_bin_side": goal_bin_side,
            "goal_bin_pose": goal_bin_pose,
            "drawer_joints": [
                _MULTI_DRAWER_SEARCH_DRAWER_JOINT_BY_DRAWER[drawer]
                for drawer in active_drawers
            ],
            "all_drawer_joints": list(_MULTI_DRAWER_SEARCH_DRAWER_JOINTS),
            "stack_count": len(active_stack_ids),
            "active_stack_ids": active_stack_ids,
            "inactive_stack_ids": [
                stack_id
                for stack_id in _MULTI_DRAWER_SEARCH_STACK_IDS
                if stack_id not in active_stack_ids
            ],
            "stack_poses": stack_poses,
            "drawer_count": len(active_drawers),
            "active_drawers": active_drawers,
            "drawer_contents": target["drawer_contents"],
            "drawer_objects": target["drawer_objects"],
            "empty_drawers": list(target["empty_drawers"]),
            "multi_item_drawers": list(target["multi_item_drawers"]),
            "inactive_drawer_bodies": [
                _MULTI_DRAWER_SEARCH_DRAWER_BODY_BY_DRAWER[drawer]
                for drawer in _MULTI_DRAWER_SEARCH_DRAWERS
                if drawer not in active_drawers
            ],
            "inactive_drawer_joints": [
                _MULTI_DRAWER_SEARCH_DRAWER_JOINT_BY_DRAWER[drawer]
                for drawer in _MULTI_DRAWER_SEARCH_DRAWERS
                if drawer not in active_drawers
            ],
            "active_object_count": len(objects),
            "inactive_object_bodies": [
                body_name
                for body_name in _MULTI_DRAWER_SEARCH_OBJECT_BODIES
                if body_name not in active_body_names
            ],
            "objects": objects,
        }
        state = RandomizationState(
            seed=seed or 0,
            object_states=object_states,
            metadata=metadata,
        )
        return self._restore_scene(model, data, state)

    def _scene_parameters(
        self, model: Any, state: RandomizationState
    ) -> tuple[dict[int, dict[str, list[float]]], set[str]]:
        """Validate the complete recorded layout without changing simulator state."""
        self._validate_full_scene(model)
        if state.scale_states:
            raise ValueError("Drawer scene replay does not support object scale overrides.")
        episode = DrawerEpisode.from_metadata(state.metadata)
        raw_poses = state.metadata["stack_poses"]
        if any(
            type(key) not in (int, str) or str(int(key)) != str(key)
            for key in raw_poses
        ):
            raise ValueError("Stack identifiers must be integers or their serialized names.")
        poses = {int(key): value for key, value in raw_poses.items()}
        stack_ids = state.metadata["active_stack_ids"]
        if (
            any(type(key) is not int for key in stack_ids)
            or len(set(stack_ids)) != len(stack_ids)
            or len(poses) != len(raw_poses)
            or set(poses) != set(stack_ids)
            or not set(stack_ids) <= set(_MULTI_DRAWER_SEARCH_STACK_IDS)
            or not stack_ids
        ):
            raise ValueError("Recorded stack poses must identify the active stacks once.")
        expected_drawers = tuple(
            _multi_drawer_search_drawer_name(stack_id, level)
            for stack_id in stack_ids for level in _MULTI_DRAWER_SEARCH_DRAWER_LEVELS
        )
        if episode.drawers != expected_drawers or episode.drawer_joints != tuple(
            _MULTI_DRAWER_SEARCH_DRAWER_JOINT_BY_DRAWER[name] for name in expected_drawers
        ):
            raise ValueError("Recorded active drawers must match their stacks and joints.")
        objects = state.metadata["objects"]
        active = {item["body"] for item in objects}
        if len(active) != len(objects) or not active <= set(_MULTI_DRAWER_SEARCH_OBJECT_BODIES):
            raise ValueError("Recorded active drawer objects must be unique known bodies.")
        by_body = {item["body"]: item for item in objects}
        for item in objects:
            if (
                item["joint"] != _MULTI_DRAWER_SEARCH_JOINT_BY_BODY[item["body"]]
                or item["drawer"] not in expected_drawers
            ):
                raise ValueError("An active object's joint or source drawer is inconsistent.")
        for target in episode.targets:
            if target.body not in active or any(
                by_body[target.body][key] != getattr(target, key)
                for key in ("joint", "drawer", "label")
            ):
                raise ValueError("Targets must match the recorded active object selection.")
        expected_joints = {*_MULTI_DRAWER_SEARCH_OBJECT_JOINTS, _MULTI_DRAWER_SEARCH_GOAL_BIN_JOINT}
        if set(state.object_states) != expected_joints:
            raise ValueError("Drawer replay must record every object and the goal-bin pose.")
        for pose in (*poses.values(), *state.object_states.values()):
            pos, quat = np.asarray(pose["pos"]), np.asarray(pose["quat"])
            if (
                pos.shape != (3,) or quat.shape != (4,)
                or not np.isfinite(pos).all() or not np.isfinite(quat).all()
                or np.linalg.norm(quat) == 0
            ):
                raise ValueError("Recorded drawer poses must be finite positions and quaternions.")
        return poses, active

    def _validate_scene(self, model: Any, state: RandomizationState) -> None:
        self._scene_parameters(model, state)

    def _restore_scene(
        self, model: Any, data: Any, state: RandomizationState
    ) -> RandomizationState:
        poses, active = self._scene_parameters(model, state)
        _multi_drawer_search_apply_stack_poses(model, poses)
        for body in _MULTI_DRAWER_SEARCH_OBJECT_BODIES:
            _multi_drawer_search_set_object_collision_enabled(model, body, body in active)
        return super()._restore_scene(model, data, state)
