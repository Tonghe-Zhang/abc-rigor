"""Mug asset discovery and scene XML assembly."""

from __future__ import annotations

import copy
import xml.etree.ElementTree as _ET
from functools import lru_cache as _lru_cache

import mujoco
import numpy as np

from abc_sim.mug_inertia import MUG_MASS_KG, surface_inertia

from ..core import (
    _format_float_list,
    _parse_float_list,
    _resolve_scene_xml_paths,
    _scale_body_subtree,
)
from .paths import _MODELS_DIR, _Path
from .xml_common import (
    _dishrack_absolutize_file_attr,
    _dishrack_asset_local_name,
    _dishrack_find_object_body,
    _dishrack_geom_world_bounds,
    _dishrack_prefixed_name,
    _dishrack_prefix_body_names,
    _dishrack_remove_dynamic_joints,
    _dishrack_rewrite_asset_refs,
)

_MUG_DEFAULT_VARIANT = "mug_0"
_MUG_BASE_SCENE_XMLS: dict[str, _Path] = {
    "mug_flip": _MODELS_DIR / "yam_mug_flip_scene.xml",
    "mug_tree": _MODELS_DIR / "yam_mug_tree_scene.xml",
}
_MUG_TASK_ASSET_ROOTS: dict[str, _Path] = {
    "mug_flip": _MODELS_DIR / "assets" / "task_mug_flip" / "mug",
    "mug_tree": _MODELS_DIR / "assets" / "task_mug_tree" / "mug",
}
_MUG_TASK_BODY_NAMES: dict[str, tuple[str, ...]] = {
    "mug_flip": ("mug_1", "mug_2", "mug_3", "mug_4"),
    "mug_tree": ("mug_1", "mug_2", "mug_3"),
}

_MUG_FLIP_TRAY_INNER_HALF_XY = (0.166, 0.112)
_MUG_FLIP_TRAY_MARGIN_M = 0.004
_MUG_FLIP_TRAY_FLOOR_Z_OFFSET = 0.008
_MUG_FLIP_SPAWN_CLEARANCE_M = 0.004
_MUG_FLIP_MUG_MARGIN_M = 0.004

def _mug_variant_names(task_name: str) -> list[str]:
    root = _MUG_TASK_ASSET_ROOTS[task_name]
    variants = [
        path.name
        for path in root.iterdir()
        if path.is_dir() and (path / "model.xml").exists()
    ]
    variants.sort(
        key=lambda name: (
            0,
            int(name[len("mug_") :]),
        )
        if name.startswith("mug_") and name[len("mug_") :].isdigit()
        else (1, name)
    )
    if not variants:
        raise FileNotFoundError(f"No mug variants found under {root}")
    return variants


def _mug_canonical_variant_name(variant_name: str) -> str:
    if variant_name == "current":
        return _MUG_DEFAULT_VARIANT
    if variant_name.isdigit():
        return f"mug_{variant_name}"
    if variant_name.startswith("mug") and variant_name[3:].isdigit():
        return f"mug_{variant_name[3:]}"
    return variant_name


def _mug_variant_dir(task_name: str, variant_name: str) -> _Path:
    variant_name = _mug_canonical_variant_name(variant_name)
    path = _MUG_TASK_ASSET_ROOTS[task_name] / variant_name
    if not (path / "model.xml").exists():
        raise FileNotFoundError(f"Missing mug variant model.xml: {path}")
    return path


def _mug_sample_variant_name(task_name: str, rng: np.random.Generator) -> str:
    variants = _mug_variant_names(task_name)
    return variants[int(rng.integers(0, len(variants)))]


def _mug_instance_prefix(variant_name: str, instance_index: int) -> str:
    return f"mug_{instance_index}_{variant_name}"


def _mug_prepare_imported_geoms(body: _ET.Element) -> None:
    for parent in body.iter():
        for child in list(parent):
            if child.tag == "inertial":
                parent.remove(child)
                continue
            if child.tag != "geom":
                continue

            name = child.get("name", "")
            child_class = child.get("class", "")
            mesh_name = child.get("mesh")
            if (
                child_class == "region"
                or name.startswith("reg_")
                or mesh_name is None
            ):
                parent.remove(child)
                continue

            is_visual = child.get("contype") == "0" and child.get("conaffinity") == "0"
            child.attrib.pop("class", None)
            if is_visual:
                child.set("group", "2")
                child.set("contype", "0")
                child.set("conaffinity", "0")
                child.set("density", "0")
                child.attrib.pop("mass", None)
                continue

            child.set("group", "3")
            child.set("rgba", "0 0 0 0")
            child.set("density", "0")
            child.set("friction", "3.0 0.03 0.003")
            child.set("condim", "6")
            child.set("solref", "0.004 1")
            child.set("solimp", "0.998 0.998 0.001")
            child.set("priority", "1")
            child.attrib.pop("mass", None)


def _mug_build_object_block(
    *,
    task_name: str,
    variant_name: str,
    instance_index: int,
    scale_factor: float,
    joint_name: str,
) -> tuple[list[_ET.Element], _ET.Element]:
    variant_name = _mug_canonical_variant_name(variant_name)
    variant_dir = _mug_variant_dir(task_name, variant_name)
    root = _ET.parse(str(variant_dir / "model.xml")).getroot()
    object_body = _dishrack_find_object_body(root)

    prefix = _mug_instance_prefix(variant_name, instance_index)
    mesh_map: dict[str, str] = {}
    texture_map: dict[str, str] = {}
    material_map: dict[str, str] = {}
    temp_asset = _ET.Element("asset")

    asset_root = root.find("asset")
    if asset_root is not None:
        asset_children = list(asset_root)
        for asset_child in asset_children:
            local_name = _dishrack_asset_local_name(asset_child)
            if asset_child.tag == "mesh":
                mesh_map[local_name] = _dishrack_prefixed_name(prefix, local_name)
            elif asset_child.tag == "texture":
                texture_map[local_name] = _dishrack_prefixed_name(prefix, local_name)
            elif asset_child.tag == "material":
                material_map[local_name] = _dishrack_prefixed_name(prefix, local_name)

        for asset_child in asset_children:
            cloned = copy.deepcopy(asset_child)
            local_name = _dishrack_asset_local_name(asset_child)
            if asset_child.tag == "mesh":
                cloned.set("name", mesh_map[local_name])
                _dishrack_absolutize_file_attr(cloned, variant_dir)
            elif asset_child.tag == "texture":
                cloned.set("name", texture_map[local_name])
                _dishrack_absolutize_file_attr(cloned, variant_dir)
            elif asset_child.tag == "material":
                cloned.set("name", material_map[local_name])
                texture_name = asset_child.get("texture")
                if texture_name and texture_name in texture_map:
                    cloned.set("texture", texture_map[texture_name])
            temp_asset.append(cloned)

    cloned_body = copy.deepcopy(object_body)
    _dishrack_remove_dynamic_joints(cloned_body)
    _dishrack_prefix_body_names(cloned_body, prefix)
    _dishrack_rewrite_asset_refs(
        cloned_body,
        mesh_map=mesh_map,
        material_map=material_map,
    )
    _mug_prepare_imported_geoms(cloned_body)

    if abs(scale_factor - 1.0) > 1e-9:
        mesh_assets = {
            mesh.get("name", ""): mesh
            for mesh in temp_asset.findall("mesh")
            if mesh.get("name")
        }
        _scale_body_subtree(
            body=cloned_body,
            factor=scale_factor,
            target_name=joint_name,
            asset_elem=temp_asset,
            mesh_assets=mesh_assets,
        )

    return list(temp_asset), cloned_body


@_lru_cache(maxsize=128)
def _mug_unit_surface_inertia(
    task_name: str, variant_name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cache unit-scale moments from a variant's transformed visual triangles."""
    assets, body = _mug_build_object_block(
        task_name=task_name,
        variant_name=variant_name,
        instance_index=0,
        scale_factor=1.0,
        joint_name="mug_inertia",
    )
    # This model only resolves mesh/body transforms; it has no dynamic joints.
    root = _ET.Element("mujoco")
    _ET.SubElement(root, "compiler", angle="radian", fusestatic="false")
    for parent in body.iter():
        for child in list(parent):
            if child.tag == "geom":
                if child.get("group") != "2":
                    parent.remove(child)
                else:
                    child.attrib.pop("material", None)
            elif child.tag not in {"body", "frame"}:
                parent.remove(child)
    mesh_names = {geom.get("mesh") for geom in body.iter("geom")}
    asset = _ET.SubElement(root, "asset")
    for item in assets:
        if item.tag == "mesh" and item.get("name") in mesh_names:
            asset.append(item)
    _ET.SubElement(root, "worldbody").append(body)
    model = mujoco.MjModel.from_xml_string(_ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    triangles = []
    for geom in range(model.ngeom):
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            raise ValueError("Imported mug visual geometry must be a mesh")
        mesh = int(model.geom_dataid[geom])
        start = int(model.mesh_vertadr[mesh])
        vertices = np.asarray(
            model.mesh_vert[start : start + int(model.mesh_vertnum[mesh])],
            dtype=np.float64,
        )
        vertices = vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        start = int(model.mesh_faceadr[mesh])
        faces = model.mesh_face[start : start + int(model.mesh_facenum[mesh])]
        triangles.append(vertices[faces])
    if not triangles:
        raise ValueError("Imported mug has no visual triangle surfaces")
    center, inertia = surface_inertia(np.concatenate(triangles))
    # The imported root's position is preserved by _scale_body_subtree.
    anchor = np.asarray(_parse_float_list(body.get("pos", "0 0 0")))
    for array in (center, inertia, anchor):
        array.setflags(write=False)
    return center, inertia, anchor


def _mug_wrapper_inertial(task_name: str, variant_name: str, scale: float) -> _ET.Element:
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Mug scale must be positive and finite")
    center, inertia, anchor = _mug_unit_surface_inertia(task_name, variant_name)
    center = anchor + scale * (center - anchor)
    inertia = scale**2 * inertia
    values = inertia[(0, 1, 2, 0, 0, 1), (0, 1, 2, 1, 2, 2)]
    return _ET.Element(
        "inertial",
        mass=str(MUG_MASS_KG),
        pos=" ".join(f"{value:.17g}" for value in center),
        fullinertia=" ".join(f"{value:.17g}" for value in values),
    )


def _mug_replace_body_contents(wrapper: _ET.Element, imported_body: _ET.Element) -> None:
    for child in list(wrapper):
        if child.tag not in {"freejoint", "joint"}:
            wrapper.remove(child)
    wrapper.append(imported_body)


def _mug_remove_inactive_bodies(root: _ET.Element, inactive_body_names: set[str]) -> None:
    if not inactive_body_names:
        return
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "body" and child.get("name") in inactive_body_names:
                parent.remove(child)


def _mug_rewrite_home_keyframe(root: _ET.Element, active_body_names: tuple[str, ...]) -> None:
    keyframe = root.find("keyframe")
    if keyframe is None:
        return
    home_key = keyframe.find("./key[@name='home']")
    if home_key is None:
        return

    raw_qpos = home_key.get("qpos", "")
    qpos_values = _parse_float_list(raw_qpos) if raw_qpos else []
    arm_qpos = qpos_values[-16:] if len(qpos_values) >= 16 else []
    object_qpos: list[float] = []
    for body_name in active_body_names:
        body = root.find(f".//body[@name='{body_name}']")
        if body is None:
            continue
        object_qpos.extend(_parse_float_list(body.get("pos", "0 0 0")))
        object_qpos.extend(_parse_float_list(body.get("quat", "1 0 0 0")))
    if object_qpos or arm_qpos:
        home_key.set("qpos", _format_float_list([*object_qpos, *arm_qpos]))


def _build_mug_scene_xml(
    *,
    task_name: str,
    mug_variant: str,
    mug_variants: list[str] | tuple[str, ...] | None = None,
    scale_states: dict[str, float],
    base_scene_xml: str | None,
    base_scene_dir: _Path | None,
) -> str:
    base_text = base_scene_xml or _MUG_BASE_SCENE_XMLS[task_name].read_text()
    base_dir = base_scene_dir or _MUG_BASE_SCENE_XMLS[task_name].parent
    if mug_variants is None:
        body_names = _MUG_TASK_BODY_NAMES[task_name]
        default_count = min(2, len(body_names))
        normalized_mug_variants = [_mug_canonical_variant_name(mug_variant)] * default_count
    else:
        normalized_mug_variants = [_mug_canonical_variant_name(str(variant)) for variant in mug_variants]
    body_names = _MUG_TASK_BODY_NAMES[task_name]
    if not 1 <= len(normalized_mug_variants) <= len(body_names):
        raise ValueError(
            f"{task_name} requires between 1 and {len(body_names)} mug variants, "
            f"got {len(normalized_mug_variants)}"
        )

    root = _ET.fromstring(base_text)
    asset_elem = root.find("asset")
    if asset_elem is None:
        raise ValueError("Mug scene XML is missing <asset>")
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Mug scene XML is missing <worldbody>")

    active_body_names = body_names[: len(normalized_mug_variants)]
    _mug_remove_inactive_bodies(root, set(body_names[len(normalized_mug_variants) :]))
    _mug_rewrite_home_keyframe(root, active_body_names)

    for instance_index, (body_name, current_mug_variant) in enumerate(
        zip(active_body_names, normalized_mug_variants)
    ):
        wrapper = worldbody.find(f".//body[@name='{body_name}']")
        if wrapper is None:
            raise ValueError(f"Mug scene XML is missing body {body_name!r}")
        joint_name = f"{body_name}_jnt"
        variant_assets, imported_body = _mug_build_object_block(
            task_name=task_name,
            variant_name=current_mug_variant,
            instance_index=instance_index,
            scale_factor=float(scale_states.get(joint_name, 1.0)),
            joint_name=joint_name,
        )
        for asset in variant_assets:
            asset_elem.append(asset)
        _mug_replace_body_contents(wrapper, imported_body)
        wrapper.append(
            _mug_wrapper_inertial(
                task_name, current_mug_variant, float(scale_states.get(joint_name, 1.0))
            )
        )

    return _resolve_scene_xml_paths(_ET.tostring(root, encoding="unicode"), base_dir)


def _mujoco_body_in_subtree(model: mujoco.MjModel, body_id: int, root_body_id: int) -> bool:
    current = int(body_id)
    while current >= 0:
        if current == root_body_id:
            return True
        parent = int(model.body_parentid[current])
        if parent == current:
            break
        current = parent
    return False


@_lru_cache(maxsize=None)
def _mug_flip_compiled_metadata(variant_name: str) -> tuple[float, float, float, float]:
    """Return collision bounds for a mug_flip asset relative to the wrapper body."""
    variant_name = _mug_canonical_variant_name(variant_name)
    xml = _build_mug_scene_xml(
        task_name="mug_flip",
        mug_variant=variant_name,
        mug_variants=[variant_name],
        scale_states={},
        base_scene_xml=None,
        base_scene_dir=None,
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    mug_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mug_1")
    if mug_body_id < 0:
        raise ValueError(f"Compiled mug_flip scene for {variant_name} is missing mug_1")

    def collect_bounds(collision_only: bool) -> tuple[np.ndarray, np.ndarray] | None:
        mins: list[np.ndarray] = []
        maxs: list[np.ndarray] = []
        for geom_id in range(model.ngeom):
            body_id = int(model.geom_bodyid[geom_id])
            if not _mujoco_body_in_subtree(model, body_id, mug_body_id):
                continue
            if collision_only and not (
                int(model.geom_contype[geom_id]) or int(model.geom_conaffinity[geom_id])
            ):
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if name.startswith("reg_"):
                continue
            lower, upper = _dishrack_geom_world_bounds(model, data, geom_id)
            mins.append(lower)
            maxs.append(upper)
        if not mins:
            return None
        return np.vstack(mins).min(axis=0), np.vstack(maxs).max(axis=0)

    bounds = collect_bounds(collision_only=True) or collect_bounds(collision_only=False)
    if bounds is None:
        raise ValueError(f"Compiled mug_flip scene for {variant_name} has no mug geoms")

    min_xyz, max_xyz = bounds
    mug_pos = np.asarray(data.xpos[mug_body_id], dtype=np.float64)
    rel_min = min_xyz - mug_pos
    rel_max = max_xyz - mug_pos
    half_x = max(abs(float(rel_min[0])), abs(float(rel_max[0])))
    half_y = max(abs(float(rel_min[1])), abs(float(rel_max[1])))
    return half_x, half_y, float(rel_min[2]), float(rel_max[2])


@_lru_cache(maxsize=None)
def _mug_plain_source_color_material_names(task_name: str) -> tuple[str, ...]:
    variant_dir = _mug_variant_dir(task_name, _MUG_DEFAULT_VARIANT)
    root = _ET.parse(str(variant_dir / "model.xml")).getroot()
    asset_root = root.find("asset")
    if asset_root is None:
        return ()

    names: list[str] = []
    for material in asset_root.findall("material"):
        rgba_attr = material.get("rgba")
        material_name = material.get("name")
        if not rgba_attr or not material_name:
            continue
        rgba = _parse_float_list(rgba_attr)
        if len(rgba) < 3:
            continue
        if max(rgba[:3]) - min(rgba[:3]) < 0.04 and min(rgba[:3]) > 0.75:
            continue
        names.append(material_name)
    return tuple(names)


def _mug_plain_color_material_names(task_name: str, instance_index: int) -> tuple[str, ...]:
    prefix = _mug_instance_prefix(_MUG_DEFAULT_VARIANT, instance_index)
    return tuple(
        _dishrack_prefixed_name(prefix, name)
        for name in _mug_plain_source_color_material_names(task_name)
    )




__all__ = [name for name in globals() if name.startswith('_') and not name.startswith('__')]
