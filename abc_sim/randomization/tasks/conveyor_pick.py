from __future__ import annotations

from pathlib import Path as _Path
from typing import Any

import mujoco
import numpy as np

from ..assets.inhand import _inhand_parse_model_xml
from ..assets.paths import _MODELS_DIR
from ..core import RandomizationState, SceneRandomizer, _parse_float_list
from ..requests import ConveyorPickResetRequest

_CONVEYOR_PICK_BASE_SCENE_XML = _MODELS_DIR / "yam_conveyor_pick_scene.xml"
_CONVEYOR_PICK_OBJECT_ASSET_ROOT = _MODELS_DIR / "assets" / "task_conveyor_pick" / "objects"
_CONVEYOR_TASK_ASSETS_PLACEHOLDER = "<!-- CONVEYOR_TASK_ASSETS_PLACEHOLDER -->"
_CONVEYOR_TASK_OBJECTS_BEGIN = "<!-- CONVEYOR_TASK_OBJECTS_BEGIN -->"
_CONVEYOR_TASK_OBJECTS_END = "<!-- CONVEYOR_TASK_OBJECTS_END -->"
_CONVEYOR_OBJECT_COUNT = 3
_CONVEYOR_OBJECT_CATEGORIES: tuple[str, ...] = (
    "boxed_food",
    "boxed_drink",
    "canned_food",
    "can",
    "yogurt",
    "bar_soap",
    "sponge",
    "cream_cheese_stick",
    "bagged_food",
    "chips",
    "cookie_dough_ball",
    "marshmallow",
    "donut",
    "bagel",
    "cupcake",
    "apple",
    "lemon",
    "lime",
    "orange",
    "tangerine",
    "tomato",
    "potato",
    "egg",
    "sandwich_bread",
    "hotdog_bun",
    "waffle",
    "pancake",
)
_CONVEYOR_OBJECT_SLOTS: tuple[tuple[float, float], ...] = (
    (0.48, 0.42),
    (0.48, 0.30),
    (0.48, 0.18),
    (0.48, 0.06),
)
_CONVEYOR_X_JITTER_M = (-0.025, 0.025)
_CONVEYOR_Y_JITTER_M = (-0.005, 0.005)
_CONVEYOR_BELT_TOP_Z = 0.805 + 0.018
_CONVEYOR_OBJECT_SPAWN_CLEARANCE_M = 0.0005
_CONVEYOR_BELT_SPEED_RANGE_M_S = (0.10, 0.20)
_CONVEYOR_BELT_SPEED_LIMITS_M_S = (0.01, 0.50)


def _conveyor_belt_speed_range(request: Any | None) -> tuple[float, float]:
    """Belt speed bounds for this reset: the defaults unless the request overrides either one."""
    reset_request = ConveyorPickResetRequest.from_value(request)
    default_low, default_high = _CONVEYOR_BELT_SPEED_RANGE_M_S
    low = default_low if reset_request.belt_speed_min_m_s is None else reset_request.belt_speed_min_m_s
    high = default_high if reset_request.belt_speed_max_m_s is None else reset_request.belt_speed_max_m_s
    lowest, highest = _CONVEYOR_BELT_SPEED_LIMITS_M_S
    if not lowest <= low <= high <= highest:
        raise ValueError(
            f"conveyor_pick belt speed bounds must satisfy {lowest} <= min <= max <= {highest} m/s, got {low}, {high}"
        )
    return low, high


def _conveyor_pick_collision_bottom_z(variant_dir: _Path) -> float:
    parsed = _inhand_parse_model_xml(variant_dir)
    bottom_z: list[float] = []
    for geom in parsed["col_geoms"]:
        geom_type = geom.get("type", "mesh")
        if geom_type != "box":
            continue
        pos = _parse_float_list(geom.get("pos", ""))
        size = _parse_float_list(geom.get("size", ""))
        if len(size) < 3:
            continue
        pos_z = pos[2] if len(pos) >= 3 else 0.0
        bottom_z.append(pos_z - size[2])

    if bottom_z:
        return min(bottom_z)

    # Newer asset generations replace the box collision proxy with coacd mesh
    # collisions plus a class="region" bounding box; its extent carries the
    # same bottom-z information.
    bbox_pos, bbox_size = parsed.get("bbox_pos"), parsed.get("bbox_size")
    if bbox_pos is not None and bbox_size is not None and len(bbox_size) >= 3:
        pos_z = bbox_pos[2] if len(bbox_pos) >= 3 else 0.0
        return pos_z - bbox_size[2]

    raise RuntimeError(
        f"conveyor_pick asset variant has no box collision proxy or region "
        f"bounding box: {variant_dir}"
    )


def _conveyor_pick_spawn_z(variant_dir: _Path) -> float:
    return (
        _CONVEYOR_BELT_TOP_Z
        + _CONVEYOR_OBJECT_SPAWN_CLEARANCE_M
        - _conveyor_pick_collision_bottom_z(variant_dir)
    )


def _conveyor_pick_get_variants(category: str) -> list[_Path]:
    cat_dir = _CONVEYOR_PICK_OBJECT_ASSET_ROOT / category
    if not cat_dir.is_dir():
        raise FileNotFoundError(f"conveyor_pick asset category is missing: {cat_dir}")

    variants: list[_Path] = []
    for variant_dir in sorted(cat_dir.iterdir()):
        if not variant_dir.is_dir():
            continue
        parsed = _inhand_parse_model_xml(variant_dir)
        if not parsed["col_geoms"]:
            raise RuntimeError(
                f"conveyor_pick asset variant has no collision geoms: {variant_dir}"
            )
        _conveyor_pick_collision_bottom_z(variant_dir)
        variants.append(variant_dir)

    if not variants:
        raise RuntimeError(f"conveyor_pick asset category has no variants: {category}")
    return variants


def _conveyor_pick_replace_object_block(base_text: str, object_xml: str) -> str:
    start = base_text.find(_CONVEYOR_TASK_OBJECTS_BEGIN)
    end = base_text.find(_CONVEYOR_TASK_OBJECTS_END)
    if start < 0 or end < 0 or end <= start:
        raise RuntimeError(
            "conveyor_pick base XML is missing conveyor object block markers"
        )
    end += len(_CONVEYOR_TASK_OBJECTS_END)
    return base_text[:start] + object_xml + base_text[end:]


def _conveyor_pick_apply_scene_transforms(xml: str, options: Any = None) -> str:
    if options is None:
        return xml
    if not (options.clean or options.mocap or options.flexible_gripper):
        return xml

    from abc_sim.scene_xml import transform_scene_xml

    transformed_xml, _ = transform_scene_xml(xml, options=options)
    return transformed_xml


def _conveyor_pick_build_xml(selections: list[dict[str, Any]]) -> str:
    base_text = _CONVEYOR_PICK_BASE_SCENE_XML.read_text()
    if _CONVEYOR_TASK_ASSETS_PLACEHOLDER not in base_text:
        raise RuntimeError(
            "conveyor_pick base XML is missing conveyor task asset placeholder"
        )

    lines_asset: list[str] = []
    lines_body: list[str] = []
    for index, selection in enumerate(selections, start=1):
        category = str(selection["category"])
        variant_dir = _Path(selection["variant_dir"]).resolve()
        x = float(selection["x"])
        y = float(selection["y"])
        z = float(selection["z"])
        yaw = float(selection["yaw"])
        parsed = _inhand_parse_model_xml(variant_dir)
        prefix = f"conveyor_obj_{index}"

        lines_asset.append(f"    <!-- Conveyor object {index}: {category}/{variant_dir.name} -->")
        for mesh in parsed["meshes"]:
            extra = {**mesh["extra"]}
            extra_str = "".join(f' {k}="{v}"' for k, v in extra.items())
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

        w, s = np.cos(yaw / 2), np.sin(yaw / 2)
        lines_body.extend(
            [
                f"    <!-- Conveyor object {index}: {category}/{variant_dir.name} -->",
                f'    <body name="conveyor_object_{index}" pos="{x:.4f} {y:.4f} {z:.4f}" quat="{w:.6f} 0 0 {s:.6f}">',
                f'      <joint name="conveyor_object_{index}_joint" type="free" damping="0"/>',
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
                f'      <geom name="conveyor_object_{index}_visual_{visual_index}" type="mesh"'
                f' {mesh_attr}{mat_attr} contype="0" conaffinity="0" group="2"'
                f' density="0" solimp="0.998 0.998 0.001" solref="0.001 1"/>'
            )

        for collision_index, collision_geom in enumerate(parsed["col_geoms"]):
            attrs: dict[str, str] = {
                "name": f"conveyor_object_{index}_geom_{collision_index}",
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

    result = base_text.replace(_CONVEYOR_TASK_ASSETS_PLACEHOLDER, "\n".join(lines_asset))
    result = _conveyor_pick_replace_object_block(result, "\n".join(lines_body))
    result = result.replace('file="i2rt_yam/', f'file="{_MODELS_DIR}/assets/i2rt_yam/')
    result = result.replace(
        'file="task_conveyor_pick/',
        f'file="{_MODELS_DIR}/assets/task_conveyor_pick/',
    )
    return result


class ConveyorPickObjectRandomizer(SceneRandomizer):
    """Reload conveyor_pick with sampled task-scoped object assets."""

    perturbations: list = []

    def __init__(self) -> None:
        super().__init__()
        self._rng = np.random.default_rng()
        self._variants: dict[str, list[_Path]] = {}

    def clone(self) -> "SceneRandomizer":
        return type(self)()

    def prepare_env(self) -> None:
        if self._env_ref is not None:
            self.randomize(self._env_ref.model, self._env_ref.data)

    def _get_variants(self, category: str) -> list[_Path]:
        if category not in self._variants:
            self._variants[category] = _conveyor_pick_get_variants(category)
        return self._variants[category]

    def _sample_selections(self, *, spawn_side_y: float) -> list[dict[str, Any]]:
        selections: list[dict[str, Any]] = []
        for x_nominal, y_nominal in _CONVEYOR_OBJECT_SLOTS[:_CONVEYOR_OBJECT_COUNT]:
            category = _CONVEYOR_OBJECT_CATEGORIES[
                int(self._rng.integers(0, len(_CONVEYOR_OBJECT_CATEGORIES)))
            ]
            variants = self._get_variants(category)
            variant_dir = variants[int(self._rng.integers(0, len(variants)))]
            z = _conveyor_pick_spawn_z(variant_dir)
            y_nominal *= spawn_side_y
            x = float(x_nominal + self._rng.uniform(*_CONVEYOR_X_JITTER_M))
            y = float(y_nominal + self._rng.uniform(*_CONVEYOR_Y_JITTER_M))
            yaw = float(self._rng.uniform(-np.pi, np.pi))
            selections.append(
                {
                    "category": category,
                    "variant_dir": variant_dir,
                    "variant": variant_dir.name,
                    "x": x,
                    "y": y,
                    "z": z,
                    "yaw": yaw,
                }
            )
        return selections

    def randomize(
        self,
        model: Any,
        data: Any,
        seed: int | None = None,
        request: Any | None = None,
    ) -> RandomizationState:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        mirrored = bool(self._rng.random() < 0.5)
        belt_direction_y = 1.0 if mirrored else -1.0
        spawn_side_y = -belt_direction_y
        selections = self._sample_selections(spawn_side_y=spawn_side_y)
        belt_speed_range_m_s = _conveyor_belt_speed_range(request)
        belt_speed_m_s = float(self._rng.uniform(*belt_speed_range_m_s))
        belt_speed_y = belt_direction_y * belt_speed_m_s
        xml = _conveyor_pick_build_xml(selections)
        xml = _conveyor_pick_apply_scene_transforms(
            xml,
            self._scene_xml_transform_options,
        )

        env = self._env_ref
        if env is not None:
            preserved_robot_qpos = env._get_reset_robot_qpos()
            env.reload_from_xml(xml)
            mujoco.mj_resetData(env.model, env.data)
            env._set_robot_qpos(preserved_robot_qpos)
            mujoco.mj_forward(env.model, env.data)

        object_states: dict[str, dict[str, list[float]]] = {}
        metadata_objects: list[dict[str, Any]] = []
        for index, selection in enumerate(selections, start=1):
            yaw = float(selection["yaw"])
            quat = [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))]
            joint_name = f"conveyor_object_{index}_joint"
            object_states[joint_name] = {
                "pos": [float(selection["x"]), float(selection["y"]), float(selection["z"])],
                "quat": quat,
            }
            metadata_objects.append(
                {
                    "index": index,
                    "category": selection["category"],
                    "variant": selection["variant"],
                    "joint": joint_name,
                }
            )

        return RandomizationState(
            seed=seed or 0,
            object_states=object_states,
            metadata={
                "objects": metadata_objects,
                "mirrored": mirrored,
                "belt_direction": (
                    "negative_y_to_positive_y"
                    if mirrored
                    else "positive_y_to_negative_y"
                ),
                "belt_direction_y": belt_direction_y,
                "spawn_side_y": spawn_side_y,
                "belt_speed_m_s": belt_speed_m_s,
                "belt_speed_y": belt_speed_y,
                "belt_speed_range_m_s": list(belt_speed_range_m_s),
            },
        )




__all__ = ["ConveyorPickObjectRandomizer"]
