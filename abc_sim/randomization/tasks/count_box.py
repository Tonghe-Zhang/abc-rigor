from __future__ import annotations

import re
from typing import Any

import mujoco
import numpy as np

from ..assets.paths import _MODELS_DIR
from ..core import RandomizationState, SceneRandomizer, _resolve_scene_xml_paths

_COUNT_BOX_BASE_SCENE_XML = _MODELS_DIR / "yam_count_into_opaque_box_scene.xml"
_COUNT_BOX_OBJECTS_BEGIN = "<!-- COUNT_BOX_OBJECTS_BEGIN -->"
_COUNT_BOX_OBJECTS_END = "<!-- COUNT_BOX_OBJECTS_END -->"
_COUNT_BOX_OBJECT_COUNT = 12
_COUNT_BOX_TABLE_Z = 0.75
_COUNT_BOX_OBJECT_COLORS: tuple[str, ...] = (
    "red",
    "yellow",
    "blue",
    "green",
    "purple",
    "orange",
)
_COUNT_BOX_OBJECT_SHAPES: tuple[str, ...] = (
    "cube",
    "sphere",
    "cylinder",
    "triangle",
)
_COUNT_BOX_OBJECT_SIZES: tuple[str, ...] = ("small", "large")
_COUNT_BOX_OBJECT_SLOTS: tuple[tuple[float, float], ...] = (
    (0.42, 0.42),
    (0.54, 0.42),
    (0.42, 0.28),
    (0.54, 0.28),
    (0.42, 0.14),
    (0.54, 0.14),
    (0.42, -0.14),
    (0.54, -0.14),
    (0.42, -0.28),
    (0.54, -0.28),
    (0.42, -0.42),
    (0.54, -0.42),
)
_COUNT_BOX_OBJECT_JITTER_M = 0.010
_COUNT_BOX_OBJECT_CLEARANCE_M = 0.006
_COUNT_BOX_MAX_PLACEMENT_TRIES = 64
_COUNT_BOX_ARM_QPOS = "0 1.047 1.047 0 0 0 0 0  0 1.047 1.047 0 0 0 0 0"
_COUNT_BOX_HOME_KEY_RE = re.compile(
    r'(<key name="home"\s+qpos=")(.*?)("\s+ctrl=)',
    re.DOTALL,
)


def _count_box_shape_spec(shape: str, size: str) -> dict[str, str]:
    large = size == "large"
    if shape == "cube":
        half = 0.031 if large else 0.022
        return {
            "type": "box",
            "size": f"{half:.3f} {half:.3f} {half:.3f}",
            "half_height": f"{half:.3f}",
            "footprint_radius": f"{(2 ** 0.5 * half):.3f}",
            "mass": "0.026" if large else "0.018",
        }
    if shape == "sphere":
        radius = 0.030 if large else 0.021
        return {
            "type": "sphere",
            "size": f"{radius:.3f}",
            "half_height": f"{radius:.3f}",
            "footprint_radius": f"{radius:.3f}",
            "mass": "0.024" if large else "0.016",
        }
    if shape == "cylinder":
        radius = 0.029 if large else 0.021
        half_height = 0.031 if large else 0.022
        return {
            "type": "cylinder",
            "size": f"{radius:.3f} {half_height:.3f}",
            "half_height": f"{half_height:.3f}",
            "footprint_radius": f"{radius:.3f}",
            "mass": "0.025" if large else "0.017",
        }
    if shape == "triangle":
        half_height = 0.024 if large else 0.018
        footprint_radius = (0.034**2 + 0.030**2) ** 0.5 if large else (0.025**2 + 0.022**2) ** 0.5
        return {
            "type": "mesh",
            "mesh": f"count_triangle_prism_{size}",
            "half_height": f"{half_height:.3f}",
            "footprint_radius": f"{footprint_radius:.3f}",
            "mass": "0.024" if large else "0.016",
        }
    raise ValueError(f"unsupported count-box shape: {shape}")


def _count_box_plural_shape(shape: str) -> str:
    if shape == "triangle":
        return "triangles"
    if shape == "sphere":
        return "spheres"
    if shape == "cylinder":
        return "cylinders"
    if shape == "cube":
        return "cubes"
    return f"{shape}s"


def _count_box_object_noun(count: int) -> str:
    return "object" if count == 1 else "objects"


def _count_box_xy_clearance_ok(
    candidate: dict[str, Any],
    placed: list[dict[str, Any]],
) -> bool:
    candidate_xy = np.array([float(candidate["x"]), float(candidate["y"])])
    candidate_radius = float(candidate["footprint_radius"])
    for other in placed:
        other_xy = np.array([float(other["x"]), float(other["y"])])
        other_radius = float(other["footprint_radius"])
        min_distance = candidate_radius + other_radius + _COUNT_BOX_OBJECT_CLEARANCE_M
        if float(np.linalg.norm(candidate_xy - other_xy)) < min_distance:
            return False
    return True


def _count_box_replace_object_block(base_text: str, object_xml: str) -> str:
    start = base_text.find(_COUNT_BOX_OBJECTS_BEGIN)
    end = base_text.find(_COUNT_BOX_OBJECTS_END)
    if start < 0 or end < 0 or end <= start:
        raise RuntimeError("count_into_opaque_box XML is missing object block markers")
    end += len(_COUNT_BOX_OBJECTS_END)
    return base_text[:start] + object_xml + base_text[end:]


def _count_box_replace_home_qpos(base_text: str, selections: list[dict[str, Any]]) -> str:
    qpos_lines = ["0"]
    for selection in selections:
        qpos_lines.append(
            f'{float(selection["x"]):.4f} {float(selection["y"]):.4f} {float(selection["z"]):.4f} '
            f'{float(selection["qw"]):.6f} 0 0 {float(selection["qz"]):.6f}'
        )
    qpos_lines.append(_COUNT_BOX_ARM_QPOS)
    qpos = "\n            ".join(qpos_lines)
    replaced, count = _COUNT_BOX_HOME_KEY_RE.subn(
        lambda match: match.group(1) + qpos + match.group(3),
        base_text,
        count=1,
    )
    if count != 1:
        raise RuntimeError("count_into_opaque_box XML is missing home key qpos")
    return replaced


def _count_box_apply_scene_transforms(xml: str, options: Any = None) -> str:
    if options is None:
        return xml
    if not (options.clean or options.mocap or options.flexible_gripper):
        return xml

    from abc_sim.scene_xml import transform_scene_xml

    transformed_xml, _ = transform_scene_xml(xml, options=options)
    return transformed_xml


def _count_box_build_xml(selections: list[dict[str, Any]]) -> str:
    base_text = _COUNT_BOX_BASE_SCENE_XML.read_text()
    lines: list[str] = [_COUNT_BOX_OBJECTS_BEGIN]
    for index, selection in enumerate(selections, start=1):
        name = str(selection["name"])
        color = str(selection["color"])
        shape = str(selection["shape"])
        size = str(selection["size"])
        spec = _count_box_shape_spec(shape, size)
        geom_attrs = [
            f'name="{name}_geom"',
            'class="count_item"',
            f'type="{spec["type"]}"',
            f'mass="{spec["mass"]}"',
            f'material="count_item_{color}"',
        ]
        if "mesh" in spec:
            geom_attrs.append(f'mesh="{spec["mesh"]}"')
        else:
            geom_attrs.append(f'size="{spec["size"]}"')
        lines.extend(
            [
                f'    <body name="{name}" pos="{float(selection["x"]):.4f} {float(selection["y"]):.4f} {float(selection["z"]):.4f}" quat="{float(selection["qw"]):.6f} 0 0 {float(selection["qz"]):.6f}">',
                f'      <joint name="{name}_joint" class="count_item"/>',
                f'      <geom {" ".join(geom_attrs)}/>',
                "    </body>",
            ]
        )
    lines.append(f"    {_COUNT_BOX_OBJECTS_END}")
    xml = _count_box_replace_object_block(base_text, "\n".join(lines))
    xml = _count_box_replace_home_qpos(xml, selections)
    return _resolve_scene_xml_paths(xml, _COUNT_BOX_BASE_SCENE_XML.parent)


def _count_box_sample_prompt(
    objects: list[dict[str, Any]],
    rng: np.random.Generator,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []

    for color in _COUNT_BOX_OBJECT_COLORS:
        matching = [obj for obj in objects if obj["color"] == color]
        if 2 <= len(matching) <= 5:
            candidates.append(
                {
                    "prompt": f"put all {color} objects in the box",
                    "prompt_type": "all_color",
                    "target_count": len(matching),
                    "target_objects": [obj["name"] for obj in matching],
                    "eligible_objects": [obj["name"] for obj in matching],
                    "attributes": {"color": color},
                }
            )

    for shape in _COUNT_BOX_OBJECT_SHAPES:
        matching = [obj for obj in objects if obj["shape"] == shape]
        if 2 <= len(matching) <= 5:
            candidates.append(
                {
                    "prompt": f"put all {_count_box_plural_shape(shape)} in the box",
                    "prompt_type": "all_shape",
                    "target_count": len(matching),
                    "target_objects": [obj["name"] for obj in matching],
                    "eligible_objects": [obj["name"] for obj in matching],
                    "attributes": {"shape": shape},
                }
            )

    for size in _COUNT_BOX_OBJECT_SIZES:
        matching = [obj for obj in objects if obj["size"] == size]
        if 2 <= len(matching) <= 8:
            candidates.append(
                {
                    "prompt": f"put all {size} objects in the box",
                    "prompt_type": "all_size",
                    "target_count": len(matching),
                    "target_objects": [obj["name"] for obj in matching],
                    "eligible_objects": [obj["name"] for obj in matching],
                    "attributes": {"size": size},
                }
            )

    for color in _COUNT_BOX_OBJECT_COLORS:
        for shape in _COUNT_BOX_OBJECT_SHAPES:
            matching = [
                obj
                for obj in objects
                if obj["color"] == color and obj["shape"] == shape
            ]
            if matching:
                plural = _count_box_plural_shape(shape)
                candidates.append(
                    {
                        "prompt": f"put all {color} {plural} in the box",
                        "prompt_type": "all_color_shape",
                        "target_count": len(matching),
                        "target_objects": [obj["name"] for obj in matching],
                        "eligible_objects": [obj["name"] for obj in matching],
                        "attributes": {"color": color, "shape": shape},
                    }
                )

    for count in (2, 3, 4):
        candidates.append(_count_box_exact_count_prompt(objects, count))

    for color in _COUNT_BOX_OBJECT_COLORS:
        matching = [obj for obj in objects if obj["color"] == color]
        if len(matching) >= 2:
            count = int(rng.integers(1, min(3, len(matching)) + 1))
            candidates.append(
                {
                    "prompt": f"put exactly {count} {color} {_count_box_object_noun(count)} in the box",
                    "prompt_type": "exact_color_count",
                    "target_count": count,
                    "target_objects": [],
                    "eligible_objects": [obj["name"] for obj in matching],
                    "attributes": {"color": color},
                }
            )

    if not candidates:
        raise RuntimeError("count_into_opaque_box generated no valid prompt candidates")
    return candidates[int(rng.integers(0, len(candidates)))]


# Fixed-count task variants request one of these exact counts.
_COUNT_BOX_REQUESTED_COUNTS = (1, 2, 3, 4)


def _count_box_requested_count(request: Any | None) -> int | None:
    if request is None:
        return None
    if not isinstance(request, dict):
        raise TypeError(f"count_into_opaque_box reset request must be a dict, got {type(request).__name__}")
    unknown = set(request) - {"target_count"}
    if unknown:
        raise ValueError(f"count_into_opaque_box reset request has unknown keys: {sorted(unknown)}")
    target_count = request.get("target_count")
    if target_count is None:
        return None
    if int(target_count) not in _COUNT_BOX_REQUESTED_COUNTS:
        raise ValueError(f"count_into_opaque_box target_count must be in {_COUNT_BOX_REQUESTED_COUNTS}, got {target_count}")
    return int(target_count)


def _count_box_exact_count_prompt(objects: list[dict[str, Any]], count: int) -> dict[str, Any]:
    """Exact-count directive over every object, as sampled for unconstrained resets."""
    if len(objects) < count:
        raise RuntimeError(f"count_into_opaque_box scene has {len(objects)} objects, fewer than the requested {count}")
    return {
        "prompt": f"put exactly {count} {_count_box_object_noun(count)} in the box",
        "prompt_type": "exact_count",
        "target_count": count,
        "target_objects": [],
        "eligible_objects": [obj["name"] for obj in objects],
        "attributes": {},
    }


class CountIntoOpaqueBoxRandomizer(SceneRandomizer):
    """Reload count_into_opaque_box with sampled primitive objects and prompt."""

    perturbations: list = []

    def clone(self) -> "SceneRandomizer":
        return type(self)()

    def prepare_env(self) -> None:
        if self._env_ref is not None:
            self.randomize(
                self._env_ref.model,
                self._env_ref.data,
                request=getattr(self._env_ref, "_task_reset_request", None),
            )

    def _sample_selections(self, rng: np.random.Generator) -> list[dict[str, Any]]:
        colors = rng.choice(_COUNT_BOX_OBJECT_COLORS, size=_COUNT_BOX_OBJECT_COUNT, replace=True).tolist()
        shapes = rng.choice(_COUNT_BOX_OBJECT_SHAPES, size=_COUNT_BOX_OBJECT_COUNT, replace=True).tolist()
        sizes = rng.choice(_COUNT_BOX_OBJECT_SIZES, size=_COUNT_BOX_OBJECT_COUNT, replace=True).tolist()
        slot_order = list(range(len(_COUNT_BOX_OBJECT_SLOTS)))
        rng.shuffle(slot_order)

        selections: list[dict[str, Any]] = []
        for index in range(_COUNT_BOX_OBJECT_COUNT):
            color = colors[index]
            shape = shapes[index]
            size = sizes[index]
            x, y = _COUNT_BOX_OBJECT_SLOTS[slot_order[index]]
            spec = _count_box_shape_spec(shape, size)
            yaw = float(rng.uniform(-np.pi, np.pi))
            for _ in range(_COUNT_BOX_MAX_PLACEMENT_TRIES):
                candidate = {
                    "name": f"count_obj_{index + 1:02d}",
                    "joint": f"count_obj_{index + 1:02d}_joint",
                    "color": color,
                    "shape": shape,
                    "size": size,
                    "x": float(x + rng.uniform(-_COUNT_BOX_OBJECT_JITTER_M, _COUNT_BOX_OBJECT_JITTER_M)),
                    "y": float(y + rng.uniform(-_COUNT_BOX_OBJECT_JITTER_M, _COUNT_BOX_OBJECT_JITTER_M)),
                    "z": float(_COUNT_BOX_TABLE_Z + float(spec["half_height"]) + 0.004),
                    "yaw": yaw,
                    "qw": float(np.cos(yaw / 2)),
                    "qz": float(np.sin(yaw / 2)),
                    "footprint_radius": float(spec["footprint_radius"]),
                }
                if _count_box_xy_clearance_ok(candidate, selections):
                    selections.append(candidate)
                    break
            else:
                raise RuntimeError(
                    "count_into_opaque_box could not place non-overlapping objects"
                )
        return selections

    def randomize(
        self,
        model: Any,
        data: Any,
        seed: int | None = None,
        request: Any | None = None,
    ) -> RandomizationState:
        requested_count = _count_box_requested_count(request)
        rng = np.random.default_rng(seed)
        selections = self._sample_selections(rng)
        prompt_info = (
            _count_box_sample_prompt(selections, rng)
            if requested_count is None
            else _count_box_exact_count_prompt(selections, requested_count)
        )
        xml = _count_box_build_xml(selections)
        xml = _count_box_apply_scene_transforms(
            xml,
            self._scene_xml_transform_options,
        )

        env = self._env_ref
        if env is not None:
            preserved_arm_state = env._get_reset_arm_state()
            env.reload_from_xml(xml)
            mujoco.mj_resetData(env.model, env.data)
            env._set_qpos_from_state(preserved_arm_state)
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
                    "color": selection["color"],
                    "shape": selection["shape"],
                    "size": selection["size"],
                }
            )

        metadata = {
            "prompt": str(prompt_info["prompt"]),
            "prompt_type": prompt_info["prompt_type"],
            "target_count": int(prompt_info["target_count"]),
            "target_objects": list(prompt_info["target_objects"]),
            "eligible_objects": list(prompt_info["eligible_objects"]),
            "attributes": dict(prompt_info["attributes"]),
            "objects": metadata_objects,
        }
        return RandomizationState(
            seed=seed or 0,
            object_states=object_states,
            metadata=metadata,
        )




__all__ = ["CountIntoOpaqueBoxRandomizer"]
