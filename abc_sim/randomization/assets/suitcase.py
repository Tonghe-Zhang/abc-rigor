"""Small rigid suitcase and size-normalized household packing objects."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache

import mujoco
import numpy as np

from ..core import _format_float_list, _scale_body_subtree
from .tabletop_distractors import DistractorSpec, build_distractor, distractor_variants
from .xml_common import _dishrack_compiled_model_bounds

CASE_HALF_XY = (0.10, 0.15)
CASE_INNER_XY = tuple(half - 0.012 for half in CASE_HALF_XY)
CASE_FLOOR_Z = 0.012
CASE_RIM_Z = 0.09
LID_LIMIT_RAD = float(np.deg2rad(110))
PACKING_CATEGORIES = ("bar_soap", "sponge", "apple", "lemon", "orange", "potato", "donut", "yogurt")
SHELL_PALETTES = {
    "navy": ((0.06, 0.13, 0.30), (0.18, 0.34, 0.55)),
    "teal": ((0.04, 0.32, 0.32), (0.20, 0.62, 0.58)),
    "burgundy": ((0.32, 0.04, 0.10), (0.65, 0.20, 0.27)),
    "mustard": ((0.48, 0.28, 0.04), (0.85, 0.62, 0.18)),
    "purple": ((0.23, 0.10, 0.38), (0.53, 0.33, 0.68)),
    "silver": ((0.30, 0.34, 0.38), (0.66, 0.70, 0.74)),
    "coral": ((0.60, 0.14, 0.10), (0.92, 0.40, 0.28)),
    "forest": ((0.08, 0.23, 0.12), (0.32, 0.47, 0.22)),
}
SHELL_PATTERNS = ("solid", "checker", "fine_checker", "bands")


@dataclass(frozen=True)
class SuitcaseTexture:
    palette: str
    pattern: str

    def __post_init__(self):
        if self.palette not in SHELL_PALETTES or self.pattern not in SHELL_PATTERNS:
            raise ValueError(f"Unknown suitcase texture: {self}.")


def build_suitcase(texture: SuitcaseTexture) -> tuple[list[ET.Element], ET.Element]:
    """Build a free case with a passive 0–110 degree lid and solid cavity walls."""
    hx, hy = CASE_HALF_XY
    ix, iy = CASE_INNER_XY
    first, second = SHELL_PALETTES[texture.palette]
    repeat = {"solid": (1, 1), "checker": (3, 4), "fine_checker": (12, 16), "bands": (1, 12)}[texture.pattern]
    asset = ET.Element("asset")
    ET.SubElement(
        asset,
        "texture",
        name="suitcase_shell_texture",
        type="2d",
        builtin="checker",
        rgb1=_format_float_list(first),
        rgb2=_format_float_list(first if texture.pattern == "solid" else second),
        width="256",
        height="256",
    )
    ET.SubElement(
        asset,
        "material",
        name="suitcase_shell",
        texture="suitcase_shell_texture",
        texrepeat="1 1",
        specular="0.25",
        shininess="0.3",
    )
    ET.SubElement(asset, "material", name="suitcase_lining", rgba="0.22 0.24 0.27 1")
    ET.SubElement(asset, "material", name="suitcase_trim", rgba="0.055 0.06 0.07 1")
    case = ET.Element("body", name="suitcase", pos="0.62 0 0.75")
    ET.SubElement(case, "freejoint", name="suitcase_joint")
    ET.SubElement(
        case,
        "inertial",
        pos="0 0 0.035",
        mass="1.2",
        diaginertia=_format_float_list(
            1.2 / 12 * np.array([(2 * hy) ** 2 + CASE_RIM_Z**2, (2 * hx) ** 2 + CASE_RIM_Z**2, 4 * (hx**2 + hy**2)])
        ),
    )

    def box(parent, name, pos, size, material="suitcase_shell", **attrs):
        if material == "suitcase_shell":
            # Separate UV vertices on each face preserve the shell texture in
            # both MuJoCo rendering and the browser's glTF mesh conversion.
            vertices = (
                np.array(
                    [
                        [-1, -1, -1],
                        [1, -1, -1],
                        [1, 1, -1],
                        [-1, 1, -1],
                        [-1, -1, 1],
                        [1, -1, 1],
                        [1, 1, 1],
                        [-1, 1, 1],
                    ]
                )
                * size
            )
            quads = np.array([[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [3, 7, 6, 2], [0, 4, 7, 3], [1, 2, 6, 5]])
            faces = np.array([[4 * i, 4 * i + 1, 4 * i + 2, 4 * i, 4 * i + 2, 4 * i + 3] for i in range(6)])
            mesh_name = f"{name}_visual_mesh"
            ET.SubElement(
                asset,
                "mesh",
                name=mesh_name,
                vertex=_format_float_list(vertices[quads].ravel()),
                face=" ".join(str(int(i)) for i in faces.ravel()),
                texcoord=_format_float_list(
                    np.tile(np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) * repeat, (6, 1)).ravel()
                ),
            )
            ET.SubElement(
                parent,
                "geom",
                name=f"{name}_visual",
                type="mesh",
                mesh=mesh_name,
                pos=_format_float_list(pos),
                material=material,
                group="2",
                contype="0",
                conaffinity="0",
                density="0",
            )
            return ET.SubElement(
                parent,
                "geom",
                name=name,
                type="box",
                pos=_format_float_list(pos),
                size=_format_float_list(size),
                friction="0.9 0.01 0.001",
                solref="0.004 1",
                group="3",
                rgba="0 0 0 0",
                **attrs,
            )
        return ET.SubElement(
            parent,
            "geom",
            name=name,
            type="box",
            pos=_format_float_list(pos),
            size=_format_float_list(size),
            material=material,
            friction="0.9 0.01 0.001",
            solref="0.004 1",
            **attrs,
        )

    box(case, "suitcase_floor", (0, 0, 0.006), (hx, hy, 0.006), "suitcase_lining")
    for sign, label in ((-1, "front"), (1, "back")):
        box(case, f"suitcase_{label}", (sign * (hx - 0.006), 0, 0.051), (0.006, hy, 0.039))
    for sign, label in ((-1, "right"), (1, "left")):
        box(case, f"suitcase_{label}", (0, sign * (hy - 0.006), 0.051), (ix, 0.006, 0.039))
    # A carrying handle is attached to the front of the lower shell.
    for sign in (-1, 1):
        box(
            case,
            f"suitcase_handle_post_{sign}",
            (-hx - 0.014, sign * 0.048, 0.044),
            (0.014, 0.008, 0.008),
            "suitcase_trim",
        )
    box(case, "suitcase_handle", (-hx - 0.028, 0, 0.044), (0.008, 0.056, 0.008), "suitcase_trim")
    lid = ET.SubElement(case, "body", name="suitcase_lid", pos=_format_float_list((hx, 0, CASE_RIM_Z)))
    ET.SubElement(
        lid,
        "joint",
        name="suitcase_hinge",
        type="hinge",
        axis="0 1 0",
        limited="true",
        range=f"0 {LID_LIMIT_RAD:.16g}",
        damping="0.06",
        frictionloss="0.015",
        solreflimit="0.004 1",
        solimplimit="0.99 0.99 0.001",
    )
    ET.SubElement(
        lid,
        "inertial",
        pos=_format_float_list((-hx, 0, 0.008)),
        mass="0.15",
        diaginertia=_format_float_list(
            0.15 / 12 * np.array([(2 * hy) ** 2 + 0.016**2, (2 * hx) ** 2 + 0.016**2, 4 * (hx**2 + hy**2)])
        ),
    )
    box(lid, "suitcase_lid_panel", (-hx, 0, 0.008), (hx, hy, 0.008))
    box(
        lid,
        "suitcase_lid_lining",
        (-hx, 0, -0.0005),
        (ix, iy, 0.0005),
        "suitcase_lining",
        contype="0",
        conaffinity="0",
        density="0",
    )
    for sign in (-1, 1):
        box(
            lid,
            f"suitcase_lid_strap_{sign}",
            (-hx, sign * hy / 3, -0.0015),
            (ix - 0.002, 0.009, 0.0005),
            "suitcase_trim",
            contype="0",
            conaffinity="0",
            density="0",
        )
    box(lid, "suitcase_lid_pull", (-2 * hx - 0.015, 0, 0.008), (0.015, 0.025, 0.008), "suitcase_trim")
    return list(asset), case


@lru_cache(maxsize=None)
def packing_object_dimensions(spec: DistractorSpec) -> tuple[float, tuple[float, float, float]]:
    """Return uniform scale and full bounds; three objects fit side by side."""
    if spec.category not in PACKING_CATEGORIES:
        raise ValueError(f"Unsupported packing object category {spec.category!r}.")
    assets, body = build_distractor(spec, 1)
    body.set("pos", "0 0 0")
    root = ET.Element("mujoco")
    ET.SubElement(root, "asset").extend(assets)
    ET.SubElement(root, "worldbody").append(body)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    lower, upper = _dishrack_compiled_model_bounds(model, data)
    dimensions = upper - lower
    scale = float(min(1.0, 0.08 / dimensions.max(), 0.055 / dimensions[2]))
    return scale, tuple(float(x) for x in dimensions * scale)


def build_packing_object(spec: DistractorSpec, index: int) -> tuple[list[ET.Element], ET.Element]:
    """Build a textured object with a solid enclosing collision proxy."""
    scale, _ = packing_object_dimensions(spec)
    assets, body = build_distractor(spec, index)
    for parent in (*assets, body):
        for elem in parent.iter():
            for key in ("name", "mesh", "texture", "material"):
                value = elem.get(key)
                if value is not None and value.startswith(f"distractor_{index}"):
                    elem.set(key, value.replace(f"distractor_{index}", f"packing_object_{index}", 1))
    asset = ET.Element("asset")
    asset.extend(assets)
    _scale_body_subtree(
        body=body,
        factor=scale,
        target_name=f"packing_object_{index}",
        asset_elem=asset,
        mesh_assets={mesh.attrib["name"]: mesh for mesh in asset.findall("mesh")},
    )
    return list(asset), body


def packing_variants(category: str) -> tuple[str, ...]:
    if category not in PACKING_CATEGORIES:
        raise ValueError(f"Unsupported packing object category {category!r}.")
    return distractor_variants(category)
