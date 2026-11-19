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
HANDLE_STYLES = ("bridge", "rounded", "tab")
STRAP_LAYOUTS = ("none", "single", "double", "cross")
SHELL_DETAILS = ("smooth", "ribs", "corner_caps")


@dataclass(frozen=True)
class SuitcaseTexture:
    palette: str
    pattern: str

    def __post_init__(self):
        if self.palette not in SHELL_PALETTES or self.pattern not in SHELL_PATTERNS:
            raise ValueError(f"Unknown suitcase texture: {self}.")


@dataclass(frozen=True)
class SuitcaseStyle:
    width: float = 0.20
    length: float = 0.30
    height: float = 0.09
    wall_thickness: float = 0.012
    floor_thickness: float = 0.012
    lid_thickness: float = 0.016
    handle: str = "bridge"
    straps: str = "double"
    detail: str = "smooth"

    def __post_init__(self):
        ranges = {
            "width": (0.19, 0.21),
            "length": (0.29, 0.31),
            "height": (0.08, 0.10),
            "wall_thickness": (0.006, 0.012),
            "floor_thickness": (0.009, 0.014),
            "lid_thickness": (0.012, 0.020),
        }
        for name, (low, high) in ranges.items():
            value = getattr(self, name)
            if not np.isfinite(value) or not low <= value <= high:
                raise ValueError(f"Suitcase {name} must be between {low} and {high} metres.")
        if self.handle not in HANDLE_STYLES or self.straps not in STRAP_LAYOUTS or self.detail not in SHELL_DETAILS:
            raise ValueError(f"Unknown suitcase style: {self}.")


def build_suitcase(
    texture: SuitcaseTexture, style: SuitcaseStyle = SuitcaseStyle()
) -> tuple[list[ET.Element], ET.Element]:
    """Build a free case with a passive 0–110 degree lid and solid cavity walls."""
    hx, hy = style.width / 2, style.length / 2
    ix, iy = hx - style.wall_thickness, hy - style.wall_thickness
    floor_z, rim_z = style.floor_thickness, style.height
    wall_half = style.wall_thickness / 2
    wall_center_z, wall_half_z = (floor_z + rim_z) / 2, (rim_z - floor_z) / 2
    lid_half = style.lid_thickness / 2
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
        pos=_format_float_list((0, 0, rim_z * 0.4)),
        mass="1.2",
        diaginertia=_format_float_list(
            1.2 / 12 * np.array([(2 * hy) ** 2 + rim_z**2, (2 * hx) ** 2 + rim_z**2, 4 * (hx**2 + hy**2)])
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

    box(case, "suitcase_floor", (0, 0, floor_z / 2), (hx, hy, floor_z / 2), "suitcase_lining")
    for sign, label in ((-1, "front"), (1, "back")):
        box(case, f"suitcase_{label}", (sign * (hx - wall_half), 0, wall_center_z), (wall_half, hy, wall_half_z))
    for sign, label in ((-1, "right"), (1, "left")):
        box(case, f"suitcase_{label}", (0, sign * (hy - wall_half), wall_center_z), (ix, wall_half, wall_half_z))
    # A carrying handle is attached to the front of the lower shell.
    handle_z = rim_z / 2
    if style.handle == "bridge":
        for sign in (-1, 1):
            box(
                case,
                f"suitcase_handle_post_{sign}",
                (-hx - 0.014, sign * 0.048, handle_z),
                (0.014, 0.008, 0.008),
                "suitcase_trim",
            )
        box(case, "suitcase_handle", (-hx - 0.028, 0, handle_z), (0.008, 0.056, 0.008), "suitcase_trim")
    elif style.handle == "rounded":
        segments = [
            (-hx, -0.045, handle_z, -hx - 0.028, -0.045, handle_z),
            (-hx - 0.028, -0.045, handle_z, -hx - 0.028, 0.045, handle_z),
            (-hx - 0.028, 0.045, handle_z, -hx, 0.045, handle_z),
        ]
        for i, segment in enumerate(segments):
            ET.SubElement(
                case,
                "geom",
                name=f"suitcase_handle_rounded_{i}",
                type="capsule",
                fromto=_format_float_list(segment),
                size="0.009",
                material="suitcase_trim",
                friction="0.9 0.01 0.001",
            )
    else:
        box(
            case,
            "suitcase_handle_tab",
            (-hx - 0.022, 0, handle_z),
            (0.022, 0.032, 0.005),
            "suitcase_lining",
        )
    lid = ET.SubElement(case, "body", name="suitcase_lid", pos=_format_float_list((hx, 0, rim_z)))
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
        pos=_format_float_list((-hx, 0, lid_half)),
        mass="0.15",
        diaginertia=_format_float_list(
            0.15
            / 12
            * np.array(
                [(2 * hy) ** 2 + style.lid_thickness**2, (2 * hx) ** 2 + style.lid_thickness**2, 4 * (hx**2 + hy**2)]
            )
        ),
    )
    box(lid, "suitcase_lid_panel", (-hx, 0, lid_half), (hx, hy, lid_half))
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
    straps = {
        "none": (),
        "single": (((-hx, 0, -0.0015), (ix - 0.002, 0.012, 0.0005)),),
        "double": tuple(((-hx, sign * hy / 3, -0.0015), (ix - 0.002, 0.009, 0.0005)) for sign in (-1, 1)),
        "cross": (
            ((-hx, 0, -0.0015), (ix - 0.002, 0.009, 0.0005)),
            ((-hx, 0, -0.0025), (0.009, iy - 0.002, 0.0005)),
        ),
    }[style.straps]
    for i, (pos, size) in enumerate(straps):
        box(
            lid,
            f"suitcase_lid_strap_{i}",
            pos,
            size,
            "suitcase_trim",
            contype="0",
            conaffinity="0",
            density="0",
        )
    if style.detail == "ribs":
        for i, y in enumerate(np.linspace(-hy * 0.7, hy * 0.7, 5)):
            box(
                lid,
                f"suitcase_lid_rib_{i}",
                (-hx, y, style.lid_thickness + 0.002),
                (hx * 0.85, 0.003, 0.002),
                "suitcase_shell",
                contype="0",
                conaffinity="0",
                density="0",
            )
    elif style.detail == "corner_caps":
        for i, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
            box(
                lid,
                f"suitcase_lid_corner_{i}",
                (-hx + sx * (hx - 0.012), sy * (hy - 0.018), style.lid_thickness + 0.001),
                (0.012, 0.018, 0.001),
                "suitcase_trim",
                contype="0",
                conaffinity="0",
                density="0",
            )
    box(lid, "suitcase_lid_pull", (-2 * hx - 0.015, 0, lid_half), (0.015, 0.025, lid_half), "suitcase_trim")
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
