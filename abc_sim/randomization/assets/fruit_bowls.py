"""Open, graspable bowls and textured fruit with rounded collision geometry."""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache

import mujoco
import numpy as np

from ..core import _format_float_list, _scale_body_subtree
from .tabletop_distractors import DistractorSpec, build_distractor, distractor_variants

BOWL_SEGMENTS = 32
POLYPROPYLENE_DENSITY_KG_M3 = 900.0
BOWL_COLORS = {
    "blue": (0.06, 0.25, 0.80),
    "white": (0.92, 0.92, 0.89),
    "red": (0.76, 0.06, 0.07),
    "yellow": (0.95, 0.72, 0.05),
    "green": (0.06, 0.55, 0.20),
    "purple": (0.48, 0.10, 0.70),
    "orange": (0.95, 0.32, 0.04),
    "pink": (0.95, 0.36, 0.59),
}
FRUIT_CATEGORIES = ("apple", "orange", "lemon", "tomato")
FRUIT_DENSITY_KG_M3 = 800.0
FRUIT_RADIUS_RANGE_M = (0.030, 0.0375)


@dataclass(frozen=True)
class BowlSpec:
    color: str
    bottom_radius: float
    rim_radius: float
    height: float
    wall: float = 0.0025
    floor: float = 0.003
    density: float = POLYPROPYLENE_DENSITY_KG_M3

    def __post_init__(self):
        if self.color not in BOWL_COLORS:
            raise ValueError(f"Unknown bowl color: {self.color!r}.")
        values = (self.bottom_radius, self.rim_radius, self.height, self.wall, self.floor, self.density)
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("Bowl dimensions and density must be finite and positive.")
        if not self.floor < self.height or not self.bottom_radius < self.rim_radius:
            raise ValueError("An open bowl requires a rising, outward-flaring wall.")


def source_bowl_spec(color="blue"):
    return BowlSpec(color, bottom_radius=0.050, rim_radius=0.080, height=0.060)


def target_bowl_spec(color="white"):
    return BowlSpec(color, bottom_radius=0.145, rim_radius=0.170, height=0.065)


def build_bowl(name: str, spec: BowlSpec) -> tuple[list[ET.Element], ET.Element]:
    """Build a flat floor, convex wall segments, and a rounded gripping rim."""
    asset = ET.Element("asset")
    ET.SubElement(
        asset,
        "material",
        name=f"{name}_color",
        rgba=_format_float_list((*BOWL_COLORS[spec.color], 1)),
        specular="0.25",
        shininess="0.3",
    )
    body = ET.Element("body", name=name, pos="0.60 0 0.75")
    ET.SubElement(body, "freejoint", name=f"{name}_joint")
    # MuJoCo infers mass, center of mass, and inertia from the compound shell.
    geom = dict(
        material=f"{name}_color",
        density=str(spec.density),
        friction="0.6 0.005 0.0005",
        solref="0.004 1",
        condim="6",
    )
    apothem_scale = 1 / np.cos(np.pi / BOWL_SEGMENTS)
    ET.SubElement(
        body,
        "geom",
        name=f"{name}_floor",
        type="cylinder",
        pos=f"0 0 {spec.floor / 2}",
        size=f"{(spec.bottom_radius + spec.wall) * apothem_scale} {spec.floor / 2}",
        **geom,
    )
    faces = "0 2 1 0 3 2 4 5 6 4 6 7 0 1 5 0 5 4 1 2 6 1 6 5 2 3 7 2 7 6 3 0 4 3 4 7"
    for i in range(BOWL_SEGMENTS):
        angles = (np.arange(2) + i - 0.5) * 2 * np.pi / BOWL_SEGMENTS
        xy = np.column_stack((np.cos(angles), np.sin(angles)))
        vertices = []
        for z, radius in ((spec.floor, spec.bottom_radius), (spec.height, spec.rim_radius)):
            for radial, endpoint in ((radius, 0), (radius + spec.wall, 0), (radius + spec.wall, 1), (radius, 1)):
                vertices.append([*(xy[endpoint] * radial * apothem_scale), z])
        mesh_name = f"{name}_wall_mesh_{i}"
        ET.SubElement(
            asset, "mesh", name=mesh_name, vertex=_format_float_list(np.asarray(vertices).ravel()), face=faces
        )
        ET.SubElement(body, "geom", name=f"{name}_wall_{i}", type="mesh", mesh=mesh_name, **geom)
        rim_xy = xy * (spec.rim_radius + spec.wall / 2) * apothem_scale
        ET.SubElement(
            body,
            "geom",
            name=f"{name}_rim_{i}",
            type="capsule",
            fromto=_format_float_list([*rim_xy[0], spec.height, *rim_xy[1], spec.height]),
            size=str(spec.wall / 2),
            **geom,
        )
    for part, z, radius in (("floor", spec.floor, spec.bottom_radius), ("rim", spec.height, spec.rim_radius)):
        ET.SubElement(
            body,
            "site",
            name=f"{name}_cavity_{part}",
            type="cylinder",
            pos=f"0 0 {z}",
            size=f"{radius} 0.001",
            rgba="0 0 0 0",
            group="4",
        )
    return list(asset), body


@dataclass(frozen=True)
class FruitSpec:
    category: str
    variant: str
    radius: float = 0.035

    def __post_init__(self):
        if self.category not in FRUIT_CATEGORIES:
            raise ValueError(f"Unsupported fruit category: {self.category!r}.")
        DistractorSpec(self.category, self.variant)
        if not np.isfinite(self.radius) or not FRUIT_RADIUS_RANGE_M[0] <= self.radius <= FRUIT_RADIUS_RANGE_M[1]:
            raise ValueError("Fruit radius must be between 30 and 37.5 mm.")


@lru_cache(maxsize=None)
def _fruit_template(category, variant):
    assets, body = build_distractor(DistractorSpec(category, variant), 1)
    body.set("pos", "0 0 0")
    root = ET.Element("mujoco")
    ET.SubElement(root, "asset").extend(copy.deepcopy(assets))
    ET.SubElement(root, "worldbody").append(copy.deepcopy(body))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    vertices = []
    for g in range(model.ngeom):
        if (
            model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
            and not model.geom_contype[g]
            and not model.geom_conaffinity[g]
        ):
            mesh = model.geom_dataid[g]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            vertices.append(
                model.mesh_vert[start : start + count] @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
            )
    points = np.concatenate(vertices)
    center = (points.min(0) + points.max(0)) / 2
    radius = float(np.linalg.norm(points - center, axis=1).max())
    for parent in body.iter("body"):
        for g in list(parent.findall("geom")):
            if g.get("contype") != "0" or g.get("conaffinity") != "0":
                parent.remove(g)
    return assets, body, center, radius


def build_fruit(spec: FruitSpec, index: int) -> tuple[list[ET.Element], ET.Element]:
    """Fit a textured fruit inside a sphere whose center is its free-body origin."""
    assets, body, center, original_radius = _fruit_template(spec.category, spec.variant)
    assets, body = copy.deepcopy(assets), copy.deepcopy(body)
    name = f"fruit_{index}"
    for parent in (*assets, body):
        for elem in parent.iter():
            for key in ("name", "mesh", "texture", "material"):
                value = elem.get(key)
                if value is not None and value.startswith("distractor_1"):
                    elem.set(key, value.replace("distractor_1", name, 1))
    asset = ET.Element("asset")
    asset.extend(assets)
    scale = spec.radius / original_radius
    _scale_body_subtree(
        body=body,
        factor=scale,
        target_name=name,
        asset_elem=asset,
        mesh_assets={mesh.attrib["name"]: mesh for mesh in asset.findall("mesh")},
    )
    for child in body.findall("body"):
        child.set("pos", _format_float_list(np.fromstring(child.attrib["pos"], sep=" ") - center * scale))
    ET.SubElement(
        body,
        "geom",
        name=f"{name}_collision",
        type="sphere",
        size=str(spec.radius),
        density=str(FRUIT_DENSITY_KG_M3),
        rgba="0 0 0 0",
        group="3",
        friction="0.35 0.003 0.0002",
        condim="6",
        priority="1",
        solref="0.004 1",
    )
    body.set("pos", f"0.4 0 {0.752 + spec.radius}")
    return list(asset), body


def fruit_variants(category):
    if category not in FRUIT_CATEGORIES:
        raise ValueError(f"Unsupported fruit category: {category!r}.")
    return distractor_variants(category)
