"""Procedural wooden shape board, colored pieces, and visible reference card."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from ..core import _format_float_list, _resolve_scene_xml_paths

SHAPES = ("circle", "square", "triangle", "hexagon")
COLORS = {
    "red": (0.82, 0.08, 0.07, 1),
    "blue": (0.06, 0.25, 0.82, 1),
    "yellow": (0.96, 0.75, 0.04, 1),
    "green": (0.12, 0.60, 0.16, 1),
    "purple": (0.52, 0.12, 0.72, 1),
    "orange": (0.98, 0.35, 0.04, 1),
    "pink": (0.95, 0.36, 0.62, 1),
    "cyan": (0.04, 0.70, 0.76, 1),
}
WOOD_COLORS = ((0.67, 0.43, 0.22, 1), (0.79, 0.58, 0.34, 1), (0.53, 0.32, 0.16, 1))
PIECE_COUNT = 16
PITCH = 0.063
BOARD_HALF = 0.140
FLOOR_HEIGHT = 0.004
WALL_HEIGHT = 0.013
PIECE_HALF_HEIGHT = 0.011
HOLE_CLEARANCE = 0.0015
PIECE_DENSITY = 600.0
PIECE_JOINTS = tuple(f"shape_piece_{i:02d}_joint" for i in range(PIECE_COUNT))


def polygon(shape: str) -> np.ndarray:
    """Counterclockwise collision outline, in metres about the piece center."""
    sides, radius, angle = {
        "circle": (32, 0.022, 0),
        "square": (4, 0.021 * np.sqrt(2), np.pi / 4),
        "triangle": (3, 0.028, np.pi / 2),
        "hexagon": (6, 0.025, 0),
    }[shape]
    angles = angle + np.arange(sides) * 2 * np.pi / sides
    return radius * np.column_stack((np.cos(angles), np.sin(angles)))


def hole_polygon(shape: str) -> np.ndarray:
    vertices = polygon(shape)
    # Scaling expands every edge by the same minimum perpendicular clearance.
    edges = np.roll(vertices, -1, axis=0) - vertices
    distances = np.abs(edges[:, 0] * vertices[:, 1] - edges[:, 1] * vertices[:, 0]) / np.linalg.norm(edges, axis=1)
    return vertices * (1 + HOLE_CLEARANCE / distances.min())


def slot_center(slot: int) -> np.ndarray:
    return PITCH * np.array((slot // 4 - 1.5, slot % 4 - 1.5))


def prism_mesh(asset: ET.Element, name: str, vertices: np.ndarray, low: float, high: float) -> str:
    """Make one convex extruded polygon with explicit outward-facing triangles."""
    count = len(vertices)
    points = np.concatenate([np.column_stack((vertices, np.full(count, z))) for z in (low, high)])
    faces = []
    for i in range(1, count - 1):
        faces.extend(((0, i + 1, i), (count, count + i, count + i + 1)))
    for i in range(count):
        j = (i + 1) % count
        faces.extend(((i, j, count + j), (i, count + j, count + i)))
    ET.SubElement(
        asset,
        "mesh",
        name=name,
        vertex=_format_float_list(points.ravel()),
        face=" ".join(map(str, np.asarray(faces).ravel())),
    )
    return name


def _ring_cells(vertices: np.ndarray):
    """Partition a square tile around its polygonal opening into convex prisms."""
    angles = np.unique(
        np.round(
            np.mod(
                np.concatenate((np.arctan2(vertices[:, 1], vertices[:, 0]), np.pi / 4 + np.arange(4) * np.pi / 2)),
                2 * np.pi,
            ),
            12,
        )
    )
    rays = np.column_stack((np.cos(angles), np.sin(angles)))
    edges = np.roll(vertices, -1, axis=0) - vertices
    normals = np.column_stack((edges[:, 1], -edges[:, 0]))
    edge_offsets = np.einsum("ij,ij->i", normals, vertices)
    denominators = rays @ normals.T
    distances = np.full_like(denominators, np.inf)
    np.divide(edge_offsets[None, :], denominators, out=distances, where=denominators > 0)
    inner = rays * distances.min(axis=1)[:, None]
    outer = rays * (PITCH / 2 / np.abs(rays).max(axis=1))[:, None]
    for i in range(len(rays)):
        j = (i + 1) % len(rays)
        yield np.array((inner[i], outer[i], outer[j], inner[j]))


def build_board(palette: tuple[str, ...], order: tuple[int, ...], wood: int):
    """Build all sixteen unique pieces, their recesses, and the reference image."""
    asset = ET.Element("asset")
    world = ET.Element("worldbody")
    board = ET.SubElement(world, "body", name="shape_board", pos="0.59 0 0.75")
    ET.SubElement(board, "freejoint", name="shape_board_joint")
    common = {"friction": "0.7 0.01 0.001", "solref": "0.004 1", "density": "600"}
    wood_rgba = _format_float_list(WOOD_COLORS[wood])
    ET.SubElement(
        board,
        "geom",
        name="shape_board_floor",
        type="box",
        size=_format_float_list((BOARD_HALF, BOARD_HALF, FLOOR_HEIGHT / 2)),
        pos=_format_float_list((0, 0, FLOOR_HEIGHT / 2)),
        rgba=wood_rgba,
        **common,
    )
    grid_half = 2 * PITCH
    for axis in range(2):
        for sign in (-1, 1):
            center = [0.0, 0.0, (FLOOR_HEIGHT + WALL_HEIGHT) / 2]
            size = [grid_half, grid_half, (WALL_HEIGHT - FLOOR_HEIGHT) / 2]
            center[axis] = sign * (BOARD_HALF + grid_half) / 2
            size[axis] = (BOARD_HALF - grid_half) / 2
            size[1 - axis] = BOARD_HALF if axis == 0 else grid_half
            ET.SubElement(
                board,
                "geom",
                name=f"shape_board_border_{axis}_{sign}",
                type="box",
                pos=_format_float_list(center),
                size=_format_float_list(size),
                rgba=wood_rgba,
                **common,
            )
    card = ET.SubElement(world, "body", name="shape_reference", pos="0.822 0 0.753")
    ET.SubElement(
        card,
        "geom",
        name="shape_reference_paper",
        type="box",
        size="0.070 0.070 0.001",
        rgba="0.99 0.99 0.97 1",
        contype="0",
        conaffinity="0",
    )
    for slot, piece in enumerate(order):
        shape = SHAPES[piece // 4]
        center = slot_center(slot)
        ET.SubElement(
            board,
            "site",
            name=f"shape_target_{piece:02d}",
            pos=_format_float_list((*center, FLOOR_HEIGHT)),
            size="0.001",
            rgba="0 0 0 0",
            group="5",
        )
        for part, cell in enumerate(_ring_cells(hole_polygon(shape))):
            mesh = prism_mesh(asset, f"shape_slot_{slot:02d}_{part}_mesh", cell, FLOOR_HEIGHT, WALL_HEIGHT)
            ET.SubElement(
                board,
                "geom",
                name=f"shape_slot_{slot:02d}_{part}",
                type="mesh",
                mesh=mesh,
                pos=_format_float_list((*center, 0)),
                rgba=wood_rgba,
                **common,
            )
        mesh = prism_mesh(asset, f"shape_reference_{slot:02d}_mesh", polygon(shape) * 0.46, 0.0011, 0.0013)
        ET.SubElement(
            card,
            "geom",
            name=f"shape_reference_{slot:02d}",
            type="mesh",
            mesh=mesh,
            pos=_format_float_list((*(center * 0.46), 0)),
            rgba=_format_float_list(COLORS[palette[piece % 4]]),
            contype="0",
            conaffinity="0",
        )
    # Matching corner marks make the board/card orientation visible from any camera.
    for body, scale, height in ((board, 1.0, WALL_HEIGHT + 0.0002), (card, 0.46, 0.0015)):
        ET.SubElement(
            body,
            "geom",
            type="cylinder",
            size=_format_float_list((0.004 * scale, 0.0002)),
            pos=_format_float_list((-0.133 * scale, -0.133 * scale, height)),
            rgba="0.10 0.07 0.04 1",
            contype="0",
            conaffinity="0",
        )
    for piece in range(PIECE_COUNT):
        shape = SHAPES[piece // 4]
        mesh = prism_mesh(
            asset,
            f"shape_piece_{piece:02d}_mesh",
            polygon(shape),
            -PIECE_HALF_HEIGHT,
            PIECE_HALF_HEIGHT,
        )
        body = ET.SubElement(
            world,
            "body",
            name=f"shape_piece_{piece:02d}",
            pos=_format_float_list((0.39 + 0.09 * (piece // 4), (-0.36, -0.25, 0.25, 0.36)[piece % 4], 0.763)),
        )
        ET.SubElement(body, "freejoint", name=PIECE_JOINTS[piece])
        ET.SubElement(
            body,
            "geom",
            name=f"shape_piece_{piece:02d}_geom",
            type="mesh",
            mesh=mesh,
            rgba=_format_float_list(COLORS[palette[piece % 4]]),
            **(common | {"density": str(PIECE_DENSITY)}),
        )
    return list(asset), list(world)


def scene_xml(palette=("red", "blue", "yellow", "green"), order=tuple(range(16)), wood=0) -> str:
    """Generate a self-contained scene using only the shared YAM mesh package."""
    models = Path(__file__).resolve().parents[2] / "models"
    root = ET.parse(models / "yam_bimanual_empty.xml").getroot()
    # Preserve the station's compiled inertias explicitly before allowing the
    # small wooden pieces to use their geometry-derived rotational inertia.
    station = mujoco.MjModel.from_xml_string(_resolve_scene_xml_paths(ET.tostring(root, encoding="unicode"), models))
    for body in root.findall(".//body"):
        body_id = station.body(body.attrib["name"]).id
        inertial = body.find("inertial")
        if inertial is not None:
            body.remove(inertial)
        ET.SubElement(
            body,
            "inertial",
            pos=_format_float_list(station.body_ipos[body_id]),
            quat=_format_float_list(station.body_iquat[body_id]),
            mass=str(station.body_mass[body_id]),
            diaginertia=_format_float_list(station.body_inertia[body_id]),
        )
    root.find("compiler").set("boundinertia", "0")
    root.set("model", "yam_shape_matching")
    assets, bodies = build_board(palette, order, wood)
    root.find("asset").extend(assets)
    root.find("worldbody").extend(bodies)
    root.find("./keyframe/key[@name='home']").attrib.pop("qpos", None)
    ET.indent(root)
    return ET.tostring(root, encoding="unicode") + "\n"
