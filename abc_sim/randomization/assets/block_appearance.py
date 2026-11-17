"""Seeded, visual-only letter-block materials using the bundled glyphs."""

from __future__ import annotations

import colorsys
import string
import xml.etree.ElementTree as ET
from collections import deque
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np

LETTERS = string.ascii_uppercase
REVISION = "block-appearance-v1"
TILE = 128
FAMILIES = ("wood_paint", "color_plastic", "white_plastic")
FINISHES = {"matte": (0.05, 0.05), "satin": (0.25, 0.25), "gloss": (0.55, 0.65)}
BLOCK_ASSETS = Path(__file__).resolve().parents[2] / "models" / "assets" / "blocks"


def luminance(rgb: list[int] | np.ndarray) -> float:
    values = np.asarray(rgb, dtype=float) / 255
    linear = np.where(
        values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4
    )
    return float(linear @ [0.2126, 0.7152, 0.0722])


def contrast(first: list[int], second: list[int]) -> float:
    low, high = sorted((luminance(first), luminance(second)))
    return (high + 0.05) / (low + 0.05)


def _ink(rng: np.random.Generator) -> list[int]:
    color = np.asarray(
        colorsys.hsv_to_rgb(rng.random(), rng.uniform(0.65, 1), rng.uniform(0.35, 0.7))
    )
    while luminance(color * 255) > 0.12:
        color *= 0.85
    return np.rint(color * 255).astype(int).tolist()


def sample_appearance(seed: int) -> dict:
    """Sample styles without consuming the goal or placement random streams."""
    if type(seed) is not int or seed < 0:
        raise ValueError(
            "Block appearance requires an explicit nonnegative integer seed."
        )
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x41505052]))
    blocks = {}
    for letter in LETTERS:
        family = str(rng.choice(FAMILIES))
        light = rng.integers(232, 251, size=3).tolist()
        if family == "wood_paint":
            body = [
                int(rng.integers(205, 231)),
                int(rng.integers(164, 191)),
                int(rng.integers(112, 145)),
            ]
            glyph = _ink(rng)
            # Dark paint remains readable on the dimmest part of the grain.
            while contrast(glyph, (np.asarray(body) * 0.90).astype(int).tolist()) < 4.5:
                glyph = (np.asarray(glyph) * 0.8).astype(int).tolist()
            panel = light
        elif family == "color_plastic":
            body, panel, glyph = _ink(rng), _ink(rng), light
        else:
            body, panel, glyph = (
                light,
                rng.integers(232, 251, size=3).tolist(),
                _ink(rng),
            )
        blocks[letter] = {
            "family": family,
            "finish": str(rng.choice(tuple(FINISHES))),
            "body_rgb": body,
            "panel_rgb": panel,
            "glyph_rgb": glyph,
            "border_rgb": _ink(rng)
            if family != "color_plastic"
            else rng.integers(185, 246, size=3).tolist(),
            "panel": str(rng.choice(("none", "square", "rounded", "disc"))),
            "border": str(rng.choice(("none", "ring", "square", "double"))),
            "stroke": int(rng.integers(0, 2)),
            "texture_seed": int(rng.integers(0, 2**32)),
        }
    return {"revision": REVISION, "seed": seed, "blocks": blocks}


def validate_appearance(record: dict) -> None:
    """Reject incomplete or unsupported replay material records."""
    if record["revision"] != REVISION or set(record["blocks"]) != set(LETTERS):
        raise ValueError("Unsupported or incomplete block appearance record.")
    for style in record["blocks"].values():
        if style["family"] not in FAMILIES or style["finish"] not in FINISHES:
            raise ValueError("Unknown block material style.")
        if style["panel"] not in ("none", "square", "rounded", "disc") or style[
            "border"
        ] not in ("none", "ring", "square", "double"):
            raise ValueError("Unknown block face decoration.")
        if type(style["stroke"]) is not int or style["stroke"] not in (0, 1):
            raise ValueError("Unsupported glyph stroke.")
        for key in ("body_rgb", "panel_rgb", "glyph_rgb", "border_rgb"):
            if len(style[key]) != 3 or any(
                type(c) is not int or not 0 <= c <= 255 for c in style[key]
            ):
                raise ValueError("Material colors must be three byte values.")
        background = (
            style["body_rgb"] if style["panel"] == "none" else style["panel_rgb"]
        )
        if contrast(style["glyph_rgb"], background) < 4.5:
            raise ValueError("Glyph/background contrast must be at least 4.5.")
        if (
            type(style["texture_seed"]) is not int
            or not 0 <= style["texture_seed"] < 2**32
        ):
            raise ValueError("Invalid procedural texture seed.")


@lru_cache(maxsize=26)
def _mesh(
    letter: str,
) -> tuple[np.ndarray, np.ndarray, list[list[tuple[int, ...]]], np.ndarray]:
    rows = (BLOCK_ASSETS / f"block_{letter}.obj").read_text().splitlines()
    vertices = np.asarray(
        [list(map(float, row.split()[1:])) for row in rows if row.startswith("v ")]
    )
    uv = np.asarray(
        [list(map(float, row.split()[1:])) for row in rows if row.startswith("vt ")]
    )
    faces = [
        [tuple(int(x) - 1 for x in item.split("/")) for item in row.split()[1:]]
        for row in rows
        if row.startswith("f ")
    ]
    front = [
        face for face in faces if all(vertices[item[0], 0] > 0.099 for item in face)
    ]
    coords = np.asarray([uv[item[1]] for face in front for item in face])
    return vertices, uv, faces, np.stack((coords.min(axis=0), coords.max(axis=0)))


def _dilate(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1)
    return np.logical_or.reduce(
        [padded[y : y + TILE, x : x + TILE] for y in range(3) for x in range(3)]
    )


@lru_cache(maxsize=1)
def glyph_masks() -> dict[str, np.ndarray]:
    """Extract only painted glyph components, excluding wood and old borders.

    The bundled atlas and OBJ UVs supply letter shapes and orientation. No
    system font, downloaded font, original face panel or wood pixel is reused.
    """
    spec = mujoco.MjSpec()
    spec.add_texture(
        name="glyph_atlas",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        file=str(BLOCK_ASSETS / "texture.png"),
        nchannel=3,
        hflip=False,
        vflip=False,
    )
    model = spec.compile()
    atlas = model.tex_data.reshape(int(model.tex_height[0]), int(model.tex_width[0]), 3)
    masks = {}
    for letter in LETTERS:
        _, _, _, bounds = _mesh(letter)
        u = np.linspace(bounds[0, 0], bounds[1, 0], TILE)
        v = np.linspace(bounds[1, 1], bounds[0, 1], TILE)
        face = atlas[
            np.rint((1 - v) * (atlas.shape[0] - 1)).astype(int)[:, None],
            np.rint(u * (atlas.shape[1] - 1)).astype(int)[None, :],
            :3,
        ].astype(float)
        chroma = face.max(axis=2) - face.min(axis=2)
        mask = (chroma > 45) & (face.min(axis=2) < 110)
        # The original square border occupies the outer 14% of this UV tile.
        # Remove it before closing, including scratches connecting it to ink.
        mask[:18] = mask[-18:] = False
        mask[:, :18] = mask[:, -18:] = False
        # Close scratches, then retain the central glyph rather than the ring.
        mask = ~_dilate(~_dilate(mask))
        seen = np.zeros_like(mask)
        glyph = np.zeros_like(mask)
        for y, x in np.argwhere(mask):
            if seen[y, x]:
                continue
            queue = deque([(int(y), int(x))])
            seen[y, x] = True
            component = []
            while queue:
                cy, cx = queue.popleft()
                component.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if (
                        0 <= ny < TILE
                        and 0 <= nx < TILE
                        and mask[ny, nx]
                        and not seen[ny, nx]
                    ):
                        seen[ny, nx] = True
                        queue.append((ny, nx))
            if len(component) > 40 and all(
                12 < cy < TILE - 13 and 12 < cx < TILE - 13 for cy, cx in component
            ):
                ys, xs = np.asarray(component).T
                glyph[ys, xs] = True
        if glyph.sum() < 150:
            raise ValueError(f"Could not extract a clean glyph for {letter}.")
        masks[letter] = glyph
    return masks


def texture_tiles(letter: str, style: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact body and face tiles used by the scene (top-down RGB)."""
    rng = np.random.default_rng(style["texture_seed"])
    y, x = np.mgrid[:TILE, :TILE] / (TILE - 1) - 0.5
    body = np.broadcast_to(
        np.asarray(style["body_rgb"], dtype=float), (TILE, TILE, 3)
    ).copy()
    if style["family"] == "wood_paint":
        phase = rng.uniform(0, 2 * np.pi)
        grain = 0.045 * np.sin(65 * x + 3 * np.sin(9 * y + phase)) + rng.uniform(
            -0.025, 0.025, (TILE, TILE)
        )
        body *= 1 + grain[..., None]
    body = np.rint(body).clip(0, 255).astype(np.uint8)
    face = body.copy()
    radius = np.sqrt(x * x + y * y)
    square = np.maximum(np.abs(x), np.abs(y))
    panel = {
        "none": np.zeros((TILE, TILE), dtype=bool),
        "square": square < 0.43,
        "rounded": (
            np.maximum(np.abs(x) - 0.34, 0) ** 2 + np.maximum(np.abs(y) - 0.34, 0) ** 2
        )
        < 0.10**2,
        "disc": radius < 0.48,
    }[style["panel"]]
    face[panel] = style["panel_rgb"]
    # Borders stay outside every glyph's support.
    border = {
        "none": np.zeros_like(panel),
        "ring": (radius > 0.46) & (radius < 0.49),
        "square": (square > 0.43) & (square < 0.47),
        "double": ((square > 0.43) & (square < 0.445))
        | ((square > 0.475) & (square < 0.49)),
    }[style["border"]]
    glyph = glyph_masks()[letter]
    if style["stroke"]:
        glyph = _dilate(glyph)
    border &= ~_dilate(_dilate(glyph))
    face[border] = style["border_rgb"]
    face[glyph] = style["glyph_rgb"]
    return body, face


def face_tile_view(letter: str, tile: np.ndarray) -> np.ndarray:
    """Read a generated tile through the original top-face UV orientation.

    Letter atlas islands have different rotations/reflections. This CPU view
    follows the negative local-X face, with decreasing Y from left to right
    and decreasing Z from top to bottom, rather than rotating every tile alike.
    """
    vertices, uv, faces, bounds = _mesh(letter)
    front = [
        face for face in faces if all(vertices[item[0], 0] < -0.099 for item in face)
    ]
    points = np.asarray([vertices[item[0], 1:] for face in front for item in face])
    coords = np.asarray([uv[item[1]] for face in front for item in face])
    transform = np.linalg.lstsq(
        np.c_[points, np.ones(len(points))], coords, rcond=None
    )[0]
    y, z = np.meshgrid(np.linspace(0.1, -0.1, TILE), np.linspace(0.1, -0.1, TILE))
    mapped = np.stack((y, z, np.ones_like(y)), axis=-1) @ transform
    normalized = (mapped - bounds[0]) / (bounds[1] - bounds[0])
    x = np.rint(normalized[..., 0] * (TILE - 1)).astype(int).clip(0, TILE - 1)
    y = np.rint((1 - normalized[..., 1]) * (TILE - 1)).astype(int).clip(0, TILE - 1)
    return tile[y, x]


def appearance_xml(xml: str) -> str:
    """Give each existing visual mesh its own compact two-tile material.

    Mesh triangles, scale, collision geoms, inertia and dynamics stay intact.
    UVs for the two original letter faces retain their original orientation.
    """
    root = ET.fromstring(xml)
    asset = root.find("asset")
    if asset is None:
        raise ValueError("Block scene is missing its assets.")
    for letter in LETTERS:
        mesh = asset.find(f"mesh[@name='block_{letter}']")
        geom = root.find(f".//geom[@name='block_{letter}_vis']")
        if mesh is None or geom is None:
            raise ValueError(f"Block scene lacks the visual mesh for {letter}.")
        vertices, uv, faces, bounds = _mesh(letter)
        new_vertices, new_uv, triangles = [], [], []
        for face in faces:
            letter_face = all(vertices[item[0], 0] > 0.099 for item in face) or all(
                vertices[item[0], 0] < -0.099 for item in face
            )
            start = len(new_vertices)
            for item in face:
                vertex = vertices[item[0]]
                new_vertices.append(vertex)
                if letter_face:
                    coord = (uv[item[1]] - bounds[0]) / (bounds[1] - bounds[0])
                    new_uv.append([0.02 + 0.96 * coord[0], 0.51 + 0.48 * coord[1]])
                else:
                    new_uv.append(
                        [
                            0.02 + 0.96 * (vertex[0] / 0.2 + 0.5),
                            0.01 + 0.48 * (vertex[2] / 0.2 + 0.5),
                        ]
                    )
            for index in range(1, len(face) - 1):
                triangles.append([start, start + index, start + index + 1])
        mesh.attrib.pop("file")
        mesh.set("vertex", " ".join(map(str, np.asarray(new_vertices).ravel())))
        mesh.set("texcoord", " ".join(map(str, np.asarray(new_uv).ravel())))
        mesh.set("face", " ".join(map(str, np.asarray(triangles).ravel())))
        name = f"block_appearance_{letter}"
        ET.SubElement(
            asset,
            "texture",
            name=name,
            type="2d",
            builtin="flat",
            width=str(TILE),
            height=str(2 * TILE),
            rgb1="1 1 1",
        )
        ET.SubElement(asset, "material", name=name, texture=name)
        geom.set("material", name)
    asset.remove(asset.find("material[@name='block_mat']"))
    asset.remove(asset.find("texture[@name='block_tex']"))
    return ET.tostring(root, encoding="unicode")


def apply_appearance(model, record: dict) -> None:
    """Fill compiled texture/material buffers before the renderer is created."""
    validate_appearance(record)
    for letter, style in record["blocks"].items():
        name = f"block_appearance_{letter}"
        texture_id = model.texture(name).id
        start = int(model.tex_adr[texture_id])
        body, face = texture_tiles(letter, style)
        # Inline UVs retain OBJ's bottom-up V. MuJoCo flips V when importing
        # an OBJ but leaves inline UVs unchanged, so reverse these tile rows.
        pixels = np.concatenate((body[::-1], face[::-1]), axis=0)
        model.tex_data[start : start + pixels.size] = pixels.ravel()
        material_id = model.material(name).id
        model.mat_specular[material_id], model.mat_shininess[material_id] = FINISHES[
            style["finish"]
        ]
