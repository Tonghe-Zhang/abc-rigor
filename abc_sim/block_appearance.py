"""Seeded letter-block appearance records and replay validation."""

from __future__ import annotations

import colorsys
import string

import numpy as np

LETTERS = string.ascii_uppercase
REVISION = "block-appearance-v1"
FAMILIES = ("wood_paint", "color_plastic", "white_plastic")
FINISHES = {"matte": (0.05, 0.05), "satin": (0.25, 0.25), "gloss": (0.55, 0.65)}


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
