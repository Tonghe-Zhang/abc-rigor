"""Opt-in background variations: real wallpaper + table surface textures on the scene's walls and table.

A background package (see ``docs/background-variations.md``) holds W wallpaper designs and T table
surface types; every (wallpaper, table) pair is one *combination*, ``id = wall_index * T + table_index``
with both type lists in sorted name order. Each design/type has one or more *instances* (the same print
photographed in different places); a draw picks the combination and the instances.

Only appearance changes. The background is a set of visual-only textured quads laid just inside the
three walls and on the table top (``contype=conaffinity=0``, no mass, worldbody-static), so collision
geometry, object placement and task physics are identical with and without it. Texture slots have a
fixed size, so a new background is written into the compiled model in place (no recompilation).

Backgrounds use their own random stream (``background_seed``), independent of task / episode seeds.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from PIL import Image

SLOTS = ("wall_far", "wall_left", "wall_right", "table")
WALL_SLOTS = SLOTS[:3]
_PREFIX = "abc_bg_"
# Seed-sequence spawn key that separates the background stream from any task / episode stream.
_STREAM_KEY = 0xB6_0001
# Visual-only geometry of the standard YAM scene (walls at x=0.9 and y=+-0.65, table top z=0.75).
_SCENE = {
    "far_x": 0.892, "left_y": 0.642, "right_y": -0.642,  # 3 mm inside the wall faces
    "near_x": 0.0, "z0": 0.75, "z1": 2.0,
    "table_x": (0.3025, 0.8975), "table_y": 0.65, "table_z": 0.7508,  # 0.8 mm above the table top
}
# Physical extent (u, v) of each slot in metres.
_SPANS = {
    "wall_far": (_SCENE["left_y"] - _SCENE["right_y"], _SCENE["z1"] - _SCENE["z0"]),
    "wall_left": (_SCENE["far_x"] - _SCENE["near_x"], _SCENE["z1"] - _SCENE["z0"]),
    "wall_right": (_SCENE["far_x"] - _SCENE["near_x"], _SCENE["z1"] - _SCENE["z0"]),
    "table": (2 * _SCENE["table_y"], _SCENE["table_x"][1] - _SCENE["table_x"][0]),
}
# MJWarp's renderer ignores material emission: a vertical wall only gets its ambient term
# (0.5 * (0.4 * 0.5 + 0.1 * 0.5) = 0.125). Wall textures uploaded to it are scaled so they match
# the emissive look of the CPU renderer.
MJWARP_WALL_AMBIENT = 0.125


@dataclass(frozen=True)
class BackgroundConfig:
    """Options for :func:`abc_sim.make_env`.

    Attributes:
        asset_dir: package directory holding ``index.json`` (local path, or an ``s3://`` URI synced once
            per machine with the AWS CLI into ``$ABC_SIM_CACHE_DIR/background_variations``).
        seed: seed of the background stream. ``None`` draws OS entropy (non-reproducible). Give every
            parallel worker its own value (e.g. ``base + rank``) for distinct, reproducible streams.
        background_id: pin one combination in ``[0, n_combinations)``; instances still vary per draw.
        resample: ``"episode"`` draws a new background on every randomized ``reset``; ``"env"`` keeps
            the first draw for the env's lifetime.
        slot_scale: texture slot resolution relative to the package's own (0.5 halves each side).
    """

    asset_dir: str
    seed: int | None = None
    background_id: int | None = None
    resample: str = "episode"
    wall_emission: float = 0.85
    table_emission: float = 0.55
    slot_scale: float = 1.0

    def __post_init__(self):
        if self.resample not in ("episode", "env"):
            raise ValueError("resample must be 'episode' or 'env'")
        if not 0 < self.slot_scale <= 1:
            raise ValueError("slot_scale must be in (0, 1]")


def slot_sizes(texel_m: float = 0.0015, scale: float = 1.0) -> dict[str, tuple[int, int]]:
    """(width, height) of every slot: the package's texel density over the slot's physical extent."""
    return {k: (max(16, round(u / texel_m * scale)), max(16, round(v / texel_m * scale))) for k, (u, v) in _SPANS.items()}


def texture_name(slot: str) -> str:
    return _PREFIX + slot


def _cache_dir() -> Path:
    base = os.environ.get("ABC_SIM_CACHE_DIR") or Path.home() / ".cache" / "abc_sim"
    return Path(base) / "background_variations"


def resolve_asset_dir(asset_dir: str | os.PathLike) -> Path:
    """Local package directory for ``asset_dir`` (syncs ``s3://`` URIs once per machine, under a file lock)."""
    s = str(asset_dir)
    if not s.startswith("s3://"):
        return _check_package(Path(s).expanduser())
    dst = _cache_dir() / hashlib.sha1(s.rstrip("/").encode()).hexdigest()[:16]
    if (dst / ".complete").exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    with _file_lock(dst.with_suffix(".lock")):
        if not (dst / ".complete").exists():
            if shutil.which("aws") is None:
                raise RuntimeError(f"syncing {s} needs the AWS CLI on PATH (or pass a local package directory)")
            tmp = Path(tempfile.mkdtemp(prefix=dst.name + ".", dir=dst.parent))
            try:
                subprocess.run(["aws", "s3", "sync", "--only-show-errors", s.rstrip("/") + "/", str(tmp)], check=True)
                _check_package(tmp)
                (tmp / ".complete").write_text(s + "\n")
                tmp.chmod(0o755)
                if dst.exists():
                    shutil.rmtree(dst)
                tmp.rename(dst)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
    return dst


@contextlib.contextmanager
def _file_lock(path: Path):
    import fcntl

    with open(path, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _check_package(root: Path) -> Path:
    if not (root / "index.json").exists():
        raise FileNotFoundError(f"no background package (index.json) in {root}")
    return root


def _resize(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if img.shape[1::-1] == size:
        return img
    return np.asarray(Image.fromarray(img).resize(size, Image.Resampling.BILINEAR, reducing_gap=2.0), dtype=np.uint8)


class BackgroundLibrary:
    """Combination index over a background package, with decoded / resampled image caches.

    ``slot_cache`` bounds how many instances' slot images are kept (each wallpaper instance is about
    5 MB at slot_scale 1, a table instance about 1 MB); batched envs raise it to keep the whole package.
    """

    def __init__(self, asset_dir: str | os.PathLike, *, slot_cache: int = 4):
        self.root = resolve_asset_dir(asset_dir)
        ix = json.loads((self.root / "index.json").read_text())
        self.texel_m = float(ix.get("texel_m", 0.0015))
        self.wall_types = sorted(ix["wallpapers"])
        self.table_types = sorted(ix["tables"])
        self._walls = {t: ix["wallpapers"][t]["instances"] for t in self.wall_types}
        self._tables = {t: ix["tables"][t]["instances"] for t in self.table_types}
        if not self.wall_types or not self.table_types:
            raise ValueError(f"empty background package at {self.root}")
        self.slot_cache = slot_cache
        self._slots: OrderedDict[tuple, dict[str, np.ndarray]] = OrderedDict()

    @property
    def n_combinations(self) -> int:
        return len(self.wall_types) * len(self.table_types)

    def combination(self, background_id: int) -> tuple[str, str]:
        if not 0 <= int(background_id) < self.n_combinations:
            raise ValueError(f"background_id must be in [0, {self.n_combinations}), got {background_id}")
        w, t = divmod(int(background_id), len(self.table_types))
        return self.wall_types[w], self.table_types[t]

    def combination_id(self, wall_type: str, table_type: str) -> int:
        return self.wall_types.index(wall_type) * len(self.table_types) + self.table_types.index(table_type)

    def draw(self, rng: np.random.Generator, background_id: int | None = None) -> dict[str, Any]:
        bid = int(rng.integers(self.n_combinations)) if background_id is None else int(background_id)
        wt, tt = self.combination(bid)
        wi = self._walls[wt][int(rng.integers(len(self._walls[wt])))]
        ti = self._tables[tt][int(rng.integers(len(self._tables[tt])))]
        return {"background_id": bid, "wallpaper": wi["id"], "table": ti["id"],
                "wallpaper_dir": wi["dir"], "table_dir": ti["dir"]}

    # ---- pixels -------------------------------------------------------------------------------
    def slot_images(self, state: dict[str, Any], scale: float = 1.0) -> dict[str, np.ndarray]:
        """(H, W, 3) uint8 per slot at the print's true scale."""
        sizes = slot_sizes(self.texel_m, scale)
        walls = self._cached(("w", state["wallpaper_dir"], scale),
                             lambda: {k: self._fit_wall(state["wallpaper_dir"], k, sizes[k]) for k in WALL_SLOTS})
        table = self._cached(("t", state["table_dir"], scale), lambda: {"table": self._tile_table(state["table_dir"], sizes["table"])})
        return {**walls, **table}

    def _cached(self, key, make):
        value = self._slots.get(key)
        if value is None:
            value = self._slots[key] = make()
            while len(self._slots) > self.slot_cache:
                self._slots.popitem(last=False)
        else:
            self._slots.move_to_end(key)
        return value

    def _read(self, rel: str) -> np.ndarray:
        with Image.open(self.root / rel) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)

    def _fit_wall(self, d: str, k: str, size: tuple[int, int]) -> np.ndarray:
        img = self._read(f"{d}/{k}.jpg")
        (span_u, span_v) = _SPANS[k]
        w, h = max(2, round(span_u / self.texel_m)), max(2, round(span_v / self.texel_m))
        H, W = img.shape[:2]
        # Extend the print beyond what the photo covers by mirroring, at true scale. The bottom row is
        # the table edge; a side wall keeps the corner it shares with the far wall.
        pad_x = max(0, w - W)
        left, right = (pad_x, 0) if k == "wall_left" else (0, pad_x)
        if pad_x or H < h:
            img = np.pad(img, ((max(0, h - H), 0), (left, right), (0, 0)), mode="symmetric")
        H, W = img.shape[:2]
        x0 = {"wall_left": W - w, "wall_right": 0}.get(k, (W - w) // 2)
        return _resize(img[H - h:, x0:x0 + w], size)

    def _tile_table(self, d: str, size: tuple[int, int]) -> np.ndarray:
        tile = self._read(f"{d}/tile.jpg")
        span_u, span_v = _SPANS["table"]
        w, h = max(2, round(span_u / self.texel_m)), max(2, round(span_v / self.texel_m))
        reps = (-(-h // tile.shape[0]), -(-w // tile.shape[1]))
        return _resize(np.tile(tile, (*reps, 1))[:h, :w], size)


def background_rng(seed: int | None) -> np.random.Generator:
    """The background stream: independent of every task / episode RNG for the same integer seed."""
    entropy = seed if seed is not None else np.random.SeedSequence().entropy
    return np.random.default_rng(np.random.SeedSequence(entropy, spawn_key=(_STREAM_KEY,)))


# ---- scene XML ---------------------------------------------------------------------------------
def apply_background_xml(
    xml: str,
    *,
    sizes: dict[str, tuple[int, int]] | None = None,
    wall_emission: float = 0.85,
    table_emission: float = 0.55,
) -> str:
    """Add fixed-size background texture slots and visual-only quads to a standard YAM scene."""
    root = ET.fromstring(xml)
    if root.find(f".//texture[@name='{texture_name('wall_far')}']") is not None:
        return xml  # already applied
    asset, world = root.find("asset"), root.find("worldbody")
    if asset is None or world is None:
        raise ValueError("Scene XML is missing <asset> or <worldbody>")
    sizes = sizes or slot_sizes()
    S = _SCENE
    z0, z1, xf, xn, yl, yr = S["z0"], S["z1"], S["far_x"], S["near_x"], S["left_y"], S["right_y"]
    tx0, tx1, ty, tz = S["table_x"][0], S["table_x"][1], S["table_y"], S["table_z"]
    # corners ordered (u0v0, u1v0, u1v1, u0v1); u = texture right, v = texture down, seen from inside
    quads = {
        "wall_far": [[xf, yl, z1], [xf, yr, z1], [xf, yr, z0], [xf, yl, z0]],
        "wall_left": [[xn, yl, z1], [xf, yl, z1], [xf, yl, z0], [xn, yl, z0]],
        "wall_right": [[xf, yr, z1], [xn, yr, z1], [xn, yr, z0], [xf, yr, z0]],
        "table": [[tx1, ty, tz], [tx1, -ty, tz], [tx0, -ty, tz], [tx0, ty, tz]],
    }
    for k in SLOTS:
        n, (w, h) = texture_name(k), sizes[k]
        ET.SubElement(asset, "texture", name=n, type="2d", builtin="flat", width=str(w), height=str(h),
                      rgb1="0.95 0.93 0.96", rgb2="0.95 0.93 0.96")
        em = table_emission if k == "table" else wall_emission
        ET.SubElement(asset, "material", name=n, texture=n, emission=f"{em:g}", specular="0", shininess="0",
                      reflectance="0", rgba="1 1 1 1")
        ET.SubElement(asset, "mesh", name=n, vertex=" ".join(f"{c:.4f}" for p in quads[k] for c in p),
                      texcoord="0 0 1 0 1 1 0 1", face="0 2 1 0 3 2", inertia="shell")
        ET.SubElement(world, "geom", name=n, type="mesh", mesh=n, material=n, group="2",
                      contype="0", conaffinity="0", density="0", mass="0")
    return ET.tostring(root, encoding="unicode")


def has_background_slots(model: mujoco.MjModel) -> bool:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, texture_name("wall_far")) >= 0


def background_texture_ids(model: mujoco.MjModel) -> dict[str, int]:
    """slot -> texture id."""
    return {k: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, texture_name(k)) for k in SLOTS}


def background_material_ids(model: mujoco.MjModel) -> dict[str, int]:
    """slot -> material id."""
    return {k: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, texture_name(k)) for k in SLOTS}


def write_background(model: mujoco.MjModel, images: dict[str, np.ndarray]) -> list[int]:
    """Write slot images into ``model.tex_data``; returns the texture ids written."""
    ids = []
    for k, tid in background_texture_ids(model).items():
        adr, n = int(model.tex_adr[tid]), int(model.tex_width[tid] * model.tex_height[tid] * model.tex_nchannel[tid])
        model.tex_data[adr:adr + n] = images[k].reshape(-1)
        ids.append(tid)
    return ids


class BackgroundController:
    """Per-env state: library, background stream, current draw."""

    def __init__(self, config: BackgroundConfig):
        self.config = config
        self.library = BackgroundLibrary(config.asset_dir)
        if config.background_id is not None:
            self.library.combination(config.background_id)  # validate early
        self.sizes = slot_sizes(self.library.texel_m, config.slot_scale)
        self.rng = background_rng(config.seed)
        self.state, self._images = self._draw(None)
        self._unused = True  # the construction-time draw serves the first reset

    def xml(self, xml: str) -> str:
        c = self.config
        return apply_background_xml(xml, sizes=self.sizes, wall_emission=c.wall_emission, table_emission=c.table_emission)

    def mjwarp_gain(self, slot: str) -> float:
        """Texture scale for MJWarp uploads (its renderer ignores emission; see MJWARP_WALL_AMBIENT)."""
        return self.config.wall_emission / MJWARP_WALL_AMBIENT if slot in WALL_SLOTS else 1.0

    def _draw(self, background_id: int | None) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        bid = self.config.background_id if background_id is None else background_id
        state = self.library.draw(self.rng, bid)
        return state, self.library.slot_images(state, self.config.slot_scale)

    def draw(self, background_id: int | None = None) -> dict[str, Any]:
        self.state, self._images = self._draw(background_id)
        self._unused = False
        return self.state

    def draw_sets(self, n: int, background_id: int | None = None) -> list[tuple[dict[str, Any], dict[str, np.ndarray]]]:
        """``n`` independent draws from the background stream (batched worlds), with their slot images."""
        return [self._draw(background_id) for _ in range(n)]

    def next_episode(self, background_id: int | None = None) -> bool:
        """Draw for a new episode; returns False when the pending construction-time draw is used."""
        if self._unused and background_id is None:
            self._unused = False
            return False
        self.draw(background_id)
        return True

    def apply(self, model: mujoco.MjModel) -> list[int]:
        """Write the current draw into ``model`` (every model compiled by a background env has the slots)."""
        if not has_background_slots(model):
            raise RuntimeError("model was compiled without background slots (compile it through MuJoCoYAMEnv)")
        return write_background(model, self._images)

    @staticmethod
    def info_of(state: dict[str, Any]) -> dict[str, Any]:
        return {k: state[k] for k in ("background_id", "wallpaper", "table")}

    def info(self) -> dict[str, Any]:
        return self.info_of(self.state)


__all__ = [
    "MJWARP_WALL_AMBIENT",
    "SLOTS",
    "BackgroundConfig",
    "BackgroundController",
    "BackgroundLibrary",
    "apply_background_xml",
    "background_material_ids",
    "background_rng",
    "background_texture_ids",
    "has_background_slots",
    "resolve_asset_dir",
    "slot_sizes",
    "write_background",
]
