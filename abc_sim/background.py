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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

# Texture slots: (width, height) at slot_scale 1. Fixed sizes make in-place swaps possible.
_SLOTS = {
    "wall_far": (1024, 1024),
    "wall_left": (768, 1024),
    "wall_right": (768, 1024),
    "table": (1024, 512),
}
_PREFIX = "abc_bg_"
# Seed-sequence spawn key that separates the background stream from any task / episode stream.
_STREAM_KEY = 0xB6_0001
# Visual-only geometry of the standard YAM scene (walls at x=0.9 and y=+-0.65, table top z=0.75).
_SCENE = {
    "far_x": 0.892, "left_y": 0.642, "right_y": -0.642,  # 3 mm inside the wall faces
    "near_x": 0.0, "z0": 0.75, "z1": 2.0,
    "table_x": (0.3025, 0.8975), "table_y": 0.65, "table_z": 0.7508,  # 0.8 mm above the table top
}
_DEFAULT_CACHE = Path(os.environ.get("ABC_SIM_CACHE_DIR", Path.home() / ".cache" / "abc_sim")) / "background_variations"


@dataclass(frozen=True)
class BackgroundConfig:
    """Options for :func:`abc_sim.make_env`.

    Attributes:
        asset_dir: package directory (local path, or an ``s3://`` URI downloaded once per machine into
            ``$ABC_SIM_CACHE_DIR/background_variations``). Required when enabled.
        seed: seed of the background stream. ``None`` draws OS entropy (non-reproducible). Give every
            parallel worker its own value (e.g. ``base + rank``) for distinct, reproducible streams.
        background_id: pin one combination in ``[0, n_combinations)``; instances still vary per draw.
        resample: ``"episode"`` draws a new background on every randomized ``reset``; ``"env"`` keeps
            the first draw for the env's lifetime.
        n_sets: texture sets compiled into the model. Set 0 is what the CPU env shows; batched MJWarp
            envs point each world at its own set (``abc_sim.make_batched_env`` sets this).
        slot_scale: texture slot resolution relative to the defaults (0.5 halves each side).
    """

    asset_dir: str
    seed: int | None = None
    background_id: int | None = None
    resample: str = "episode"
    wall_emission: float = 0.85
    table_emission: float = 0.55
    n_sets: int = 1
    slot_scale: float = 1.0

    def __post_init__(self):
        if self.resample not in ("episode", "env"):
            raise ValueError("resample must be 'episode' or 'env'")
        if self.n_sets < 1:
            raise ValueError("n_sets must be >= 1")
        if not 0 < self.slot_scale <= 1:
            raise ValueError("slot_scale must be in (0, 1]")


def slot_sizes(scale: float = 1.0) -> dict[str, tuple[int, int]]:
    return {k: (max(16, round(w * scale)), max(16, round(h * scale))) for k, (w, h) in _SLOTS.items()}


def texture_name(slot: str, set_index: int = 0) -> str:
    """Name of a background texture: set 0 keeps the plain name its materials reference."""
    return f"{_PREFIX}{slot}" if set_index == 0 else f"{_PREFIX}{set_index}_{slot}"


def material_name(slot: str) -> str:
    return f"{_PREFIX}{slot}"


def resolve_asset_dir(asset_dir: str | os.PathLike) -> Path:
    """Local package directory for ``asset_dir`` (downloads ``s3://`` URIs once, under a file lock)."""
    s = str(asset_dir)
    if not s.startswith("s3://"):
        return _find_index_dir(Path(s).expanduser())
    dst = _DEFAULT_CACHE / hashlib.sha1(s.rstrip("/").encode()).hexdigest()[:16]
    done = dst / ".complete"
    if done.exists():
        return _find_index_dir(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with _file_lock(dst.with_suffix(".lock")):
        if not done.exists():
            tmp = Path(tempfile.mkdtemp(prefix=dst.name + ".", dir=dst.parent))
            try:
                _s3_download(s.rstrip("/") + "/", tmp)
                _find_index_dir(tmp)
                if dst.exists():
                    shutil.rmtree(dst)
                tmp.rename(dst)
                done.write_text(s + "\n")
            finally:
                if tmp.exists():
                    shutil.rmtree(tmp, ignore_errors=True)
    return _find_index_dir(dst)


@contextlib.contextmanager
def _file_lock(path: Path):
    import fcntl

    with open(path, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _s3_download(uri: str, dst: Path) -> None:
    if shutil.which("aws"):
        subprocess.run(["aws", "s3", "sync", "--only-show-errors", "--exclude", "code/*", uri, str(dst)], check=True)
        return
    import boto3  # optional dependency, only for s3:// asset dirs without the AWS CLI

    bucket, _, prefix = uri[len("s3://"):].partition("/")
    s3 = boto3.client("s3")
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            rel = obj["Key"][len(prefix):]
            if not rel or rel.endswith("/") or rel.startswith("code/"):
                continue
            out = dst / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, obj["Key"], str(out))


def _find_index_dir(root: Path) -> Path:
    """The package root holds index.json; a parent with version subfolders resolves to the newest."""
    if (root / "index.json").exists():
        return root
    subs = sorted(p for p in root.glob("*/index.json"))
    if subs:
        return subs[-1].parent
    raise FileNotFoundError(f"no background package (index.json) under {root}")


class BackgroundLibrary:
    """Combination index over a background package."""

    def __init__(self, asset_dir: str | os.PathLike):
        self.root = resolve_asset_dir(asset_dir)
        ix = json.loads((self.root / "index.json").read_text())
        self.texel_m = float(ix.get("texel_m", 0.0015))
        self.wall_types = sorted(ix["wallpapers"])
        self.table_types = sorted(ix["tables"])
        self._walls = {t: ix["wallpapers"][t]["instances"] for t in self.wall_types}
        self._tables = {t: ix["tables"][t]["instances"] for t in self.table_types}
        if not self.wall_types or not self.table_types:
            raise ValueError(f"empty background package at {self.root}")
        self._cache: dict[tuple[str, str], np.ndarray] = {}
        self._slots: dict[tuple, dict[str, np.ndarray]] = {}

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
        """(H, W, 3) uint8 per slot, at the scene's true print scale (recent instances cached)."""
        S, sizes = _SCENE, slot_sizes(scale)
        walls = self._slots.get(("w", state["wallpaper_dir"], scale))
        if walls is None:
            walls = {}
            for k in ("wall_far", "wall_left", "wall_right"):
                span_u = (S["left_y"] - S["right_y"]) if k == "wall_far" else (S["far_x"] - S["near_x"])
                walls[k] = self._fit_wall(state["wallpaper_dir"], k, span_u, S["z1"] - S["z0"], sizes[k])
            self._remember(("w", state["wallpaper_dir"], scale), walls)
        table = self._slots.get(("t", state["table_dir"], scale))
        if table is None:
            span = (2 * S["table_y"], S["table_x"][1] - S["table_x"][0])
            table = {"table": self._tile_table(state["table_dir"], *span, sizes["table"])}
            self._remember(("t", state["table_dir"], scale), table)
        return {**walls, **table}

    def _remember(self, key, value) -> None:
        if len(self._slots) >= 12:  # <= ~50 MB per process at slot_scale 1
            self._slots.pop(next(iter(self._slots)))
        self._slots[key] = value

    def _read(self, rel: str) -> np.ndarray:
        key = (rel, "")
        img = self._cache.get(key)
        if img is None:
            img = _read_rgb(self.root / rel)
            if len(self._cache) > 24:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = img
        return img

    def _fit_wall(self, d: str, k: str, span_u: float, span_v: float, slot: tuple[int, int]) -> np.ndarray:
        img = self._read(f"{d}/{k}.jpg")
        w, h = max(2, round(span_u / self.texel_m)), max(2, round(span_v / self.texel_m))
        H, W = img.shape[:2]
        if H < h or W < w:  # extend the print beyond what the photo covers (mirror), true scale kept
            img = np.pad(img, ((max(0, h - H), 0), (0, max(0, w - W)), (0, 0)), mode="symmetric")
        H, W = img.shape[:2]
        x0 = (W - w) // 2
        if k == "wall_left":  # the far end of a side wall meets the far wall: keep that corner
            x0 = W - w
        elif k == "wall_right":
            x0 = 0
        crop = img[H - h:, x0:x0 + w]  # bottom row = table height
        return _resize(crop, slot)

    def _tile_table(self, d: str, span_u: float, span_v: float, slot: tuple[int, int]) -> np.ndarray:
        tile = self._read(f"{d}/tile.jpg")
        w, h = max(2, round(span_u / self.texel_m)), max(2, round(span_v / self.texel_m))
        reps = (int(np.ceil(h / tile.shape[0])), int(np.ceil(w / tile.shape[1])))
        return _resize(np.tile(tile, (reps[0], reps[1], 1))[:h, :w], slot)


def _read_rgb(path: Path) -> np.ndarray:
    try:
        from PIL import Image

        with Image.open(path) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)
    except ImportError:
        import imageio.v3 as iio

        return np.asarray(iio.imread(path))[..., :3].astype(np.uint8)


def _resize(img: np.ndarray, slot: tuple[int, int]) -> np.ndarray:
    W, H = slot
    try:
        from PIL import Image

        return np.asarray(Image.fromarray(img).resize((W, H), Image.Resampling.BILINEAR, reducing_gap=2.0), dtype=np.uint8)
    except ImportError:  # nearest-neighbour fallback
        ys = (np.arange(H) * img.shape[0] / H).astype(int)
        xs = (np.arange(W) * img.shape[1] / W).astype(int)
        return img[ys][:, xs]


def background_rng(seed: int | None) -> np.random.Generator:
    """The background stream: independent of every task / episode RNG for the same integer seed."""
    entropy = seed if seed is not None else np.random.SeedSequence().entropy
    return np.random.default_rng(np.random.SeedSequence(entropy, spawn_key=(_STREAM_KEY,)))


# ---- scene XML ---------------------------------------------------------------------------------
def apply_background_xml(
    xml: str,
    *,
    wall_emission: float = 0.85,
    table_emission: float = 0.55,
    n_sets: int = 1,
    slot_scale: float = 1.0,
) -> str:
    """Add fixed-size background texture slots and visual-only quads to a standard YAM scene.

    Set 0's textures carry the materials and quads; sets 1.. are extra texture slots that batched
    renderers select per world through the materials' texture ids.
    """
    root = ET.fromstring(xml)
    if root.find(f".//texture[@name='{_PREFIX}wall_far']") is not None:
        return xml  # already applied
    asset = root.find("asset")
    world = root.find("worldbody")
    if asset is None or world is None:
        raise ValueError("Scene XML is missing <asset> or <worldbody>")
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
    for k, (w, h) in slot_sizes(slot_scale).items():
        n = material_name(k)
        for j in range(n_sets):
            ET.SubElement(asset, "texture", name=texture_name(k, j), type="2d", builtin="flat", width=str(w),
                          height=str(h), rgb1="0.95 0.93 0.96", rgb2="0.95 0.93 0.96")
        em = table_emission if k == "table" else wall_emission
        ET.SubElement(asset, "material", name=n, texture=n, emission=f"{em:g}", specular="0", shininess="0",
                      reflectance="0", rgba="1 1 1 1")
        ET.SubElement(asset, "mesh", name=n, vertex=" ".join(f"{c:.4f}" for p in quads[k] for c in p),
                      texcoord="0 0 1 0 1 1 0 1", face="0 2 1 0 3 2", inertia="shell")
        ET.SubElement(world, "geom", name=n, type="mesh", mesh=n, material=n, group="2",
                      contype="0", conaffinity="0", density="0", mass="0")
    if hasattr(ET, "indent"):
        ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


def has_background_slots(model: mujoco.MjModel) -> bool:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, texture_name("wall_far")) >= 0


def background_texture_ids(model: mujoco.MjModel, set_index: int = 0) -> dict[str, int]:
    """slot -> texture id of one set (missing slots omitted)."""
    out = {}
    for k in _SLOTS:
        tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, texture_name(k, set_index))
        if tid >= 0:
            out[k] = tid
    return out


def background_material_ids(model: mujoco.MjModel) -> dict[str, int]:
    out = {}
    for k in _SLOTS:
        mid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, material_name(k))
        if mid >= 0:
            out[k] = mid
    return out


def write_background(model: mujoco.MjModel, images: dict[str, np.ndarray], set_index: int = 0) -> list[int]:
    """Write slot images into ``model.tex_data`` (one set); returns the texture ids written."""
    ids = []
    for k, img in images.items():
        tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, texture_name(k, set_index))
        if tid < 0:
            continue
        w, h, c = int(model.tex_width[tid]), int(model.tex_height[tid]), int(model.tex_nchannel[tid])
        if img.shape[:2] != (h, w):
            img = _resize(img, (w, h))
        if c == 4:
            img = np.concatenate([img, np.full(img.shape[:2] + (1,), 255, np.uint8)], -1)
        adr = int(model.tex_adr[tid])
        model.tex_data[adr:adr + w * h * c] = img.reshape(-1)
        ids.append(tid)
    return ids


class BackgroundController:
    """Per-env state: library, background stream, current draw."""

    def __init__(self, config: BackgroundConfig):
        self.config = config
        self.library = BackgroundLibrary(config.asset_dir)
        if config.background_id is not None:
            self.library.combination(config.background_id)  # validate early
        self.rng = background_rng(config.seed)
        self.state: dict[str, Any] | None = None
        self._images: dict[str, np.ndarray] | None = None
        self._unused = False  # a draw made before the first reset serves that reset

    def xml(self, xml: str) -> str:
        c = self.config
        return apply_background_xml(xml, wall_emission=c.wall_emission, table_emission=c.table_emission,
                                    n_sets=c.n_sets, slot_scale=c.slot_scale)

    def draw(self, background_id: int | None = None) -> dict[str, Any]:
        bid = self.config.background_id if background_id is None else background_id
        self.state = self.library.draw(self.rng, bid)
        self._images = self.library.slot_images(self.state, self.config.slot_scale)
        self._unused = False
        return self.state

    def draw_sets(self, n: int) -> list[tuple[dict[str, Any], dict[str, np.ndarray]]]:
        """``n`` independent draws from the background stream (batched worlds), with their slot images."""
        out = []
        for _ in range(n):
            state = self.library.draw(self.rng, self.config.background_id)
            out.append((state, self.library.slot_images(state, self.config.slot_scale)))
        return out

    def library_draw(self, background_id: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        """One draw of a pinned combination (instances from the stream), with its slot images."""
        state = self.library.draw(self.rng, background_id)
        return state, self.library.slot_images(state, self.config.slot_scale)

    @staticmethod
    def info_of(state: dict[str, Any]) -> dict[str, Any]:
        return {k: state[k] for k in ("background_id", "wallpaper", "table")}

    def next_episode(self, background_id: int | None = None) -> bool:
        """Draw for a new episode; returns False when the pending construction-time draw is used."""
        if self._unused and background_id is None:
            self._unused = False
            return False
        self.draw(background_id)
        return True

    def apply(self, model: mujoco.MjModel) -> list[int]:
        if self.state is None:
            self.draw()
            self._unused = True
        return write_background(model, self._images) if has_background_slots(model) else []

    def info(self) -> dict[str, Any] | None:
        if self.state is None:
            return None
        return {k: self.state[k] for k in ("background_id", "wallpaper", "table")}


__all__ = [
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
    "texture_name",
    "write_background",
]
