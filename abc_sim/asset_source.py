"""Download-on-demand asset packs for externally registered tasks.

Large task assets (meshes, textures, flex models) are NOT committed to this repository.  A task declares an
:class:`AssetPack` -- a name plus a relative archive / directory name -- and the *location* the pack is fetched from is
configured at run time, never hard-coded:

    ABC_SIM_ASSET_SOURCE=<base>             base location for every pack
    ABC_SIM_ASSET_SOURCE_<PACK>=<location>  per-pack override (PACK upper-cased, non-alphanumerics -> "_")

``<base>`` / ``<location>`` may be

* a local directory (or ``file://`` URL) holding ``<archive>`` (a ``.tar``/``.tar.gz``/``.tgz``/``.zip`` file, or an
  already-extracted directory),
* an ``http(s)://`` URL prefix (resumable download, optional sha256 / size check),
* an ``s3://`` prefix (fetched with the AWS CLI; credentials are the caller's).

A per-pack override names the pack itself (the archive / directory); the base names the folder that contains it.
Packs are resolved once per machine into ``$ABC_SIM_CACHE_DIR/assets/<pack>-<hash>/`` under a file lock with a
``.complete`` marker; local extracted directories are used in place (no copy).
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

SOURCE_ENV = "ABC_SIM_ASSET_SOURCE"
_ARCHIVE_SUFFIXES = (".tar.gz", ".tgz", ".tar", ".zip")


def cache_root() -> Path:
    base = os.environ.get("ABC_SIM_CACHE_DIR") or Path.home() / ".cache" / "abc_sim"
    return Path(base) / "assets"


def _env_key(name: str) -> str:
    return f"{SOURCE_ENV}_" + re.sub(r"[^A-Za-z0-9]", "_", name).upper()


@dataclass(frozen=True)
class AssetPack:
    """A named bundle of task assets fetched on first use.

    ``archive`` is the file or directory name relative to the configured source (e.g. ``"bread_tongs_v2.tar.gz"`` or
    ``"bread_tongs_v2"``).  ``subdir`` selects a folder inside the extracted pack as the returned root.
    """

    name: str
    archive: str
    sha256: str | None = None
    size: int | None = None
    subdir: str = ""

    def location(self, source: str | None = None) -> str:
        """The configured location of this pack (raises when nothing is configured)."""
        override = os.environ.get(_env_key(self.name))
        if override:
            return override
        base = source or os.environ.get(SOURCE_ENV)
        if not base:
            raise RuntimeError(
                f"asset pack {self.name!r} has no source: set {SOURCE_ENV}=<dir | http(s):// | s3:// prefix> "
                f"(or {_env_key(self.name)}=<location of {self.archive}>)"
            )
        return _join(base, self.archive)

    def path(self, source: str | None = None) -> Path:
        """Local root of the pack, fetching / extracting it once if needed."""
        root = resolve(self.location(source), name=self.name, sha256=self.sha256, size=self.size)
        out = root / self.subdir if self.subdir else root
        if not out.exists():
            raise FileNotFoundError(f"asset pack {self.name!r}: no {self.subdir!r} in {root}")
        return out


def _join(base: str, rel: str) -> str:
    if "://" in base:
        return base.rstrip("/") + "/" + rel.lstrip("/")
    return str(Path(base).expanduser() / rel)


def _is_archive(loc: str) -> bool:
    return loc.lower().rstrip("/").endswith(_ARCHIVE_SUFFIXES)


def resolve(location: str, *, name: str = "pack", sha256: str | None = None, size: int | None = None) -> Path:
    """Local directory for ``location`` (see the module docstring)."""
    if location.startswith("file://"):
        location = urllib.parse.unquote(urllib.parse.urlparse(location).path)
    if "://" not in location:
        p = Path(location).expanduser()
        if p.is_dir():
            return p
    dst = cache_root() / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', name)}-{hashlib.sha1(location.encode()).hexdigest()[:12]}"
    if (dst / ".complete").exists():   # already fetched: works offline / after the source moved
        return dst
    if "://" not in location and not Path(location).expanduser().exists():
        raise FileNotFoundError(f"asset pack {name!r}: {location} does not exist")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with _file_lock(dst.with_suffix(".lock")):
        if (dst / ".complete").exists():
            return dst
        tmp = Path(tempfile.mkdtemp(prefix=dst.name + ".", dir=dst.parent))
        try:
            if _is_archive(location):
                arc = _fetch_file(location, dst.parent / (dst.name + ".part" + _suffix(location)), sha256, size)
                _extract(arc, tmp)
                if "://" in location:
                    arc.unlink(missing_ok=True)
            elif location.startswith("s3://"):
                _aws(["s3", "sync", "--only-show-errors", location.rstrip("/") + "/", str(tmp)])
            else:
                raise ValueError(f"asset pack {name!r}: {location} is neither an archive nor a directory prefix")
            (tmp / ".complete").write_text(location + "\n")
            if dst.exists():
                shutil.rmtree(dst)
            tmp.rename(dst)
        finally:
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)
    return dst


def _suffix(loc: str) -> str:
    low = loc.lower()
    return next(s for s in _ARCHIVE_SUFFIXES if low.endswith(s))


def _fetch_file(location: str, part: Path, sha256: str | None, size: int | None) -> Path:
    if "://" not in location:
        src = Path(location).expanduser()
        _verify(src, sha256, size)
        return src
    if location.startswith("s3://"):
        _aws(["s3", "cp", "--only-show-errors", location, str(part)])
    elif location.startswith(("http://", "https://")):
        _http_download(location, part, size)
    else:
        raise ValueError(f"unsupported asset location scheme: {location.split('://')[0]}://")
    _verify(part, sha256, size)
    return part


def _http_download(url: str, part: Path, size: int | None, chunk: int = 1 << 20) -> None:
    have = part.stat().st_size if part.exists() else 0
    if size is not None and have == size:
        return
    req = urllib.request.Request(url, headers={"User-Agent": "abc_sim-assets"})
    if have:
        req.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(req, timeout=60) as resp:
        mode = "ab" if have and resp.status == 206 else "wb"
        with open(part, mode) as fh:
            shutil.copyfileobj(resp, fh, chunk)


def _verify(path: Path, sha256: str | None, size: int | None) -> None:
    if size is not None and path.stat().st_size != size:
        raise RuntimeError(f"{path.name}: size {path.stat().st_size} != expected {size}")
    if sha256:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for blk in iter(lambda: fh.read(1 << 22), b""):
                h.update(blk)
        if h.hexdigest() != sha256.lower():
            raise RuntimeError(f"{path.name}: sha256 mismatch")


def _extract(archive: Path, dest: Path) -> None:
    dest = dest.resolve()
    if archive.name.lower().endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            for m in zf.namelist():
                if not (dest / m).resolve().is_relative_to(dest):
                    raise RuntimeError(f"unsafe path in {archive.name}: {m}")
            zf.extractall(dest)
        return
    with tarfile.open(archive) as tf:
        tf.extractall(dest, filter="data")


def _aws(args: list[str]) -> None:
    if shutil.which("aws") is None:
        raise RuntimeError("s3:// asset sources need the AWS CLI on PATH (or use a local / http(s) source)")
    subprocess.run(["aws", *args], check=True)


@contextlib.contextmanager
def _file_lock(path: Path):
    import fcntl

    with open(path, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
