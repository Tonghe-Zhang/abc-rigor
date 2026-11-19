"""Household clutter and open trash bins for tabletop task distractors."""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache

import mujoco

from ..core import _format_float_list
from .paths import _MODELS_DIR
from .xml_common import _dishrack_compiled_model_bounds

# Each category is sampled with equal probability, then a variant within it.
# Bottle-shaped containers are omitted so "the bottles" identifies the targets.
DISTRACTOR_CATEGORIES = (
    "mug",
    "bowl",
    "coffee_cup",
    "can",
    "canned_food",
    "boxed_food",
    "boxed_drink",
    "cereal",
    "jar",
    "yogurt",
    "bar_soap",
    "sponge",
    "apple",
    "lemon",
    "orange",
    "tomato",
    "potato",
    "donut",
    "cupcake",
    "trash_bin",
)
TRASH_BIN_VARIANTS = ("round_orange", "blue_mesh")
_CLUTTER_ROOT = _MODELS_DIR / "assets" / "task_grab_clutter" / "objects"


@lru_cache(maxsize=None)
def distractor_variants(category: str) -> tuple[str, ...]:
    """Return the installed variants of an approved distractor category."""
    if category not in DISTRACTOR_CATEGORIES:
        raise ValueError(f"Unknown distractor category {category!r}.")
    if category == "trash_bin":
        return TRASH_BIN_VARIANTS
    variants = tuple(sorted(p.parent.name for p in (_CLUTTER_ROOT / category).glob("*/model.xml")))
    if not variants:
        raise FileNotFoundError(f"No distractor assets installed for {category!r}.")
    return variants


@dataclass(frozen=True)
class DistractorSpec:
    """Portable asset identity; no installation-specific paths are recorded."""

    category: str
    variant: str

    def __post_init__(self) -> None:
        if self.variant not in distractor_variants(self.category):
            raise ValueError(f"Unknown distractor variant {self.category}/{self.variant}.")


def _household_template(spec: DistractorSpec) -> tuple[ET.Element, ET.Element]:
    directory = _CLUTTER_ROOT / spec.category / spec.variant
    source = ET.parse(directory / "model.xml").getroot()
    body = ET.Element("body", name="object")
    # The clutter pack supplies lightweight box proxies. Its source XML also
    # retains unused collision references whose meshes are outside this pack.
    for geom in source.findall(".//worldbody//geom"):
        if geom.get("class") == "region":
            continue
        visual = geom.get("contype") == "0" and geom.get("conaffinity") == "0"
        if not visual and geom.get("class") != "collision":
            continue
        cloned = copy.deepcopy(geom)
        cloned.attrib.pop("class", None)
        cloned.set("name", f"geom_{len(body)}")
        if visual:
            cloned.set("group", "2")
            cloned.set("density", "0")
            cloned.attrib.pop("mass", None)
        else:
            cloned.set("rgba", "0 0 0 0")
            # Enclose the visible object, including handles, at table contacts.
            bbox = source.find('.//geom[@name="reg_bbox"]')
            cloned.set("pos", bbox.attrib["pos"])
            cloned.set("size", bbox.attrib["size"])
        body.append(cloned)
    used_meshes = {g.get("mesh") for g in body}
    asset = copy.deepcopy(source.find("asset"))
    for elem in list(asset):
        if elem.tag == "mesh" and elem.get("name") not in used_meshes:
            asset.remove(elem)
        elif "file" in elem.attrib:
            elem.set("file", str((directory / elem.attrib["file"]).resolve()))
    return asset, body


def _trash_bin_template(variant: str) -> tuple[ET.Element, ET.Element]:
    scene = "yam_bottles_scene.xml" if variant == "round_orange" else "yam_put_bottles_scene.xml"
    source = ET.parse(_MODELS_DIR / scene).getroot()
    body = copy.deepcopy(source.find('.//body[@name="bin_container"]'))
    body.set("pos", "0 0 0")
    body.remove(body.find("freejoint"))
    # Resolve the source scene's geom defaults before importing its bin.
    for geom in body.findall("geom"):
        cls = geom.attrib.pop("class", None)
        if cls is not None:
            defaults = source.find(f".//default[@class='{cls}']/geom")
            for key, value in defaults.attrib.items():
                geom.attrib.setdefault(key, value)
            if variant == "round_orange":
                defaults = source.find(".//default[@class='task_bin']/geom")
                for key, value in defaults.attrib.items():
                    geom.attrib.setdefault(key, value)
        if geom.get("contype") != "0":
            geom.set("group", "3" if variant == "blue_mesh" else "0")
            if variant == "blue_mesh":
                geom.set("rgba", "0 0 0 0")
    asset = ET.Element("asset")
    meshes = {g.attrib["mesh"] for g in body.findall("geom") if "mesh" in g.attrib}
    materials = {g.attrib["material"] for g in body.findall("geom") if "material" in g.attrib}
    for elem in source.find("asset"):
        if (elem.tag == "mesh" and elem.get("name") in meshes) or (
            elem.tag == "material" and elem.get("name") in materials
        ):
            cloned = copy.deepcopy(elem)
            if "file" in cloned.attrib:
                cloned.set("file", str((_MODELS_DIR / "assets" / cloned.attrib["file"]).resolve()))
            asset.append(cloned)
    return asset, body


@lru_cache(maxsize=None)
def _template(spec: DistractorSpec) -> tuple[ET.Element, ET.Element, tuple[float, float]]:
    asset, body = _trash_bin_template(spec.variant) if spec.category == "trash_bin" else _household_template(spec)
    root = ET.Element("mujoco")
    root.append(copy.deepcopy(asset))
    ET.SubElement(root, "worldbody").append(copy.deepcopy(body))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    lower, upper = _dishrack_compiled_model_bounds(model, data)
    center = (lower + upper) / 2
    body.set("pos", _format_float_list([-center[0], -center[1], -lower[2]]))
    half_xy = (upper[:2] - lower[:2]) / 2
    return asset, body, tuple(float(x) for x in half_xy)


def distractor_half_extents(spec: DistractorSpec) -> tuple[float, float]:
    """Conservative XY bounds of both the visual and collision geometry."""
    return _template(spec)[2]


def build_distractor(spec: DistractorSpec, index: int) -> tuple[list[ET.Element], ET.Element]:
    """Build a free object whose origin is its bottom-centre in the upright pose."""
    asset, body, _ = _template(spec)
    asset, body = copy.deepcopy(asset), copy.deepcopy(body)
    prefix = f"distractor_{index}"
    names = {e.attrib["name"] for parent in (asset, body) for e in parent.iter() if "name" in e.attrib}
    for parent in (asset, body):
        for elem in parent.iter():
            for attr in ("name", "mesh", "material", "texture"):
                if elem.get(attr) in names:
                    elem.set(attr, f"{prefix}_{elem.attrib[attr]}")
    wrapper = ET.Element("body", name=prefix, pos="0.6 0 0.752")
    ET.SubElement(wrapper, "freejoint", name=f"{prefix}_joint")
    wrapper.append(body)
    return list(asset), wrapper
