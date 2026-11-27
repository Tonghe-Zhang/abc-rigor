"""Standard abc_sim station cameras + one-model YAM hardware variants for plugin task scenes.

Plugin scenes (tongs packages, cap/jar portables) carry their own robot build, so before this module they rendered
their own cameras and silently dropped one-model's hardware randomization (``environment_factory`` / ``variant_cfg``
arrive through ``abc_sim.make_env(**kwargs)`` and were swallowed).  Here:

1. ``standardize(spec)`` grafts the standard abc_sim station camera bodies -- ``top_bar_center`` (with
   ``top_camera_d405`` / camera ``top``) under ``gate_collision`` and ``{side}_camera_d405`` (camera ``left`` /
   ``right``) under ``{side}_link_6`` -- copied at runtime from a standard abc_sim task scene (``camera_reference_xml``,
   task ``drawer``; the ``empty`` station hides the housings).  Nothing about the hardware is re-typed here.
2. ``compose(spec, variant)`` then applies ``one_model.sim.abc.hardware.YamVariant(variant).compose(spec)`` -- the
   same call one-model's ``LinearGripperEnv`` makes -- so all 5 one-model variants (abc_crank, yam_cell_d405,
   yam_cell_zed, abc_box_d405, molmoact2) apply to plugin scenes exactly as to built-in abc_sim tasks.

After either step the policy cameras are always the model cameras ``top`` / ``left`` / ``right`` (``POLICY_CAMERAS``).
"""

from __future__ import annotations

import functools
import logging
import os
from typing import Any

import mujoco
import numpy as np

logger = logging.getLogger(__name__)

POLICY_CAMERAS = {"base_0_rgb": "top", "left_wrist_0_rgb": "left", "right_wrist_0_rgb": "right"}
SIDES = ("left", "right")
_ASSET_KINDS = ("meshes", "materials", "textures", "hfields", "skins")
_NON_TREE = ("actuators", "sensors", "tendons", "equalities", "pairs", "excludes", "keys", "texts", "numerics",
             "tuples", "plugins")


# ---------------------------------------------------------------------------------------------------- reference
def reference_xml() -> str:
    """abc_sim's empty bimanual station (task ``empty``): the source of the standard camera hardware."""
    from abc_sim.task_registry import get_task_scene_xml

    p = get_task_scene_xml("empty")
    if p is None:
        raise FileNotFoundError("abc_sim task 'empty' has no scene xml")
    return str(p)


CAMERA_REFERENCE_TASK = "drawer"


def camera_reference_xml() -> str:
    """Source of the grafted camera bodies.  abc_sim's ``empty`` scene hides the D405 housings (rgba 0, group 3) while
    the standard task scenes (``drawer``, ``blocks``, ``jenga``, ...) render them (group 2, rgba 0.22 0.22 0.24): the
    bodies are otherwise identical (same pose, camera, ``{side}_cam_col`` collision sphere).  Task scenes are the
    reference here so the wrist / top housings are visible to the cameras as in every abc_sim task."""
    from abc_sim.task_registry import get_task_scene_xml

    p = get_task_scene_xml(CAMERA_REFERENCE_TASK)
    if p is None:
        raise FileNotFoundError(f"abc_sim task {CAMERA_REFERENCE_TASK!r} has no scene xml")
    return str(p)


def _load_abs(path: str) -> mujoco.MjSpec:
    s = mujoco.MjSpec.from_file(path)
    base = os.path.join(os.path.dirname(path), s.meshdir or "")
    for m in s.meshes:
        if m.file and not os.path.isabs(m.file):
            m.file = os.path.join(base, m.file)
    tbase = os.path.join(os.path.dirname(path), s.texturedir or "")
    for t in s.textures:
        if t.file and not os.path.isabs(t.file):
            t.file = os.path.join(tbase, t.file)
    return s


def _subtree_bodies(b) -> list:
    out = [b]
    for c in b.bodies:
        out += _subtree_bodies(c)
    return out


def _graft(dst: mujoco.MjSpec, parent, ref_path: str, name: str) -> None:
    """Copy body ``name`` (and its subtree) of the reference scene under ``parent`` at its reference local pose.
    Assets it references are copied under an ``abc_std_`` name when the destination already has that name."""
    ref = _load_abs(ref_path)
    src = ref.body(name)
    if src is None:
        raise KeyError(f"reference station {ref_path} has no body {name!r}")
    # attach copies every element of the child spec: keep only the subtree and the assets it uses
    for kind in _NON_TREE:
        for el in list(getattr(ref, kind, ())):
            ref.delete(el)
    geoms = [g for b in _subtree_bodies(src) for g in b.geoms]
    used = {"meshes": {g.meshname for g in geoms if g.meshname}, "materials": {g.material for g in geoms if g.material}}
    used["textures"] = {t for m in ref.materials if m.name in used["materials"] for t in m.textures if t}
    for kind in _ASSET_KINDS:
        have = {a.name for a in getattr(dst, kind)}
        for a in list(getattr(ref, kind)):
            if a.name not in used.get(kind, ()):
                ref.delete(a)
            elif a.name in have:
                new = "abc_std_" + a.name
                for g in geoms:
                    if kind == "meshes" and g.meshname == a.name:
                        g.meshname = new
                    if kind == "materials" and g.material == a.name:
                        g.material = new
                if new in have:          # grafted before (same reference asset): reuse it
                    ref.delete(a)
                else:
                    a.name = new
    frame = parent.add_frame(pos=src.pos, quat=src.quat)
    src.pos, src.quat = [0, 0, 0], [1, 0, 0, 0]
    frame.attach_body(src, "", "")


def _ref_table(ref_path: str):
    """(centre xy, half xy, top z) of the standard station's play_table box, compiled (world frame)."""
    rm = mujoco.MjSpec.from_file(ref_path).compile()
    d = mujoco.MjData(rm)
    mujoco.mj_kinematics(rm, d)
    b = rm.body("play_table").id
    g = next(g for g in range(rm.ngeom) if rm.geom_bodyid[g] == b and rm.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX)
    return d.geom_xpos[g][:2].copy(), rm.geom_size[g][:2].copy(), float(d.geom_xpos[g][2] + rm.geom_size[g][2])


def _match_table(spec: mujoco.MjSpec, ref_path: str, name: str = "table") -> None:
    """Give the scene's table-top box the xy footprint of the standard station's play_table, so it ends at the gate /
    arm-mount rack like every abc_sim task.  The box is a world geom ``name`` (tongs packages) or the box of a
    ``play_table`` body (station rebuilds of real episodes, which carry their station's longer table); its top height
    and thickness are kept.  Then the visual walls / photo planes are fitted to the same footprint."""
    c, h, ref_top = _ref_table(ref_path)
    g = next((x for x in spec.worldbody.geoms if x.name == name and x.type == mujoco.mjtGeom.mjGEOM_BOX), None)
    parent = np.zeros(3)
    if g is None and spec.body("play_table") is not None:
        pb = spec.body("play_table")
        if any(abs(q) > 1e-9 for q in list(pb.quat)[1:]):
            raise ValueError("scene play_table body is rotated; cannot fit it to the standard table")
        g = next((x for x in pb.geoms if x.type == mujoco.mjtGeom.mjGEOM_BOX), None)
        parent = np.asarray(list(pb.pos), float)
    if g is not None:
        top = float(parent[2] + g.pos[2] + g.size[2])
        if abs(ref_top - top) > 1e-3:
            logger.warning("scene table top z %.4f != standard play_table top %.4f", top, ref_top)
        g.pos = [float(c[0] - parent[0]), float(c[1] - parent[1]), g.pos[2]]
        g.size = [float(h[0]), float(h[1]), g.size[2]]
    _fit_walls(spec)
    _fit_photo_planes(spec, c, h)


def _fit_photo_planes(spec: mujoco.MjSpec, tc, th) -> None:
    """Planar visual meshes of the world (non-colliding world mesh geoms, one bounding-box extent ~0: e.g. a real
    station's photographed table / walls, ``bg_table`` / ``bg_wall_*``) are rescaled and moved onto the standard
    footprint, keeping their texture: a horizontal plane onto the play_table top (xy of the standard table), a far
    (+x facing) wall onto ``background._SCENE`` far_x over the side walls' y span, side walls onto left_y / right_y
    over near_x..far_x.  Planes behind the arm mounts (centre x < near_x) only get their y snapped (side planes) /
    their y span matched (back plane).  All heights
    of walls are set to ``_SCENE`` z0..z1."""
    from abc_sim.background import _SCENE as S

    W = mujoco.mjtGeom.mjGEOM_MESH
    cand = [g for g in spec.worldbody.geoms if g.type == W and not g.contype and not g.conaffinity and g.meshname]
    if not cand:
        return
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    meshes = {x.name: x for x in spec.meshes}
    users = {}
    for g in spec.geoms:
        if g.meshname:
            users[g.meshname] = users.get(g.meshname, 0) + 1
    for g in cand:
        i = m.geom(g.name).id if g.name else None
        if i is None:
            continue
        R = d.geom_xmat[i].reshape(3, 3)
        half_l = m.geom_aabb[i, 3:]
        half_w = np.abs(R) @ half_l
        cen = d.geom_xpos[i] + R @ m.geom_aabb[i, :3]
        flat = int(np.argmin(half_w))
        if half_w[flat] > 1e-3 or users.get(g.meshname, 0) != 1:
            continue
        lo, hi = cen - half_w, cen + half_w
        tlo, thi = lo.copy(), hi.copy()
        if flat == 2:                                  # table photo
            tlo[:2], thi[:2] = tc - th, tc + th
        elif cen[0] < S["near_x"]:                     # behind the mounts: only snap the side plane
            if flat == 1:
                tlo[1] = thi[1] = S["left_y"] if cen[1] > 0 else S["right_y"]
            else:
                tlo[1], thi[1] = S["right_y"], S["left_y"]
        elif flat == 0:                                # far wall
            tlo[0] = thi[0] = S["far_x"]
            tlo[1], thi[1] = S["right_y"], S["left_y"]
            tlo[2], thi[2] = S["z0"], S["z1"]
        else:                                          # side wall
            tlo[1] = thi[1] = S["left_y"] if cen[1] > 0 else S["right_y"]
            tlo[0], thi[0] = S["near_x"], S["far_x"]
            tlo[2], thi[2] = S["z0"], S["z1"]
        sw = np.ones(3)
        for k in range(3):
            if k != flat:
                sw[k] = (thi[k] - tlo[k]) / max(hi[k] - lo[k], 1e-9)
        # world-axis scale -> mesh-local axes (planes are axis aligned: R is a signed permutation)
        perm = np.argmax(np.abs(R), axis=0)            # local axis j -> world axis perm[j]
        sl = sw[perm]
        mesh = meshes[g.meshname]
        mesh.scale = (np.asarray(list(mesh.scale), float) * sl).tolist()
        # centre: the compiled centre scales with the mesh about the geom frame origin
        new_cen = d.geom_xpos[i] + R @ (m.geom_aabb[i, :3] * sl)
        g.pos = (np.asarray(list(g.pos), float) + (tlo + thi) / 2 - new_cen).tolist()


def _fit_walls(spec: mujoco.MjSpec) -> None:
    """Visual wall quads of a scene (world meshes ``wall_far`` / ``wall_left`` / ``wall_right``) snapped onto the
    standard station's background planes (``abc_sim.background._SCENE``: the far wall at the gate rack, the side walls
    at the play_table +-y edges), keeping each quad's vertex order (its texture UVs).  Package walls placed for an
    older, larger table otherwise cut through the standard table or leave a gap behind it."""
    from abc_sim.background import _SCENE as S

    for m in spec.meshes:
        if m.name not in ("wall_far", "wall_left", "wall_right"):
            continue
        v = np.asarray(list(m.uservert), float).reshape(-1, 3)
        if not len(v):
            continue
        hi = v.max(0) - 1e-6

        def snap(a, axis, lo_val, hi_val):
            v[:, axis] = np.where(a >= hi[axis], hi_val, lo_val)

        x, y, z = v[:, 0].copy(), v[:, 1].copy(), v[:, 2].copy()
        snap(z, 2, S["z0"], S["z1"])
        if m.name == "wall_far":
            v[:, 0] = S["far_x"]
            snap(y, 1, S["right_y"], S["left_y"])
        else:
            v[:, 1] = S["left_y"] if m.name == "wall_left" else S["right_y"]
            snap(x, 0, S["near_x"], S["far_x"])
        m.uservert = v.ravel().tolist()


def _lim(flag) -> int:
    return mujoco.mjtLimited.mjLIMITED_TRUE if flag else mujoco.mjtLimited.mjLIMITED_FALSE


def _match_arms(spec: mujoco.MjSpec, ref_path: str) -> None:
    """Arm mounts (``{side}_arm`` local pose) and arm joint ranges as in the standard station: task packages built
    from a per-station calibration (e.g. a real yam-NN mount offset) are put back on the standard hardware."""
    ref = mujoco.MjSpec.from_file(ref_path)
    rm = ref.compile()          # compiled values: the reference sets ranges / limits via defaults, autolimits, inheritrange
    for s in SIDES:
        a, r = spec.body(f"{s}_arm"), ref.body(f"{s}_arm")
        if a is not None and r is not None:
            a.pos, a.quat = list(r.pos), list(r.quat)
        rb = _subtree_bodies(r) if r is not None else []
        for j in (j for b in rb for j in b.joints):
            sj = spec.joint(j.name)
            if sj is not None:
                i = rm.joint(j.name).id
                sj.range = rm.jnt_range[i].tolist()
                sj.limited = _lim(rm.jnt_limited[i])
    # gripper gravity compensation as in the standard station: its link_6 descendants with mass (fingers) carry the
    # arm controller's gravcomp; the scene's own gripper bodies under link_6 (not cameras) get that same value
    for s in SIDES:
        r6, s6 = ref.body(f"{s}_link_6"), spec.body(f"{s}_link_6")
        if r6 is None or s6 is None:
            continue
        gcs = {float(rm.body_gravcomp[rm.body(b.name).id]) for b in _subtree_bodies(r6)[1:]
               if "camera" not in b.name and rm.body_mass[rm.body(b.name).id] > 1e-3}
        if len(gcs) != 1:
            logger.warning("standard %s gripper gravcomp not uniform (%s); scene gripper left as is", s, gcs)
            continue
        gc = gcs.pop()
        for b in _subtree_bodies(s6)[1:]:
            if "camera" not in b.name:
                b.gravcomp = gc
    arm_joints = {j.name for s in SIDES if ref.body(f"{s}_arm") is not None
                  for b in _subtree_bodies(ref.body(f"{s}_arm")) for j in b.joints if not b.name.endswith("finger")}
    have = {a.name: a for a in spec.actuators}
    for ra in ref.actuators:
        a = have.get(ra.name)
        if a is not None and ra.target in arm_joints and a.target == ra.target:
            i = rm.actuator(ra.name).id
            a.inheritrange = 0
            a.ctrlrange, a.ctrllimited = rm.actuator_ctrlrange[i].tolist(), _lim(rm.actuator_ctrllimited[i])
            a.forcerange, a.forcelimited = rm.actuator_forcerange[i].tolist(), _lim(rm.actuator_forcelimited[i])
            a.gainprm, a.biasprm = rm.actuator_gainprm[i].tolist(), rm.actuator_biasprm[i].tolist()
            a.gear = rm.actuator_gear[i].tolist()


def standardize(spec: mujoco.MjSpec, ref_path: str | None = None) -> mujoco.MjSpec:
    """Replace the scene's own top / wrist camera hardware by the standard abc_sim station's (in place).

    Requires the abc robot naming (``gate_collision``, ``{side}_link_6``).  World cameras of the task scene named
    ``top`` / ``left`` / ``right`` (scene preview cameras) are renamed ``scene_<name>`` so the standard names are unique.
    """
    ref_path = ref_path or reference_xml()
    bodies = {b.name: b for b in spec.bodies}
    for name in ("top_bar_center", "top_camera_d405", *(f"{s}_camera_d405" for s in SIDES)):
        if name in bodies:
            spec.delete(bodies[name])
    for c in list(spec.cameras):
        if c.name in POLICY_CAMERAS.values():
            c.name = "scene_" + c.name
    bodies = {b.name: b for b in spec.bodies}
    missing = [n for n in ("gate_collision", *(f"{s}_link_6" for s in SIDES)) if n not in bodies]
    if missing:
        raise KeyError(f"standardize: scene has no {missing} (needs the abc_sim robot / station naming)")
    _match_table(spec, ref_path)
    _match_arms(spec, ref_path)
    cam_ref = camera_reference_xml()
    _graft(spec, bodies["gate_collision"], cam_ref, "top_bar_center")
    for s in SIDES:
        _graft(spec, bodies[f"{s}_link_6"], cam_ref, f"{s}_camera_d405")
    return spec


# ---------------------------------------------------------------------------------------------------- variants
def _hardware():
    try:
        from one_model.sim.abc import hardware
    except ImportError as e:   # optional dependency: only needed when a variant is requested
        raise ImportError("a YAM hardware variant was requested but one_model.sim.abc.hardware is not importable "
                          "(put one-model on PYTHONPATH; it needs pydantic)") from e
    return hardware


def resolve_variant(kwargs: dict[str, Any], task_kwargs: dict[str, Any] | None = None, seed: int | None = None):
    """The one-model ``YamVariantCfg`` requested for this env, or None (= standard abc_sim station, ``abc_crank``
    cameras).  Sources, first hit wins:

    * ``variant_cfg=`` (a YamVariantCfg, its dict, or a variant name), as one-model passes it;
    * ``environment_factory=partial(LinearGripperEnv, variant_cfg=..., gripper_cfg=...)`` -- what one-model's
      ``AbcScene.make_env`` forwards through ``abc_sim.make_env``;
    * ``hardware_variant=`` in kwargs or task_kwargs: a name, a dict, or ``"random"`` (one-model's
      ``sample_yam_variant`` over all variants with the episode seed).
    """
    task_kwargs = task_kwargs or {}
    v = kwargs.get("variant_cfg")
    factory = kwargs.get("environment_factory")
    if v is None and isinstance(factory, functools.partial):
        v = factory.keywords.get("variant_cfg")
    if v is None:
        v = kwargs.get("hardware_variant", task_kwargs.get("hardware_variant"))
    if v is None:
        return None
    hw = _hardware()
    if isinstance(v, hw.YamVariantCfg):
        return v
    if isinstance(v, str) and v == "random":
        return hw.sample_yam_variant(hw.all_yam_variants(), 0 if seed is None else int(seed))
    if isinstance(v, str):
        return hw.YamVariantCfg(name=v)
    if isinstance(v, dict):
        return hw.YamVariantCfg(**v)
    if hasattr(v, "model_dump"):     # a YamVariantCfg from another import of one_model
        return hw.YamVariantCfg(**v.model_dump())
    raise TypeError(f"unsupported hardware variant {v!r}")


def gripper_kind(kwargs: dict[str, Any], variant) -> str | None:
    """'linear' | 'crank' | None (= task default).  one-model: variant.gripper 'scene' (abc_crank) keeps the ABC
    crank gripper, 'linear_4310' -> linear; an environment_factory with a gripper_cfg and no variant -> linear."""
    if variant is not None:
        return "crank" if variant.gripper == "scene" else "linear"
    factory = kwargs.get("environment_factory")
    if isinstance(factory, functools.partial) and factory.keywords.get("gripper_cfg") is not None:
        return "linear"
    return None


def compose(spec: mujoco.MjSpec, variant=None) -> mujoco.MjSpec:
    """standardize + (optional) one-model YamVariant.compose, in place, before compile."""
    standardize(spec)
    if variant is not None:
        _hardware().YamVariant(variant).compose(spec)
    return spec


def camera_sizes(variant, default: tuple[int, int]) -> dict[str, tuple[int, int]]:
    """(height, width) per policy camera: the variant's native sizes (one-model default_camera_sizes) else default."""
    if variant is None:
        return {k: tuple(default) for k in POLICY_CAMERAS}
    sizes = variant.default_camera_sizes()
    out = {}
    for key, cam in POLICY_CAMERAS.items():
        s = sizes[cam]
        out[key] = (int(s[0]), int(s[1])) if not hasattr(s, "height") else (int(s.height), int(s.width))
    return out


__all__ = ["POLICY_CAMERAS", "camera_reference_xml", "camera_sizes", "compose", "gripper_kind", "reference_xml",
           "resolve_variant",
           "standardize"]
