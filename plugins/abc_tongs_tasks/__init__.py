"""abc_sim plugin: the three tongs tasks (bag / place / pick) on mujoco 4.

    ABC_SIM_PLUGINS=abc_tongs_tasks:register
    ABC_SIM_ASSET_SOURCE=<dir | s3:// prefix holding the three packs>     (or per-pack ABC_SIM_ASSET_SOURCE_<PACK>)

    env = abc_sim.make_env("tongs_bread_bag", render_cameras=False)     # or tongs_bread_place / tongs_pick
    obs, info = env.reset(seed=3)          # obs["prompt"]: a seeded paraphrase naming the target
    obs, r, term, trunc, info = env.step(a14)     # ONE physics step; a14 = 6 left joints, left grip, 6 right, right grip

Gripper action follows abc_sim / deploy-anything: 1 = open, 0 = closed (scaled to the tongs-scene finger range).
Rewards are the packages' own: 1 success, -0.5 robot (gripper / arm) touching food or, for pick, lifting the WRONG
object, 0 other failures, else shaping <= 0.99.  reward_mode="shaped01" maps [-0.5, 1] -> [0, 1] for GRPO.
Asset locations are configured at run time only (abc_sim.asset_source); nothing here names a bucket.
"""
from __future__ import annotations

import os
import re
import sys
from typing import Any

import mujoco
import numpy as np

import abc_hw
from abc_sim.asset_source import AssetPack
from abc_sim.plugins import ExternalTask, register_task
from abc_sim.task_eval.base import TaskEvalResult
from abc_sim.task_specs import SimTaskSpec

FINGER_OPEN = 0.0475          # policy gripper 1.0 -> 47.5 mm finger travel (deploy-anything convention)
PACKS = {
    "bag": AssetPack(name="bread_tongs_bag", archive="bread_tongs_env/v2_20261123"),
    "place": AssetPack(name="bread_tongs_place", archive="bread_place_env/v2_20261126"),
    "pick": AssetPack(name="tongs_pick", archive="pick_anything_tongs/v2_20261126"),
}
# Two registered tasks (user design 2026-11-27):
#   tongs_bread_bag   the real2sim bread -> paper bag task, unchanged (full tongs / bread / bag / wallpaper / table DR);
#   tongs_pick_place  "use the tongs to pick up anything and place it anywhere": per seed one sub-task drawn from MIX
#                     (bread + paper bag upweighted to 50%), each with its own package evaluator and DR.
#                     task_kwargs preset="bag" | "place" | "pick" pins the sub-task; task_kwargs mix={...} reweights.
TASKS = {"bag": "tongs_bread_bag", "mix": "tongs_pick_place"}
MIX = {"bag": 0.5, "place": 0.25, "pick": 0.25}
MIX_STREAM = 5101
_ENV_KEYS = ("grasp_hold_s", "preset", "mix", "randomize_background", "background_dir", "background_id", "layout",
             "tongs_pad_torsion", "grip_kp", "grip_kv", "grip_force", "grip_delay_s", "replay_layout", "replay_tex_dir")
LAYOUT_STREAM = 7741
# Torsional friction (m) of the explicit gripper-pad <-> tongs pairs.  The packs hard-code 0.012 (not fitted).
# 2026-11-27 fit on the calibrated cd198 open-loop replay (real in-hand tongs pose is rigid): torsion 0.003 lets the
# tongs twist 10.9 deg p50 in hand and bags nothing; 0.012 twists 2.7 deg; >= 0.025 plateaus at 0.7-1.0 deg (rigid)
# with both breads bagged.  0.05 = the value of the calibrated per-episode replays.
TONGS_PAD_TORSION = 0.05
# Gripper finger servo as identified on the real fleet (pack code/sim4/replay.py + tongs_ident.py, distributions/
# physics_identified_v1.json grip_force_cap_n): position servo kp 5000 N/m, kv 50 N s/m, force cap 40 N.  Real
# saturated grips on the tongs plateau at 46-54 N sim-equivalent.  The packs compile the gripper at kp 100 / kv 10
# with no cap, i.e. ~1-2 N squeeze on a tongs handle (tongs weigh 0.19 kg): the tongs slipped out of every eval grasp.
# Fleet free-air check 2026-11-28 (abc_hw.GRIP_DELAY_S): the real jaw also trails the command by 2 frames (66.7 ms),
# applied here as a pure command delay.  Override with task_kwargs grip_kp / grip_kv / grip_force / grip_delay_s
# (0 = no delay).
GRIP_KP, GRIP_KV, GRIP_FORCE, GRIP_DELAY_S = abc_hw.GRIP_KP, abc_hw.GRIP_KV, abc_hw.GRIP_FORCE, abc_hw.GRIP_DELAY_S


def mix_kind(seed: int, mix: dict | None = None, preset: str | None = None) -> str:
    """Sub-task for ``seed`` in tongs_pick_place (own stream: scene draws are those of the sub-task's own seed)."""
    if preset:
        if preset not in PACKS:
            raise ValueError(f"preset must be one of {sorted(PACKS)}")
        return preset
    w = dict(MIX if mix is None else mix)
    ks = [k for k in PACKS if w.get(k, 0) > 0]
    p = np.asarray([float(w[k]) for k in ks])
    return ks[int(np.random.default_rng([int(seed), MIX_STREAM]).choice(len(ks), p=p / p.sum()))]


# ======================================================================================================== prompts
BAG_PROMPTS = (
    "Use the tongs to place {bread} into the paper bag",
    "Grab the tongs, and use them to grip {bread} and drop it into the paper bag",
    "Pick up the tongs, grab {bread} with them and put it in the paper bag",
    "With the tongs, move {bread} into the paper bag",
    "Take the tongs and bag {bread}",
    "Using the tongs, pick up {bread} and place it inside the paper bag",
    "Grab the tongs, pinch {bread} and dump it into the bag",
    "Hold the tongs, lift {bread} and put it into the brown paper bag",
    "Use the tongs to pack {bread} into the paper bag",
    "Pick up the tongs to grab {bread} and place it in the paper bag",
)
PLACE_PROMPTS = (
    "Use the tongs to place {bread} {where}",
    "Grab the tongs, and use them to grip {bread} and put it {where}",
    "Pick up the tongs to grab {bread} and place it {where}",
    "With the tongs, move {bread} {where}",
    "Take the tongs, pick up {bread} and set it down {where}",
    "Using the tongs, lift {bread} and put it {where}",
    "Hold the tongs, grab {bread} and drop it {where}",
    "Use the tongs to move {bread} so that it ends up {where}",
)
PICK_PROMPTS = (
    "Grab the tongs and pick up {obj}",
    "Use the tongs to pick up {obj}",
    "Pick up the tongs, grip {obj} with them and lift it",
    "With the tongs, lift {obj} off the table",
    "Take the tongs and use them to grab {obj}",
    "Pick up the tongs to grab {obj} and hold it up",
    "Using the tongs, pinch {obj} and raise it",
    "Grab the tongs, and use them to grip {obj} and lift it into the air",
)
DIRECTIONS = {"in": "in {ref}", "on": "on top of {ref}", "front": "in front of {ref}", "behind": "behind {ref}",
              "left": "to the left of {ref}", "right": "to the right of {ref}"}

# pick catalogue suffix -> noun phrase by LOOK (what a camera sees), not by internal id
PICK_NOUNS = {
    "blue_cylinder": "blue cylinder", "toy_vehicle": "toy vehicle", "yellow_block": "yellow block",
    "yellow_cube": "yellow cube", "orange": "orange", "poker_chip_stack": "stack of poker chips",
    "scrub_brush": "scrub brush", "toy_car": "toy car", "bit_1": "screwdriver bit", "bit_10": "screwdriver bit",
    "glove": "glove", "screwdriver": "screwdriver", "playing_card_box": "playing card box",
    "bunch_10": "bunch of bananas", "bunch_7": "bunch of bananas", "magenta_paint_jar_a": "magenta paint jar",
    "magenta_paint_jar_b": "magenta paint jar", "orange_paint_jar": "orange paint jar", "red_paint_jar": "red paint jar",
    "stapler": "stapler", "teal_paint_jar": "teal paint jar", "yellow_handled_tool": "yellow-handled tool",
    "yellow_paint_jar": "yellow paint jar", "watch": "watch", "croissant": "croissant", "lobed_bun": "lobed bun",
    "ridged_bun": "ridged bun", "sugar_bun": "sugar bun", "light_blue_case": "light blue case",
    "pen_holder": "pen holder", "clear_tape_roll": "roll of clear tape", "mango_orange_large": "orange mango",
    "mango_orange_small": "orange mango", "mango_red_1": "red mango", "mango_red_2": "red mango",
    "mango_yellow": "yellow mango", "block_e": "letter E block", "block_n": "letter N block",
    "block_o": "letter O block", "pink_1": "pink block", "pink_5": "pink block", "wood_end": "wooden block",
    "glass_jar_clear": "clear glass jar", "glass_jar_dark_lid": "glass jar with the dark lid",
    "block_7": "number 7 block", "block_A": "letter A block", "block_N_pink": "pink letter N block",
    "block_O_blue": "blue letter O block", "block_yellow": "yellow block", "toy_spade": "toy spade",
}


def pick_noun(name: str) -> str:
    suf = name.split("__")[-1]
    return PICK_NOUNS.get(suf, suf.replace("_", " "))


def _the(label: str) -> str:
    return label if label.startswith(("the ", "a ")) else f"the {label}"


def pick_object_phrase(items: list[dict], target: int = 0) -> str:
    """Unambiguous phrase for items[target]; items: dict(noun, vol, y).  Robot frame: +y is the robot's left."""
    t = items[target]
    same = [i for i, it in enumerate(items) if it["noun"] == t["noun"]]
    if len(same) == 1:
        return _the(t["noun"])
    vols = [items[i]["vol"] for i in same]
    if len(same) == 2 and max(vols) > 1.15 * min(vols):
        return f"the {'larger' if t['vol'] == max(vols) else 'smaller'} {t['noun']}"
    ys = [items[i]["y"] for i in same]
    side = "leftmost" if t["y"] == max(ys) else "rightmost" if t["y"] == min(ys) else None
    if side:
        return f"the {side} {t['noun']}" if len(same) > 2 else f"the {t['noun']} on the {side[:-4]}"
    return f"the {t['noun']} between the other two"


# ======================================================================================================== pack import
_LOADED_ROOT: dict[str, str] = {}


def _use_pack(kind: str) -> str:
    """Put this pack's code first on sys.path; drop modules imported from the OTHER packs (same module names)."""
    root = str(PACKS[kind].path())
    code = os.path.join(root, "code", "sim4")
    if not os.path.isdir(code):
        raise FileNotFoundError(f"asset pack {PACKS[kind].name}: no code/sim4 in {root}")
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None) or ""
        if f.endswith(".py") and os.sep + os.path.join("code", "sim4") + os.sep in f and not f.startswith(code + os.sep):
            del sys.modules[name]
    while code in sys.path:
        sys.path.remove(code)
    sys.path.insert(0, code)
    _LOADED_ROOT[kind] = root
    return code


def _wrong_pick_cls():
    import bread_reward as BR
    import pick_reward as PR

    class WrongPickEvaluator(PR.PickTongsEvaluator):
        """Pick evaluator + wrong-object failure: a DISTRACTOR squeezed by both tongs heads, tongs in hand, nothing else
        touching it and lifted >= wrong_lift_mm -> failure "distractor<k>_picked_wrong_object", reward -0.5."""

        def __init__(self, wrong_lift_mm: float = 30.0, **kw):
            super().__init__(**kw)
            self.wrong_lift = float(wrong_lift_mm) / 1000.0

        def _goal_setup(self, m, d):
            super()._goal_setup(m, d)
            self.dz0 = {k: float(self._centre(d, k)[2]) for k in self.breads if k != PR.OBJ}

        def _extra_step(self, m, d):
            super()._extra_step(m, d)
            for k, z0 in self.dz0.items():
                s = self.touch.get(BR.BREAD0 + int(k), set())
                if not ({BR.T_HEAD_L, BR.T_HEAD_R} <= s) or any(not BR._tongs(x) for x in s) or not self.held:
                    continue
                if float(self._centre(d, k)[2]) - z0 >= self.wrong_lift:
                    self._fail(f"distractor{k}_picked_wrong_object")

        @property
        def robot_failed(self):
            return bool(super().robot_failed) or any(
                f["why"].endswith("_picked_wrong_object") for f in self.failures)

    return WrongPickEvaluator


def _strict_tongs_grasp(ev, hold_steps: int):
    """Make the evaluator's tongs-grasp shaping (0.1 * w_grasp) require a STABLE PICKUP instead of the pack's
    'both fingertips touched the handle once' (tongs_grasp_ever): the tongs held (both fingertips of an arm on the
    handle, the pack's hold grace), touching no support / bread, hub >= tongs_lift_mm above its last rest height,
    continuously for ``hold_steps`` physics steps -> latched ``tongs_grasp_stable``.  Instance-level wrap: the cached
    pack code is not edited.  Failures / stages / the pack's own metrics are unchanged."""
    import bread_reward as BR

    if getattr(ev, "_strict_grasp", False):
        return ev
    ev._strict_grasp, ev.stable_run, ev.grasp_stable = True, 0, False
    step0, reward0, metrics0, reset0 = ev.after_physics_step, ev._reward, ev._metrics, ev.after_reset

    def after_reset(m, d):
        reset0(m, d)
        ev.stable_run, ev.grasp_stable = 0, False

    def after_physics_step(m, d):
        step0(m, d)
        touch = getattr(ev, "touch", {}) or {}
        tcont = set().union(*[touch.get(c, set()) for c in (BR.T_HUB, BR.T_ARM_L, BR.T_ARM_R, BR.T_HEAD_L, BR.T_HEAD_R)])
        tsup = bool(tcont & set(BR.SUPPORT)) or any(BR._bread(c) for c in tcont)
        up = float(d.xpos[ev.tongs_root][2]) > ev.tz_rest + ev.tlift
        ev.stable_run = ev.stable_run + 1 if (ev.held and any(ev.grasp_now.values()) and not tsup and up) else 0
        if ev.stable_run >= hold_steps:
            ev.grasp_stable = True

    def _reward():
        ge = ev.grasp_ever
        ev.grasp_ever = {k: bool(ev.grasp_stable) for k in ge}
        try:
            return reward0()
        finally:
            ev.grasp_ever = ge

    def _metrics():
        out = metrics0()
        out.update(tongs_grasp_stable=bool(ev.grasp_stable), tongs_stable_run_steps=int(ev.stable_run),
                   tongs_touched=any(dict(ev.grasp_ever).values()))
        return out

    ev.after_reset, ev.after_physics_step, ev._reward, ev._metrics = after_reset, after_physics_step, _reward, _metrics
    return ev


def _install_station_hook(variant) -> None:
    """Make the pack's robot_v4.attach_robot also graft the standard abc_sim station cameras and apply the one-model
    hardware variant (abc_hw.compose) on the scene spec, before the package compiles it."""
    import robot_v4

    orig = getattr(robot_v4.attach_robot, "_abc_orig", robot_v4.attach_robot)

    def attach_robot(scene, *a, **k):
        out = orig(scene, *a, **k)
        abc_hw.compose(scene, variant)
        return out

    attach_robot._abc_orig = orig
    robot_v4.attach_robot = attach_robot


def _set_gripper_servo(m, kp: float, kv: float, fmax: float) -> list[int]:
    """Identified finger servo on the compiled model (see GRIP_KP): every ``*_gripper`` position actuator."""
    return abc_hw.set_gripper_servo(m, kp, kv, fmax)


def _install_torsion_hook(torsion: float) -> None:
    """Make the pack's bread_env.add_gripper_tongs_pairs use the fitted pad <-> tongs torsion (``TONGS_PAD_TORSION``)."""
    import bread_env

    orig = getattr(bread_env.add_gripper_tongs_pairs, "_abc_orig", bread_env.add_gripper_tongs_pairs)

    def add_gripper_tongs_pairs(*a, **k):
        k.setdefault("torsion", float(torsion))
        return orig(*a, **k)

    add_gripper_tongs_pairs._abc_orig = orig
    bread_env.add_gripper_tongs_pairs = add_gripper_tongs_pairs

# calibrated-replay bread contact / look (the replay scene builder's values: MU_OBJ, foam solref, projective texture)
_REPLAY_MU, _REPLAY_SOLREF, _REPLAY_SOLIMP, _REPLAY_EMIT = 0.8, [0.004, 1.0], [0.9, 0.95, 0.001, 0.5, 2.0], 0.8


def _quat_yaw(yaw: float) -> list:
    return [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))]


def _replay_tex_mesh(spec, body, name, V, F, uv, png, mass):
    m = spec.add_mesh()
    m.name = name
    m.uservert = np.asarray(V, float).ravel().tolist()
    m.userface = np.asarray(F, int).ravel().tolist()
    m.usertexcoord = np.asarray(uv, float).ravel().tolist()
    m.userfacetexcoord = np.asarray(F, int).ravel().tolist()
    t = spec.add_texture()
    t.name, t.type, t.file = name, mujoco.mjtTexture.mjTEXTURE_2D, os.path.abspath(png)
    mat = spec.add_material()
    mat.name = name
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = t.name
    mat.emission, mat.specular, mat.shininess, mat.rgba = _REPLAY_EMIT, 0.0, 0.0, [1, 1, 1, 1]
    g = body.add_geom()
    g.name, g.type, g.meshname, g.material = name, mujoco.mjtGeom.mjGEOM_MESH, name, name
    g.friction, g.condim, g.solref, g.solimp = [_REPLAY_MU, 0.01, 0.001], 4, _REPLAY_SOLREF, _REPLAY_SOLIMP
    g.mass = mass
    return g


def _replay_lay(path: str, tex_dir: str | None) -> dict:
    """A calibrated real-episode layout json (code/replay_cal/layouts/lay_*.json: dome breads with projective episode
    textures, tray, bag, tongs), its texture / uv paths remapped by basename into ``tex_dir`` when given."""
    import json
    L = json.load(open(path))
    if tex_dir:
        for o in L["objects"].values():
            for k in ("tex", "uv"):
                if o.get(k):
                    o[k] = os.path.join(tex_dir, os.path.basename(o[k]))
    for o in L["objects"].values():
        for k in ("tex", "uv"):
            if o.get(k) and not os.path.exists(o[k]):
                raise FileNotFoundError(f"replay layout {path}: {k} {o[k]} missing (set task_kwargs replay_tex_dir)")
    return L


def _install_replay_hook(L: dict | None) -> None:
    """tongs_bread_bag on a real episode's calibrated layout + assets (e.g. cd198, the layout of the successful
    calibrated replays) instead of a sampled one.  The pack builds everything as usual (flex bag, tongs, robot), with:
    its layout draw (bread_env_v4.sample_layout_robot) returning the episode's tray / bag / tongs x, y, yaw; its tray and
    bag dims (+ bag texture) the episode's; its library breads replaced by the episode's fitted dome breads (settled
    pose, projective texture, fleet bread mass, MU_OBJ 0.8 -- the replay values), keeping the names bread0..N-1.
    L=None restores the pack behaviour.  Physics is the pack's; only the scene content changes."""
    import bread_env as BE
    import bread_env_v4 as BE4

    lo = getattr(BE4.sample_layout_robot, "_abc_orig", BE4.sample_layout_robot)
    bo = getattr(BE4.BreadEnvV4._build, "_abc_orig", BE4.BreadEnvV4._build)
    if L is None:
        BE4.sample_layout_robot, BE4.BreadEnvV4._build = lo, bo
        return
    objs = L["objects"]
    bag = next(o for o in objs.values() if o.get("kind") == "bag")
    breads = [objs[k] for k in sorted(objs, key=int) if objs[k].get("kind") == "bread"]
    tr = L["tray"]["pose"]

    def sample_layout_robot(*a, **k):
        return {"tray": [tr[0], tr[1], tr[2]], "bag": list(bag["pose"][:3]),
                "tongs": [L["tongs"]["pos"][0], L["tongs"]["pos"][1], L["tongs"]["yaw"]]}

    def _build(self, w, h):
        import json
        import bread_mesh as bm
        idf = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(BE.__file__)))),
                           "ident", "dists.json")
        mass = float(json.load(open(idf))["sim_params"]["bread_mass_kg"]) if os.path.exists(idf) else 0.0378
        old = BE.TRAY, BE.BAG
        BE.TRAY = dict(old[0], a=tr[3], b=tr[4], floor=L["tray"].get("floor", old[0]["floor"]),
                       rim=L["tray"].get("rim", old[0]["rim"]), rgba=tuple(L["tray"].get("vis_rgba", old[0]["rgba"][:3])) + (1,))
        BE.BAG = dict(old[1], a=bag["pose"][3], b=bag["pose"][4], c=bag["pose"][5],
                      handle_top=bag.get("handle_top", old[1]["handle_top"]), png=bag["tex"], uv=bag["uv"])
        self.bag_v = None                       # episode bag texture (BE.BAG png / uv), not a fleet bag look
        try:
            spec = bo(self, w, h)
        finally:
            BE.TRAY, BE.BAG = old
        for b in [b for b in spec.bodies if re.fullmatch(r"bread\d+", b.name or "")]:
            spec.delete(b)
        new = []
        for k, o in enumerate(breads):
            x, y, yaw, a, b_, c = o["pose"]
            body = spec.worldbody.add_body()
            body.name, body.pos, body.quat = f"bread{k}", list(o["xyz"]), list(o.get("quat", _quat_yaw(yaw)))
            body.add_freejoint().name = f"bread{k}_free"
            V, F = bm.dome(a, b_, c, o["shape"]["n"], o["shape"]["p"])
            _replay_tex_mesh(spec, body, f"rbread{k}_g", V, F, np.load(o["uv"]), o["tex"], mass)
            new.append(dict(name=f"replay_{k}", cls="replay", label=o.get("label", "bread"), density_kg_m3=float("nan"),
                            mass_scale=1.0, mass_kg=mass, extents=[2 * a, 2 * b_, c],
                            pose=[x, y, float(body.pos[2]), yaw], layer=0, friction=_REPLAY_MU))
        self.breads[:] = new
        self.flex = 0
        return spec

    sample_layout_robot._abc_orig, _build._abc_orig = lo, bo
    BE4.sample_layout_robot, BE4.BreadEnvV4._build = sample_layout_robot, _build


def _check_cameras(m, cams: dict) -> None:
    missing = [c for c in cams.values() if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, c) < 0]
    if missing:
        raise RuntimeError(f"policy cameras {missing} missing from the compiled scene")


# ======================================================================================================== env
class TongsAbcEnv:
    """abc_sim env surface (abc_sim.plugins.SimEnvSurface) over one tongs package task.  step() = ONE physics step."""

    # policy key -> model camera: the standard abc_sim station cameras (or a one-model YAM hardware variant's), grafted
    # into the package scene by abc_hw at build time
    POLICY_CAMERAS = dict(abc_hw.POLICY_CAMERAS)
    CAMERAS = tuple(POLICY_CAMERAS.values())

    def __init__(self, kind: str, spec: SimTaskSpec | None = None, *, seed: int = 0, prompt: str | None = None,
                 max_seconds: float = 60.0, reward_mode: str = "native", terminate_on_success: bool = True,
                 terminate_on_failure: bool = True, task_kwargs: dict | None = None, **_ignored: Any):
        if kind not in PACKS and kind != "mix":
            raise ValueError(kind)
        self.task_name = TASKS.get(kind, kind)
        self.mode, self.kind, self.spec = kind, kind, spec
        self.fixed_prompt = prompt
        self.max_seconds = float(max_seconds)
        self.reward_mode = reward_mode
        self.term_success, self.term_failure = bool(terminate_on_success), bool(terminate_on_failure)
        self.task_kwargs = dict(task_kwargs or {})
        # one-model hardware randomization: variant_cfg / environment_factory=partial(..., variant_cfg=) / hardware_variant
        self._hw_kwargs = {k: _ignored[k] for k in ("variant_cfg", "environment_factory", "hardware_variant") if k in _ignored}
        self.variant = None
        self.task = None
        self.prompt: str | None = None
        self.cur_step = 0
        self.seed = None
        self._build(int(seed))

    # ---------------------------------------------------------------------------------------------- build
    def _build(self, seed: int) -> None:
        if self.mode == "mix":
            self.kind = mix_kind(seed, self.task_kwargs.get("mix"), self.task_kwargs.get("preset"))
            if hasattr(self, "_cat") and self.kind != "pick":
                del self._cat
        _use_pack(self.kind)
        kw = {k: v for k, v in self.task_kwargs.items() if k not in _ENV_KEYS}
        kw["max_seconds"] = self.max_seconds
        self.variant = abc_hw.resolve_variant(self._hw_kwargs, self.task_kwargs, seed)
        grip = abc_hw.gripper_kind(self._hw_kwargs, self.variant)
        if grip is not None:
            kw.setdefault("robot", grip)
        _install_station_hook(self.variant)
        _install_torsion_hook(self.task_kwargs.get("tongs_pad_torsion", TONGS_PAD_TORSION))
        rl = self.task_kwargs.get("replay_layout") if self.kind == "bag" else None
        _install_replay_hook(_replay_lay(rl, self.task_kwargs.get("replay_tex_dir")) if rl else None)
        if rl:
            kw.setdefault("n_breads", 4)
        if self.kind == "bag":
            import bread_tasks as BT
            kw.setdefault("bag_model", "flex")   # MuJoCo 4 shell-flex paper bag (bag5b, CE sampled); "rigid" = box proxy
            self.task = BT.BreadTask("bag", seed=seed, **kw)
        elif self.kind == "place":
            import bread_tasks_v2 as BT2
            self.task = BT2.BreadTask("place", seed=seed, **kw)
        else:
            import pick_task as PT
            t = PT.PickTask(seed=seed, **kw)
            ev = _wrong_pick_cls()(**t.eval_kw)
            ev._bind(t.model)
            ev.after_reset(t.model, t.data)
            t.evaluator, t.last = ev, ev.evaluate_model_data(t.model, t.data)
            self.task = t
        _check_cameras(self.task.model, self.POLICY_CAMERAS)
        gids = _set_gripper_servo(self.task.model, float(self.task_kwargs.get("grip_kp", GRIP_KP)),
                                  float(self.task_kwargs.get("grip_kv", GRIP_KV)), float(self.task_kwargs.get("grip_force", GRIP_FORCE)))
        self._gdelay = abc_hw.GripperDelay(gids, float(self.task_kwargs.get("grip_delay_s", GRIP_DELAY_S)),
                                           float(self.task.model.opt.timestep))
        self.layout_info = self._scatter(seed)
        t = self.task
        _strict_tongs_grasp(t.evaluator, int(round(float(self.task_kwargs.get("grasp_hold_s", 0.5)) / float(t.model.opt.timestep))))
        t.last = t.evaluator.evaluate_model_data(t.model, t.data)
        self.seed, self.cur_step = seed, 0
        m = self.model
        self._grip = np.array([i for i, n in enumerate(self.task.action_names) if n.endswith("gripper")])
        self._jq = np.asarray(self.task._jq)
        self._last_reward = float(np.asarray(self.task.last.reward).reshape(-1)[0])
        self.prompt = self.fixed_prompt or self._sample_prompt(np.random.default_rng([seed, 9311]))
        mujoco.mj_forward(m, self.data)

    def _scatter(self, seed: int) -> dict | None:
        """Reach-gated layout randomization (abc_sim.layout): absolute AND relative placement / yaw of every free object
        group.  On by default for tongs_pick_place; tongs_bread_bag keeps its package layout unless layout="scatter".
        The evaluator is re-bound afterwards so its reset references (rest poses, place directive) see the new layout."""
        want = self.task_kwargs.get("layout", "scatter" if self.mode == "mix" else "package")
        if want in (None, False, "package", "none"):
            return None
        from abc_sim import layout
        t, m, d = self.task, self.task.model, self.task.data
        info = layout.scatter(m, d, np.random.default_rng([int(seed), LAYOUT_STREAM]))
        if not info.get("moved"):
            return info
        kw = dict(getattr(t, "eval_kw", {}) or {})
        if self.kind == "bag":
            import bread_reward as BR
            ev = BR.BreadTongsEvaluator(**kw)
            ev._bind(m)
        elif self.kind == "place":
            import bread_place_reward_v2 as BP2
            import bread_tasks as BT
            if getattr(t, "directive_fixed", None):
                t.directive = dict(t.directive_fixed)
                ev = BP2.BreadTongsPlaceEvaluatorV2(**t.directive, **kw)
                ev._bind(m)
            else:
                t.directive, ev = BP2.pick_directive_v2(np.random.default_rng([int(seed), BT.DIRECTIVE_STREAM]), m, d, **kw)
        else:
            ev = _wrong_pick_cls()(**kw)
            ev._bind(m)
        ev.after_reset(m, d)
        t.evaluator = ev
        t.t0, t.n_steps = float(d.time), 0
        t.last = ev.evaluate_model_data(m, d)
        return info

    @property
    def model(self) -> mujoco.MjModel:
        return self.task.model

    @property
    def data(self) -> mujoco.MjData:
        return self.task.data

    @property
    def evaluator(self):
        return self.task.evaluator

    @property
    def directive(self) -> dict:
        return dict(self.task.directive or {})

    # ---------------------------------------------------------------------------------------------- prompts
    def target_phrase(self) -> str:
        p = self.task.params
        if self.kind == "pick":
            objs = p["pick"]["objects"]
            cat = {c["name"]: c for c in self._catalogue()}
            items = []
            for o in objs:
                c = cat.get(o["name"], {})
                vol = float(c.get("width_mm", 1)) * float(c.get("length_mm", 1)) * float(c.get("height_mm", 1))
                y = float(self.data.xpos[self.model.body(o["body"]).id][1])
                items.append(dict(noun=pick_noun(o["name"]), vol=vol, y=y))
            return pick_object_phrase(items, 0)
        breads = p["breads"]
        if self.kind == "bag":
            ks = [int(k) for k in self.evaluator.targets]
            labels = [_the(breads[k]["label"]) for k in ks if k < len(breads)]
            if len(labels) == 1:
                return labels[0]
            return "all the bread" if len(labels) > 2 else " and ".join(labels)
        return _the(breads[int(self.directive["mover"])]["label"])

    def _catalogue(self):
        import pick_env as PE
        if not hasattr(self, "_cat"):
            self._cat = PE.catalogue()
        return self._cat

    def place_where(self) -> str:
        D, p = self.directive, self.task.params
        ref = D["reference"]
        if ref == "tray":
            ref_s = "the tray"
        elif ref == "tray2":
            col = ((p.get("containers") or {}).get("tray2") or {}).get("colour", "")
            ref_s = f"the empty {col.replace('_', ' ')} tray".replace("  ", " ")
        elif ref == "bowl":
            ref_s = "the bowl"
        elif re.fullmatch(r"\d+", str(ref)):
            ref_s = _the(p["breads"][int(ref)]["label"])
        else:
            ref_s = f"the {str(ref).replace('_', ' ')}"
        return DIRECTIONS.get(D["direction"], D["direction"] + " {ref}").format(ref=ref_s)

    def _sample_prompt(self, rng) -> str:
        if self.kind == "bag":
            return str(rng.choice(BAG_PROMPTS)).format(bread=self.target_phrase())
        if self.kind == "place":
            return str(rng.choice(PLACE_PROMPTS)).format(bread=self.target_phrase(), where=self.place_where())
        return str(rng.choice(PICK_PROMPTS)).format(obj=self.target_phrase())

    # ---------------------------------------------------------------------------------------------- env surface
    def reset(self, seed: int | None = None, options: dict | None = None, randomize: bool = True):
        seed = int(self.seed if (seed is None or not randomize) else seed)
        if self.task is None or seed != self.seed or self.cur_step > 0:
            self._build(seed)
        return self._obs(), self._info()

    def _ctrl(self, action) -> np.ndarray:
        a = np.asarray(action, float).reshape(-1)
        if a.shape[0] != self.model.nu:
            raise ValueError(f"action must have {self.model.nu} entries ({self.task.action_names})")
        a = a.copy()
        a[self._grip] = a[self._grip] * FINGER_OPEN
        return np.clip(a, self.task.action_low, self.task.action_high)

    def apply_action(self, action, physics_steps: int = 1) -> None:
        m, d, ev = self.model, self.data, self.evaluator
        c = self._ctrl(action)
        g = self._gdelay.ids
        keep = np.ones(len(c), bool)
        keep[g] = False
        d.ctrl[keep] = c[keep]
        for _ in range(int(physics_steps)):
            self._gdelay.step(d, c[g])          # gripper command reaches the servo GRIP_DELAY_S late (real transport delay)
            mujoco.mj_step(m, d)
            ev.after_physics_step(m, d)
            self.cur_step += 1
            if ev.terminated:
                break

    def step(self, action, render_obs: bool = True):
        self.apply_action(action, 1)
        res = self.task.last = self.evaluator.evaluate_model_data(self.model, self.data)
        info = self._info(res)
        ev = self.evaluator
        terminated = bool((ev.ever_success and self.term_success) or (ev.failures and self.term_failure))
        truncated = (not terminated) and bool(info["time_limit_reached"])
        return self._obs(), float(info["task_reward"]), terminated, truncated, info

    def _map_reward(self, r: float) -> float:
        return (r + 0.5) / 1.5 if self.reward_mode == "shaped01" else r

    def evaluate_task(self) -> TaskEvalResult:
        r = self.evaluator.evaluate_model_data(self.model, self.data)
        rew = np.asarray(r.reward, np.float32).reshape(-1)
        return TaskEvalResult(reward=np.asarray([self._map_reward(float(rew[0]))], np.float32),
                              success=np.asarray(r.success, bool).reshape(-1), metrics=dict(r.metrics))

    def _info(self, res=None) -> dict:
        res = res if res is not None else self.evaluator.evaluate_model_data(self.model, self.data)
        met = dict(res.metrics)
        r = self._map_reward(float(np.asarray(res.reward).reshape(-1)[0]))
        ok = bool(np.asarray(res.success).reshape(-1)[0])
        failed = bool(self.evaluator.failures)
        t = float(self.data.time - self.task.t0)
        met.update(ever_failed=failed, fail_reason=";".join(str(f.get("why")) for f in self.evaluator.failures) or None, reward=r, directive=self.directive, prompt=self.prompt, subtask=self.kind,
                   task=self.task_name, seed=self.seed)
        return dict(task_reward=r, task_success=ok, task_failed=failed, task_eval=met, episode_step=self.cur_step,
                    subtask=self.kind,
                    sim_time=t, time_limit_reached=t >= self.max_seconds, tongs_grasped=bool(met.get("tongs_grasp_stable")), tongs_touched=bool(met.get("tongs_touched")),
                    robot_touch_failure=bool(met.get("robot_touch_failure")))

    def _obs(self) -> dict:
        return dict(prompt=self.prompt, state=self.get_state(), qpos=self.data.qpos[self._jq].copy(),
                    time=float(self.data.time - self.task.t0))

    def get_state(self) -> np.ndarray:
        q = self.data.qpos[self._jq].astype(np.float64).copy()
        q[self._grip] = q[self._grip] / FINGER_OPEN
        return q

    def forget_arm_state(self) -> None:
        pass

    def render(self, cam: str = "top", w: int = 640, h: int = 480, renderer=None):
        return self.task.render(cam, w, h, renderer)

    def close(self) -> None:
        self.task = None


# ======================================================================================================== registration
def _factory(kind: str):
    def make(*, task=None, prompt=None, seed=0, max_episode_steps=None, terminate_on_success=True,
             terminate_on_failure=True, reward_mode="native", task_kwargs=None, max_seconds=None, **_ignored):
        if max_seconds is None:   # max_episode_steps counts physics steps (timestep 2 ms) when given
            max_seconds = 60.0 if not max_episode_steps else float(max_episode_steps) * 0.002
        return TongsAbcEnv(kind, task if isinstance(task, SimTaskSpec) else None, seed=int(seed or 0),
                           prompt=prompt, max_seconds=max_seconds, reward_mode=reward_mode,
                           terminate_on_success=terminate_on_success, terminate_on_failure=terminate_on_failure,
                           task_kwargs=task_kwargs)
    return make


_SPEC_TEXT = {
    "bag": ("Use the tongs to place the bread into the paper bag", ("tongs bag bread", "bread tongs bag"),
            "tongs bread -> paper bag (real2sim of the offline task; mujoco 4, per-seed tongs / bread / bag / "
            "wallpaper / table DR; the env prompt is a paraphrase naming the target)"),
    "mix": ("Use the tongs to pick up the named object and place it as instructed",
            ("tongs pick place", "tongs pick anything place anywhere", "tongs pick and place"),
            "tongs pick anything / place anywhere (mujoco 4): per seed 50% bread -> paper bag, 25% bread placed "
            "relative to tray / bowl / bread, 25% pick any of ~50 objects (4-8 per scene); the env prompt names "
            "the target; task_kwargs preset / mix"),
}


def register(replace: bool = True) -> None:
    for kind, env_task in TASKS.items():
        prompt, aliases, desc = _SPEC_TEXT[kind]
        spec = SimTaskSpec(name=env_task, env_task=env_task, prompt=prompt, aliases=aliases, max_chunks=60,
                           description=desc)
        packs = (PACKS[kind],) if kind in PACKS else tuple(PACKS.values())
        register_task(ExternalTask(env_task=env_task, specs=(spec,), env_factory=_factory(kind),
                                   assets=packs, description=desc), replace=replace)
    for old in ("tongs_bread_place", "tongs_pick"):     # superseded by tongs_pick_place presets
        from abc_sim.plugins import unregister_task
        unregister_task(old)


__all__ = ["MIX", "PACKS", "TASKS", "mix_kind", "TongsAbcEnv", "pick_noun", "pick_object_phrase", "register"]
