"""abc_sim plugin: open a screw-cap container (paint jar / bottle / tethered-cap bottle) on mujoco 4.

    ABC_SIM_PLUGINS=abc_cap_tasks:register
    ABC_SIM_ASSET_SOURCE=<dir | s3:// prefix holding golden_restore/019cd8ca/portable>   (or ABC_SIM_ASSET_SOURCE_CAP_OPEN)

    env = abc_sim.make_env("open_cap", render_cameras=False)
    obs, info = env.reset(seed=3)        # per-seed scene: jars swapped / bottles added from the 70-variant catalogue
    obs, r, term, trunc, info = env.step(a14)     # ONE physics step (2 ms); a14 = 6 left joints, left grip, 6 right, right grip

Actions / state are in dataset units (the recorded YAM episode, replay.npz): q + joint_offset, gripper 1 = open
(x 0.0475 m finger travel).  Reward (native): 1 success (target lid unscrewed and off: no lid-jar contact for 0.1 s and
lid >= 5 mm off its seat), -0.5 a DIFFERENT container opened (terminate), 0 target container knocked over / off the
table, else shaping <= 0.9:  0.05 robot touched the target lid + 0.1 lid pinched by one gripper + 0.1 container held by
the other gripper + 0.65 * clip(max unscrew twist / that variant's audited free angle, 0, 1).
"""
from __future__ import annotations

import hashlib
import json
import logging
import lzma
import os
import shutil
import sys
import tempfile
import threading
from collections import OrderedDict
from typing import Any

import mujoco
import numpy as np

import abc_hw
from abc_sim.asset_source import AssetPack, cache_root
from abc_sim.plugins import ExternalTask, register_task
from abc_sim.task_eval.base import TaskEvalResult
from abc_sim.task_specs import SimTaskSpec

logger = logging.getLogger(__name__)

PACK = AssetPack(name="cap_open", archive="golden_restore/019cd8ca/portable")
TASK = "open_cap"
DT_FREE = 0.1            # s without lid-container contact
LIFT_FREE = 0.005        # m lid off its seat (container frame)
DEFAULT_FREE_DEG = 228.0
TWIST_DEADBAND_DEG = 2.0
ORIG_JARS = ("magenta_paint_jar_a", "magenta_paint_jar_b", "orange_paint_jar", "red_paint_jar", "teal_paint_jar",
             "yellow_paint_jar")
ORIG_COLOUR = {"magenta_paint_jar_a": "magenta", "magenta_paint_jar_b": "magenta", "orange_paint_jar": "orange",
               "red_paint_jar": "red", "teal_paint_jar": "teal", "yellow_paint_jar": "yellow"}

PROMPTS = (
    "Open {obj}", "Unscrew the {lid} of {obj}", "Take the {lid} off {obj}", "Twist off the {lid} of {obj}",
    "Remove the {lid} from {obj}", "Unscrew {obj} and lift the {lid} off", "Open {obj} by unscrewing its {lid}",
    "Hold {obj} and twist its {lid} off",
)


# ======================================================================================================== pack + build
_LOCK = threading.Lock()


def _work_port() -> str:
    """A pod-local port: symlinks to the pack + an unpacked scene.mjb (the pack itself is never written)."""
    root = str(PACK.path())
    w = os.path.join(str(cache_root().parent), "cap_work", hashlib.sha1(root.encode()).hexdigest()[:12], "port")
    with _LOCK:
        if not os.path.exists(os.path.join(w, "scene.mjb")):
            os.makedirs(w, exist_ok=True)
            for f in os.listdir(root):
                if f in ("scene.mjb", "scene.mjb.xz") or f.startswith("."):
                    continue
                d = os.path.join(w, f)
                if not os.path.lexists(d):
                    os.symlink(os.path.join(root, f), d)
            src = os.path.join(root, "scene.mjb")
            tmp = os.path.join(w, f"scene.mjb.{os.getpid()}.tmp")
            if os.path.exists(src):
                shutil.copyfile(src, tmp)
            else:
                with lzma.open(src + ".xz") as fi, open(tmp, "wb") as fo:
                    shutil.copyfileobj(fi, fo, 1 << 24)
            os.replace(tmp, os.path.join(w, "scene.mjb"))
    return w


def _jarvar(port: str):
    v = os.path.join(port, "variants")
    if v not in sys.path:
        sys.path.insert(0, v)
    old, sys.dont_write_bytecode = sys.dont_write_bytecode, True   # never drop __pycache__ into the pack
    try:
        import jarvar
    finally:
        sys.dont_write_bytecode = old
    return jarvar


_CACHE: OrderedDict = OrderedDict()
_BUILD_LOCK = threading.Lock()
CACHE_N = int(os.environ.get("ABC_CAP_SCENE_CACHE", "4"))


def sample_layout(seed: int, catalog: dict, audit: dict, opts: dict) -> tuple[list, list]:
    rng = np.random.default_rng([int(seed), 4417])
    ok = [k for k in catalog if audit.get(k, {}).get("pass", True)]
    jars = [k for k in ok if catalog[k]["family"] == "paint_jar" and float(catalog[k].get("sr", 9)) <= 1.7]
    p_swap = float(opts.get("p_swap", 0.5))
    swaps = [(j, str(rng.choice(jars))) for j in ORIG_JARS if rng.random() < p_swap]
    nadd = int(rng.integers(0, int(opts.get("max_adds", 3)) + 1))
    adds = [(str(rng.choice(ok)), None) for _ in range(nadd)]
    return swaps, adds


def build(seed: int, opts: dict | None = None, variant=None) -> dict:
    """Scene for ``seed``: dict(model, qpos0, ctrl0, R, log, port).  Built in a temp dir that is removed afterwards.

    The portable's robot is put on the standard abc_sim station hardware (cameras, arm mounts, joint ranges, arm
    actuators) and the requested one-model YAM variant (``abc_hw.compose``) before jarvar adds objects and compiles."""
    opts = dict(opts or {})
    vkey = None if variant is None else json.dumps(variant.metadata(), sort_keys=True)
    key = (int(seed), json.dumps(opts, sort_keys=True), vkey)
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    port = _work_port()
    JV = _jarvar(port)
    cat = json.load(open(os.path.join(port, "variants", "catalog.json")))
    audit = {a["name"]: a for a in json.load(open(os.path.join(port, "variants", "audit.json")))}
    swaps, adds = sample_layout(seed, cat, audit, opts)
    tmp = tempfile.mkdtemp(prefix=f"cap_s{seed}_", dir=os.path.dirname(port))
    rebuild = JV.rebuild

    specs = []

    def std_rebuild(m0, *a, **k):
        sp = abc_hw.compose(rebuild(m0, *a, **k), variant)
        specs.append(sp)            # the final scene's spec (jarvar adds objects to it, then compiles)
        return sp

    _BUILD_LOCK.acquire()
    JV.rebuild = std_rebuild
    try:
        while True:
            try:
                m, log = JV.build_scene(tmp, swaps=swaps, adds=adds, seed=int(seed), port=port, catalog=cat,
                                        stickers=bool(opts.get("stickers", True)), strap_slack=(0.0, 60.0))
                break
            except RuntimeError as e:
                if "no free table spot" not in str(e) or not adds:
                    raise
                adds = adds[:-1]
        R = dict(np.load(os.path.join(tmp, "replay.npz")))
    finally:
        JV.rebuild = rebuild
        _BUILD_LOCK.release()
        shutil.rmtree(tmp, ignore_errors=True)
    out = dict(model=m, R=R, log=log, audit=audit, catalog=cat, swaps=swaps, adds=adds, spec=specs[-1] if specs else None)
    _CACHE[key] = out
    while len(_CACHE) > max(1, CACHE_N):
        _CACHE.popitem(last=False)
    return out


# ======================================================================================================== fixture
FIXTURE = "obj_file_holder"     # the portable's file holder (bookshelf); its support group = holder + the books in it
P_FIXTURE = 0.3                 # probability the fixture group is in a seed's scene


def _fixture_group(m: mujoco.MjModel, d: mujoco.MjData, root: str):
    """The fixture and what it holds: free root bodies (not the robot) whose centre of mass lies inside the xy
    footprint (mesh-vertex hull) of ``root`` and above its base.  Objects merely touching / leaning on it are not
    members (they stay scene objects).  No member names are assumed."""
    from abc_sim import layout
    rid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, root)
    if rid < 0:
        return None
    pts = layout._body_points(m, d, layout._subtree(m, rid))
    H = layout._hull(pts[:, :2])
    z0 = float(pts[:, 2].min())
    robot = layout._robot_bodies(m, layout.DEFAULT_ARMS)
    roots = [rid] + [b for b in range(1, m.nbody) if b != rid and b not in robot and layout._is_free_root(m, b)
                     and d.xipos[b][2] > z0 and layout._point_poly_dist(d.xipos[b][:2], H) < 0]
    bodies = [b for r in roots for b in layout._subtree(m, r)]
    pts = layout._body_points(m, d, bodies)
    hull = layout._hull(pts[:, :2])
    geoms = {g for b in bodies for g in range(m.body_geomadr[b], m.body_geomadr[b] + m.body_geomnum[b])}
    return layout._Group(roots, bodies, geoms, pts, hull, hull.mean(0), [m.body(r).name for r in roots])


def without_fixture(sc: dict, root: str = FIXTURE) -> dict:
    """The scene with the fixture group deleted from the model (recompiled; qpos0 / keyframes without its joints).
    Cached on the scene dict."""
    ck = "_without_" + root
    if ck in sc:
        return sc[ck]
    m, R = sc["model"], sc["R"]
    d = mujoco.MjData(m)
    d.qpos[:] = R["qpos0"]
    mujoco.mj_forward(m, d)
    g = _fixture_group(m, d, root)
    if g is None:
        raise KeyError(f"open_cap: fixture {root!r} not in the scene")
    drop = np.zeros(m.nq, bool)
    for r in g.roots:
        a = m.jnt_qposadr[m.body_jntadr[r]]
        drop[a:a + 7] = True
    dropv = np.zeros(m.nv, bool)
    for r in g.roots:
        a = m.jnt_dofadr[m.body_jntadr[r]]
        dropv[a:a + 6] = True
    if sc.get("spec") is None:
        raise RuntimeError("open_cap: scene spec not kept; cannot remove the fixture")
    spec = sc["spec"].copy()
    gone = set()
    for b in g.bodies:
        gone.add(m.body(b).name)
    ggeo = {m.geom(i).name for i in g.geoms if m.geom(i).name}
    for el in list(spec.pairs):
        if el.geomname1 in ggeo or el.geomname2 in ggeo:
            spec.delete(el)
    for el in list(spec.excludes):
        if el.bodyname1 in gone or el.bodyname2 in gone:
            spec.delete(el)
    for n in g.names:
        spec.delete(spec.body(n))
    for k in spec.keys:
        if len(k.qpos) == m.nq:
            k.qpos = np.asarray(k.qpos)[~drop]
        if len(k.qvel) == m.nv:
            k.qvel = np.asarray(k.qvel)[~dropv]
    m2 = spec.compile()
    if m2.nq != int((~drop).sum()) or m2.nu != m.nu:
        raise RuntimeError(f"open_cap: fixture removal changed more than the group ({m.nq}->{m2.nq}, nu {m.nu}->{m2.nu})")
    R2 = dict(R)
    R2["qpos0"] = np.asarray(R["qpos0"])[~drop]
    act = np.asarray(R["actuator_ids"], int)
    if [m.actuator(int(i)).name for i in act] != [m2.actuator(int(i)).name for i in act]:
        raise RuntimeError("open_cap: actuator order changed by the fixture removal")
    out = dict(sc, model=m2, R=R2, fixture=None, removed=list(g.names))
    sc[ck] = out
    return out


def place_fixture(m, d, rng, root: str = FIXTURE, edge: float = 0.03, gap: float = 0.03, tries: int = 200) -> dict | None:
    """Fixture group at the BACK (+x) table edge, fully on the table, upright as built, at a uniformly random y
    that keeps ``gap`` from every robot point (arms at their reset pose).  The other free objects are re-placed by
    ``layout.scatter`` right after, so they are not obstacles here; but a group the new footprint would land on
    (within ``gap``) is first moved into the footprint the fixture vacated, so scatter still sees it as its own
    group instead of merging it into the fixture's kept support group.  None if no spot is clear of the arms or a
    covered group has no room in the vacated footprint."""
    from abc_sim import layout
    g = _fixture_group(m, d, root)
    if g is None:
        return None
    tc, ts, _ = layout.table_box(m, d)
    tlo, thi = tc - ts + edge, tc + ts - edge
    lo, hi = g.hull.min(0), g.hull.max(0)
    ylo, yhi = tlo[1] - lo[1], thi[1] - hi[1]           # shifts that keep the footprint inside the table
    if yhi < ylo:
        raise RuntimeError(f"open_cap: fixture {g.names} wider than the table")
    robot = layout._robot_bodies(m, layout.DEFAULT_ARMS)
    rp = layout._body_points(m, d, sorted(robot))[:, :2]
    others = layout._groups(m, d, robot, exclude=tuple(g.names))     # grouped before anything moves
    dx = thi[0] - hi[0]
    q0 = d.qpos.copy()

    def vacate(H):
        """Moves (group, centre) putting every group H lands on into the vacated footprint, or None."""
        covered = [o for o in others if layout._poly_dist(H, o.hull) < gap]
        placed, moves = [H] + [o.hull for o in others if o not in covered], []
        for o in sorted(covered, key=lambda o: -float(np.ptp(o.hull, 0).prod())):
            olo, ohi = g.hull.min(0) - o.hull.min(0) + o.centre, g.hull.max(0) - o.hull.max(0) + o.centre
            if np.any(ohi < olo):
                return None
            for _ in range(50):
                c = rng.uniform(olo, ohi)
                oh = o.hull - o.centre + c
                if all(layout._poly_dist(oh, p) >= gap for p in placed):
                    break
            else:
                return None
            placed.append(oh)
            moves.append((o, c))
        return moves

    n_arm = n_vac = 0
    for _ in range(int(tries)):                     # random y; a y whose covered groups do not fit is redrawn
        sh = np.array([dx, rng.uniform(ylo, yhi)])
        if min(layout._point_poly_dist(p, g.hull + sh) for p in rp) <= gap:
            n_arm += 1
            continue
        moves = vacate(g.hull + sh)
        if moves is not None:
            break
        n_vac += 1
    else:
        logger.warning("open_cap fixture: no y in %d tries (%d hit the arms, %d no room to vacate)", tries, n_arm, n_vac)
        return None
    for o, c in moves:
        layout._move(m, d, o, q0, c, 0.0)
    layout._move(m, d, g, q0, g.centre + sh, 0.0)
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    return dict(names=g.names, centre=[round(float(v), 4) for v in g.centre + sh],
                vacated=[o.names for o, _ in moves])


# ======================================================================================================== containers
def containers(scene: dict) -> list[dict]:
    """Every screw-cap container in the scene: dict(body, lid, family, colour, cap, variant, free_deg, lid_word)."""
    m, log, cat, audit = scene["model"], scene["log"], scene["catalog"], scene["audit"]
    out = []
    for j in ORIG_JARS:
        sw = log["swaps"].get(j)
        var = sw["variant"] if sw else None
        col = (cat[var].get("color") if var else ORIG_COLOUR[j]) or "paint"
        out.append(dict(body=f"obj_{j}", lid=f"obj_{j}__lid", family="paint_jar", colour=col, cap=None, variant=var))
    for b, a in log["adds"].items():
        c = cat[a["variant"]]
        out.append(dict(body=b, lid=f"{b}__lid", family=c["family"], colour=c.get("color"), cap=c.get("cap"),
                        variant=a["variant"]))
    for c in out:
        c["free_deg"] = float(audit.get(c["variant"] or "", {}).get("free_angle_deg") or DEFAULT_FREE_DEG)
        c["lid_word"] = "lid" if c["family"] == "paint_jar" else "cap"
        c["bid"], c["lid_bid"] = m.body(c["body"]).id, m.body(c["lid"]).id
        # rigid group of the container (tethered: + ring) -- contacts with these keep the lid "on"
        c["group"] = {c["bid"]} | {m.body(n).id for n in (f"{c['body']}__tring",) if _has_body(m, n)}
    return out


def _has_body(m, name: str) -> bool:
    try:
        m.body(name)
        return True
    except KeyError:
        return False


COLOUR_WORDS = {"sky": "light blue", "natural": "translucent", "mint": "mint green", "lime": "lime green",
                "cool_white": "white", "warm_white": "off-white"}


def _w(s: str | None) -> str:
    return COLOUR_WORDS.get(s or "", (s or "").replace("_", " "))


def noun(c: dict) -> str:
    if c["family"] == "paint_jar":
        return f"{_w(c['colour'])} paint jar"
    if not c.get("cap"):
        return f"{_w(c['colour'])} bottle with the {'tethered ' if c['family'] == 'tethered' else ''}cap"
    return f"{_w(c['colour'])} bottle with the {_w(c['cap'])} cap"


def phrase(items: list[dict], k: int) -> str:
    """Unambiguous phrase for items[k] (dict noun, y).  Robot frame: +y = the robot's left."""
    t = items[k]
    same = [i for i, it in enumerate(items) if it["noun"] == t["noun"]]
    if len(same) == 1:
        return f"the {t['noun']}"
    ys = sorted(((items[i]["y"], i) for i in same), reverse=True)
    pos = [i for _, i in ys].index(k)
    if len(same) == 2:
        return f"the {t['noun']} on the {'left' if pos == 0 else 'right'}"
    word = "leftmost" if pos == 0 else "rightmost" if pos == len(same) - 1 else None
    return f"the {word} {t['noun']}" if word else f"the {t['noun']} in the middle"


# ======================================================================================================== evaluator
class CapEvaluator:
    """Per-physics-step lid tracker for every container; reward / success / failures for the target."""

    def __init__(self, m: mujoco.MjModel, cs: list[dict], target: int, table_z: float):
        self.cs, self.target, self.table_z = cs, int(target), float(table_z)
        names = [m.body(b).name for b in range(m.nbody)]
        self.robot = np.array([n.startswith(("left_", "right_", "pad_left", "pad_right")) for n in names])
        self.pads = {arm: {m.body(f"pad_{arm}_tip_left").id, m.body(f"pad_{arm}_tip_right").id}
                     for arm in ("left", "right")}
        self.reset_counters()

    def reset_counters(self):
        n = len(self.cs)
        self.twist = np.zeros(n)
        self.max_twist = np.zeros(n)
        self.prev = np.zeros(n)
        self.nfree = np.zeros(n, int)
        self.free = np.zeros(n, bool)
        self.seat = [None] * n
        self.z0 = np.zeros(n)
        self.touch_lid = self.pinch_lid = self.hold_jar = False
        self.failures: list[dict] = []
        self.ever_success = False
        self.dt = None

    def _rel(self, d, c):
        Rj = d.xmat[c["bid"]].reshape(3, 3)
        Rl = d.xmat[c["lid_bid"]].reshape(3, 3)
        Rr = Rj.T @ Rl
        p = Rj.T @ (d.xpos[c["lid_bid"]] - d.xpos[c["bid"]])
        return float(np.arctan2(Rr[1, 0], Rr[0, 0])), p, float(Rr[2, 2])

    def after_reset(self, m, d):
        self.reset_counters()
        self.dt = float(m.opt.timestep)
        for i, c in enumerate(self.cs):
            th, p, _ = self._rel(d, c)
            self.prev[i], self.seat[i], self.z0[i] = th, p, float(d.xpos[c["bid"]][2])

    def after_physics_step(self, m, d):
        nc = d.ncon
        b1 = m.geom_bodyid[d.contact.geom1[:nc]]
        b2 = m.geom_bodyid[d.contact.geom2[:nc]]
        pairs = set(zip(b1.tolist(), b2.tolist())) | set(zip(b2.tolist(), b1.tolist()))
        tgt = self.cs[self.target]
        for i, c in enumerate(self.cs):
            th, p, cz = self._rel(d, c)
            dth = (th - self.prev[i] + np.pi) % (2 * np.pi) - np.pi
            self.prev[i] = th
            if cz > 0.85:                       # lid roughly coaxial: count the twist about the container axis
                self.twist[i] += dth
                self.max_twist[i] = max(self.max_twist[i], self.twist[i])
            touching = any((c["lid_bid"], g) in pairs for g in c["group"])
            off = float(np.linalg.norm(p - self.seat[i])) >= LIFT_FREE
            self.nfree[i] = 0 if (touching or not off) else self.nfree[i] + 1
            if self.nfree[i] * self.dt >= DT_FREE and not self.free[i]:
                self.free[i] = True
                if i != self.target:
                    self._fail(f"opened_wrong_container:{c['body'][4:]}", robot=True)
        lid_hits = {b for (a, b) in pairs if a == tgt["lid_bid"]}
        jar_hits = {b for (a, b) in pairs if a == tgt["bid"]}
        if any(self.robot[b] for b in lid_hits):
            self.touch_lid = True
        pinch = [arm for arm, pd in self.pads.items() if pd <= lid_hits]
        hold = [arm for arm, pd in self.pads.items() if pd <= jar_hits]
        self.pinch_lid |= bool(pinch)
        self.hold_jar |= bool(hold) and (not pinch or bool(set(hold) - set(pinch)))
        if self.free[self.target]:
            self.ever_success = True
        Rj = d.xmat[tgt["bid"]].reshape(3, 3)
        if not self.ever_success and (Rj[2, 2] < 0.5 or d.xpos[tgt["bid"]][2] < self.table_z - 0.05):
            self._fail("target_knocked_over")

    def _fail(self, why: str, robot: bool = False):
        if not any(f["why"] == why for f in self.failures):
            self.failures.append(dict(why=why, robot=robot))

    @property
    def terminated(self) -> bool:
        return bool(self.ever_success or self.failures)

    @property
    def robot_failed(self) -> bool:
        return any(f["robot"] for f in self.failures)

    def progress(self) -> float:
        c = self.cs[self.target]
        tw = self.max_twist[self.target] - np.deg2rad(TWIST_DEADBAND_DEG)      # contact jitter is not progress
        return float(np.clip(tw / np.deg2rad(c["free_deg"] - TWIST_DEADBAND_DEG), 0.0, 1.0))

    def evaluate(self) -> TaskEvalResult:
        if self.ever_success:
            r = 1.0
        elif self.robot_failed:
            r = -0.5
        elif self.failures:
            r = 0.0
        else:
            r = 0.05 * self.touch_lid + 0.1 * self.pinch_lid + 0.1 * self.hold_jar + 0.65 * self.progress()
        t = self.target
        met = dict(target=self.cs[t]["body"][4:], target_variant=self.cs[t]["variant"],
                   twist_deg=float(np.rad2deg(self.twist[t])), max_twist_deg=float(np.rad2deg(self.max_twist[t])),
                   free_angle_deg=self.cs[t]["free_deg"], progress=self.progress(), touched_lid=self.touch_lid,
                   pinched_lid=self.pinch_lid, held_container=self.hold_jar, lid_free=bool(self.free[t]),
                   opened=[c["body"][4:] for c, f in zip(self.cs, self.free) if f],
                   failures=[f["why"] for f in self.failures], robot_touch_failure=self.robot_failed)
        return TaskEvalResult(reward=np.asarray([r], np.float32), success=np.asarray([self.ever_success]),
                              metrics=met)


# ======================================================================================================== env
class CapAbcEnv:
    """abc_sim env surface (abc_sim.plugins.SimEnvSurface) for open_cap.  step() = ONE physics step."""

    # policy key -> model camera: the standard abc_sim station cameras (or a one-model YAM variant's), see abc_hw
    POLICY_CAMERAS = dict(abc_hw.POLICY_CAMERAS)
    CAMERAS = tuple(POLICY_CAMERAS.values())

    def __init__(self, spec: SimTaskSpec | None = None, *, seed: int = 0, prompt: str | None = None,
                 max_seconds: float = 60.0, reward_mode: str = "native", terminate_on_success: bool = True,
                 terminate_on_failure: bool = True, task_kwargs: dict | None = None, **_ignored: Any):
        self.spec, self.fixed_prompt = spec, prompt
        self.max_seconds, self.reward_mode = float(max_seconds), reward_mode
        self.term_success, self.term_failure = bool(terminate_on_success), bool(terminate_on_failure)
        self.opts = dict(task_kwargs or {})
        self.prompt: str | None = None
        self.cur_step, self.seed, self.scene = 0, None, None
        self.model = self.data = None
        # one-model hardware randomization: variant_cfg / environment_factory=partial(..., variant_cfg=) / hardware_variant
        self._hw_kwargs = {k: _ignored[k] for k in ("variant_cfg", "environment_factory", "hardware_variant") if k in _ignored}
        self.variant = None
        self._build(int(seed))

    def _build(self, seed: int) -> None:
        self.variant = abc_hw.resolve_variant(self._hw_kwargs, self.opts, seed)
        grip = abc_hw.gripper_kind(self._hw_kwargs, self.variant)
        if grip is not None and not getattr(self, "_grip_warned", False):
            self._grip_warned = True
            logger.warning("open_cap: gripper %r requested by the hardware variant; the portable keeps its own "
                           "(real-calibrated) fingers -- cameras / station hardware follow the variant", grip)
        sc = build(seed, {k: v for k, v in self.opts.items() if k in ("p_swap", "max_adds", "stickers")}, self.variant)
        fx = self.opts.get("fixture", FIXTURE)
        p_fx = float(self.opts.get("p_fixture", P_FIXTURE))
        self.fixture_present = bool(fx) and np.random.default_rng([seed, 5521]).random() < p_fx
        if fx and not self.fixture_present:
            sc = without_fixture(sc, fx)
        if self.scene is not sc:
            self.scene, self.model = sc, sc["model"]
            self.data = mujoco.MjData(self.model)
            R = sc["R"]
            self.joint_offset = np.asarray(R["joint_offset"], float)
            self.grip_idx = np.asarray(R["grip_idx"], int)
            self.grip_scale = float(R["grip_scale"])
            self.act_ids = np.asarray(R["actuator_ids"], int)
            self.act_bias = np.asarray(R["act_bias"], float)
            self.qadr = self.model.jnt_qposadr[self.model.actuator_trnid[self.act_ids, 0]]
            self.cs = containers(sc)
            missing = [c for c in self.POLICY_CAMERAS.values()
                       if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, c) < 0]
            if missing:
                raise RuntimeError(f"policy cameras {missing} missing from the compiled scene")
        m, d, R = self.model, self.data, sc["R"]
        mujoco.mj_resetData(m, d)
        d.qpos[:] = R["qpos0"]
        d.qvel[:] = 0
        d.ctrl[:] = np.asarray(R["ctrl"][0], float)
        mujoco.mj_forward(m, d)
        self.layout_info = None
        self.fixture_info = None
        if self.opts.get("layout", "scatter") not in (None, False, "package", "none"):
            from abc_sim import layout   # reach-gated absolute + relative layout randomization (shared rule)
            if self.fixture_present:     # back edge, random y; then an obstacle the other groups scatter around
                self.fixture_info = place_fixture(m, d, np.random.default_rng([seed, 5522]), fx)
                if self.fixture_info is None:
                    raise RuntimeError(f"open_cap seed {seed}: no back-edge spot for fixture {fx!r} clear of the arms")
            self.layout_info = layout.scatter(m, d, np.random.default_rng([seed, 7741]),
                                              keep=(fx,) if self.fixture_present else ())
        rng = np.random.default_rng([seed, 9311])
        target = int(self.opts["target"]) if "target" in self.opts else int(rng.integers(len(self.cs)))
        tz = float(np.min([d.xpos[c["bid"]][2] for c in self.cs]))
        self.evaluator = CapEvaluator(m, self.cs, target, tz)
        self.evaluator.after_reset(m, d)
        self.seed, self.cur_step, self.t0 = seed, 0, float(d.time)
        items = [dict(noun=noun(c), y=float(d.xpos[c["bid"]][1])) for c in self.cs]
        self.target_phrase = phrase(items, target)
        self.prompt = self.fixed_prompt or str(rng.choice(PROMPTS)).format(
            obj=self.target_phrase, lid=self.cs[target]["lid_word"])

    # ---------------------------------------------------------------------------------------------- env surface
    def reset(self, seed: int | None = None, options: dict | None = None, randomize: bool = True):
        seed = int(self.seed if (seed is None or not randomize) else seed)
        self._build(seed)
        return self._obs(), self._info()

    def _ctrl(self, action) -> np.ndarray:
        a = np.asarray(action, float).reshape(-1)
        if a.shape[0] != len(self.act_ids):
            raise ValueError(f"action must have {len(self.act_ids)} entries")
        q = a + self.joint_offset
        q[self.grip_idx] *= self.grip_scale
        lo, hi = self.model.actuator_ctrlrange[self.act_ids].T
        lim = self.model.actuator_ctrllimited[self.act_ids].astype(bool)
        c = q + self.act_bias
        return np.where(lim, np.clip(c, lo, hi), c)

    def apply_action(self, action, physics_steps: int = 1) -> None:
        m, d, ev = self.model, self.data, self.evaluator
        d.ctrl[self.act_ids] = self._ctrl(action)
        for _ in range(int(physics_steps)):
            mujoco.mj_step(m, d)
            ev.after_physics_step(m, d)
            self.cur_step += 1
            if ev.terminated:
                break

    def step(self, action, render_obs: bool = True):
        self.apply_action(action, 1)
        info = self._info()
        ev = self.evaluator
        terminated = bool((ev.ever_success and self.term_success) or (ev.failures and self.term_failure))
        truncated = (not terminated) and bool(info["time_limit_reached"])
        return self._obs(), float(info["task_reward"]), terminated, truncated, info

    def _map_reward(self, r: float) -> float:
        return (r + 0.5) / 1.5 if self.reward_mode == "shaped01" else r

    def evaluate_task(self) -> TaskEvalResult:
        r = self.evaluator.evaluate()
        return TaskEvalResult(reward=np.asarray([self._map_reward(float(r.reward[0]))], np.float32),
                              success=r.success, metrics=dict(r.metrics))

    def _info(self) -> dict:
        res = self.evaluator.evaluate()
        met = dict(res.metrics)
        r = self._map_reward(float(res.reward[0]))
        t = float(self.data.time - self.t0)
        failed = bool(self.evaluator.failures)
        met.update(ever_failed=failed, fail_reason=";".join(f["why"] for f in self.evaluator.failures) or None,
                   reward=r, prompt=self.prompt,
                   scene=dict(seed=self.seed, swaps=self.scene["swaps"], adds=[a for a, _ in self.scene["adds"]],
                             fixture=self.fixture_info if self.fixture_present else None))
        return dict(task_reward=r, task_success=bool(res.success[0]), task_failed=failed, task_eval=met,
                    episode_step=self.cur_step, sim_time=t, time_limit_reached=t >= self.max_seconds,
                    robot_touch_failure=bool(met["robot_touch_failure"]))

    def _obs(self) -> dict:
        return dict(prompt=self.prompt, state=self.get_state(), qpos=self.data.qpos[self.qadr].copy(),
                    time=float(self.data.time - self.t0))

    def get_state(self) -> np.ndarray:
        q = self.data.qpos[self.qadr].astype(np.float64) - self.act_bias - self.joint_offset
        q[self.grip_idx] = (self.data.qpos[self.qadr][self.grip_idx] - self.act_bias[self.grip_idx]) / self.grip_scale \
            - self.joint_offset[self.grip_idx]
        return q

    def hold_action(self) -> np.ndarray:
        """Dataset-unit action reproducing the reset ctrl (a do-nothing policy)."""
        c = np.asarray(self.scene["R"]["ctrl"][0], float)[self.act_ids] - self.act_bias
        c[self.grip_idx] /= self.grip_scale
        return c - self.joint_offset

    def forget_arm_state(self) -> None:
        pass

    def render(self, cam: str = "top", w: int = 640, h: int = 480, renderer=None):
        r = renderer or mujoco.Renderer(self.model, h, w)
        r.update_scene(self.data, camera=self.POLICY_CAMERAS.get(cam, cam))
        img = r.render()
        if renderer is None:
            r.close()
        return img

    def close(self) -> None:
        self.data = None


# ======================================================================================================== registration
def _make(*, task=None, prompt=None, seed=0, max_episode_steps=None, terminate_on_success=True,
          terminate_on_failure=True, reward_mode="native", task_kwargs=None, max_seconds=None, **_ignored):
    if max_seconds is None:   # max_episode_steps counts physics steps (2 ms) when given
        max_seconds = 60.0 if not max_episode_steps else float(max_episode_steps) * 0.002
    return CapAbcEnv(task if isinstance(task, SimTaskSpec) else None, seed=int(seed or 0), prompt=prompt,
                     max_seconds=max_seconds, reward_mode=reward_mode, terminate_on_success=terminate_on_success,
                     terminate_on_failure=terminate_on_failure, task_kwargs=task_kwargs)


def register(replace: bool = True) -> None:
    spec = SimTaskSpec(name=TASK, env_task=TASK, prompt="Unscrew and remove the cap of the named container",
                       aliases=("open the bottle cap", "open cap", "unscrew the cap"), max_chunks=120,
                       description="open a screw-cap paint jar / bottle / tethered-cap bottle (mujoco 4, per-seed "
                                   "scene from 70 audited variants; the env prompt names the target)")
    register_task(ExternalTask(env_task=TASK, specs=(spec,), env_factory=_make, assets=(PACK,),
                               description=spec.description), replace=replace)


__all__ = ["PACK", "TASK", "CapAbcEnv", "CapEvaluator", "build", "containers", "noun", "phrase", "register"]
