"""Reach-gated spatial layout randomization for table-top scenes (shared by task plugins).

Rule for new tasks: objects are scattered in absolute table coordinates AND in their relative placement and direction
(any yaw, any left/right/front/back order), so a policy cannot rely on a fixed "tray left of bag" arrangement.  Every
placed object stays where at least one arm can reach it with a top-down-ish grasp (inverse kinematics on the scene's own
robot), so seeds never put objects in corners the arms cannot reach.

What moves: every free-joint root body (with all its children, e.g. flex vertex bodies of a paper bag).  Free bodies
resting on another free body (bread on a tray, books in a holder, a lid on a jar) are grouped with their support by
contact / proximity (or an equality constraint) and moved rigidly together; a group is a rigid SE(2) move (xy + yaw about z) of all its roots, so its height and
the way it rests are kept.

Checks for each layout: convex footprints (from geom bounding boxes) inside the table and apart by ``gap``; every root
of a group inside the eroded reach map of some arm at its height; clear of the arm bases; no contact with the robot or
between groups after the move; and the scene stays at rest over a short settle (no group moves more than
``settle_tol``).  A layout that fails is resampled; after ``restarts`` tries the original layout is kept (logged).

Use::

    from abc_sim import layout
    info = layout.scatter(model, data, np.random.default_rng([seed, 7741]))   # after the scene has settled
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

logger = logging.getLogger(__name__)

DOWN = np.array([0.0, 0.0, -1.0])


# ---------------------------------------------------------------------------------------------------- geometry
def _cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _hull(pts: np.ndarray) -> np.ndarray:
    pts = np.unique(np.round(pts, 6), axis=0)
    if len(pts) < 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def half(ps):
        out = []
        for p in ps:
            while len(out) >= 2 and _cross2(out[-1] - out[-2], p - out[-2]) <= 0:
                out.pop()
            out.append(p)
        return out[:-1]

    return np.array(half(pts) + half(pts[::-1]))


def _poly_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Signed separation of two convex polygons along the best SAT axis (>0 apart, <0 overlapping)."""
    best = -np.inf
    for P in (a, b):
        e = np.roll(P, -1, axis=0) - P
        n = np.stack([e[:, 1], -e[:, 0]], 1)
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        pa, pb = a @ n.T, b @ n.T
        sep = np.maximum(pb.min(0) - pa.max(0), pa.min(0) - pb.max(0))
        best = max(best, float(sep.max()))
    return best


def _point_poly_dist(p: np.ndarray, P: np.ndarray) -> float:
    e = np.roll(P, -1, axis=0) - P
    t = np.clip(np.einsum("ij,ij->i", p - P, e) / (np.einsum("ij,ij->i", e, e) + 1e-12), 0, 1)
    d = float(np.min(np.linalg.norm(P + t[:, None] * e - p, axis=1)))
    inside = np.all(_cross2(e, p - P) >= 0)
    return -d if inside else d


def _rotz(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


# ---------------------------------------------------------------------------------------------------- scene parsing
def _subtree(m, root: int) -> list[int]:
    out = []
    for b in range(root, m.nbody):
        p = b
        while p and p != root:
            p = m.body_parentid[p]
        if p == root:
            out.append(b)
    return out


def _is_free_root(m, b: int) -> bool:
    return (m.body_parentid[b] == 0 and m.body_jntnum[b] > 0
            and int(m.jnt_type[m.body_jntadr[b]]) == int(mujoco.mjtJoint.mjJNT_FREE))


def _geom_corners(m, d, g: int) -> np.ndarray:
    c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
    s = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float)
    loc = c + s * h
    return d.geom_xpos[g] + loc @ d.geom_xmat[g].reshape(3, 3).T


def _body_points(m, d, bodies: list[int]) -> np.ndarray:
    pts = [d.xpos[b][None] for b in bodies]
    for b in bodies:
        for g in range(m.body_geomadr[b], m.body_geomadr[b] + m.body_geomnum[b]):
            if m.geom_contype[g] or m.geom_conaffinity[g] or m.geom_group[g] < 3:
                pts.append(_geom_corners(m, d, g))
    return np.concatenate(pts, 0)


@dataclass
class _Group:
    roots: list[int]
    bodies: list[int]
    geoms: set
    pts: np.ndarray        # world points of the footprint (3d)
    hull: np.ndarray       # xy hull, world
    centre: np.ndarray     # xy
    names: list[str]


def _support_groups(m, d, roots: list[int], near: float = 0.003) -> list[list[int]]:
    """Union free roots that belong together: touching (any contact) or within ``near`` of each other (a lid on its
    thread, books standing in a holder, bread on a tray), or coupled by an equality constraint (a tethered cap's
    strap pinned to its lid)."""
    owner = {}
    for r in roots:
        for b in _subtree(m, r):
            owner[b] = r
    parent = {r: r for r in roots}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        if a is not None and b is not None and a != b:
            parent[find(a)] = find(b)

    for i in range(d.ncon):
        c = d.contact[i]
        if c.geom[0] >= 0 and c.geom[1] >= 0:
            union(owner.get(m.geom_bodyid[c.geom[0]]), owner.get(m.geom_bodyid[c.geom[1]]))
    for i in range(m.neq):
        t = int(m.eq_type[i])
        if t in (int(mujoco.mjtEq.mjEQ_CONNECT), int(mujoco.mjtEq.mjEQ_WELD)):
            if int(m.eq_objtype[i]) == int(mujoco.mjtObj.mjOBJ_SITE):
                b1, b2 = m.site_bodyid[m.eq_obj1id[i]], m.site_bodyid[m.eq_obj2id[i]]
            else:
                b1, b2 = m.eq_obj1id[i], m.eq_obj2id[i]
        elif t == int(mujoco.mjtEq.mjEQ_JOINT):
            b1 = m.jnt_bodyid[m.eq_obj1id[i]]
            b2 = m.jnt_bodyid[m.eq_obj2id[i]] if m.eq_obj2id[i] >= 0 else -1
        else:
            continue
        union(owner.get(int(b1)), owner.get(int(b2)))
    geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in owner and (m.geom_contype[g] or m.geom_conaffinity[g])]
    if geoms:
        box = np.array([np.r_[c.min(0), c.max(0)] for c in (_geom_corners(m, d, g) for g in geoms)])
        order = np.argsort(box[:, 0])
        fromto = np.zeros(6)
        for ii, i in enumerate(order):
            gi, ri = geoms[i], owner[m.geom_bodyid[geoms[i]]]
            for j in order[ii + 1:]:
                if box[j, 0] > box[i, 3] + near:
                    break
                gj = geoms[j]
                rj = owner[m.geom_bodyid[gj]]
                if find(ri) == find(rj) or np.any(box[j, :3] > box[i, 3:] + near) or np.any(box[i, :3] > box[j, 3:] + near):
                    continue
                if mujoco.mj_geomDistance(m, d, gi, gj, near, fromto) < near:
                    union(ri, rj)
    groups: dict[int, list[int]] = {}
    for r in roots:
        groups.setdefault(find(r), []).append(r)
    return list(groups.values())


def _robot_bodies(m, arms) -> set[int]:
    out = set()
    for arm in arms:
        b = m.site_bodyid[m.site(arm.grasp_site).id]
        while m.body_parentid[b]:
            b = m.body_parentid[b]
        out |= set(_subtree(m, b))
    return out


def table_geom(m) -> int:
    """The table-top box: a world geom named ``table`` / ``bg_table`` / ``*table*``, else the largest world box."""
    names = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "": g for g in range(m.ngeom)}
    for n in ("table", "bg_table"):
        if n in names:
            return names[n]
    boxes = [g for g in range(m.ngeom) if m.geom_bodyid[g] == 0 and int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_BOX)]
    named = [g for g in boxes if "table" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "")]
    pool = named or boxes
    if not pool:
        raise ValueError("layout: no table geom found")
    return max(pool, key=lambda g: float(m.geom_size[g][0] * m.geom_size[g][1]))


def table_box(m, d) -> tuple[np.ndarray, np.ndarray, float]:
    """``(centre xy, half extents xy, top z)`` of the table top, in world (works for box and mesh tables)."""
    tid = table_geom(m)
    if int(m.geom_type[tid]) == int(mujoco.mjtGeom.mjGEOM_BOX):
        return d.geom_xpos[tid][:2].copy(), m.geom_size[tid][:2].copy(), float(d.geom_xpos[tid][2] + m.geom_size[tid][2])
    c = _geom_corners(m, d, tid)
    lo, hi = c.min(0), c.max(0)
    return (lo[:2] + hi[:2]) / 2, (hi[:2] - lo[:2]) / 2, float(hi[2])


# ---------------------------------------------------------------------------------------------------- reach map
@dataclass
class Arm:
    grasp_site: str
    tcp_site: str


DEFAULT_ARMS = (Arm("left_grasp_site", "left_tcp_site"), Arm("right_grasp_site", "right_tcp_site"))


def _arm_chain(m, arm: Arm) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    b = m.site_bodyid[m.site(arm.grasp_site).id]
    joints = []
    while b:
        joints += list(range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b]))
        b = m.body_parentid[b]
    joints = [j for j in sorted(joints) if int(m.jnt_type[j]) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))]
    base_body = m.jnt_bodyid[joints[0]]
    return np.array(joints), np.array([m.jnt_dofadr[j] for j in joints]), np.array([m.jnt_qposadr[j] for j in joints]), base_body


def _ik(m, d, arm, chain, target, approach, q0, iters=60):
    joints, dofs, qadr, _ = chain
    gs, ts = m.site(arm.grasp_site).id, m.site(arm.tcp_site).id
    lo, hi = m.jnt_range[joints].T
    lim = m.jnt_limited[joints].astype(bool)
    q = q0.copy()
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    for _ in range(iters):
        d.qpos[qadr] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        pg, pt = d.site_xpos[gs], d.site_xpos[ts]
        a = (pg - pt) / (np.linalg.norm(pg - pt) + 1e-12)
        ep, er = target - pg, np.cross(a, approach)
        if np.linalg.norm(ep) < 0.002 and np.linalg.norm(er) < 0.03:
            break
        mujoco.mj_jacSite(m, d, jp, jr, gs)
        J = np.vstack([jp[:, dofs], 0.3 * jr[:, dofs]])
        e = np.concatenate([ep, 0.3 * er])
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
        n = np.linalg.norm(dq)
        q = q + dq * min(1.0, 0.3 / (n + 1e-12))
        q = np.where(lim, np.clip(q, lo, hi), q)
    d.qpos[qadr] = q
    mujoco.mj_kinematics(m, d)
    pg, pt = d.site_xpos[gs], d.site_xpos[ts]
    a = (pg - pt) / (np.linalg.norm(pg - pt) + 1e-12)
    return q, float(np.linalg.norm(target - pg)), float(np.degrees(np.arccos(np.clip(a @ DOWN, -1, 1))))


@dataclass
class ReachMap:
    xs: np.ndarray
    ys: np.ndarray
    zs: np.ndarray
    ok: np.ndarray          # (arm, z, x, y) bool, eroded

    def reachable(self, p: np.ndarray) -> bool:
        iz = int(np.argmin(np.abs(self.zs - p[2])))
        ix, iy = int(np.round((p[0] - self.xs[0]) / (self.xs[1] - self.xs[0]))), \
            int(np.round((p[1] - self.ys[0]) / (self.ys[1] - self.ys[0])))
        if not (0 <= ix < len(self.xs) and 0 <= iy < len(self.ys)):
            return False
        return bool(self.ok[:, iz, ix, iy].any())

    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        any_ok = self.ok.any((0, 1))
        ix, iy = np.nonzero(any_ok)
        if not len(ix):
            raise ValueError("layout: the reach map is empty (no table point reachable by any arm)")
        return np.array([self.xs[ix.min()], self.ys[iy.min()]]), np.array([self.xs[ix.max()], self.ys[iy.max()]])


_REACH_CACHE: dict[str, ReachMap] = {}


def reach_map(m, d, arms=DEFAULT_ARMS, *, step=0.04, zs=(0.04, 0.12), max_tilt=45.0, tol=0.005,
              erode=1) -> ReachMap:
    """Per-arm map of table points a top-down-ish grasp (approach within ``max_tilt`` of vertical, position error
    < ``tol``) reaches, at heights ``zs`` above the table top; eroded by ``erode`` cells as a margin.  Cached per robot
    (in memory and under ``$ABC_SIM_CACHE_DIR/layout_reach``)."""
    tc, ts, top = table_box(m, d)
    chains = [_arm_chain(m, a) for a in arms]
    key_src = [top, step, zs, max_tilt, tol, erode]
    for a, (joints, _, qadr, base) in zip(arms, chains):
        key_src += [a.grasp_site, d.xpos[base].round(4).tolist(), m.jnt_range[joints].round(4).tolist(),
                    m.site_pos[m.site(a.grasp_site).id].round(4).tolist(), m.body_pos[m.jnt_bodyid[joints]].round(4).tolist()]
    key = hashlib.sha1(repr(key_src).encode()).hexdigest()[:16]
    if key in _REACH_CACHE:
        return _REACH_CACHE[key]
    cdir = Path(os.environ.get("ABC_SIM_CACHE_DIR", "/tmp/abc_cache")) / "layout_reach"
    f = cdir / f"{key}.npz"
    if f.exists():
        z = np.load(f)
        rm = ReachMap(z["xs"], z["ys"], z["zs"], z["ok"])
        _REACH_CACHE[key] = rm
        return rm
    xs = np.arange(tc[0] - ts[0] + step / 2, tc[0] + ts[0], step)
    ys = np.arange(tc[1] - ts[1] + step / 2, tc[1] + ts[1], step)
    zw = top + np.asarray(zs, float)
    raw = np.zeros((len(arms), len(zw), len(xs), len(ys)), bool)
    d2 = mujoco.MjData(m)
    d2.qpos[:] = d.qpos
    for ai, (arm, chain) in enumerate(zip(arms, chains)):
        base = d.xpos[chain[3]][:2]
        home = d.qpos[chain[2]].copy()
        for zi, z in enumerate(zw):
            q_prev = home
            for ix, x in enumerate(xs):
                for iy, y in enumerate(ys if ix % 2 == 0 else ys[::-1]):
                    jy = iy if ix % 2 == 0 else len(ys) - 1 - iy
                    p = np.array([x, y, z])
                    r = np.linalg.norm(p[:2] - base)
                    if r < 0.12 or r > 0.75:
                        continue
                    radial = np.append((p[:2] - base) / r, 0.0)
                    for tilt in (np.radians(30), 0.0, np.radians(max_tilt)):
                        app = np.cos(tilt) * DOWN + np.sin(tilt) * radial
                        for q0 in (q_prev, home):
                            q, err, ang = _ik(m, d2, arm, chain, p, app, q0)
                            if err < tol and ang <= max_tilt + 1e-6:
                                raw[ai, zi, ix, jy] = True
                                q_prev = q
                                break
                        if raw[ai, zi, ix, jy]:
                            break
    ok = raw.copy()
    for _ in range(erode):
        e = ok.copy()
        e[..., 1:, :] &= ok[..., :-1, :]
        e[..., :-1, :] &= ok[..., 1:, :]
        e[..., :, 1:] &= ok[..., :, :-1]
        e[..., :, :-1] &= ok[..., :, 1:]
        ok = e
    rm = ReachMap(xs, ys, zw, ok)
    cdir.mkdir(parents=True, exist_ok=True)
    np.savez(f, xs=xs, ys=ys, zs=zw, ok=ok, raw=raw)
    _REACH_CACHE[key] = rm
    return rm


# ---------------------------------------------------------------------------------------------------- scatter
def _groups(m, d, robot: set[int], exclude=()) -> list[_Group]:
    roots = [b for b in range(1, m.nbody) if _is_free_root(m, b) and b not in robot
             and mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) not in exclude]
    out = []
    for rs in _support_groups(m, d, roots):
        bodies = [b for r in rs for b in _subtree(m, r)]
        pts = _body_points(m, d, bodies)
        hull = _hull(pts[:, :2])
        geoms = {g for b in bodies for g in range(m.body_geomadr[b], m.body_geomadr[b] + m.body_geomnum[b])}
        out.append(_Group(rs, bodies, geoms, pts, hull, hull.mean(0),
                          [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, r) for r in rs]))
    return out


def _move(m, d, g: _Group, q_orig: np.ndarray, centre: np.ndarray, yaw: float) -> None:
    R = _rotz(yaw)
    qz = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    for r in g.roots:
        a = m.jnt_qposadr[m.body_jntadr[r]]
        p = q_orig[a:a + 3]
        d.qpos[a:a + 2] = centre + R @ (p[:2] - g.centre)
        d.qpos[a + 2] = p[2]
        qn = np.zeros(4)
        mujoco.mju_mulQuat(qn, qz, q_orig[a + 3:a + 7])
        d.qpos[a + 3:a + 7] = qn


def scatter(m, d, rng: np.random.Generator, *, arms=DEFAULT_ARMS, gap=0.025, edge=0.03, base_clear=0.14,
            tries=200, restarts=12, settle_steps=150, settle_tol=0.015, exclude=(), keep=()) -> dict:
    """Scatter the free objects of a settled scene in place (see the module docstring).  ``exclude``: body names that
    are left out entirely; ``keep``: names of roots whose group stays where it is (an obstacle for the others).
    Returns ``{"moved": bool, "groups": [...], "attempts": n}``."""
    mujoco.mj_forward(m, d)
    robot = _robot_bodies(m, arms)
    groups = _groups(m, d, robot, exclude)
    rm = reach_map(m, d, arms)
    tc, ts, _ = table_box(m, d)
    tlo, thi = tc - ts + edge, tc + ts - edge
    lo, hi = rm.bbox()
    lo, hi = np.maximum(lo, tlo), np.minimum(hi, thi)
    bases = [d.xpos[_arm_chain(m, a)[3]][:2].copy() for a in arms]
    state0 = np.zeros(mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(m, d, state0, mujoco.mjtState.mjSTATE_INTEGRATION)
    q0 = d.qpos.copy()
    fixed = [g for g in groups if set(g.names) & set(keep)]
    movers = [g for g in groups if g not in fixed]
    movers.sort(key=lambda g: -float(np.ptp(g.hull, 0).prod()))     # big first, order then shuffled within ties
    robot_geoms = {g for b in robot for g in range(m.body_geomadr[b], m.body_geomadr[b] + m.body_geomnum[b])}
    owner = {gg: i for i, g in enumerate(movers + fixed) for gg in g.geoms}
    root_rel = {id(g): [(q0[m.jnt_qposadr[m.body_jntadr[r]]:m.jnt_qposadr[m.body_jntadr[r]] + 3] -
                         np.append(g.centre, 0)) for r in g.roots] for g in movers}
    for attempt in range(1, restarts + 1):
        mujoco.mj_setState(m, d, state0, mujoco.mjtState.mjSTATE_INTEGRATION)
        order = list(movers)
        if attempt > 1 or rng.random() < 0.5:
            rng.shuffle(order)
        placed = [g.hull for g in fixed]
        poses, fail = {}, None
        for g in order:
            ok = False
            for _ in range(tries):
                c = rng.uniform(lo, hi)
                yaw = float(rng.uniform(-np.pi, np.pi))
                R = _rotz(yaw)
                hull = c + (g.hull - g.centre) @ R.T
                if np.any(hull < tlo) or np.any(hull > thi):
                    continue
                if any(_point_poly_dist(b, hull) < base_clear for b in bases):
                    continue
                if any(_poly_dist(hull, h) < gap for h in placed):
                    continue
                roots_w = [np.append(c + R @ rel[:2], rel[2]) for rel in root_rel[id(g)]]
                if not all(rm.reachable(p) for p in roots_w):
                    continue
                ok = True
                break
            if not ok:
                fail = f"no spot for {g.names}"
                break
            placed.append(hull)
            poses[id(g)] = (c, yaw)
        if fail:
            continue
        for g in movers:
            _move(m, d, g, q0, *poses[id(g)])
        d.qvel[:] = 0
        try:
            mujoco.mj_forward(m, d)
        except mujoco.FatalError as e:      # e.g. a contact explosion overflowing the arena
            fail = f"forward failed ({str(e).splitlines()[0]})"
            continue
        bad = None
        for i in range(d.ncon):
            c = d.contact[i]
            if c.geom[0] < 0 or c.geom[1] < 0 or c.dist > 0:
                continue
            a, b = int(c.geom[0]), int(c.geom[1])
            if (a in robot_geoms and b in owner) or (b in robot_geoms and a in owner):
                bad = "robot contact"
                break
            if a in owner and b in owner and owner[a] != owner[b] and c.dist < -0.002:
                bad = "objects overlap"
                break
        if bad:
            fail = bad
            continue
        before = {r: d.xpos[r].copy() for g in movers for r in g.roots}
        try:
            for _ in range(settle_steps):
                mujoco.mj_step(m, d)
        except mujoco.FatalError as e:
            fail = f"settle failed ({str(e).splitlines()[0]})"
            continue
        drift = max(float(np.linalg.norm(d.xpos[r] - p)) for r, p in before.items()) if before else 0.0
        if not np.isfinite(d.qpos).all() or drift > settle_tol:
            fail = f"unsettled drift {drift:.3f}"
            continue
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        info = [dict(names=g.names, xy=[round(float(v), 4) for v in poses[id(g)][0]],
                     yaw_deg=round(float(np.degrees(poses[id(g)][1])), 1)) for g in movers]
        return dict(moved=True, attempts=attempt, drift=round(drift, 4), groups=info)
    mujoco.mj_setState(m, d, state0, mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(m, d)
    logger.warning("layout.scatter: kept the original layout after %d attempts (%s)", restarts, fail)
    return dict(moved=False, attempts=restarts, reason=fail, groups=[])


__all__ = ["Arm", "DEFAULT_ARMS", "ReachMap", "reach_map", "scatter", "table_box", "table_geom"]
