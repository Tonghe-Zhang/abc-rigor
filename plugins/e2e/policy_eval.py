"""Offline, multi-process policy eval of abc_sim plugin tasks against a streaming (protocol v6) policy server.

Each worker process owns one env and one websocket session per episode; seeds are spread over ``--workers``.  The
client is self-contained (msgpack + websockets) so it runs in the light mujoco-4 venv without deploy-anything.

Action space ``xdof_bimanual_chunk_rel_eef_rot6d_measured_anchor``: per arm (left 0-9, right 10-19) a position delta
(3) and a rot6d relative rotation (6) w.r.t. the anchor state sent with the observation, and an absolute gripper
(0 open / 1 closed).  Targets are turned into joint targets by damped-least-squares IK on the scene's own arm and
played at 30 Hz with linear interpolation (2 ms physics).

    python policy_eval.py --server ws://127.0.0.1:8000 --task tongs_bread_bag --seeds 0-15 --workers 8 --out DIR
    python policy_eval.py --mock-server 8765            # tiny hold-policy server for smoke tests

Writes ``DIR/episodes.jsonl`` (one line per episode), ``DIR/summary.json`` and H.264 videos ``DIR/seed_<s>.mp4``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path

import numpy as np

PV = 6
# standard abc_sim / deploy-anything harness: top/left/right at native 450x600 (yam_abc_catalog_mujoco.yaml)
CAMS = (("base_0_rgb", (450, 600)), ("left_wrist_0_rgb", (450, 600)), ("right_wrist_0_rgb", (450, 600)))
FFMPEG = os.environ.get("FFMPEG", "/tmp/mj4/bin/ffmpeg")


# ---------------------------------------------------------------------------------------------------- wire
def _enc(o):
    if isinstance(o, np.ndarray):
        a = np.ascontiguousarray(o)
        return {"__ndarray__": True, "data": a.tobytes(), "dtype": a.dtype.str, "shape": list(a.shape)}
    if isinstance(o, np.generic):
        return o.item()
    return o


def _dec(o):
    if isinstance(o, dict) and o.get("__ndarray__"):
        return np.frombuffer(o["data"], dtype=np.dtype(o["dtype"])).reshape(o["shape"])
    return o


def pack(t, sid, **f):
    import msgpack
    return msgpack.packb({"protocol_version": PV, "type": t, "session_id": sid, **f}, use_bin_type=True, default=_enc)


def unpack(b):
    import msgpack
    return msgpack.unpackb(b, raw=False, object_hook=_dec)


# ---------------------------------------------------------------------------------------------------- geometry
def rot6d_to_mat(r):
    a, b = np.asarray(r[:3], float), np.asarray(r[3:6], float)
    c0 = a / (np.linalg.norm(a) + 1e-12)
    c2 = np.cross(c0, b / (np.linalg.norm(b) + 1e-12))
    c2 /= np.linalg.norm(c2) + 1e-12
    c1 = np.cross(c2, c0)
    c1 /= np.linalg.norm(c1) + 1e-12
    c2 = np.cross(c0, c1)
    return np.stack([c0, c1, c2], 1)


def mat_to_rot6d(R):
    return np.asarray(R)[:, :2].T.reshape(-1)


class Rig:
    """State / IK helper bound to a TongsAbcEnv / CapAbcEnv (14 actions: arm j1-6 + gripper, per arm)."""

    ARM_IDX = (np.arange(0, 6), np.arange(7, 13))
    GRIP_IDX = (6, 13)

    def __init__(self, env):
        import mujoco
        from abc_sim import layout
        self.mj, self.env = mujoco, env
        m = env.model
        self.arms = layout.DEFAULT_ARMS
        self.chains = []
        for a in self.arms:
            joints, dofs, qadr, base = layout._arm_chain(m, a)
            keep = [i for i, j in enumerate(joints) if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE)][:6]
            self.chains.append((joints[keep], dofs[keep], qadr[keep], base))
        self.sites = [m.site(a.grasp_site).id for a in self.arms]
        self.d2 = mujoco.MjData(m)

    def state(self) -> np.ndarray:
        d, s = self.env.data, self.env.get_state()
        out = []
        for k, sid in enumerate(self.sites):
            out += [d.site_xpos[sid], mat_to_rot6d(d.site_xmat[sid].reshape(3, 3)), [1.0 - float(s[self.GRIP_IDX[k]])]]
        return np.concatenate(out).astype(np.float32)

    def joint_offset(self, k):   # action-space joint value minus qpos (CapAbcEnv works in offset joint coordinates)
        return self.env.get_state()[self.ARM_IDX[k]] - self.env.data.qpos[self.chains[k][2]]

    def ik(self, k, p_tgt, R_tgt, q0, iters=40):
        m, d = self.env.model, self.d2
        joints, dofs, qadr, _ = self.chains[k]
        lo, hi = m.jnt_range[joints].T
        lim = m.jnt_limited[joints].astype(bool)
        sid = self.sites[k]
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        q = q0.copy()
        for _ in range(iters):
            d.qpos[qadr] = q
            self.mj.mj_kinematics(m, d)
            self.mj.mj_comPos(m, d)
            ep = p_tgt - d.site_xpos[sid]
            Rc = d.site_xmat[sid].reshape(3, 3)
            Re = R_tgt @ Rc.T
            er = 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
            if np.linalg.norm(ep) < 5e-4 and np.linalg.norm(er) < 5e-3:
                break
            self.mj.mj_jacSite(m, d, jp, jr, sid)
            J = np.vstack([2.0 * jp[:, dofs], 1.0 * jr[:, dofs]])
            e = np.concatenate([2.0 * ep, 1.0 * er])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(6), e)
            n = np.linalg.norm(dq)
            q = q + dq * min(1.0, 0.25 / (n + 1e-12))
            q = np.where(lim, np.clip(q, lo, hi), q)
        return q

    def decode(self, row, anchor, q_prev):
        """One 20-d action row -> 14-d env action (joint targets in action coordinates + env gripper 1=open)."""
        out = np.zeros(14)
        for k in range(2):
            o = 10 * k
            p = anchor[o:o + 3] + row[o:o + 3]
            R = rot6d_to_mat(row[o + 3:o + 9]) @ rot6d_to_mat(anchor[o + 3:o + 9])
            off = self.joint_offset(k)
            q = self.ik(k, p, R, q_prev[k])
            q_prev[k] = q
            out[self.ARM_IDX[k]] = q + off
            out[self.GRIP_IDX[k]] = float(np.clip(1.0 - row[o + 9], 0.0, 1.0))
        return out


# ---------------------------------------------------------------------------------------------------- episode
class Video:
    def __init__(self, path, h, w, fps):
        self.p = subprocess.Popen([FFMPEG, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                   "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt",
                                   "yuv420p", "-crf", "23", str(path)], stdin=subprocess.PIPE)

    def add(self, img):
        self.p.stdin.write(np.ascontiguousarray(img, np.uint8).tobytes())

    def close(self):
        self.p.stdin.close()
        self.p.wait()


def _tongs_contacts(env):
    """(tongs geom, other geom/body) pairs in contact, for geoms on bodies named tongs*."""
    import mujoco
    m, d = env.model, env.data
    out = set()
    for i in range(d.ncon):
        c = d.contact[i]
        for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
            ba = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[a])) or ""
            if ba.startswith("tongs"):
                bb = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[b])) or ""
                if not bb.startswith("tongs"):
                    ga = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(a)) or ba
                    gb = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(b)) or bb
                    out.add(f"{ga}|{gb}")
    return sorted(out)


async def run_episode(args, seed: int, out: Path) -> dict:
    import mujoco
    import websockets
    from abc_sim.plugins import maybe_get_external_task

    t_start = time.time()
    ext = maybe_get_external_task(args.task)
    if ext is None:
        raise SystemExit(f"unknown task {args.task!r} (is ABC_SIM_PLUGINS set?)")
    tk = json.loads(args.task_kwargs) if args.task_kwargs else None
    env = ext.env_factory(task=args.task, seed=seed, max_seconds=args.max_seconds, task_kwargs=tk,
                          terminate_on_success=True, terminate_on_failure=True)
    env.reset(seed=seed)
    rig = Rig(env)
    camnames = {mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(env.model.ncam)}
    cams = []
    pmap = getattr(env, "POLICY_CAMERAS", None)
    if not pmap:
        raise RuntimeError(f"{type(env).__name__} has no POLICY_CAMERAS map; refusing to guess cameras {sorted(camnames)}")
    for key, hw in CAMS:
        name = pmap.get(key)
        if name not in camnames:
            raise RuntimeError(f"policy camera {key} -> {name!r} not in model cameras {sorted(camnames)}")
        cams.append((key, name, hw))
    print(f"[cams] {[(k, n, hw) for k, n, hw in cams]}", flush=True)
    rends = {hw: mujoco.Renderer(env.model, *hw) for _, _, hw in cams}

    def images():
        return [np.ascontiguousarray(env.render(n, hw[1], hw[0], rends[hw])) for _, n, hw in cams]

    prompt = env.prompt
    sid = f"eval-{seed}-{uuid.uuid4().hex[:8]}"
    rec = dict(task=args.task, seed=seed, prompt=prompt, policy=args.server, reward=0.0, max_reward=-1.0,
               success=False, ever_success=False, tongs_grasped=False, robot_touch_failure=False, failed=False,
               chunks=0, text_turns=0, rows=0, sim_time=0.0, end="time_limit", error=None)
    vid = Video(out / f"seed_{seed}.mp4", 450, 600, max(1, 30 // max(1, args.video_every))) if args.video else None
    q_prev = [env.data.qpos[c[2]].copy() for c in rig.chains]
    trace = [] if args.trace else None
    hq = [int(env.model.jnt_qposadr[j]) for j in range(env.model.njnt)
          if "tongs_hinge" in (mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_JOINT, j) or "")]
    dt = float(env.model.opt.timestep)
    spr = 1.0 / (30.0 * dt)          # physics steps per action row
    acc = 0.0
    a_prev = None
    seq = 0
    info = {}
    ws = None
    try:
        async def connect():
            w = await websockets.connect(args.server, max_size=None, open_timeout=60, ping_interval=None)
            return w, unpack(await asyncio.wait_for(w.recv(), 120))

        ws, hello = await connect()
        if True:
            assert hello["type"] == "server_hello", hello.get("type")
            stride = int(args.replan_stride or hello["replan_stride"])
            offset = int(hello.get("action_start_offset_steps") or 0) if args.start_offset < 0 else args.start_offset
            sched = {}                   # row index -> (action row, anchor state); rows start at obs + offset
            rec["server"] = {k: hello.get(k) for k in ("action_space_id", "action_horizon", "action_dim", "replan_stride",
                                                     "action_start_offset_steps", "camera_keys", "checkpoint_sha256")}
            extra = {"camera_shapes": [[*hw, 3] for _, _, hw in cams]} if hello.get("max_image_tokens") is not None else {}
            async def start(sid_):
                await ws.send(pack("start_episode", sid_, prompt=prompt, robot_id="abc_sim_yam_bimanual",
                                   action_space_id=hello["action_space_id"], seed=int(seed), **extra))
                while True:      # skip leftovers of a closed session until this one is acknowledged
                    st = unpack(await asyncio.wait_for(ws.recv(), 120))
                    if st["type"] == "error":
                        raise RuntimeError(f"server error {st.get('code')}: {st.get('message')}")
                    if st.get("session_id") == sid_ and st["type"] != "episode_closed":
                        return

            await start(sid)
            sess_chunks = 0
            done = False
            while not done:
                if args.session_chunks and sess_chunks >= args.session_chunks:
                    # the server's recurrent context (K/V) is finite: roll over to a fresh session, same episode
                    # (one session per connection on the streaming server: close, reconnect, start again)
                    try:
                        await ws.send(pack("close_episode", sid))
                        await ws.close()
                    except Exception:   # noqa: BLE001
                        pass
                    ws, _ = await connect()
                    sid = f"eval-{seed}-{uuid.uuid4().hex[:8]}"
                    await start(sid)
                    sess_chunks = 0
                    rec["sessions"] = rec.get("sessions", 1) + 1
                anchor = rig.state()
                imgs = images()
                chunk = None
                text = ""
                for _attempt in range(8):
                    seq += 1
                    obs = dict(observation_seq=seq, capture_time_ns=time.monotonic_ns(), deadline_budget_ms=600000,
                               images=imgs, state=anchor)
                    if args.action_temperature is not None:
                        obs["action_temperature"] = float(args.action_temperature)
                    await ws.send(pack("observation", sid, **obs))
                    t_end = time.time() + args.chunk_timeout
                    while time.time() < t_end:
                        msg = unpack(await asyncio.wait_for(ws.recv(), max(1.0, t_end - time.time())))
                        ty = msg["type"]
                        if ty == "action_chunk" and msg.get("observation_seq") == seq:
                            chunk = np.asarray(msg["actions"], float)
                            rec.setdefault("timing", msg.get("timing"))
                            break
                        if ty == "observation_status" and msg.get("observation_seq") == seq and \
                                msg.get("status") in ("rejected", "stale"):
                            break
                        if ty == "episode_closed":
                            rec["end"] = "server_closed"
                            done = True
                            break
                        if ty == "error":
                            raise RuntimeError(f"server error {msg.get('code')}: {msg.get('message')}")
                        if ty == "text_delta" and msg.get("observation_seq") == seq:
                            text += str(msg.get("text", ""))
                        if ty == "text_done" and msg.get("observation_seq") == seq:
                            rec["text_turns"] += 1     # a reasoning turn ends without actions: observe again
                            rec["last_text"] = text[-300:]
                            text = ""
                            break
                    if chunk is not None or done:
                        break
                if done:
                    break
                if chunk is None:
                    raise RuntimeError("no action chunk after 8 observations")
                rec["chunks"] += 1
                sess_chunks += 1
                t0 = rec["rows"]
                for i, row in enumerate(chunk):
                    sched[t0 + offset + i] = (row, anchor)
                for t in range(t0, t0 + stride):
                    if t in sched:
                        row_t, anc_t = sched.pop(t)
                        a = rig.decode(row_t, anc_t, q_prev)
                    else:                # nothing scheduled yet (first chunk's start offset): hold
                        a = a_prev.copy() if a_prev is not None else env.get_state().astype(float).copy()
                    a0 = a if a_prev is None else a_prev
                    acc += spr
                    n = int(acc)
                    acc -= n
                    for i in range(n):
                        f = (i + 1) / n
                        ai = a0 + f * (a - a0)
                        ai[list(rig.GRIP_IDX)] = a[list(rig.GRIP_IDX)]
                        if i < n - 1:
                            env.apply_action(ai, 1)
                            continue
                        _, r, term, trunc, info = env.step(ai)
                    a_prev = a
                    rec["rows"] += 1
                    if trace is not None:
                        st = env.get_state()
                        trace.append(dict(row=rec["rows"], t=float(env.data.time),
                                          cmd=[float(a[g]) for g in rig.GRIP_IDX], state=[float(st[g]) for g in rig.GRIP_IDX],
                                          hinge=[float(env.data.qpos[q]) for q in hq],
                                          tg=bool(info.get("tongs_grasped")),
                                          contacts=_tongs_contacts(env), qpos=[float(x) for x in env.data.qpos]))
                    if vid is not None and rec["rows"] % max(1, args.video_every) == 0:
                        vid.add(env.render(cams[0][1], 600, 450, rends[(450, 600)]))
                    rec["reward"] = float(r)
                    rec["max_reward"] = max(rec["max_reward"], float(r))
                    rec["ever_success"] |= bool(info.get("task_success"))
                    rec["tongs_grasped"] |= bool(info.get("tongs_grasped"))
                    rec["robot_touch_failure"] |= bool(info.get("robot_touch_failure"))
                    if term or trunc:
                        rec["end"] = "success" if info.get("task_success") else "failure" if info.get("task_failed") \
                            else "terminated" if term else "time_limit"
                        done = True
                        break
                if rec["chunks"] >= args.max_chunks:
                    rec["end"] = "max_chunks"
                    done = True
            try:
                await ws.send(pack("close_episode", sid))
                await ws.close()
            except Exception:   # noqa: BLE001
                pass
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()[-1500:]
    finally:
        if ws is not None:       # never leave a connection open: the server holds one session slot per connection
            try:
                await ws.close()
            except Exception:   # noqa: BLE001
                pass
        if vid is not None:
            vid.close()
        if trace is not None:
            with open(out / f"seed_{seed}_trace.jsonl", "w") as fh:
                for t_ in trace:
                    fh.write(json.dumps(t_) + "\n")
        for r_ in rends.values():
            r_.close()
    rec["success"] = bool(info.get("task_success", False)) or rec["ever_success"]
    rec["failed"] = bool(info.get("task_failed", False))
    rec["sim_time"] = float(info.get("sim_time", 0.0))
    rec["task_eval"] = {k: v for k, v in dict(info.get("task_eval") or {}).items()
                        if isinstance(v, (int, float, str, bool)) or v is None}
    rec["wall_s"] = round(time.time() - t_start, 1)
    return rec


def _worker(args, seeds, out, q):
    for s in seeds:
        try:
            rec = asyncio.run(run_episode(args, s, out))
        except Exception as e:  # noqa: BLE001
            rec = dict(task=args.task, seed=s, error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-1500:])
        q.put(rec)


def _seeds(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def summarize(recs):
    ok = [r for r in recs if not r.get("error")]
    n = len(ok)
    f = lambda k: round(sum(bool(r.get(k)) for r in ok) / n, 4) if n else None  # noqa: E731
    return dict(episodes=len(recs), errors=len(recs) - n, success_rate=f("success"), tongs_grasp_rate=f("tongs_grasped"),
                robot_touch_failure_rate=f("robot_touch_failure"), failure_rate=f("failed"),
                mean_final_reward=round(float(np.mean([r["reward"] for r in ok])), 4) if n else None,
                mean_max_reward=round(float(np.mean([r["max_reward"] for r in ok])), 4) if n else None,
                ends={e: sum(r.get("end") == e for r in ok) for e in sorted({r.get("end") for r in ok})})


def main_eval(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = _seeds(args.seeds)
    nw = max(1, min(args.workers, len(seeds)))
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    servers = args.server.split(",")     # the streaming server holds one session at a time: one server per worker
    procs = []
    for i in range(nw):
        wa = argparse.Namespace(**vars(args))
        wa.server = servers[i % len(servers)]
        procs.append(ctx.Process(target=_worker, args=(wa, seeds[i::nw], out, q), daemon=True))
    for p in procs:
        p.start()
    recs = []
    with open(out / "episodes.jsonl", "a") as fh:
        while len(recs) < len(seeds):
            rec = q.get()
            recs.append(rec)
            fh.write(json.dumps(rec, default=str) + "\n")
            fh.flush()
            print(json.dumps({k: rec.get(k) for k in ("seed", "end", "reward", "max_reward", "success", "tongs_grasped",
                                                      "robot_touch_failure", "chunks", "wall_s", "error")}), flush=True)
    for p in procs:
        p.join(timeout=30)
    s = dict(task=args.task, server=args.server, seeds=args.seeds, **summarize(recs))
    (out / "summary.json").write_text(json.dumps(s, indent=1))
    print("SUMMARY", json.dumps(s), flush=True)


# ---------------------------------------------------------------------------------------------------- mock server
async def _mock_handler(ws):
    H, D = 30, 20
    await ws.send(pack("server_hello", "mock", checkpoint_sha256="0" * 64, config_sha256="0" * 64,
                       camera_keys=[c[0] for c in CAMS], camera_shapes=[[*c[1], 3] for c in CAMS], state_dim=20,
                       action_space_id="xdof_bimanual_chunk_rel_eef_rot6d_measured_anchor", action_horizon=H,
                       action_dim=D, replan_stride=15, action_start_offset_steps=0, max_context_tokens=0))
    sid = None
    turn = 0
    async for b in ws:
        m = unpack(b)
        if m["type"] == "start_episode":
            sid = m["session_id"]
            await ws.send(pack("episode_started", sid))
        elif m["type"] == "observation":
            st = np.asarray(m["state"], np.float32)
            row = np.zeros(D, np.float32)
            for o in (0, 10):
                row[o + 3:o + 9] = [1, 0, 0, 0, 1, 0]
                row[o + 9] = st[o + 9]
            if os.environ.get("MOCK_POLICY") == "lower":    # move both grasp sites 10 cm down over the chunk
                acts = np.repeat(row[None], H, 0)
                acts[:, 2] = acts[:, 12] = -np.linspace(0, 0.1, H)
            else:
                acts = np.repeat(row[None], H, 0)
            turn += 1
            await ws.send(pack("observation_status", sid, observation_seq=m["observation_seq"], status="accepted"))
            await ws.send(pack("action_chunk", sid, turn_id=turn, observation_seq=m["observation_seq"],
                               capture_time_ns=m["capture_time_ns"], action_space_id=
                               "xdof_bimanual_chunk_rel_eef_rot6d_measured_anchor", actions=acts, timing={}))
        elif m["type"] == "close_episode":
            await ws.send(pack("episode_closed", sid, reason="client"))
            return


async def _mock_main(port):
    import websockets
    async with websockets.serve(_mock_handler, "127.0.0.1", port, max_size=None):
        print(f"mock server on {port}", flush=True)
        await asyncio.Future()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="ws://127.0.0.1:8000", help="comma-separated; worker i uses server i % n")
    ap.add_argument("--task", default="tongs_bread_bag")
    ap.add_argument("--task-kwargs", default=None)
    ap.add_argument("--seeds", default="0-7")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="/tmp/policy_eval")
    ap.add_argument("--max-seconds", type=float, default=60.0)
    ap.add_argument("--max-chunks", type=int, default=400)
    ap.add_argument("--session-chunks", type=int, default=120, help="start a fresh server session every N chunks (finite K/V context); 0 = never")
    ap.add_argument("--replan-stride", type=int, default=0)
    ap.add_argument("--start-offset", type=int, default=-1, help="rows between observation and chunk row 0 (-1: server's)")
    ap.add_argument("--chunk-timeout", type=float, default=120.0)
    ap.add_argument("--action-temperature", type=float, default=None)
    ap.add_argument("--video", type=int, default=1)
    ap.add_argument("--video-every", type=int, default=3)
    ap.add_argument("--mock-server", type=int, default=0)
    ap.add_argument("--trace", type=int, default=0, help="write seed_N_trace.jsonl: gripper cmd/state + tongs hinge per row")
    args = ap.parse_args()
    if args.mock_server:
        asyncio.run(_mock_main(args.mock_server))
        return
    main_eval(args)


if __name__ == "__main__":
    sys.exit(main())
