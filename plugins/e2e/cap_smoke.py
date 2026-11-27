"""e2e: open_cap via abc_sim.make_env -- build, prompt, hold rollout, evaluator probes (test-harness lid moves), video."""
import json, os, subprocess, sys, time
import numpy as np, mujoco
import abc_sim
from abc_sim.task_specs import get_task_spec
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
seeds = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 1, 2]
rep = {}
assert get_task_spec("open the bottle cap").env_task == "open_cap"
for s in seeds:
    t = time.time(); env = abc_sim.make_env("open_cap", render_cameras=False, seed=s); tb = time.time() - t
    obs, info = env.reset(seed=s)
    a = env.hold_action(); st = env.get_state()
    r = dict(build_s=round(tb, 1), prompt=env.prompt, n_containers=len(env.cs), target=info["task_eval"]["target"],
             variant=info["task_eval"]["target_variant"], free_deg=info["task_eval"]["free_angle_deg"],
             state_vs_hold=float(np.abs(st - a).max()), r0=info["task_reward"])
    frames = []
    rnd = mujoco.Renderer(env.model, 360, 480)
    t = time.time()
    for k in range(1500):          # 3 s of holding
        obs, rew, term, trunc, info = env.step(a)
        if k % 17 == 0: frames.append(env.render("base_0_rgb", 480, 360, rnd))
        if term: break
    r.update(hold_steps=k + 1, hold_sps=round((k + 1) / (time.time() - t)), hold_reward=rew, hold_term=term,
             hold_fail=info["task_eval"]["failures"], max_twist=info["task_eval"]["max_twist_deg"])
    # --- evaluator probe (harness only): spin the target lid about the jar axis, then lift it off
    ev = env.evaluator; c = env.cs[ev.target]; m, d = env.model, env.data
    j = m.body(c["lid_bid"]).jntadr[0]; qa = m.jnt_qposadr[j]
    assert m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE, "lid is not a free body"
    Rj = d.xmat[c["bid"]].reshape(3, 3)
    def setlid(ang, dz):
        q = np.zeros(4); mujoco.mju_axisAngle2Quat(q, Rj[:, 2], ang)
        q0 = d.qpos[qa + 3:qa + 7].copy(); qn = np.zeros(4); mujoco.mju_mulQuat(qn, q, q0)
        d.qpos[qa + 3:qa + 7] = qn; d.qpos[qa:qa + 3] += Rj[:, 2] * dz; d.qvel[:] = 0
        mujoco.mj_forward(m, d)
    p0 = d.qpos[qa:qa + 7].copy()
    for i in range(60):       # 0.5 x free angle in 60 harness steps
        setlid(np.deg2rad(c["free_deg"]) * 0.5 / 60, 0.0); ev.after_physics_step(m, d)
    r["probe_half_reward"] = float(env.evaluate_task().reward[0])
    d.qpos[qa:qa + 3] += Rj[:, 2] * 0.05; mujoco.mj_forward(m, d)
    for i in range(80): ev.after_physics_step(m, d)
    rr = env.evaluate_task(); r["probe_off_reward"] = float(rr.reward[0]); r["probe_off_success"] = bool(rr.success[0])
    # wrong container probe
    env.reset(seed=s); ev = env.evaluator; m, d = env.model, env.data
    o = (ev.target + 1) % len(env.cs); c2 = env.cs[o]; qa2 = m.jnt_qposadr[m.body(c2["lid_bid"]).jntadr[0]]
    d.qpos[qa2 + 2] += 0.08; mujoco.mj_forward(m, d)
    for i in range(80): ev.after_physics_step(m, d)
    rr = env.evaluate_task(); r.update(probe_wrong_reward=float(rr.reward[0]), probe_wrong_fail=rr.metrics["failures"],
                                       wrong_terminated=ev.terminated)
    rnd.close()
    mp4 = f"{out}/cap_s{s}_hold.mp4"
    p = subprocess.Popen(["/tmp/mj4/bin/ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", "480x360", "-r", "10", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", mp4],
                         stdin=subprocess.PIPE)
    p.stdin.write(np.stack(frames).tobytes()); p.stdin.close(); p.wait()
    r["video"] = mp4; rep[s] = r; print(json.dumps({s: r}), flush=True); env.close()
json.dump(rep, open(f"{out}/report.json", "w"), indent=1)
