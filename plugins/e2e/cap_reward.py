"""e2e: open_cap reward stages through abc_sim.make_env (harness lid placement only, test-side)."""
# run with PYTHONPATH=<abc-rigor>:<abc-rigor>/plugins and ABC_SIM_PLUGINS=abc_cap_tasks:register
import mujoco, abc_sim, numpy as np
env = abc_sim.make_env("open_cap", render_cameras=False)
def inner(e):
    while not hasattr(e, "evaluator"): e = getattr(e, "env", None) or getattr(e, "_env")
    return e
def lid_jnt(e, c):
    m = e.model; j = m.body_jntadr[c["lid_bid"]]
    assert m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE, "lid not free-jointed"
    return m.jnt_qposadr[j], m.jnt_dofadr[j]
def run(e, a, secs):
    out = None
    for _ in range(int(secs / e.model.opt.timestep)):
        _, r, term, trunc, info = env.step(a, render_obs=False)
        out = (r, term, info)
        if term: break
    return out
def case(name, place):
    env.reset(seed=1); e = inner(env); ev = e.evaluator; c = e.cs[ev.target]; m, d = e.model, e.data
    a = e.hold_action()
    r, term, info = run(e, a, 0.3)
    print(f"[{name}] settle: r={r:.3f} term={term}", flush=True)
    qa, va = lid_jnt(e, c)
    print(f"[{name}] jar z {d.xpos[c['bid']][2]:.4f} lid z {d.xpos[c['lid_bid']][2]:.4f} table top {ev.table_top:.4f}", flush=True)
    place(e, c, d, qa); d.qvel[va:va+6] = 0; mujoco.mj_forward(m, d)
    r, term, info = run(e, a, 0.15)
    t = info["task_eval"]
    print(f"[{name}] +0.15s r={r:.3f} unmounted={t['lid_unmounted']} landed={t['lid_landed']} term={term} fail={t['fail_reason']}", flush=True)
    r, term, info = run(e, a, 1.5)
    t = info["task_eval"]
    print(f"[{name}] end r={r:.3f} success={info['task_success']} unmounted={t['lid_unmounted']} landed={t['lid_landed']} term={term} fail={t['fail_reason']}", flush=True)
    return r, info
def beside(e, c, d, qa):            # lid set on the table 8 cm from its jar (same orientation)
    p = d.xpos[c["bid"]].copy(); d.qpos[qa:qa+3] = [p[0] - 0.08, p[1], d.qpos[qa+2] - (p[2] - e.evaluator.table_top) + 0.005]
def above(e, c, d, qa):             # lid 15 cm above the jar side: falls, then lands
    p = d.xpos[c["bid"]].copy(); d.qpos[qa:qa+3] = [p[0] - 0.08, p[1], d.qpos[qa+2] + 0.10]
def offtable(e, c, d, qa):          # lid dropped beyond the table edge
    d.qpos[qa:qa+3] = [d.qpos[qa], 1.5, e.evaluator.table_top + 0.05]
r, i = case("beside", beside);   assert i["task_success"] and r == 1.0
r, i = case("above", above);     assert i["task_success"] and r == 1.0
r, i = case("offtable", offtable); assert not i["task_success"] and r == 0.0 and "lid_off_table" in i["task_eval"]["fail_reason"]
env.reset(seed=1); e = inner(env); ev = e.evaluator
ev.pinch_lid = ev.hold_jar = True; ev.max_twist[ev.target] = np.deg2rad(e.cs[ev.target]["free_deg"])
r = float(e.evaluate_task().reward[0]); print("cap pre-unmount (grab + full twist):", round(r, 3)); assert abs(r - 0.7) < 1e-5
ev.free[ev.target] = True; r = float(e.evaluate_task().reward[0]); print("cap unmounted not landed:", round(r, 3)); assert abs(r - 0.9) < 1e-5
print("E2E PASS")
