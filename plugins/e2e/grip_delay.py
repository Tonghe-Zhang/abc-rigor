"""e2e: all 3 plugin tasks build with the fleet gripper (5000/50/40) and the gripper command arrives 2 frames late."""
import numpy as np, abc_sim
import abc_hw
for task in ("tongs_bread_bag", "tongs_pick_place", "open_cap"):
    env = abc_sim.make_env(task, seed=1)
    u = env.unwrapped if hasattr(env, "unwrapped") else env
    inner = getattr(u, "_external", None) or getattr(u, "env", None) or u
    obs, _ = env.reset(seed=1)
    src = [o for o in (inner, getattr(inner, "env", None)) if o is not None and hasattr(o, "_gdelay")][0]
    m, d, gd = src.model, src.data, src._gdelay
    print(task, "kp", m.actuator_gainprm[gd.ids, 0], "kv", -m.actuator_biasprm[gd.ids, 2], "cap", m.actuator_forcerange[gd.ids, 1], "delay steps", gd.n)
    st = np.asarray(obs["state"], float).copy()
    c0 = d.ctrl[gd.ids].copy()
    a = st.copy()
    gi = [6, 13]
    a[gi] = 0.0                                     # close both grippers
    for k in range(gd.n + 3):
        src.apply_action(a, 1)
        if k in (0, gd.n - 2, gd.n - 1, gd.n):
            print(f"  step {k+1}: ctrl {d.ctrl[gd.ids].round(4)} (reset ctrl {c0.round(4)})")
    assert np.allclose(d.ctrl[gd.ids], 0, atol=1e-6) and gd.n == 33
print("E2E_OK delay")
