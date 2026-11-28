import time, json, sys, numpy as np
import abc_sim
from abc_sim.task_specs import get_task_spec
for name in sys.argv[1:]:
    t=time.time(); env=abc_sim.make_env(name, render_cameras=False, seed=1)
    print(name, type(env).__name__, "build %.1fs"%(time.time()-t), "dt", env.model.opt.timestep)
    for s in (1,2):
        obs,info=env.reset(seed=s); print("  seed",s,"|",obs["prompt"],"| r",round(info["task_reward"],3))
    a=env.get_state(); a=np.asarray(a); 
    for i in range(200): obs,r,te,tr,info=env.step(a)
    print("  200 steps r",round(r,3),"term",te,"metrics keys",sorted(k for k in info["task_eval"] if "tong" in k or "fail" in k)[:8])
    print("  eval", env.evaluate_task().reward, "state", np.round(env.get_state(),3)[[6,13]])
    if name=="tongs_pick": print("  catalogue keys", sorted(env._catalogue()[0].keys()))
    env.close()
