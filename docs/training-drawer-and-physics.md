# Drawer training and explicit physics conditions

`multi_drawer_search_2_shaped` is a separate training task. It samples exactly two
ordered targets from the native drawer-search object/layout distribution. Native
`multi_drawer_search` still samples 2–4 targets. Neither task changes the
environment horizon: training callers can select a 120-second simulation budget
with their own rollout controller. With the default 0.002-second physics tick and
17 ticks per action, a caller can use `max_episode_steps=3529` (119.986 seconds).
This cap is an experiment setting, not a task success condition.

## Reward and prompt sequence

Let `C` be the number of ordered targets already deposited in the goal bin and
`N` the target count. The native retrieval score is
`B = (C + p_current) / N`, where `p_current` is the existing XY proximity score
of the next uncompleted target. Completion gives `B=1`. Both target count and
current-target selection are updated before calculating proximity; a deposited
target cannot simultaneously earn completion and current-target credit.

The training task returns `0.75 * B + drawer_bonus` until completion, then exactly
`1`. Opening each previously unopened drawer adds `0.05`, with at most `0.20`
earned across the episode. An opening is eligible only if the current target's
source drawer had not been opened **before** that observation. Thus opening the
target's own drawer can earn credit, and later unrelated openings cannot.
Every opened drawer is remembered, including ineligible ones. Credit is retained
when the target changes; seeing a later target's drawer early prevents future
exploration credit for that target. Simultaneous openings use the pre-observation
history and target; all eligible first openings count toward the same cap.

The opening threshold is half the drawer's slide-joint range: the native scene
allows 0–0.216 m of outward travel, giving a 0.108 m threshold. Tiny motions do not
count. This is an opening proxy, not a visibility test. Finding an object alone
does not count as retrieving it: the existing three-dimensional goal-bin
containment condition still controls completion.

The evaluator is the authority for both reward and runtime prompts and updates
on every native physics tick. Ordered deposit progress remains latched, including
transient bin entries, as in the existing sequence semantics. Reward-history
replay must start from reset and provide chronological physics observations.
A final qpos snapshot or a 30 Hz video does not contain enough information to
reconstruct unique openings or transient deposits. Batched MJWarp environments
already reject drawer tasks because they require a per-step Python runtime.

## Friction profile

Physics changes are independent of task selection:

```python
import abc_sim

env = abc_sim.make_env(
    "multi_drawer_search_2_shaped",
    physics_profile="round2_low_friction_v1",
    max_episode_steps=3529,
)
```

Without `physics_profile`, models retain their original coefficients.
`round2_low_friction_v1` caps sliding friction
at **1.5**, preserving already-lower values and all torsional/rolling values.
It covers geoms, flexes, explicit contact pairs in both tangent directions, and
global contact overrides. Geom priorities remain unchanged. MuJoCo selects geom
coefficients by priority or componentwise maximum, so capping each source also
caps the resulting contact. The implementation follows the MuJoCo 3.8.1 collision
driver's parameter combination rules.

The environment applies the profile to each compiled model before creating
`MjData`, renderers, or Warp copies. XML reloads and compiled-model reloads use
the same hook. The public `abc_sim.physics.apply_physics_profile(model, profile)`
supports custom registered task builders: call it before building data or copying
the model to another simulator. Applying a profile repeatedly is idempotent.
Do not change profiles on a running environment; create a new one.

The full profile definition is recorded in reset `info["physics_profile"]`,
randomization metadata, and `capture_state()["physics_profile"]`. Save it next to
the scene XML; in-memory changes do not rewrite the original XML.

```python
from abc_sim.physics import PhysicsProfile

recorded = episode_metadata["physics_profile"]
profile = None if recorded is None else PhysicsProfile.from_dict(recorded)
replay_env = abc_sim.make_env(task, physics_profile=profile)
```

`SceneRandomizer.apply` reapplies the recorded profile to standalone models and
rejects mismatches with an already-bound environment, avoiding stale renderer or
Warp copies. Replay adapters outside this package must pass the recorded profile
when constructing their environments. Pin task ID, evaluator ID, profile
definition, simulator revision, and the caller's horizon in experiment metadata;
scores from different physics conditions are separate evaluation conditions.
