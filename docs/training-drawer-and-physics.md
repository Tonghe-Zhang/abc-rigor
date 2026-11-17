# Drawer training and explicit physics conditions

`multi_drawer_search_2_shaped` is a separate training task. It samples exactly two
ordered targets from the native drawer-search object/layout distribution. Native
`multi_drawer_search` still samples 2–4 targets. Neither task changes the
environment horizon: training callers can select a 120-second simulation budget
with their own rollout controller. With the default 0.002-second physics tick and
17 ticks per action, a caller can use `max_episode_steps=3529` (119.986 seconds).
This cap is an experiment setting, not a task success condition.

The training task's two-target count is a fixed constraint. Empty caller
randomization options retain it; an explicit incompatible count is rejected
before reset mutates the environment. Native task options remain configurable.
Saved replay metadata must identify the same fixed count and is validated before
replay mutates the environment. The variant has a distinct prompt alias,
preserving native prompt lookup.
The declared count must match the actual ordered sequence; malformed, repeated,
or inconsistently associated targets are rejected before replay changes the
running episode. Explicit native counts 1–4 sample enough objects to satisfy the
request. The no-count native sampling distribution and RNG order are unchanged.
Each active object's recorded label identifies the requested object. Ordered
targets must repeat that label consistently with their body, joint, and source
drawer; replay rejects contradictory labels before changing the running episode.

A fresh `reset(randomize=False)` prepares the compiled initial episode before
clearing dynamics, RNG state or runtime history. The native compiled fallback
has one target and is available only when it satisfies the task's fixed
constraints. A task fixing two targets must first sample an episode or replay a
recording; it cannot silently initialize a one-target task. Sampling defaults
remain distinct from fixed constraints. Reset, replay and compiled-state
inspection use the same definition preparer, and the runtime receives that
validated definition without constructing a separate fallback.

## Reward and prompt sequence

Let `C` be the number of ordered targets already deposited in the goal bin and
`N` the target count. The native retrieval score is
`B = (C + p_current) / N`, where `p_current` is the existing XY proximity score
of the next uncompleted target. Completion gives `B=1`. Both target count and
current-target selection are updated before calculating proximity; a deposited
target cannot simultaneously earn completion and current-target credit.
Goal containment uses the complete bin-to-site transform, including the site's
local orientation and center. These model constants are cached on binding, so
rotated nonsquare regions use the same qpos-only observation path. The XY
proximity score remains measured from the transformed world-space goal center.
The progress source validates that the configured goal site belongs directly to
the configured bin freejoint's body. This read-only model contract applies at
source construction, episode preparation and configuration, and before public
XML/model replacement changes live state. Sampler readiness reuses the same
relationship check while retaining its complete scene-inventory requirements.
Task rebinding constructs and validates its prospective runtime/source before
committing task fields, so rejected configured goals retain the running task,
prompt, cached score and progress history.
Malformed goal ownership is rejected even when scoring or sampling is disabled,
including compiled initialization and native scoring before reset.

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

One drawer progress source advances runtime prompts on every native physics
tick. When reward evaluation is configured, that evaluator is the same source;
the environment observes it once and shares the result. Score queries read the
last observed result without advancing history again. Native scoring can inspect
a task-compatible compiled initial state before reset; shaped scoring requires its sampled
episode metadata. Ordered deposit progress remains latched, including
transient bin entries, as in the existing sequence semantics. Reward-history
replay must start from reset and provide chronological physics observations.
A final qpos snapshot or a 30 Hz video does not contain enough information to
reconstruct unique openings or transient deposits. Batched MJWarp environments
already reject drawer tasks because they require a per-step Python runtime.

Custom specs can omit reward evaluation:

```python
from abc_sim import SimTaskSpec, make_env

env = make_env(
    SimTaskSpec(
        name="drawer_preview",
        env_task="multi_drawer_search",
        prompt="find the target",
    ),
    render_cameras=False,
)
env.reset(seed=6)
```

This uses native scene validity and ordered deposit progress. Randomized resets,
retained resets and recorded replay continue to update the live target prompt.
`evaluate_task()` returns `None`; steps return zero reward and omit task scoring
fields, so `terminate_on_success` does not terminate these unscored tasks.
Caller horizons still apply. `EpisodeTaskEvaluator` declares read-only episode
preparation and later configuration, and `TaskProgressRuntime` declares the
shared physics observation source. The drawer runtime requires
`DrawerProgressSource`: an optional configured evaluator must supply ordered
drawer progress and its joint requirements. Unsupported evaluator/runtime or
randomizer/task combinations are rejected explicitly at binding.

These rewards are persistent progress **scores**, not increments. `step` returns
the current score; `step_chunk` returns the last action's score. Neither method
aggregates an episode return. An episodic RL caller can assign the final observed
score once, or explicitly track and assign the maximum observed score. These are
different objectives because current-target proximity can decrease. The
environment does not select either convention for the caller.

Summed discounted scores need separate timing and terminal treatment. With
`terminate_on_success=True`, waiting at a score of `0.575` for one action before
completing gives `0.575 + gamma`, exceeding immediate completion's `1` when
`gamma > 0.425`. This incentive is conditional on that summed-return objective
and early termination. The default is `terminate_on_success=False`; callers can
retain completed scenes through their fixed horizon. Record the chosen score
aggregation, horizon, and termination settings with the training objective.

Nonrandomized resets restore the current episode's recorded initial physical
scene while clearing deposit and opening history. This includes stack poses,
active-object collision masks, object/bin positions, and inactive placements;
compiled defaults are not the retained episode. Runtime prompts and optional scoring
consume the same validated target definition. The first shaped episode requires a randomized
reset or recorded-state replay so target-to-drawer metadata is available.
Fresh replay binds its recorded sequence before scoring; it does not recover
mid-episode history from positions alone. Drawer scoring reads qpos and cached
model constants, so physics-rate observation does not add position refreshes.
Recorded drawer replay restores those same model fields after any intervening
episode or into a fresh environment. Serialized stack identifiers are normalized
on ingestion. Replay uses the shared scene-restoration path used by sampling and
retained reset; it does not depend on a previous episode's object selection.
Successful bound replay records the restorer that owns this initial scene.
Retained resets use that restorer independently of the sampling randomizer, so
full-XML replay environments may keep `enable_task_randomizer=False`. A rejected
replay leaves the previous restorer in place.

Drawer scene readiness follows the bound model's inventory. Complete assembled
scenes are validated and reused; native templates are assembled before sampling.
Incomplete or mixed assembled inventories are rejected before reset clears
dynamics or retained episode state. Readiness and episode preparation share
the selected progress source's read-only drawer-joint validation. The shaped task requires
limited slides with zero lower travel bounds on every potentially sampled drawer,
including currently inactive stacks. Native drawer tasks retain their existing
custom slide-limit semantics. This applies to direct full-XML construction
and to randomized resets after replay. `enable_task_randomizer=False` skips
initial sampler attachment; an explicit randomized reset can initialize a sampler
and subsequent resets reuse it. When starting with a native template and sampling
disabled, perform a randomized reset before applying a full-scene recording.
Bound drawer replay expects that assembled inventory.
The template's goal-bin joint and goal-site ownership are checked before
assembly. The assembled candidate is compiled once and validated before binding;
rejected candidates leave the live model, data, evaluator, renderer and episode
state in place.

Explicit replay with `keep_arm_state=True`, or retained reset with
`options={"keep_arm_state": True}`, carries the current arm and gripper pose.
This includes both physical fingers on each gripper; the 14D policy projection
does not include every physical robot joint. Scene-object poses still come from
the sampled or recorded initial scene.
The default restores configured arm initialization. Both choices clear
velocities, controls, time, step count, and chronological drawer history.
Task runtimes may supply a default robot-pose initializer after scene restoration.
It is skipped when a pose is explicitly preserved. Conveyor's default still opens
both grippers and sets their opening controls; preserved-pose reset/replay keeps
all four finger positions and leaves reset controls cleared. Ongoing conveyor
roller and object controls are unchanged.

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
Warp copies. This is the shared final replay entry point: task implementations
override `_restore_scene`, which runs only after physics validation. Rejection
precedes model reloads, randomizer selections, and data changes. A recording
without profile metadata means native physics and is rejected by a bound
low-friction environment. Bound replay then initializes fresh dynamics and the
default arm pose, restores recorded object poses, and finalizes episode state.
Recordings contain initial scene randomization, not mid-episode robot state.
Standalone callers must supply a freshly constructed
model with the recording's native coefficients before profile application.
Replay adapters outside this package must pass the recorded profile
when constructing their environments. Pin task ID, evaluator ID, profile
definition, simulator revision, and the caller's horizon in experiment metadata;
scores from different physics conditions are separate evaluation conditions.
