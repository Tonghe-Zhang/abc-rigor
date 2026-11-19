# ABC Sim

`abc_sim` is the self-contained simulator package for ABC tasks. It owns the
MuJoCo scenes, task registry, randomization, reward/success evaluation, camera
rendering, and Gym-style environment API. Sim eval, RL, rendered observations,
task randomization, and new sim environments all work from this package alone.

**Contents**

- [Quickstart](#quickstart)
- [Environment API](#environment-api)
- [Step vs Step Chunk](#step-vs-step-chunk)
- [Tasks and Randomization](#tasks-and-randomization)
- [Rewards and Success](#rewards-and-success)
- [Sim Eval](#sim-eval)
  - [Eval flags](#eval-flags)
  - [Batched eval](#batched-eval)
  - [Scored tasks](#scored-tasks)
  - [The put-bottles task and its aliases](#the-put-bottles-task-and-its-aliases)
  - [Prompt defaults](#prompt-defaults)
  - [The multi-task sim policy](#the-multi-task-sim-policy)
  - [Finetuned per-task checkpoints](#finetuned-per-task-checkpoints)
  - [Matching the production eval numbers](#matching-the-production-eval-numbers)
- [Sim-only install](#sim-only-install)
- [Assets](#assets)
- [Rendering](#rendering)
- [Adding a New Env or Task](#adding-a-new-env-or-task)
- [RL Usage](#rl-usage)

## Quickstart

Install project dependencies from the repository root:

```bash
uv sync
```

Download simulator assets before creating environments — either all of them, or
just what one task needs:

```bash
uv run prepare.py --sim
uv run prepare.py --sim-task put_plastic_bottles_in_bin
```

Create and step an environment:

```python
import abc_sim

spec = abc_sim.get_task_spec("put_bottles")
env = abc_sim.make_env(
    task=spec.env_task,
    prompt=spec.prompt,
    camera_height=168,
    camera_width=224,
    max_episode_steps=236 * 15,
    terminate_on_success=True,
)

obs, reset_info = env.reset(seed=0, randomize=True)
obs, reward, terminated, truncated, step_info = env.step(env.action_space.sample())
env.close()
```

### Pack and pour fruit

`pack_and_pour_fruit` starts with three 60–75 mm fruits, a bowl with a 16 cm
opening and 6 cm height, and a shallow 34 cm receiving bowl. Apple, orange, lemon,
and tomato asset variants have textured visuals enclosed by spherical
collision geometry. Both bowls have solid floors and open, outward-flaring
walls. The receiver is large enough to catch all three fruits together.

```python
env = abc_sim.make_env(task="pack_and_pour_fruit", render_cameras=False)
env.reset(seed=0)
print(env.prompt)
```

Install assets with `uv run prepare.py --sim-task pack_and_pour_fruit`.

Both bowls use polypropylene density **900 kg/m³**, **2.5 mm walls** and a
**3 mm base**. MuJoCo infers their mass, center of mass and inertia from the
floor, convex wall segments and rounded rim geoms. The empty source bowl is
approximately **79.7 g**, and the receiver is **330.9 g**. The source's outer
diameter is about **16.6 cm**. Force allowances and support thresholds use the
compiled body weights.
The seed controls identities, fruit radii, five separated poses, two distinct
named colors from eight choices, and six prompt templates. Prompts name the
realized colors, such as “Place all the fruits into the blue bowl, then pour
the fruits from the blue bowl into the large white bowl. Finally, put the
empty blue bowl back on the table.”
Fruit collision geometry uses 800 kg/m³ density, giving 90–177 g per fruit;
mass and inertia scale with the sampled radius.
Serialized randomization restores geometry, colors, poses and language.
Reset options under `randomization` accept `fruits` (three
`category`/`variant`/`radius` dictionaries), `source_color`, `target_color`,
and `prompt_index`.

Reward stages are chronological:

| Stage | Maximum | Qualification |
| --- | ---: | --- |
| Packing | 0.2 | 0.2/3 per fruit in a supported pile in the small bowl, released and settled for 0.5 s |
| Transport | 0.2 | At least one qualified fruit lifted in the grasped bowl, then the load carried above or into the receiver |
| Emptying | 0.25 | Fraction of the carried load observed leaving the tilted bowl's opening after transport, while grasped or supported by the receiver |
| Catching | 0.25 | 0.25/3 per poured fruit settled, released and supported in the receiver for 0.5 s |
| Return | 0.1 | The carried load validly poured out and the small bowl empty, upright, released and directly supported by the table for 0.5 s |

Putting fruit directly into the receiver earns **zero**. Nesting the packed
small bowl inside the receiver earns **0.4 before force deductions** and
no pouring credit. Packing and catching remain proportional to all three fruits.
Packing two, transporting them, pouring both into the receiver, and returning
the empty bowl earns **0.85 before force deductions**:
`0.2 * 2/3 + 0.2 + 0.25 + 0.25 * 2/3 + 0.1`.
One fruit completing the same sequence earns 0.7. These are partial completions;
binary success still requires all three caught and the source bowl returned.
Transporting an empty bowl earns zero.

Packing and transport retain their completed milestones.
Before all three are packed, each qualified fruit retains its partial packing
credit while its center stays inside the small bowl, including while rolling
or being carried. This tolerates contact-solver penetration after qualification.
Its center leaving the bowl clears that fruit's partial credit until a qualified
bowl lift records it in the carried load; packing credit for carried fruit
persists through pouring. The fruits can
qualify at different times or on separate trips. Each fruit must be qualified,
contained and supported, or already recorded in a carried load, to complete
the permanent packing milestone. Fruits may
protrude above either bowl's rim or stack on other fruits, with a contact-support
path to that bowl. Above-rim fruit centers must stay over the opening; unsupported
fruit hovering over it cannot qualify.
Transport has a 0.1 s dwell; a nonempty bowl of qualified fruit must first have
been lifted at least 5 cm with a bowl grasp. Grasp acquisition requires both opposing
fingers of one hand contacting the bowl for 50 ms, tolerating gaps of up to
20 ms per finger during acquisition. Once acquired, either
finger can carry load; losing both contacts for more than 20 ms ends the hold.
Rim contact is measured independently and does not gate reward. Transport
allows a horizontal center distance up to the receiver's rim radius (17 cm),
with the source mouth's center at least 1 cm above the upright receiver's rim.
All fruit in the pending carried load must remain inside during the dwell, and the bowl must be
grasped or supported by the receiver. This allows base overhang and movement
during placement. Nesting also qualifies when the source floor is inside the receiver's
flared cavity, with upward support from the receiver and both bowls upright and
settled. The observer remembers fruit membership from qualified lifts; spilling
or manually removing one cannot shrink the emptying denominator. Later loads
add their fruit to this set and must earn transport proof for those fruits.
Emptying is normalized across this remembered set. A fruit placed directly
in the receiver cannot use another fruit's transport proof to earn catching.
Pouring requires
at least 30° tilt and the fruit's center leaving the opening's upper column, with no
direct robot-fruit contact. The source must either remain grasped and lifted
5 cm or be supported by the receiver within the transport zone. Receiver
support means at least 30% of the empty source bowl's weight carried by the
upright, table-supported receiver. This permits fruit to finish rolling out
after release onto the receiver; support gaps up to 20 ms are tolerated while
the source remains in the transport zone. A brushed fruit can re-arm for pouring
after 50 ms released onto the supported pile, tolerating 20 ms support gaps while
its collision bounds remain contained. A hand-carried fruit cannot.
A spill can earn emptying credit but cannot earn catching credit
after contacting the table. Re-entering the small bowl clears a fruit's
pouring credit. Removing a caught fruit clears its current catching credit.
Containment checks each sphere against the finite polygonal walls and floor,
allowing 1 mm of side-wall solver penetration. The opening's vertical column
admits above-rim piles; contact support is required for packing and catching.
Fruit settling limits `linear_speed + radius * angular_speed` to 3 cm/s;
bowls must move slower than 3 cm/s and 0.25 rad/s.
Return credit requires every fruit in the carried load to have validly poured
out and no fruit remaining in the small bowl. Its base must receive at least 60% of its
own weight from the table, its upright axis within about 11.5° of vertical,
and no robot contact throughout the 0.5 s dwell. Any table location qualifies.
Lifting or touching the bowl again revokes only this component. Catching every
fruit is independent of return credit; full task success requires both.

For a three-fruit load, gentle handling reserves 0.045 per fruit, 0.1275 for the
small bowl and 0.0375 for the receiver. A smaller load divides the fruit half
of emptying's force-sensitive share among its members; packing and catching
retain their three-fruit denominators. The `force_gentle_credit` metric reports
the current per-entity shares, totaling 0.3. Every solved native tick sums the magnitudes of all
external contact forces, including friction; opposing forces add. Dose is
`max(load - allowance, 0) * dt`, and 5 N·s consumes an entity's entire
gentle-handling share. Fruit allowance is 15 N. Bowls add their own weight
and twice the total fruit weight, accounting for contents and support
reaction. Damage never recovers before reset, and remains separate from
binary task completion. `force_allowance_n` and `force_dose_budget_ns`
are configurable evaluator options.

Metrics report the five reward components, per-fruit packing/emptying/catching,
stage milestones, retained packing masks, qualification masks and dwell times,
bowl support/readiness, containment/support/pour masks, bowl/rim grasp, transport
eligibility and dwell, return eligibility/dwell, center-distance transport zone, receiver support,
mouth crossings, qualified departures, floor containment, and load/peak/
excess-dose/potential-penalty columns in order
`[fruit_1, fruit_2, fruit_3, source_bowl, target_bowl]`.
Potential force penalties describe the loss if all affected stages are earned.
Use independent native environments for rollout workers: qpos-only batch
evaluation cannot reconstruct this contact history and raises explicitly.

### Pack and close a small suitcase

`pack_suitcase` starts with a free, rigid suitcase opened between **105 and
110 degrees**, plus exactly three loose household objects on the table.
The passive hinge has a **0–110 degree joint limit**; gravity holds the lid
against the opening stop and closes it after it passes vertical. MuJoCo's
joint stops are compliant, so solved angles can exceed a limit by a small
solver tolerance. The case has a carrying handle, a lid pull tab, solid
walls and floor, and an enclosing lid. The nominal footprint is 20 × 30 cm.
Seeded construction varies width **19–21 cm**, length **29–31 cm**, wall height
**8–10 cm**, wall thickness **6–12 mm**, floor thickness **9–14 mm**, and lid
thickness **12–20 mm**. Three handle shapes (bridge, rounded, tab), four strap
layouts (none, single, double, cross), and three shell details (smooth, ribs,
corner caps) give 36 construction combinations before continuous dimensions
and textures. Every combination has room for three maximum-size objects.

The seed selects one of **32 shell appearances** (eight palettes × four
procedural texture patterns), case construction and position/yaw, lid opening angle, object
identities/poses, and one of **12 equivalent prompts**. Seed 0 says:
“put all objects on the table into the suitcase and close it”.
Object categories are soap, sponge, apple, lemon, orange, potato, donut,
and yogurt. Uniform scaling limits each object to 8 cm in any dimension
and 5.5 cm in height, leaving room for all three side by side.

```python
env = abc_sim.make_env(task="pack_suitcase", render_cameras=False)
env.reset(seed=42, options={"randomization": {
    "texture": {"palette": "navy", "pattern": "checker"},
    "style": {"handle": "rounded", "straps": "none", "detail": "ribs", "wall_thickness": 0.007},
    "lid_angle_deg": 107,
}})
```

Optional `objects` is a list of three `{category, variant}` mappings from
the approved library. Replay restores all appearances, construction dimensions, object identities,
poses, opening angle and language from serialized `RandomizationState`.
Install the existing object pack with `uv run prepare.py --sim-task pack_suitcase`.

Packing earns **0.3 total**, or **0.1 per object**. Its entire collision bounds must be inside
the cavity, with a solved upward support path to the case floor, no robot
contact, and low linear/angular velocity for **0.5 uninterrupted seconds**.
The suitcase must be upright, settled, and supported on the tabletop for
initial packing qualification. Afterward, the object keeps its packing credit
while fully contained, including during case motion, lifting, or contact.
Leaving the cavity clears that qualification and requires settling again.
Containment uses the generated case's actual walls, floor and rim. Objects may
touch the side walls, allowing up to **1 mm** of contact-solver penetration.
Closing earns **0.4** and requires all three objects packed and the lid within
**3 degrees of closed**, with the case and contents released, the case supported
and settled, and the hinge settled
for another 0.5 seconds. Closing an empty or partly packed case adds no
credit. Removing an item or reopening the lid revokes current credit;
`ever_success` records history only.

Staying in place earns **0.3**, measured from the lower shell's pose at episode
reset. Translation is the full 3D distance; rotation is the shortest 3D angular
difference, independent of the lid angle. Translation keeps full credit through
**5 cm**, then decreases linearly to zero at **20 cm**. Rotation keeps full
credit through **5 degrees**, then decreases linearly to zero at **45 degrees**.
The staying reward is `0.3 * min(translation_fraction, rotation_fraction)`,
available only while the lid is **within 3 degrees of closed**. Opening the lid
removes this component immediately. It reflects the current/final pose:
returning the case restores pose credit while closed. An open reset scene earns zero.
Binary success requires packing, settled closure, and the case within both
pose deadbands.

Gentle handling accounts for **30%** of packing and closing: **0.03 per
object** and **0.12 for the suitcase**. On every native physics tick each
entity accumulates `max(contact_force_n - allowance_n, 0) * dt`.
The penalty is its gentle-handling credit times `min(dose / 5 N*s, 1)`,
and persists until reset. Each object's allowance is **15 N**. The case's
allowance is **15 N + its weight + twice the combined weight of all three
objects**: contents push down on the case and also increase its table
reaction. This fixed, episode-specific allowance prevents ordinary loaded
resting contact from continuously spending credit.

Loads sum magnitudes of all external contact forces including friction;
opposing fingers add. The lid and lower shell count as one suitcase entity,
so their internal contacts do not count. A press between an object and the
case loads both entities. Joint torques are not part of this contact-load
metric. Unpacked objects and incomplete closure earn zero rather than a
separate negative penalty; a completed task within both pose deadbands with
all force credit spent earns 0.79. Force penalties do not change binary success
or the staying-in-place component.
These are provisional shaping defaults, not physical damage thresholds.
Evaluator options are `object_force_allowance_n`,
`case_extra_force_allowance_n`, and `force_dose_budget_ns`.
Force metrics have columns `[object_1, object_2, object_3, suitcase]` and
include `contact_force_n`, `peak_force_n`, `force_threshold_n`,
`excess_force_n`, `force_dose_ns`, and `force_penalty`.
Pose metrics include `suitcase_translation_m`, `suitcase_rotation_deg`,
`suitcase_translation_fraction`, `suitcase_rotation_fraction`,
`suitcase_staying_fraction`, and `suitcase_staying_reward`.

Scoring requires native per-physics-tick observations; qpos-only/MJWarp
scoring cannot reconstruct the required support and dwell history.

### Stand three water bottles upright

`lift_bottles_upright` uses the ABC station with exactly three water bottles and
1–3 household distractors. Each bottle is sampled independently from 14 water-bottle
variants (`bottle_5` through `bottle_18`) and starts lying on its side.
Distractors come from 20 categories / 75 variants: mugs, bowls, coffee cups, cans,
canned food, food boxes, drink cartons, cereal, jars, yogurt, soap bars, sponges,
apples, lemons, oranges, tomatoes, potatoes, donuts, cupcakes, and trash bins.
The trash bins include the round orange bin and the blue mesh bin from the bottle
tasks. Categories are sampled without replacement; bins are distractors, not goals.
Full visual/collision footprints fit the table, with clearance from other objects
and the reset arm pose. Install its assets with
`uv run prepare.py --sim-task lift_bottles_upright`, then select the task through
`abc_sim.make_env(task="lift_bottles_upright")`.

Seed 0 prompts “lift the bottles upright”. Consecutive seeds cycle through
sixteen equivalent instructions, including “stand all three water bottles
upright” and “lift each bottle and set it down upright”. Each instruction is
also a task alias; reset records the selected wording in the episode metadata.
The seed controls bottle variants, distractor count/identities, positions, headings,
and wording. Repeating a seed reproduces the scene; serialized randomization
states restore all object identities and poses, including across changed counts.

To vary placement while retaining the same objects, pin the three bottle variants
and distractor identities through reset options:

```python
env.reset(seed=42, options={"randomization": {
    "bottle_variants": ["bottle_5", "bottle_11", "bottle_18"],
    "distractors": [
        {"category": "trash_bin", "variant": "blue_mesh"},
        {"category": "apple", "variant": "apple_10"},
    ],
}})
```

`distractor_variants(category)` in
`abc_sim.randomization.assets.tabletop_distractors` lists installed choices.
With no identity overrides, all assets are resampled each episode. The same seed
in all worlds already produces matching models. Assets use their native sizes.

Reward combines upright placement and gentle handling for each bottle.
Each bottle must satisfy all of these for **0.5 uninterrupted seconds**:

- Cap-up axis within 14° of vertical, base origin within 8 mm of table height
  (0.75 m), x in [0.34, 0.84] and y in [-0.53, 0.53].
- Upward solved tabletop contact at its base carrying at least 80% of its weight.
- No contact with the robot, other bottles, or distractors.
- Linear speed at most 0.03 m/s and angular speed at most 0.25 rad/s.

Any interruption clears that bottle's dwell and credit. All three must qualify
simultaneously for success. Historical maximum/ever-success metrics are diagnostics;
they do not preserve reward after a bottle falls. A bottle held upright, propped
against clutter, bouncing, or supported inside a trash bin earns no credit.
The table uses a finite collision box so objects pushed over the edge fall.

Each qualified bottle can earn **1/3**, of which **0.1** depends on its cumulative
excess contact-force dose. At every physics tick, sum the magnitudes of the 3D contact
forces against **all external bodies**, including normal and friction forces.
Two opposing 8 N fingers contribute 16 N; table, clutter and other-bottle impacts
also count. Torque components have units N·m and are excluded.

```text
force_i = sum over contacts ||force_xyz||
dose_i += max(force_i - 15 N, 0) * dt
penalty_i = 0.1 * min(dose_i / (2.5 N*s), 1)
reward_i = qualified_i * (1/3 - penalty_i)
reward = sum_i reward_i
```

Loads at or below **15 N** spend no handling credit. Above that threshold,
the deduction grows at **0.04 reward per N·s of excess force**, capped at 0.1.
For example, 20 N for 100 ms deducts 0.02, 20 N for 500 ms deducts the full 0.1,
and a single 2 ms tick at 80 N deducts 0.0052. Integration uses simulation
time and every solved tick; the reset baseline and score reads add no dose.
Separated bursts accumulate. That loss cannot recover after release, waiting,
falling or re-standing; only a new episode/reset clears it. The threshold keeps
resting weight and low-load holding from consuming the credit indefinitely.
Unqualified bottles earn zero, including their gentle-handling component.
Binary success still means all three qualify; reward can be 0.7 at success if
all three exhausted their gentle-handling credit. The evaluator accepts
`force_penalty_start_n` (default 15 N) and `force_dose_budget_ns` (default 2.5 N·s)
through `evaluator_options`. Metrics expose `bottle_contact_force_n`,
`bottle_excess_force_n`, `bottle_force_dose_ns`, `bottle_peak_force_n`,
`bottle_force_penalty`, and `bottle_reward`.
The peak remains diagnostic; it does not determine the penalty. The dose
adjusts the qualified-bottle score, not a separate negative step reward.

`mj_contactForce` reads MuJoCo's solved contact forces in Newtons, the same source
as the built-in `<contact>` sensor. Reading all contacts avoids a fixed sensor
slot limit and reuses the support/release checks above. Regression tests compare
the aggregate against that sensor under rest, a known load, and a balanced
squeeze. A `<force>` sensor measures the body-to-parent wrench and can miss
balanced compression; `<touch>` sums normal forces within a site but omits
friction. These are simulator contact loads, sensitive to timestep, solver and
collision/contact parameters; they are not calibrated physical bottle stress.

The evaluator observes **every solved native physics tick** through `env.step`,
`env.apply_action`, or `env.advance_physics`. Evaluation is a read of cached state;
repeated reads cannot accumulate dwell. Direct `mujoco.mj_step` callers must also
invoke the observer exactly once per tick. Qpos-only scoring is explicitly
unsupported. Independent native environments support seeded RL workers, including
identical scene seeds across 16 workers. ABC's current batched MJWarp backend does
not support these physics observers and rejects this task.

For a runnable example:

```bash
uv run python abc_sim/example_api_rollout.py --task put_bottles --render-cameras
```

## Environment API

`abc_sim.make_env(...)` returns a Gymnasium-compatible MuJoCo environment.

```python
env = abc_sim.make_env(
    task="put_bottles",
    render_cameras=True,
    camera_height=168,
    camera_width=224,
    max_episode_steps=3000,
    terminate_on_success=True,
)
```

The stable API contract is:

- `obs, info = env.reset(seed=..., randomize=True)`
- `obs, reward, terminated, truncated, info = env.step(action_14d)`
- `obs, history, reward, terminated, truncated, info = env.step_chunk(action_chunk)`
- `env.action_space.shape == (14,)`
- `env.chunk_action_space.shape == (chunk_dim, 14)`

The 14D policy-space action/state layout is:

```text
[left_j1..left_j6, left_gripper, right_j1..right_j6, right_gripper]
```

Observations are a dictionary with:

- `state`: `(14,)` float32 policy-space state.
- `images`: per-camera `(3, H, W)` uint8 RGB images when rendering is enabled.
- `prompt`: task prompt string.
- `masks`: per-camera availability masks.
- `camera_timestamps`: per-camera timestamps.

`env.observation_space`, `env.action_space`, and `env.chunk_action_space` are
defined for downstream Gym/RL integrations.

## Step vs Step Chunk

Use `step(...)` when the controller or RL policy emits one action at a time:

```python
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

Use `step_chunk(...)` when a behavior cloning policy emits a fixed action
chunk:

```python
chunk = policy(obs)  # shape: (chunk_dim, 14)
obs, history, reward, terminated, truncated, info = env.step_chunk(chunk)
```

`step_chunk(...)` repeatedly calls the single-action stepping path and returns
the final observation plus a `history` dictionary for the intermediate states.

## Tasks and Randomization

Task metadata lives in `abc_sim/task_specs.py`. Use the registry helpers to
resolve canonical task names, prompts, scene XMLs, randomizers, and evaluators:

```python
spec = abc_sim.get_task_spec("put_bottles")
print(spec.name, spec.env_task, spec.prompt)
```

Randomization is enabled by default for tasks with registered randomizers:

```python
obs, info = env.reset(seed=123, randomize=True)
randomization = info["randomization"]
```

`randomization` contains the sampled world metadata needed for replay or fixed
eval sets. Passing `randomize=False` resets the scene without sampling a new
world.

Custom `SimTaskSpec` objects can also be passed directly to `make_env`. See
[optional seeded spelling goals](../docs/seeded-spelling.md) for a configuration
that samples two, three, or four distinct letters without adding catalogue tasks.

`make_env(..., randomize_background=True, background_variation_asset_dir=...)` turns on
[background variations](../docs/background-variations.md). This opt-in, appearance-only randomization puts real wallpaper and table surface textures on the walls and the table; objects and physics are unchanged. `background_seed` drives a background stream that is independent of the task seed, and `background_id` pins one wallpaper × table combination.

## Rewards and Success

Custom tasks can opt into [conservative free-placement spelling](../docs/free-placement-spelling.md),
which uses native contacts and credits only unambiguous letter positions.

Task evaluators live under `abc_sim/task_eval/`. If a task has an evaluator,
`env.step(...)` and `env.step_chunk(...)` return task metrics in `info`:

- `info["task_reward"]`: scalar reward.
- `info["task_success"]`: boolean success flag.
- `info["task_eval"]`: evaluator-specific metrics.

Episode stopping is controlled by environment configuration:

- `max_episode_steps`: returns `truncated=True` when the action horizon is hit.
- `terminate_on_success=True`: returns `terminated=True` when the evaluator
  reports success.

You can also query the evaluator directly:

```python
result = env.evaluate_task()
if result is not None:
    print(result.to_info(squeeze=True))
```

## Sim Eval

Policy evaluation drives this same public environment API through the repo's
eval entrypoint, `eval_policy.py`. The default `--task
put_plastic_bottles_in_bin` and every other simulation task run through this
package's catalogue. The old spelling `put_bottles` remains an alias for the
same catalogue task; it no longer selects a separate scene or scoring rule. The
sim assets are not checked in, so download them first (see [Assets](#assets)):

```bash
uv run prepare.py --sim
```

```bash
# Put bottles in the bin, 20 worlds, one video per rollout.
uv run eval_policy.py \
    --checkpoint cache/bottles_75k.pt \
    --task put_plastic_bottles_in_bin \
    --num-worlds 20 --save-video --video-every-n-actions 15

# Any other task in the catalogue, by name or alias.
uv run eval_policy.py \
    --checkpoint cache/bottles_75k.pt \
    --task turn_mug_right_side_up \
    --num-worlds 20 --save-video --video-every-n-actions 15

# Output: $REPO/outputs/sim_eval_<task>/
#   summary.json     — success_rate, num_success (mean_max_bottles_in_bin
#                      only for the bottles tasks, which count bottles)
#   world_*.mp4      — per-world rollout videos (with --save-video)
```

The checkpoints to evaluate — your own finetunes, the 75k-step bottles-only
policy, or the multi-task sim policy — are described in the
[top-level README](../README.md#evaluation).

### Eval flags

- `--num-worlds N` — independent random scenes (default 5; use 50+ for
numbers you want to quote — at a true 10% success rate a 5-world run most
often reads 0).
- `--num-chunks N` — action chunks per rollout; each chunk is
`--execute-chunk-dim` actions (defaults: 236 chunks × 15 = 3540 sim steps,
the horizon the production sim-eval dashboard measured checkpoints with).
- `--rtc` / `--no-rtc` (default on) — RTC evaluation: each inference receives
the next `--rtc-prefix-length` (default 4) not-yet-executed actions and overlaps
inference with their execution. `--no-rtc` runs a synchronous loop.
- `--parallel-worlds N` — step N worlds together in MJWarp physics (default
0: one CPU MuJoCo world at a time). `--num-worlds` must be a multiple of N;
see [Batched eval](#batched-eval).
- `--randomization JSON` — reset request for the task randomizer, applied to
every world, e.g. `'{"bottle_count": 6, "randomize_variants": false,
"randomize_scales": false}'` (fields: `abc_sim/randomization/requests.py`).
For `conveyor_pick`, `'{"belt_speed_min_m_s": 0.05, "belt_speed_max_m_s": 0.30}'`
changes the belt speed bounds (default 0.10–0.20 m/s; either bound alone works).
Sorting accepts `nut_count` and/or `bolt_count` (1–20 each), or
`blocks_per_color` (1–9). Omitted counts retain the existing sampled ranges.
Sorting evaluation reports each world's scoring denominator as `active_object_count`.
- `--diffusion-steps N` — flow-matching Euler steps per inference
(default 10, matches production).
- `--checkpoint` — path to the `.pt` checkpoint to evaluate.
- `--norm-stats-path` — explicit `norm_stats.json` (otherwise uses the
one bundled in the checkpoint).
- `--fast-inference` / `--no-fast-inference` (default on) — bf16 +
torch.compile + CUDA-graph captured `sample_actions`. ~5× faster
inference; first call pays a one-time ~25 s compile cost.
- `--vanilla-physics` / `--no-vanilla-physics` (default off) — no-op:
every task already steps physics in CPU MuJoCo. Kept for command-line
compatibility.

Note that the first launch compiles MJWarp's CUDA kernels (~1 min).

`--camera-backend mujoco` renders with CPU MuJoCo instead of MJWarp (`blender`
renders with Cycles, see [Rendering](#rendering)): slow, but
it runs anywhere MuJoCo does, which makes it the way to smoke-test on a laptop
(including macOS). On a headless Linux box, set `MUJOCO_GL=egl`.

### Batched eval

`--parallel-worlds N` steps N worlds together in MJWarp physics with one
batched policy call per chunk; RTC, prefix conditioning, `--save-video`, and
`summary.json` are unchanged. A batch shares one compiled model, so pin the
randomization that would recompile it (object counts, variants, scales) with
`--randomization`; poses and colors still vary per world. The sorting tasks
batch with `{"park_inactive": true, "bin_visual_style": "stackable"}`, which
keeps every candidate object in the scene and parks the unused ones behind
the robot. Tasks with a per-step runtime (conveyor pick, ball tray, multi-drawer search) and evaluators
that need MjData or per-episode targets are sequential-only.

MJWarp is a different physics engine: absolute success rates differ from the
CPU MuJoCo loop (put-bottles with the 200k policy: 0.71–0.77 batched vs 0.54
sequential on the same pinned scenes), runs are not seed-reproducible, and every
world starts from the home pose. Compare checkpoints within one backend.

On one H100 a 100-world put-bottles eval takes about 8 minutes after a one-off
compile per batch width (~15 min cold, ~1 min cached), against about an hour
sequentially; the ray-traced render (~7 ms per world per chunk) is the ceiling.

### Scored tasks

All but four tasks ship with a success evaluator, so their `success_rate` is a
real measurement. The unscored four — `put_markers_in_top_drawer`,
`put_markers_in_middle_drawer`, `put_markers_in_bottom_drawer`, and
`build_wood_block_tower` — still roll out and record videos, but have no way to
score themselves: they warn at startup and report reward 0 and success 0
throughout. Watch the videos to judge those.

`ball_tray_balancing` asks the policy to keep the ball balanced on the tray for
10 to 15 seconds, then gently place the tray on the table. The hold counts while
the tray centre is at least 5 cm above its reset height, with the ball within
the tray and no gripper touching it. Placing the tray with both handles gripped
after 10–18 seconds up, then keeping the ball on it for 1 second, succeeds:
1.0 for the asked 10–15 seconds, down to 0.8 at 18 seconds (18 s covers how long
the demonstrations hold). Placing earlier keeps half the partial reward reached
(0.9 × seconds / 10), and placing later keeps the reward reached (falling from
0.9 after 15 s to 0.5 at 23 s), both without success; touching or
losing the ball after 10 seconds scores 0. Success latches and ends the rollout
early. `--num-chunks` sets the time budget for completing the task;
it does not set the required hold duration.

See [the success conditions](../docs/success_conditions.md) for the scoring
details. Earlier ball-tray results used a maintenance criterion and are not
directly comparable with the current hold-and-lower success rate.

Task names, aliases, and prompts are listed in `abc_sim/task_specs.py`, and an
unrecognized `--task` fails immediately with the full catalogue in the error.

### The put-bottles task and its aliases

`put_bottles` and `water_bottles` are aliases of the catalogue task
`put_plastic_bottles_in_bin`; every spelling builds the same environment and
scores with the same shared evaluator, which scores each bottle's centre of
mass against the tapered interior measured from the bin's collision mesh, with
a height cap based on the bottle's own collision reach. (Until August 2026,
`put_bottles` selected a separate scene that `eval_policy.py` built itself —
6 bottles near unit scale from a synthetic home pose. That native path is
retired; numbers measured in it live in the archived eval summaries and are
reproducible from git history, and are not comparable with catalogue-task
numbers, which randomize 2–6 bottles, bin scale up to 1.4×, and start from the
real robot's teleoperation pose.)

### Prompt defaults

The prompt defaults to `"sim "` plus the task's own prompt, so
`--task turn_mug_right_side_up` is prompted `sim turn mug right side up`. That is
how sim episodes are labelled in the training mixture. If `model.pt` has a
neighbouring `model.json` metadata sidecar, its `sim_prompt_map` supplies the
exact prompt that checkpoint saw for each task during training. An explicit
`--prompt` wins over the sidecar, and the sidecar wins over the catalogue
default.

### The multi-task sim policy

`abc_dit_xl_200k_model.pt` is the DiT-XL parent behind `--load-pretrained`
finetuning (see the [top-level README](../README.md#abc-dit-training)), trained on the
3.5k-hour real mixture plus 98 hours of sim across five tasks. It is also the
checkpoint to evaluate those five tasks with:

```bash
uv run prepare.py --pretrained --sim
uv run eval_policy.py \
    --checkpoint cache/abc_dit_xl_200k_model.pt \
    --task put_plastic_bottles_in_bin \
    --num-worlds 20 --save-video
```

The five tasks it saw, and the prompt each one's episodes carried:

| `--task` | prompt it trained under |
| --- | --- |
| `put_plastic_bottles_in_bin` | `sim throw plastic bottles in bin` |
| `sweep_away_paper_scraps_from_table` | `sim sweep away paper scraps from the table` |
| `turn_mug_right_side_up` | `sim turn mug right side up` |
| `load_plates_into_dish_rack` | `sim load plates into tabletop dish rack` |
| `hang_mug_on_mug_rack` | `sim hang the mug on the mug rack` |

The training mixture remapped its sim task names, and prompts are derived from the name after
the remap, so the put-bottles episodes were labelled with the *throw* prompt.
The release ships an `abc_dit_xl_200k_model.json` sidecar
recording the map; `eval_policy.py` reads it back and prints the prompt it chose.
Keep the sidecar beside the `.pt` — which is what `--pretrained` does — and the
defaults are correct. This checkpoint's `max_action_prefix` is 8 during training.

Two caveats specific to it:

- The genuine throw-bottles sim scene (`--task bottles`) was left out of the
  training mixture as physically distinct from the real throw data, so expect
  roughly 1 in 4 there. Evaluate `put_plastic_bottles_in_bin` instead.

Read `mean_max_progress` from `summary.json` for numbers comparable to the
paper's sim-progress figures, and `success_rate` for the strict metric.

### Finetuned per-task checkpoints

Every sim task below has a checkpoint finetuned from `abc_dit_xl_200k_model.pt`
for 25k steps on that task's sim_224 episodes, published with its rollout
videos and eval stats. They are described by a checkpoint manifest
(`https://abc-data.timehorizons.org/finetuned_sim/manifest.json`): per task the
published steps with sizes and sha256, the recommended step (best combined
eval successes — not always 25k), and the exact prompt to evaluate under.
Download through it (no credentials needed; sha256-verified):

```bash
uv run prepare.py --sim-checkpoint-list       # what's published, with results
uv run prepare.py --sim-checkpoint pour       # the recommended step
uv run prepare.py --sim-checkpoint pour@25000 # a specific step
# Lands in cache/finetuned_sim/<task>/<step>.pt (~8 GB: model-only, all any
# eval or viz needs) and prints the exact eval command. Pass
# --sim-checkpoint-full-state for the ~24 GB training-state file (optimizer
# included, valid for --resume-from). Raw URLs follow
#   https://abc-data.timehorizons.org/finetuned_sim/<task>/checkpoints/<step>.pt
# with the model-only variant beside it at .../<step>_model.pt
```

Expected results, measured as strict `success_rate` over 20 randomized worlds
per seed (two seeds: 20260511 and 20260512, the default RTC evaluation, the
default 236-chunk budget), with `mean_max_progress` beside them. The prompt column is
the exact string the checkpoint trained under -- evaluate with it (for the
directive tasks the env rewrites the live prompt each reset, so any seed
prompt works):

| `<task>` (URL path) | trained + eval prompt | success (seed A / B) | progress |
| --- | --- | --- | --- |
| `pour` | `sim pouring beads` | 13/20 / 16/20 | 0.65 / 0.80 |
| `sim_inhand_transfer_the_item_to_other_side` | `sim inhand transfer the item to other side` | 18/20 / 18/20 | 0.96 / 0.97 |
| `put_relative` | per-episode directive | 5/20 / 10/20 | 0.73 / 0.83 |
| `grab_clutter` | per-episode directive | 5/20 / 3/20 | 0.25 / 0.15 |
| `conveyor_pick` | `conveyor pick` | 2/20 / 2/20 | 0.38 / 0.37 |
| `sim_sweep_away_paper_scraps_from_the_table` | `sim sweep away paper scraps from the table` | 2/20 / 1/20 | 0.26 / 0.14 |
| `count_into_opaque_box` | per-episode directive | 1/20 / 0/20 | 0.28 / 0.17 |
| `lego_blocks_sorting` | `lego blocks sorting` | 0/20 / 1/20 (400 chunks, see below) | 0.74 / 0.75 |
| `ball_tray_balancing` | `grab the tray handles and keep the ball balanced on the tray` | 1/20 / 0/20 (historical maintenance criterion; see below) | 1.00 / 1.00 |
| `nuts_bolts_sorting` | `nuts bolts sorting` | 0/20 / 0/20 | 0.08 / 0.12 |
| `sim_put_the_plastic_bottles_in_the_bin` | `sim put the plastic bottles in the bin` | **19/20 / 18/20 (20k)** | 0.99 / 0.95 |
| `sim_load_the_plates_into_the_dish_rack` | `sim load the plates into the dish rack` | **15/20 / 14/20 (20k)** | 0.85 / 0.87 |

put-bottles and dishrack peak at 20k (found by 5k-interval sweeps), so their
`checkpoints/20000.pt` is published beside `checkpoints/25000.pt` and is the
one to reach for.

Every checkpoint above is retrained on the v3 release
(the action-alignment fix documented in the release bucket's
`notes/action_alignment.md`); rows update as each task's retrain is evaluated,
and each task's superseded checkpoints move to `checkpoints_pre_v3/` alongside.
Rows before the conveyor retrain were measured under MuJoCo 3.6; the repo now
pins 3.8, and contact-heavy tasks are sensitive to the engine step (the
dishrack 20k checkpoint re-measures at 12/20 on seed A under 3.8).

The published ball-tray row and progress values were measured under the
previous maintenance criterion. Every world earned full hold-duration credit,
but the criterion asked the policy to hold roughly 4x longer than any
demonstration, making the zero a horizon artifact. These measurements predate
the sampled hold-and-lower criterion and require a new evaluation to report
current success. For LEGO, 73% of its demonstrations are longer than the default
236-chunk budget, so its row is measured at `--num-chunks 400` (its published
eval summaries too); at the default budget every world times out mid-sort. The
inverted-convention and directive-relabel history behind nuts_bolts_sorting
and put_relative is documented in their evaluator docstrings.

How each training prompt is derived (and why eval must match it): an episode's
`episode_metadata.json` `prompt_timeline` wins when it has one, resolved per
frame -- `multi_drawer_search` advances through 2-4 targets mid-episode; else
`instruction` when that carries a real directive; otherwise the dataset
`task_name` with underscores as spaces. Two
historical traps worth knowing: the 200k parent pretrained the put-bottles
scene under the THROW wording (`sim throw plastic bottles in bin`) and
dishrack under `sim load plates into tabletop dish rack` -- those sidecar
prompts apply when evaluating the PARENT, while a checkpoint finetuned on the
sim_224 episodes trains and evaluates under the dataset-derived wording
(`sim put the plastic bottles in the bin`, `sim load the plates into the dish
rack`).

To reproduce a finetune end to end (any dataset task name from
`prepare.py --sim-data-list`):

```bash
# 1. Episodes + assets into a fresh per-task cache root, parent alongside.
uv run prepare.py --sim-data lego_blocks_sorting --cache /data/ft/lego/cache
uv run prepare.py --pretrained --cache /data/ft/lego/cache

# 2. 25k steps on 2 GPUs -- the recipe every published checkpoint used
#    (batch 90/GPU, LR 1e-4, checkpoints every 5k under finetune_checkpoints/).
torchrun --standalone --nproc-per-node 2 train.py \
    --cache-root /data/ft/lego/cache \
    --load-pretrained --mixture-preset sim_task \
    --flow.max-action-prefix 8 --train-steps 25000 \
    --val-every 5000 --val-batches 40 --ckpt-every 5000

# 3. Evaluate with the prompt the run trained under (printed at startup).
uv run eval_policy.py \
    --checkpoint /data/ft/lego/cache/finetune_checkpoints/25000.pt \
    --task lego_blocks_sorting --prompt "lego blocks sorting" \
    --num-worlds 20 --save-video --video-every-n-actions 15
```

### Matching the production eval numbers

The 236-chunk horizon matches the production sim-eval protocol. The tables in
this README are measured under the default RTC evaluation; production-era
dashboards used an older synchronous inference mode, so their absolute numbers
are not directly comparable (background: the release bucket's
notes/action_alignment.md).
Two reading notes when comparing against those numbers:

- `mean_max_progress` — mean over worlds of the best instantaneous progress
  fraction — is the statistic the production dishrack eval aggregated, and the
  right column to compare for any task where progress can be undone (a plate
  knocked back out, or still touching the gripper at the end, subtracts from
  `mean_reward` but not from `mean_max_progress`). `success_rate` is stricter
  than both: every object placed simultaneously, at any instant. For the
  current ball-tray task, reward increases with uninterrupted hold progress
  up to 0.9, stays at 0.9 until a new table contact, then reaches 1.0 on latched
  success. Its published historical maintenance results used different reward
  semantics and should be read under that earlier criterion.
- Training remaps some sim task names before they become prompts, so two
  catalogue tasks read best under a prompt that differs from their spec
  default. For checkpoints trained on the xdof + sim mixtures: the put-bottles
  scene's sim data trained as `--prompt "sim throw plastic bottles in bin"`,
  and hang-mug as `--prompt "sim hang the mug on the mug rack"`. Mug flip,
  dishrack, and sweep defaults already match training.

## Sim-only install

`abc_sim/pyproject.toml` packages the simulator alone (MuJoCo + NumPy), without
the training stack the root project pulls in. Install it editable from a
checkout, because scenes resolve assets next to the package:

```bash
pip install -e abc_sim            # or: uv pip install -e abc_sim
python prepare.py --sim-task ball_tray_balancing   # needs only tyro
```

## Assets

Large simulator meshes and textures are not committed to git. They ship as 37
per-package tarballs listed in `abc_sim/models/assets_manifest.json` (640.5 MB to
download, 1.7 GB unpacked) and install into `abc_sim/models/assets/`:

```bash
uv run prepare.py --sim                                   # every package
uv run prepare.py --sim-task load_plates_into_dish_rack   # one task's packages
uv run prepare.py --sim-list                              # sizes and file counts
```

`--sim-task` takes task names, aliases, or prompts — anything
`abc_sim.get_task_spec` resolves — and installs only the packages that task's
scene loads:

| Task | Packages | Download |
|---|---|---|
| `throw_plastic_bottles_in_bin` | `i2rt_yam` | 4.6 MB |
| `ball_tray_balancing` | `i2rt_yam` | 4.6 MB |
| `sort_lego_blocks` | `i2rt_yam`, `task_bins` | 4.7 MB |
| `sort_nuts_and_bolts` | `i2rt_yam`, `task_bins`, `task_nuts_bolts` | 4.9 MB |
| `sweep_away_paper_scraps_from_table` | `i2rt_yam`, `brush_flat`, `dustpan`, `garbage_can`, `paper_ball` | 8.5 MB |
| `turn_mug_right_side_up` | `i2rt_yam`, `mug`, `tray`, `task_mug_flip` | 18.1 MB |
| `multi_drawer_search` | `i2rt_yam`, `blocks`, `task_bins`, `task_multi_drawer_search` | 26.6 MB |
| `put_plastic_bottles_in_bin` | `i2rt_yam`, `task_water_bottles` | 45.7 MB |
| `load_plates_into_dish_rack` | `i2rt_yam`, `task_dishrack` | 58.2 MB |
| `grab_specific_object_from_clutter` | `i2rt_yam`, `task_bins`, `task_grab_clutter` | 72.9 MB |

Sizes are decimal MB, the same convention `--sim-list` prints. The table covers
the ten most-downloaded scenes; the other mapped tasks (`inhand_transfer`,
`conveyor_pick`, `count_into_opaque_box`, `put_relative`, `mug_tree`, `pour`,
`chess`, `blocks`) resolve the same way, with `inhand_transfer` additionally
needing the RoboCasa object packs described below. `task_bins`, the bin fixtures the search and sorting tasks
share, is a different package from `bin`, which no mapped task uses — the two are
easy to mix up once abbreviated, so the table spells them out.

`--sim` and `--sim-task` are a union, not a filter, so
`--sim --sim-task sort_lego_blocks` installs all 37 packages rather than that
task's 2. Pass `--sim-task` on its own to get the download in the table.

Every archive is checked against the manifest SHA256 before it is unpacked, and
each installed package records its hash under
`abc_sim/models/assets/.abc_sim_asset_packages/`, so reruns skip work that is
already done. Add `--sim-force` to reinstall, or `--sim-package <name> <name>`
to install packages by manifest name (space-separated; repeating the flag keeps
only the last name).

A package that cannot be fetched or fails its hash check does not stop the run:
the others still install, the closing summary lists each failure with its
reason, and the exit status is nonzero so a partial install is not mistaken for
a complete one. Re-running retries only what is missing.

To install from tarballs you already have (an offline mirror, a shared
filesystem, a package that is not on the mirror yet), point `--sim-source` at
the directory. Anything it does not carry still comes over https:

```bash
uv run prepare.py --sim --sim-source /path/to/archives
```

That fallback makes a partial directory useful — a handful of tarballs top up an
otherwise complete install without any extra flags — but it does mean
`--sim-source` is not an offline switch. Combined with `--sim` on a fresh clone,
a directory holding four archives installs those four and fetches the other 33
from the mirror. Name the packages you want with `--sim-package` if that is not
what you meant.

The `inhand_transfer` scene additionally needs RoboCasa's public object packs.
Those come from `utexas.box.com` rather than the ABC mirror and are a 2.8 GB
download (7.5 GB unpacked), so they are opt-in and `--sim` leaves them out:

```bash
uv run prepare.py --sim-robocasa
```

## Rendering

Live environment rendering is selected through `camera_backend`:

```python
env = abc_sim.make_env(
    task="put_bottles",
    render_cameras=True,
    camera_backend="mujoco",
    camera_height=168,
    camera_width=224,
)
```

Supported paths:

- `camera_backend="mujoco"`: standard MuJoCo renderer for local/single-env
  rendering. Requires a working OpenGL context.
- `camera_backend="mjwarp"`: GPU renderer used by policy evaluation.
- `camera_backend="blender"`: path-traced (Blender Cycles) renders of the same
  cameras, one Blender process per environment, for evaluating policies under a
  photoreal domain. About 190 ms per three-camera observation at 224x168 on an
  H100 (16 samples, denoised) plus ~4 s of startup per reset, so one world at a
  time. Needs a Blender 4.2+ binary (`BLENDER=/path/to/blender`), `usd-core` in
  the Python environment and optionally `ABC_HDRI` for a studio HDRI;
  `ABC_BLENDER_SAMPLES` / `ABC_BLENDER_DENOISER` trade quality for speed. See
  [abc_sim/rendering/blender/README.md](rendering/blender/README.md).
- `camera_backend="madrona"`: not supported in this release. The Madrona
  renderer needs a build that is not part of the public package, so use
  `mujoco` or `mjwarp` instead.

**Headless GPU boxes: check your EGL vendor.** Many cloud images ship the CUDA
userland without NVIDIA's OpenGL libraries, and `MUJOCO_GL=egl` then silently
falls back to Mesa's software rasterizer — everything works, just 10–40× slower
wherever the `mujoco` backend renders (we measured 8 vs 90+ frames/s). Check
`ls /usr/share/glvnd/egl_vendor.d/`: if there is no `10_nvidia.json`, install
the GL package matching `nvidia-smi`'s driver version, e.g.
`sudo apt-get install libnvidia-gl-580-server` for a 580.x server driver (no
reboot needed for headless EGL). On multi-GPU machines also set
`MUJOCO_EGL_DEVICE_ID` to the physical GPU index — EGL enumerates all GPUs
regardless of `CUDA_VISIBLE_DEVICES`.

The camera renderers behind these backends live under `abc_sim/rendering/`.

## Adding a New Env or Task

At minimum, a new task needs:

1. A MuJoCo scene XML under `abc_sim/models/`.
2. Any required meshes/textures under `abc_sim/models/assets/`.
3. A `SceneTaskSpec` entry in `abc_sim/task_registry.py`.
4. A `SimTaskSpec` entry in `abc_sim/task_specs.py`.

Most tasks should also add:

- A randomizer under `abc_sim/randomization/tasks/`, then register it in
  `abc_sim/randomization/registry.py`.
- A task evaluator under `abc_sim/task_eval/`, then register it in
  `abc_sim/task_eval/registry.py`.
- Runtime hooks under `abc_sim/task_runtime/` if the environment has dynamic
  behavior beyond ordinary MuJoCo stepping.
- Asset package metadata in `abc_sim/models/assets_manifest.json` if the new
  task needs large mesh or texture files, plus an entry in `prepare.py`'s
  `SIM_TASK_PACKAGES` map so `--sim-task` can install them.

Keep simulator-owned behavior inside `abc_sim`. Teleop/data-collection logic
belongs outside this package.

## RL Usage

For RL, use one-action stepping and Gymnasium termination semantics:

```python
import abc_sim

env = abc_sim.make_env(
    task="put_bottles",
    render_cameras=False,
    max_episode_steps=1000,
    terminate_on_success=True,
)

obs, reset_info = env.reset(seed=0, randomize=True)
done = False
while not done:
    action = policy(obs)  # shape: (14,)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()
```

Use `info["task_success"]` and `info["task_eval"]` for success-conditioned
metrics, curricula, or offline analysis. Use `reset_info["randomization"]` to
record and replay fixed worlds across policies.
