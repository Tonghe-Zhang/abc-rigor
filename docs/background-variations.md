# Background variations

`randomize_background` is an opt-in, appearance-only domain randomization. It replaces the plain walls and table top with textures photographed from real robot stations: a wallpaper design on the three walls and a table surface on the table. It is off by default; with it off, the compiled scene is exactly the same as before.

**Visual only.** Objects, collision geometry and task physics do not change. The background is four textured quads laid just inside the three walls and on the table top:
- `contype = conaffinity = 0`;
- no mass;
- attached to the world body.

Trajectories with and without backgrounds are bit-identical for the same seeds and actions (`tests/test_background.py`).

- [Quick start](#quick-start)
- [Parameters](#parameters)
- [Per-episode API](#per-episode-api)
- [Combinations, ids and seeding](#combinations-ids-and-seeding)
- [Large-scale parallel runs](#large-scale-parallel-runs)
- [Rendering backends and batched envs](#rendering-backends-and-batched-envs)
- [Background package layout](#background-package-layout)
- [Viewer](#viewer)
- [Troubleshooting](#troubleshooting)

## Quick start

```python
import abc_sim

env = abc_sim.make_env(
    task="sort_lego_blocks",
    randomize_background=True,
    background_variation_asset_dir="/path/to/background_package",   # or an s3:// URI
    background_seed=1234,
)
obs, info = env.reset(seed=0)
print(info["background"])   # {'background_id': 617, 'wallpaper': '<design>/<instance>', 'table': '<type>/<instance>'}
obs, info = env.reset(seed=1)   # a new background, drawn from the background stream
```

The eval CLI takes the same options:

```bash
uv run eval_policy.py --checkpoint <ckpt> --task sort_lego_blocks \
    --randomize_background \
    --background_variation_asset_dir /path/to/background_package \
    --background_seed 1234
```

Flags accept both spellings, `--randomize_background` and `--randomize-background`.

## Parameters

| Parameter | `make_env` | Eval CLI | Type | Default |
|---|---|---|---|---|
| Enable | `randomize_background` | `--randomize_background` | bool | `False` |
| Asset package | `background_variation_asset_dir` | `--background_variation_asset_dir` | str / path | `None` |
| Background seed | `background_seed` | `--background_seed` | int or `None` | `None` |
| Pinned combination | `background_id` | `--background_id` | int or `None` | `None` |
| When to redraw | `background_resample` | (not exposed) | `"episode"` or `"env"` | `"episode"` |

### `randomize_background`

The master switch, off by default.

- **Off:** nothing about the scene changes: no extra geoms, no extra textures, no package needed. Passing `background_seed` or `background_id` without it raises a `ValueError`, so a typo can't silently do nothing.
- **On:** the env adds the four background texture slots to every scene it compiles (base scene and randomizer reloads). Every randomized `reset` then shows a background from the package (see `background_resample`).

### `background_variation_asset_dir`

Where the background package lives. It is required when `randomize_background=True`; there is no built-in default location.

- **Local directory:** used as is. It must be the package itself, the directory holding `index.json` (for example `.../background_variations/v1`).
- **`s3://bucket/prefix` URI:** synced once per machine into `$ABC_SIM_CACHE_DIR/background_variations/<hash>/` (default `~/.cache/abc_sim/...`).
  - The sync runs under a file lock, so hundreds of workers on one node download one copy; the rest wait, then reuse it.
  - It needs the AWS CLI on `PATH` (`aws s3 sync`), with credentials from the usual AWS environment. The URI must point at the package itself, the prefix holding `index.json`.
  - A completed sync is marked with a `.complete` file. Delete that cache directory to force a fresh download.

### `background_seed`

Seed of the background stream. This random stream is used only to choose backgrounds.

- It is independent of the task / episode `seed` you give `reset(seed=...)`:
  - the same `background_seed` gives the same background sequence whatever task seeds you use;
  - the same task seed gives the same objects whatever `background_seed` you use.
  - So task and background randomness are fully decoupled.
- `None` (default) draws OS entropy in `make_env`: every process gets a different, non-reproducible sequence. The eval CLI instead derives it from `--seed`, so evals stay reproducible; the stream is still separate from the task stream.
- Episode `k` of an env uses draw `k` of its stream, which makes runs reproducible: same `background_seed`, same order of randomized resets → same backgrounds.

### `background_id`

Pins one wallpaper × table combination, an integer in `[0, n_combinations)` (54 × 16 = 864 for the current package).

- **What stays fixed:** the wallpaper design and the table type. The instance within each (the same print photographed in different stations and lighting) is still drawn from the background stream, so a pinned id still varies slightly in lighting and placement.
- **Out of range:** raises a `ValueError` when the env is created.
- **Unset (`None`):** every draw picks the combination uniformly at random.
- **Per episode instead of per env:** use `reset(options={"background_id": k})`, see below.

### `background_resample`

When a new background is drawn:

- **`"episode"`** (default): every randomized `reset` (`randomize=True`, the default) draws a new background.
- **`"env"`:** the first draw is kept for the env's lifetime.

In both modes:
- Replay resets (`reset(randomize=False)`) keep the current background (unless `options={"background_id": k}` is given), and still report it in `info["background"]`.
- An explicit `options={"background_id": k}` always applies.
- In `make_batched_env`, every world gets its own background and all of them are redrawn on every batched `reset`; see [batched envs](#rendering-backends-and-batched-envs).

## Per-episode API

| Call | Effect |
|---|---|
| `env.reset(seed=s)` | Randomized reset. With `background_resample="episode"`, also draws the next background from the background stream. |
| `env.reset(seed=s, options={"background_id": k})` | Shows combination `k` for this episode (instances still drawn from the stream). |
| `env.reset(seed=s, randomize=False)` | Replay reset: keeps the current background. |
| `info["background"]` | Returned by every reset when enabled: `{"background_id", "wallpaper", "table"}`. The last two are instance ids (`design/instance`, `type/instance`). Log it with your episode to reproduce or analyse it. |
| `env.background_info` | The current draw, the same dict; `None` when disabled. |
| `env.set_background(k=None)` | Draws (or pins `k`) and shows it immediately, without a reset. Raises when disabled. |

## Combinations, ids and seeding

A package has `W` wallpaper designs and `T` table surface types:

```
background_id = wallpaper_index * T + table_index      # both indices into the sorted type names
```

```python
from abc_sim.background import BackgroundLibrary

lib = BackgroundLibrary("/path/to/background_package")
lib.n_combinations                 # 864
lib.combination(617)               # ('<wallpaper design>', '<table type>')
lib.combination_id(w, t)           # the reverse
lib.wall_types, lib.table_types    # the sorted name lists the ids index into
```

Because both lists are sorted, ids are stable for a given package version. A new package version that adds types renumbers them, so log `info["background"]`: it records names, not only the id.

The stream is `np.random.default_rng(SeedSequence(background_seed, spawn_key=(0xB60001,)))`, a different stream from `default_rng(background_seed)`. Each draw takes the combination (uniform over all ids, unless pinned), then one instance of the wallpaper design and one of the table type (uniform).

## Large-scale parallel runs

```python
env = abc_sim.make_env(
    task=task,
    randomize_background=True,
    background_variation_asset_dir="s3://<bucket>/<prefix>",   # synced once per node
    background_seed=BASE_BG_SEED + global_rank,                # distinct stream per worker
)
for episode in range(n):
    obs, info = env.reset(seed=task_seed(global_rank, episode))  # task seeds stay yours
    log(episode, info["background"])
```

- **Distinct, reproducible streams:** give each worker its own seed, `base + rank`. With `background_seed=None`, every worker draws OS entropy: distinct but not reproducible.
- **Reproducibility:** the background sequence depends only on `background_seed` and the number of randomized resets, never on task seeds or worker scheduling.
- **Cost:** a background swap is a texture write plus a texture re-upload, with no model recompilation. On a CPU-only machine (software OSMesa GL, 224×168 cameras, `build_wood_block_tower`), `reset` + observation took 1419 ms median with backgrounds against 1354 ms without, so about +65 ms. A hardware GL context uploads faster.
- **Memory:** the four slots add about 6 MB of texture data to each compiled model, and the few most recent instances' slot images are cached. The same run measured +40 MB RSS at construction, and +47 MB against +39 MB without backgrounds over 40 resets. Use `background_slot_scale=0.5` to quarter the texture memory on large CPU fleets.

## Rendering backends and batched envs

| Backend | Per-episode swap |
|---|---|
| `camera_backend="mujoco"` (CPU) | In place: only the four background textures are re-uploaded. |
| `camera_backend="mjwarp"` | In place: the four textures are replaced in the live MJWarp render context (with the wall gain below). |
| `madrona`, `blender` | These copy textures when created, so a swap rebuilds the camera provider. That's correct but slower. |
| `make_batched_env` (MJWarp worlds) | Per world. See [Batched MJWarp envs](#batched-mjwarp-envs). |

### Batched MJWarp envs

Every world of a batched env shows its own background, and each `reset(seeds=...)` redraws all of them from the background stream.

```python
env = abc_sim.make_batched_env(
    "sort_lego_blocks", num_worlds=64,
    randomize_background=True,
    background_variation_asset_dir="/path/to/background_package",
    background_seed=1000 + rank,
    background_pool_size=64,     # texture sets in the model; world w uses set w % pool
    background_slot_scale=0.5,   # batched cameras are low-res: half-size slots, about 8 MB GPU texture per set
)
obs = env.reset(seeds=episode_seeds)               # env.background_info: one dict per world
obs = env.reset(seeds=episode_seeds, options={"background_id": 17})   # every world shows combination 17
```

**How it works.** The render context holds `min(num_worlds, background_pool_size)` texture sets: set 0 in the model's four background textures, and the rest appended to the render context only, so the MuJoCo model stays small. MJWarp's renderer reads each world's material texture ids (`mat_texid`) per world, so world `w`'s four background materials point at set `w % n_sets`. A reset draws one background per set and replaces just those textures in the live render context; the decoded package is cached, so redraws don't re-read files. Geometry, bodies and the physics model are untouched; trajectories are bit-identical with and without backgrounds (`tests/test_background.py::BatchedTest`).

| Parameter (`make_batched_env`) | Default | Meaning |
|---|---|---|
| `background_pool_size` | `64` | Maximum distinct backgrounds per batch. With more worlds than sets, world `w` shares set `w % n_sets`. Lower it to save GPU memory. |
| `background_slot_scale` | `0.5` | Texture resolution relative to the package's. 1.0 is about 33 MB of GPU texture per set (MJWarp stores float RGBA). |

The other options (`background_seed`, `background_id`, `background_variation_asset_dir`) mean the same as for `make_env`.

MJWarp's renderer does not apply material emission: a vertical wall only gets its ambient term (0.125). Wall textures uploaded to MJWarp, whether batched or the `mjwarp` live backend, are therefore scaled by `wall_emission / 0.125`, so walls look as they do in the CPU renderer. Tables are lit from above and need no scaling.

**Texture slots** are sized from the package's texel density over each surface, about 1.5 mm per pixel for the current package, so nothing is upsampled. Wall textures keep the print's true scale, and the table tile repeats at its physical size.

| Slot | Size at `background_slot_scale=1` (current package) |
|---|---|
| Far wall | 856 × 833 |
| Side walls | 595 × 833 each |
| Table | 867 × 397 |

**Lighting:** walls render with material emission 0.85 and the table with 0.55. This keeps the photographed colours while objects still cast shadows on the table. `BackgroundConfig(wall_emission=..., table_emission=...)` changes this when you construct `MuJoCoYAMEnv` yourself.

**Other live backends** (`madrona`, `blender`) copy textures when they are created, so a per-episode swap rebuilds their camera provider. `mujoco` and `mjwarp` swap in place.

## Background package layout

```
index.json
wallpapers/<design>/<instance>/wall_far.jpg  wall_left.jpg  wall_right.jpg
tables/<type>/<instance>/tile.jpg            # seamless tile
```

`index.json`:

```json
{"texel_m": 0.0015,
 "wallpapers": {"<design>": {"instances": [{"id": "<design>/<instance>", "dir": "wallpapers/<design>/<instance>"}]}},
 "tables":     {"<type>":   {"instances": [{"id": "<type>/<instance>",   "dir": "tables/<type>/<instance>"}]}}}
```

Other keys are ignored. To build your own package, use this layout. Wall images are bottom-aligned to the table height (bottom row = table edge) at `texel_m` metres per pixel.

## Viewer

```bash
MUJOCO_GL=egl python scripts/view_backgrounds.py \
    --background_variation_asset_dir /path/to/background_package \
    --background_seed 0 --max_tasks 40 --port 8765
```

Open `http://localhost:8765/`; forward the port from a remote machine. Use `MUJOCO_GL=osmesa` without a GPU.

- **Grid:** a tile for every catalogue task, each in its own background (task `i` uses background seed `base + i`).
- **Task view:** click a tile for:
  - the top, left-wrist or right-wrist cameras, or a free orbit camera (drag to orbit, shift/right-drag to pan, wheel to zoom);
  - **next background**;
  - **set id** to pin a combination;
  - **re-sample task**: new objects under the same background.

| Viewer flag | Default | Meaning |
|---|---|---|
| `--background_variation_asset_dir` | required | As above. |
| `--background_seed` | `0` | Base seed; task `i` uses `base + i`. |
| `--tasks` | all | Only these task names. |
| `--max_tasks` | `40` | Cap on tasks loaded at once (CPU / memory). |
| `--port`, `--host` | `8765`, `127.0.0.1` | Server address. |
| `--width`, `--height` | `640`, `480` | Base render size; the large view renders at 1.5×. |

## Troubleshooting

- **`randomize_background=True needs background_variation_asset_dir`:** pass the package location; there is no default.
- **`background_id / background_seed need randomize_background=True`:** the background options only make sense with the switch on.
- **`no background package (index.json) under ...`:** the path points to the wrong level, or the S3 sync was incomplete. Delete `$ABC_SIM_CACHE_DIR/background_variations/<hash>` and retry.
- **Every worker shows the same backgrounds:** they share one `background_seed`. Use `base + rank`.
