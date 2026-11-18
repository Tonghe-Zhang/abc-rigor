# Background variations

`randomize_background` is an opt-in, appearance-only domain randomization. It replaces the plain walls and table top with textures photographed from real robot stations: a wallpaper design on the three walls and a table surface on the table. It is off by default; with it off, the compiled scene is exactly the same as before.

Objects, collision geometry and task physics do not change. The background is four visual-only textured quads, laid just inside the three walls and on the table top. They have `contype=conaffinity=0`, no mass, and belong to the world body. Trajectories with and without backgrounds are bit-identical for the same seeds and actions; `tests/test_background.py` checks this.

## Options

| `make_env` / eval config | default | meaning |
|---|---|---|
| `randomize_background` (`--randomize-background`) | `False` | Turn background variations on. |
| `background_variation_asset_dir` (`--background-variation-asset-dir`) | `None` | Background package: a local directory, or an `s3://` URI. An S3 URI is downloaded once per machine into `$ABC_SIM_CACHE_DIR/background_variations` (default `~/.cache/abc_sim`) under a file lock, so many workers on a node share one copy. Required when enabled. |
| `background_seed` (`--background-seed`) | `None` | Seed of the background stream. It is independent of the task / episode `seed` given to `reset`. `None` uses OS entropy. |
| `background_id` (`--background-id`) | `None` | Pin one combination in `[0, n_combinations)`. |
| `background_resample` | `"episode"` | `"episode"` draws a new background on every randomized `reset`; `"env"` keeps the first draw. |

```python
env = abc_sim.make_env(
    task="sort_lego_blocks",
    randomize_background=True,
    background_variation_asset_dir="/path/to/background_package",
    background_seed=1000 + worker_rank,
)
obs, info = env.reset(seed=episode_seed)        # info["background"] = {"background_id", "wallpaper", "table"}
obs, info = env.reset(seed=episode_seed, options={"background_id": 17})   # pin one for this episode
env.set_background()                              # draw the next background now, without a reset
```

The eval CLI (`abc_minimal.config.SimEvalConfig`) takes the same options:

```bash
uv run eval_policy.py --checkpoint ... --randomize-background \
    --background-variation-asset-dir /path/to/background_package --background-seed 7
```

## Combinations and seeding

A package has `W` wallpaper designs and `T` table surface types. Each (wallpaper, table) pair is one combination:

`background_id = wallpaper_index * T + table_index`

Both lists are in sorted name order. `BackgroundLibrary(dir).combination(i)` returns the pair, and `combination_id(w, t)` gives the reverse. Each design or type has one or more instances (the same print photographed under different light and placements). A draw picks the combination uniformly, then one instance of each.

The background stream is a NumPy generator seeded with `SeedSequence(background_seed, spawn_key=(0xB60001,))`. It never consumes, or is consumed by, the task randomizer's RNG. As a result:
- the same `background_seed` gives the same background sequence for any task seeds;
- the same task seed gives the same objects for any background seed.

Episode `k` of an env uses draw `k`. Replay resets (`reset(randomize=False)`) keep the current background.

**Large-scale parallel runs.** Give every worker its own `background_seed`, for example `base + rank`, for distinct reproducible streams. Use `None` for non-reproducible draws.

**Batched (`make_batched_env`).** All worlds of one batched env share one texture set, so a batched env has one background (`background_resample="env"` is forced). Vary it across batched envs.

## Rendering

Texture slots have fixed sizes:

| slot | size |
|---|---|
| far wall | 1024×1024 |
| side walls | 768×1024 each |
| table | 1024×512 |

A new background is written into the compiled model in place. The CPU MuJoCo renderer re-uploads just those textures. The other live camera backends (`mjwarp`, `madrona`, `blender`) copy textures when they are created, so a per-episode change rebuilds them.

Walls keep the print's true scale: the package is metric, at 1.5 mm per texel. The table tile is repeated at its physical size.

## Package layout

```
index.json                       {"texel_m", "wallpapers": {type: {"instances": [{"id", "dir", ...}]}}, "tables": {...}}
wallpapers/<type>/<instance>/    wall_far.jpg  wall_left.jpg  wall_right.jpg
tables/<type>/<instance>/        tile.jpg (seamless)
```

If the given directory has no `index.json` but contains version subfolders (`v1/`, `v2/`, ...), the newest one is used.

## Viewer

```bash
MUJOCO_GL=egl python scripts/view_backgrounds.py --background-variation-asset-dir /path/to/background_package --port 8765
```

This serves a browser page at `http://localhost:8765/`. It shows a grid of every task, each in its own background (task `i` uses background seed `base + i`). Click a task for its top or wrist cameras or a free orbit camera, draw the next background, pin an id, or re-sample the task under the same background. `--max-tasks` caps how many tasks are loaded at once.
