# External sim tasks (plugins) and download-on-demand assets

New sim tasks can be added **without editing `abc_sim`** and **without committing large assets to git**. Everything a
task needs comes from one `ExternalTask` registration in your own package:

```python
from abc_sim.plugins import ExternalTask, register_task
from abc_sim.asset_source import AssetPack
from abc_sim.task_specs import SimTaskSpec

PACK = AssetPack(name="my_task_assets", archive="my_task_assets_v1.tar.gz", sha256="...")   # where: configured at run time

def build_scene(seed: int, options: dict):          # -> MJCF string | mujoco.MjSpec | mujoco.MjModel
    root = PACK.path()                              # fetched + cached on first use
    ...

register_task(ExternalTask(
    env_task="my_task",
    specs=(
        SimTaskSpec(name="my_task", env_task="my_task", prompt="pick up the red cube",
                    aliases=("grab the red cube",), evaluator_name="my_task"),
    ),
    scene_builder=build_scene,                       # or scene_xml=... / env_factory=...
    builder_options={"n_objects": 4},
    evaluators={"my_task": MyEvaluator},             # TaskEvaluator protocol (abc_sim/task_eval/base.py)
    assets=(PACK,),
))
```

After that the task behaves like a built-in one: `make_env("my_task")`, `make_env("grab the red cube")`,
`list_task_specs()`, `get_task_spec(...)`, `make_task_evaluator(...)`, `env.evaluate_task()` and the eval / RL tools
that use them.

## Making the scene: pick one mode

| mode | field | when |
|---|---|---|
| static XML | `scene_xml=Path` or a zero-arg callable returning a path (resolved lazily, so assets are fetched only when the scene is built) | a fixed scene file; optional `randomizer=` (zero-arg factory of a `SceneRandomizer`) |
| builder | `scene_builder=(seed, options) -> str \| MjSpec \| MjModel` | procedural layouts, MjSpec-built flex objects, a different scene for every seed |
| factory | `env_factory=(task=SimTaskSpec, **make_env kwargs) -> env` | the task cannot run in `MuJoCoYAMEnv`; the env must implement `abc_sim.plugins.SimEnvSurface` |

**Builder mode.** `make_env(task, seed=s, builder_options={...})` builds seed `s` (default 0). Every randomized
`env.reset(seed=s)` rebuilds the scene for `s` and keeps the robot's reset pose. The seed and options are stored in
`RandomizationState.metadata["scene_builder"]`, so replaying a recorded state rebuilds the same scene. The evaluator is
re-created for every rebuilt model, so evaluators may cache body / geom ids in `__init__`.

**Names** may not shadow built-in scene tasks, task names / aliases / prompts or evaluators; re-registering the same
`env_task` needs `replace=True`.

## Discovery

Plugins are loaded lazily, on the first task lookup:

* Python entry points, group `abc_sim.tasks` (a module to import, or a zero-arg callable that registers):

  ```toml
  [project.entry-points."abc_sim.tasks"]
  my_tasks = "my_package.abc_tasks:register"
  ```

* `ABC_SIM_PLUGINS=my_package.abc_tasks:register,other_module` (comma-separated `module` or `module:function`).

A plugin that fails to import is logged and skipped. It does not break the built-in tasks.

## Assets: configured location, never committed

`AssetPack(name, archive, sha256=None, size=None, subdir="")` names a pack. **Where** it lives is set by whoever runs
the task:

```bash
export ABC_SIM_ASSET_SOURCE=/data/abc_assets              # a local folder holding <archive> (file or extracted dir)
export ABC_SIM_ASSET_SOURCE=https://example.org/abc_assets  # http(s) prefix: resumable download
export ABC_SIM_ASSET_SOURCE=s3://some-bucket/abc_assets     # s3 prefix: fetched with the AWS CLI, your credentials
export ABC_SIM_ASSET_SOURCE_MY_TASK_ASSETS=/elsewhere/my_task_assets_v1   # per-pack override (name upper-cased)
```

* Local extracted folders are used in place. Archives (`.tar`, `.tar.gz`, `.tgz`, `.zip`) and remote folders are
  fetched once into `$ABC_SIM_CACHE_DIR/assets/<pack>-<hash>/` under a file lock, with a `.complete` marker. A cached
  pack keeps working offline.
* `sha256` / `size` are checked before extraction. Tar members are extracted with the `data` filter and zip paths are
  checked for traversal.
* To move the assets (for example to a public bucket when open-sourcing), only the configured source changes. No code
  changes.

`prepare.py` prefetches packs for external tasks:

```bash
uv run prepare.py --sim-task my_task --sim-asset-source /data/abc_assets
```

## Evaluators

An evaluator class is constructed as `cls(model=..., spec=..., **spec.evaluator_options, **evaluator_options)` and
follows `abc_sim.task_eval.base.TaskEvaluator` (`reset`, `evaluate_qpos_batch`). The opt-in protocols in the same file
(`PhysicsObservedTaskEvaluator` for contact / flex state that is not in `qpos`, `EpisodeTaskEvaluator`) work for
external tasks too. Results are `TaskEvalResult(reward=(B,), success=(B,), metrics={...})`.

## Tests

`tests/test_external_tasks.py` has toy plugins for all three modes, replay of builder seeds, collision checks,
`ABC_SIM_PLUGINS` loading, and the asset source (local, archive + cache + checksum, missing source).
