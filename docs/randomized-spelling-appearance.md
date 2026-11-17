# Randomized spelling appearances

The opt-in tasks `spell_random_2_appearance`, `spell_random_3_appearance`, and
`spell_random_4_appearance` sample a sequence of distinct uppercase letters and
independent visual styles for all 26 blocks. An explicit nonnegative reset seed
is required. Existing fixed-word tasks retain their original assets and defaults.

```python
from abc_sim import make_env

env = make_env(task="spell_random_3_appearance", render_cameras=False)
observation, info = env.reset(seed=31)
print(observation["prompt"])
appearance = info["randomization"].metadata["block_appearance"]
```

The goal uses the existing `SpellingGoal` stream. Appearance has a separate
versioned stream, so the same seed preserves goal and block placement relative
to the existing seeded-reset configuration. Neither the target letters nor
their order enter appearance sampling; any letter can receive any style.

Each block independently samples:

- Painted wood, saturated colored plastic with light ink, or light plastic with
  colored ink.
- Matte, satin, or gloss material parameters.
- Body, panel, glyph, and border palettes; no letter-specific palette.
- No panel, square, rounded, or circular panel; no border, ring, square, or
  double border.
- Original or slightly thickened glyph strokes and a procedural wood-grain seed.

The glyph masks come from the existing bundled block atlas and OBJ UVs. They
contain only the cleaned letter shapes, without the old wood texture or border.
No fonts or assets are downloaded. Font families do not change. Glyph/background
contrast is at least 4.5 in the generated albedo; borders leave a gap around ink.
This is not a claim about legibility under every lighting or camera condition.

The existing visual mesh triangles and scales remain intact; only their UVs
and materials change. Collision geoms, friction, mass, inertia, joints,
actuators, placements, and reward/evaluator configuration do not change.
The registered evaluator remains native `spell_word` with a sampled sequence;
the existing separate free-placement scoring helper is unchanged.

`RandomizationState.metadata["block_appearance"]` records the revision, seed,
and every realized per-block style. Replay applies those styles without
resampling. A single compiled appearance template is reused; its compact
per-block textures replace the old shared atlas. Each reset fills the complete
texture/material buffers before rebuilding the camera backend, ensuring both
CPU and GPU camera providers receive the current materials. Switching back to
a legacy task restores its original scene with either `randomize=True` or
`randomize=False`. Constructor wrist-camera FOV and clearance positions survive
appearance resets, replay, and restoration; reloading a cached model does not
apply the clearance adjustment again. Camera clearance uses each incoming
model's own mounts, including changed replacement XML. Calibration records use
weak references so discarded models are not retained.
Model caches call `env.copy_model(model)` to preserve the original camera mounts
across template copies; copying a calibrated model outside that API does not
carry its calibration record.

Native MuJoCo rendering uses the sampled matte/satin/gloss material parameters.
The current Blender material conversion replaces those finish parameters with
fixed shader settings. Blender preserves the sampled textures and colors, but
does not reproduce the recorded finish variation.

Applying a saved spelling `RandomizationState` through the bound randomizer
first calls `initialize_randomization()` to reset transient physics, including
velocities, controls and simulation time, and start arms at configured `init_q`.
The saved state records object poses, not robot poses. After restoring materials
and object poses, `finalize_randomization(state)` binds the recorded goal,
prompt, metadata and a fresh evaluator. Replaying after stepping therefore
starts the same initial scene and can immediately be scored or stepped.
Normal reset shares both operations and retains its explicit
`options={"keep_arm_state": True}` option to carry the current arm pose.
Initial replay does not reconstruct history-dependent rewards from an arbitrary
mid-episode qpos snapshot. A fresh appearance task needs a randomized reset or
a saved-state replay; `reset(randomize=False)` alone cannot supply its goal or
appearance.
No generated texture files are needed for replay; the recorded parameters and
bundled glyph assets suffice.

Use independent environments for these tasks. `BatchedWarpYAMEnv` shares texture
buffers across worlds and therefore rejects appearance-randomized resets before
GPU initialization, including explicit reset overrides. Different per-world
textures in that shared-model renderer are not implemented here.

## CPU validation and preview

The simulator and CPU tests use the declared MuJoCo, NumPy, and Gymnasium
dependencies. MuJoCo's PNG texture loader decodes the bundled glyph atlas.
The optional preview uses Pillow, supplied by the root project's ImageIO
dependency, for images and labels.

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest \
  tests.test_block_appearance tests.test_spelling_goals \
  tests.test_spelling_placement tests.test_spelling_placement_physics
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/preview_block_appearance.py \
  --output-dir /tmp/spelling-appearance-preview
```

The preview directory must not already exist. Its contact sheet shows actual
sampled face/body tiles for the same letters across all three material families,
plus every cleaned glyph. Matte/gloss differences require lighting in a 3D
renderer; the RGB sheet only displays their recorded labels. CPU tests verify
physical parameter equality, seed independence, metadata replay, glyph
extraction, contrast, and legacy restoration. Replaying after real physics
steps checks fresh dynamics and exact repeated first-action motion for native
and randomized appearances. A recording camera-provider double checks material
and calibration delivery, replacement cleanup, and reset ordering without
rendering. A separate CPU check samples the
original and generated compiled meshes through their UVs on both letter faces
for asymmetric glyphs; it checks orientation without relying on a texture-buffer
row-reversal assertion. These checks do not establish 3D lighting or GPU renderer
qualification. No GPU rendering is required.
