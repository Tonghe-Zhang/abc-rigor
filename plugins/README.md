# External abc_sim task plugins

Registered through `ABC_SIM_PLUGINS=abc_tongs_tasks:register,abc_cap_tasks:register` with `plugins/` on `PYTHONPATH`.
Assets are not in git: they are downloaded on demand from `ABC_SIM_ASSET_SOURCE` (a dir or an S3 prefix) into
`ABC_SIM_CACHE_DIR`.

| task | package |
|---|---|
| `tongs_bread_bag` | bread + tongs into a paper bag (flex bag by default) |
| `tongs_pick_place` | tongs pick / place presets with 4-8 objects and paraphrased prompts |
| `open_cap` | unscrew jar / bottle / tethered caps |

`abc_hw.py` puts every plugin scene on the standard abc_sim station, read at runtime from task `empty`:
the top / wrist camera bodies are grafted from it, the table box and the photo walls / planes are fitted to its
`play_table` footprint and `abc_sim.background._SCENE`, and the arm mounts, joint ranges, arm actuator gains and the
gripper gravity compensation are copied from it. Then one-model's `YamVariant(variant).compose(spec)` applies, so
all 5 one-model hardware variants (abc_crank, yam_cell_d405, yam_cell_zed, abc_box_d405, molmoact2) randomize
plugin tasks the same way as built-in tasks. A policy camera that is missing raises an error; there is no fallback.

Known differences from the standard station: the plugin grippers keep their own calibrated fingers (tongs:
linear kp 100, ctrl 0-0.0475; open_cap: ctrl 0-0.0475, vs the standard crank kp 400, 0-0.0495), and a warning is
logged when a variant asks for another gripper. open_cap keeps its portable's solver options (Euler / elliptic /
impratio 10).

`e2e/policy_eval.py` runs a policy server against a task and writes `episodes.jsonl` and H.264 videos.
