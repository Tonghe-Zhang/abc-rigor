# Revised success conditions

- **Ball tray (default):** prompt `keep the ball balanced on the tray for 10
  to 15 seconds, then gently place the tray on the table`. The hold runs while
  the tray is at least 5 cm above its initial height, the ball is in the
  tray-local region, and no gripper touches the ball. Air time runs from the
  start of the hold until the tray is back below 5 cm (it resumes if the tray
  goes back up). While up, reward is 0.9 × air time / 10 s, 0.9 through 15 s,
  then falls linearly to 0.5 by 23 s. Touchdown (tray–table contact below 5 cm)
  ends the attempt. With both handles gripped (0.1 s grace) and the ball then
  on the tray and untouched for 1 s, it succeeds if the air time is 10–18 s
  (`max_hold_s`) and touchdown came within 5 s of it (`max_lowering_s`).
  Success scores 1.0 for 10–15 s, then falls linearly to 0.8 at 18 s, so the
  asked 10–15 s is best. Otherwise there is no success: a placement before
  10 s keeps half the reward reached, and a later one keeps all of it (at most
  0.75, below any success). The limits come from the 2,039 demonstrations that
  lift the tray: air time is 13.7 s at the median and 17.6 s at p95, and
  touchdown follows the final drop within 5.3 s at p95, so each limit passes
  about 95% of them. Touchdown without both handles keeps
  at most 0.5. Touching or losing the ball before 10 s restarts the hold;
  after 10 s, or in the 1 s after touchdown, it scores 0.
- **Ball tray (`hold_window_s=None`):** sample uniformly from `[5, 15.5)` seconds, then round down to
  the nearest half-second. The 21 requirements from 5 through 15 seconds are
  equally likely. Sample duration independently of the tray and ball scales.
  Record this duration and use it for both scoring and the
  episode prompt: `balance the ball in the tray for at least X seconds`.
  Format whole seconds without `.0`. Keep the tray at least 5 cm
  above its initial height and the ball within the existing tray-local region
  for that duration without interruption. An invalid sample restarts the hold.
  Once the hold completes, ignore the ball and require a new tray–table contact
  at a later timestamp. Existing contact or repeated evaluation of the same
  timestamp does not count as lowering. Completion is latched.
  The environment owns the evaluator and shares its initial and control-step
  results with the runtime and step rewards. Ball-tray scoring refreshes
  position-dependent state once per control step (normally 34 ms), and at the
  end of any shorter explicit physics advance, so positions, site transforms,
  and contacts agree. Lapses shorter than that sampling interval may be missed.
  Runtime controls still run every physics step. Tasks without
  `requires_step_updates` incur no extra position refresh while stepping;
  contact-based evaluators refresh when explicitly scored. Evaluator resets
  and batch-width changes clear temporal state while retaining the configured duration.
  `evaluate_model_data` uses simulation time; qpos-only callers must supply
  `time_s` explicitly.
- **Chess:** the selected target squares must be occupied by upright pieces
  of the matching color and type. Pieces are interchangeable across the entire
  same-color/type set. Match selected squares first, one-to-one. By default,
  every non-target home square must also remain filled; vacated home squares
  reduce reward by one target's worth until restored. `require_home_squares=False`
  restores selected-target-only success and reward. Retain the board-relative position,
  height, and upright tolerances.
  Both chess scenes place the white pieces on the robot's side to match the
  intended real-world board orientation. Piece identities, `color_mode` and
  `target_colors` agree with the visible materials: white is `light_piece`,
  black is `dark_piece`. Body, joint, geom and mesh identifiers are renamed
  together; positions, mesh files and physical geometry are unchanged.
  Chess appearance therefore differs from simulation data rendered before
  this PR's color swap.
- **Relative placement:** allow zero separation along the requested direction.
  Keep the 30 cm upper limit, 8 cm perpendicular limit, and existing placement
  height check.
- **In-hand transfer:** require object–table contact and no object–gripper
  contact in place of the height limit. Keep the table footprint and opposite
  side requirements. Reset metadata remains the normal source of the start side.
- **LEGO sorting:** allow a block center up to 5 mm in Euclidean distance outside
  its matching bin's goal box. This is not a separate 5 mm extension on each
  axis. Preserve the exact wrong-bin exclusion.

The following overrides belong only to the separate historical success
calculation, not these persistent evaluator changes:

- Use the first mapped video frame to determine the in-hand transfer start side.
- For dish-rack scoring, ignore rack and gripper contact while retaining the
  upright and spatial tests.

CPU regressions: `python -m unittest discover -s tests -p test_success_conditions.py -v`.
