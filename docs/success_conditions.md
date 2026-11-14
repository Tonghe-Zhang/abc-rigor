# Revised success conditions

- **Ball tray:** sample uniformly from `[5, 15.5)` seconds, then round down to
  the nearest half-second. The 21 requirements from 5 through 15 seconds are
  equally likely. Sample duration independently of the tray and ball scales.
  Record this duration and use it for both scoring and the
  episode prompt: `balance the ball in the tray for at least X seconds`.
  Format whole seconds without `.0`. Keep the tray at least 5 cm
  above its initial height and the ball within the existing tray-local region
  for that duration without interruption. An invalid sample restarts the hold.
  Once the hold completes, ignore the ball and require a new tray–table contact
  at a later timestamp. Existing contact or repeated evaluation of the same
  timestamp does not count as lowering. Completion is latched. Physics-step updates
  prevent an interruption between control observations from being missed.
  The environment owns the evaluator and shares its initial and physics-step
  results with the runtime and step rewards. After each integration step it
  refreshes position-dependent state so live scoring uses positions, site
  transforms, and contacts from the same simulation state. Evaluator resets
  and batch-width changes clear temporal state while retaining the configured duration.
  `evaluate_model_data` uses simulation time; qpos-only callers must supply
  `time_s` explicitly.
- **Chess:** only the selected target squares must be occupied by upright pieces
  of the matching color and type. Pieces are interchangeable across the entire
  same-color/type set. Match selected squares first, one-to-one; unselected
  squares contribute only to full-board diagnostics. Reward is the fraction
  of selected targets filled. Retain the existing board-relative position,
  height, and upright tolerances.
  In both chess scenes, swap the pieces' dark and light materials while
  preserving their names, positions, meshes, and evaluator identities.
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
