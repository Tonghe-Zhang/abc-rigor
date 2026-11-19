# Mug hanging reward

The public `hang_mug_on_mug_rack` task uses `mug_on_rack_v2`. Each of the
one to three active mugs earns an equal share only after all of these hold
continuously for 0.5 seconds:

- A rack branch passes fully through the handle opening.
- The same branch carries at least half the mug's weight in upward contact load.
- Neither arm touches that mug or the rack.
- Mug and rack speeds stay below 2 cm/s and 0.2 rad/s.
- Mug–rack collision penetration stays below 2 mm.

Removing a mug, touching it again, or moving it breaks its qualification
immediately and restarts its dwell. Touching the shared rack breaks every
mug's qualification. Switching branches restarts the per-branch dwell.
No success latch preserves reward after the required arrangement is lost.

## Geometry and variants

The evaluator measures each realized, scaled collision mesh. All available
mugs have a handle normal along local X. The enclosed hole in the YZ
collision silhouette identifies the handle; sub-pixel seams between convex
pieces are closed at 0.25 mm resolution. Triangle intersections along the
material bordering the hole determine its asymmetric front and back faces.

A branch centerline must cross the opening throughout that thickness, and
both ends must extend at least 1 mm beyond the handle faces. The branch's
actual convex mesh section at the middle plane must fit the opening,
including its polygon edges. A 0.75 mm boundary tolerance accounts for
raster resolution and contact indentation. Deep penetration is separately
rejected using solved contacts.

There are six measured branch colliders. The base and trunk cannot qualify.
All sixteen styles in the default pool have collision-free threaded
placements at both 0.9 and 1.1 scale. Styles `mug_1`, `mug_7`, and `mug_10`
have openings too small for these branches and are excluded from default
sampling. Their assets remain available for replay and explicit inspection.

## Runtime contract

Every native physics tick updates threading, contact support, release,
motion and the dwell timer. Repeated result reads do not advance time.
Reset clears every timer and success diagnostic. Qpos-only scoring and
batched Warp cannot supply this history and reject the evaluator.

Metrics include threading and qualification by peg, support forces,
robot contacts, penetration, stability and dwell time. Reward remains
`num_mugs_hanging / num_active_mugs`; success requires every active mug.
The original `mug_on_rack` evaluator remains available for callers that
explicitly pin historical recipes.
