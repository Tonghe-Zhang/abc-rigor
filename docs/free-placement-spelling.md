# Conservative free-placement spelling

Choose the `abc-spelling-free-placement-v1` evaluator explicitly on a custom
`SimTaskSpec` to score settled placements in an unmarked row:

```python
from abc_sim import SimTaskSpec, make_env

spec = SimTaskSpec(
    name="my_spelling_task",
    env_task="blocks",
    prompt="spell the requested letter sequence",
    evaluator_name="abc-spelling-free-placement-v1",
    evaluator_options=(("sequence_length", 3),),
    reset_request=(("sequence_length", 3),),
)
env = make_env(spec, render_cameras=False)
observation, info = env.reset(seed=37)
```

The reset request samples two, three or four distinct letters. Its realized goal
and prompt come from the seeded spelling randomizer. Fixed-word tasks, including
`spell_cat`, use `spell_word`. The public random appearance tasks use
[`abc-spelling-placement-v2`](spelling-placement-v2.md).

The evaluator considers the geometry of every row occupant, including wrong,
unreadable and held blocks. At least one block must move from its reset position.
It credits only letters whose positions agree under every admissible row
interpretation. For target `ABC`, a settled row `XBC` earns `2/3`; a row containing
only adjacent `BC` earns zero because its absolute positions are ambiguous.
An interior gap can establish positions: `A_C` with a two-slot gap earns `2/3`.
Competing rows and extra occupants also affect the interpretation.

Credited blocks must show a letter face upward, receive direct table contact
support of at least half their weight, and remain below the configured velocity
limits for 0.1 simulated seconds. Contact load from finger subtrees disqualifies
a block until release and settling. This contact scope does not include wrist
or camera bodies.

Contact observations run after every solved native MuJoCo step. Scoring runs
every 17 steps; `evaluate_task()` reads the latest scheduled result and raises
until the first result is ready. `step()` and `step_chunk()` require a control
decimation of at least 17 for this evaluator. Low-level callers can use
`apply_action(action, physics_steps=...)` or `advance_physics(...)` with shorter
increments and read the score when ready. `apply_action` uses native actuator
mapping and runtime callbacks, increments the action count once, and returns no
observation or termination result.

Qpos-only evaluation and batched MJWarp replay cannot provide the required
solved contacts and continuous settling history, so they reject this evaluator.
The custom task does not impose an episode duration or RL batch recipe.
