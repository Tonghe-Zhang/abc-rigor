# Random spelling placement reward

`spell_random_2_appearance`, `spell_random_3_appearance`, and
`spell_random_4_appearance` use `abc-spelling-placement-v2`. Reset reward is zero.
The seeded goal, appearances, cube physics and prompts retain their existing
distributions.

Each requested letter has a share of `1 / sequence_length`:

- Correct position in the output row: full placement credit.
- Present in the row but in the wrong position: 25% placement credit.
- Absent from the row, untouched, held, unsettled or unsupported: zero.
- An unrequested letter earns zero and still occupies its geometric position.

For target `ZXBI`, `ZXBL` earns 0.75 and `XZBI` earns 0.625 before force deductions.
Placement credit is recomputed from the current arrangement.

## Physical qualification and row geometry

A letter must have experienced robot contact since reset and be at least 2 cm
away from its initial planar position. Initial constraint solves and passive
settling cannot establish robot interaction. Its letter face must point upward
(absolute local-X/up dot at least 0.85). For 0.1 continuous seconds it must be
released from both entire arm subtrees, directly supported by table contact
carrying at least half its weight, and moving no faster than 1 cm/s and 0.1 rad/s.

Row detection uses all cubes, including unrequested, untouched and held
occupants. A row has at least two cubes, at most 3 cm of lateral spread and
2.5–6 cm spacing per occupied or empty slot. Among connected candidate rows,
the row with the most deliberately placed cubes supplies the output; ties
retain only credit common to every candidate, without choosing by letter identity.
Reading starts at the first occupied position toward decreasing scene Y.
Thus an incomplete `ZXB` prefix earns 0.75, and `Z_BI` with an unambiguous empty
second slot also earns 0.75. A singleton does not establish a row.

Co-projected cubes within 5 mm reserve one conflicted slot. That slot earns
zero without shifting its neighbors: stacking the first target on a wrong
cube leaves the other three correct letters' 0.75 credit intact.

Longer gaps admit missing slots. If multiple interpretations fit, full positional
credit requires agreement across them; an eligible requested letter otherwise
retains its quarter credit. Overlong rows have ambiguous endpoints, so adding
distractors cannot earn full reward by selecting the best substring.

## Excess contact force

Every native physics tick sums the magnitudes of all contact-force vectors on
each cube, including table, other cubes and both arms. Opposing fingers add
rather than cancel. Each cube accumulates:

```text
dose += max(total_contact_force_N - 15, 0) * dt_seconds
damage = min(dose / 5, 1)
letter_reward = placement_credit * (1 - 0.30 * damage)
```

The 30% handling reserve disappears after one second at 20 N, or half a second
at 25 N. Damage persists until reset, including when the cube leaves the row
or changes position. It reduces only that cube's earned credit. Success reports
correct physical spelling independently of accumulated handling damage.

Metrics expose raw placement credit, per-letter rewards, qualification, current
and peak loads, accumulated dose and actual force deductions. Repeated reads do
not advance history. Native observations run on every physics tick; missing or
duplicate observations fail. Qpos-only and batched Warp evaluation reject this
contact-dependent evaluator.

The explicit `abc-spelling-free-placement-v1` evaluator and fixed-word
`spell_word` tasks keep their recorded definitions. Callers using versioned v1
training recipes must select the current catalogue tasks to adopt v2.
