"""Per-letter spelling progress in a physically assembled, unmarked row."""

from __future__ import annotations

import string
from dataclasses import dataclass
from itertools import product

import numpy as np

from abc_sim.spelling_placement import FreePlacementSpellingCfg


@dataclass(frozen=True)
class SpellingPlacementV2Cfg:
    """Geometry, release, settling and irreversible contact-dose parameters."""

    revision: str = "abc-spelling-placement-v2"
    row_width_m: float = 0.03
    gap_min_m: float = 0.025
    gap_max_m: float = 0.06
    slot_overlap_m: float = 0.005
    displacement_m: float = 0.02
    upright_dot_min: float = 0.85
    minimum_weight_fraction: float = 0.5
    max_linear_speed: float = 0.01
    max_angular_speed: float = 0.1
    settling_seconds: float = 0.1
    robot_force_tolerance: float = 1e-6
    misplaced_fraction: float = 0.25
    gentle_fraction: float = 0.30
    force_allowance_n: float = 15.0
    force_dose_budget_ns: float = 5.0

    def __post_init__(self) -> None:
        if self.revision != "abc-spelling-placement-v2":
            raise ValueError("Unsupported spelling placement revision.")
        FreePlacementSpellingCfg(
            **{
                name: getattr(self, name)
                for name in FreePlacementSpellingCfg.__dataclass_fields__
                if name != "revision"
            }
        )
        values = (
            self.misplaced_fraction,
            self.gentle_fraction,
            self.force_allowance_n,
            self.force_dose_budget_ns,
            self.slot_overlap_m,
        )
        if (
            not np.isfinite(values).all()
            or not 0 < self.misplaced_fraction < 1
            or not 0 <= self.gentle_fraction < 1
            or self.force_allowance_n < 0
            or self.force_dose_budget_ns <= 0
            or not 0 < self.slot_overlap_m < self.gap_min_m
        ):
            raise ValueError("Invalid spelling partial-credit or force-dose configuration.")


def score_spelling_placement(
    cfg: SpellingPlacementV2Cfg,
    *,
    word: str,
    letters: list[str],
    positions: list[list[float]],
    placed: list[bool],
    eligible: list[bool],
) -> dict:
    """Score one assembled row, retaining wrong and ineligible occupants.

    Rows are connected geometric components with at least two occupants.
    The component with the most deliberately placed blocks is the output row;
    tied components retain only credit common to every interpretation.
    Reading starts at the first occupied position, toward decreasing scene Y.
    Longer gaps admit empty slots. An overlong row has ambiguous endpoints.
    Co-projected occupants reserve one conflicted slot without changing the
    indices or credit of neighboring letters.
    Correct positions must agree across all admissible gap/endpoint choices.
    Requested, placed, eligible letters in the row otherwise earn quarter credit.
    """
    count, length = len(letters), len(word)
    if (
        not 2 <= length <= 4
        or len(set(word)) != length
        or not set(word) <= set(string.ascii_uppercase)
        or len(set(letters)) != count
        or any(letter not in string.ascii_uppercase or len(letter) != 1 for letter in letters)
        or not set(word) <= set(letters)
    ):
        raise ValueError("Spelling requires distinct uppercase letters and a complete target inventory.")
    positions = np.asarray(positions, dtype=np.float64)
    if (
        positions.shape != (count, 3)
        or not np.isfinite(positions).all()
        or any(len(flags) != count or any(type(flag) is not bool for flag in flags) for flags in (placed, eligible))
    ):
        raise ValueError("Spelling requires finite positions and explicit placement/eligibility flags.")
    placed = np.asarray(placed)
    eligible = placed & np.asarray(eligible)
    delta = np.abs(positions[:, None, :2] - positions[None, :, :2])
    adjacent = (delta[..., 0] <= cfg.row_width_m) & (delta[..., 1] <= (length - 1) * cfg.gap_max_m)
    remaining, candidates = set(range(count)), []
    while remaining:
        component, frontier = set(), {min(remaining)}
        while frontier:
            index = frontier.pop()
            component.add(index)
            remaining.remove(index)
            frontier.update(int(other) for other in np.flatnonzero(adjacent[index]) if other in remaining)
            frontier.difference_update(component)
        indices = sorted(component, key=lambda index: (-positions[index, 1], index))
        if len(indices) >= 2 and placed[indices].any():
            candidates.append(indices)

    progress = np.zeros(count)
    diagnostics, interpretations = [], []
    largest = max((int(placed[row].sum()) for row in candidates), default=0)
    for row in candidates:
        selected = int(placed[row].sum()) == largest
        groups = []
        for index in row:
            if groups and positions[groups[-1][0], 1] - positions[index, 1] <= cfg.slot_overlap_m:
                groups[-1].append(index)
            else:
                groups.append([index])
        gaps = -np.diff([positions[group, 1].mean() for group in groups])
        max_span = max(length, len(groups))
        options = [
            [jump for jump in range(1, max_span - len(groups) + 2) if cfg.gap_min_m <= gap / jump <= cfg.gap_max_m]
            for gap in gaps
        ]
        row_scores = []
        if np.ptp(positions[row, 0]) <= cfg.row_width_m:
            for jumps in product(*options):
                span = 1 + sum(jumps)
                if span > max_span:
                    continue
                slots = np.concatenate(([0], np.cumsum(jumps)))
                for offset in range(min(0, length - span), 1):
                    values = np.zeros(count)
                    for group, slot in zip(groups, slots + offset, strict=True):
                        if len(group) != 1:
                            continue
                        index = group[0]
                        if eligible[index] and letters[index] in word:
                            values[index] = (
                                1.0 if 0 <= slot < length and letters[index] == word[slot] else cfg.misplaced_fraction
                            )
                    row_scores.append(values)
        diagnostics.append(
            {"occupants": [letters[index] for index in row], "selected": selected, "interpretations": len(row_scores)}
        )
        if selected:
            # A selected malformed row has no admissible placement interpretation.
            interpretations.extend(row_scores if row_scores else [np.zeros(count)])
    if interpretations:
        progress = np.min(interpretations, axis=0)
    return {
        "revision": cfg.revision,
        "letter_progress": progress.tolist(),
        "correct_placements": int(np.count_nonzero(progress == 1)),
        "misplaced_letters": [letter for letter, value in zip(letters, progress, strict=True) if 0 < value < 1],
        "reward": float(progress.sum() / length),
        "success": bool(np.count_nonzero(progress == 1) == length),
        "rows": diagnostics,
    }
