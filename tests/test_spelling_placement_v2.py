"""Independent per-letter credit, row ambiguity and manipulation qualification."""

import itertools
import string
import unittest

import numpy as np
from abc_sim.spelling_placement_v2 import SpellingPlacementV2Cfg, score_spelling_placement


def frame(row, *, word="ZXBI", gap=0.04):
    letters = list(string.ascii_uppercase)
    positions = [[1 + 0.25 * i, 1, 0.765] for i in range(26)]
    placed = [False] * 26
    for slot, letter in enumerate(row):
        if letter == "_":
            continue
        index = letters.index(letter)
        positions[index] = [0.42, 0.12 - gap * slot, 0.765]
        placed[index] = True
    return dict(word=word, letters=letters, positions=positions, placed=placed, eligible=[True] * 26)


class SpellingPlacementV2Tests(unittest.TestCase):
    def score(self, sample):
        return score_spelling_placement(SpellingPlacementV2Cfg(), **sample)

    def test_exact_observed_zxbl_row_scores_three_quarters(self):
        sample = frame("ZXBL")
        observed = {
            "Z": [0.42210811638544515, 0.050285854528097695, 0.76499858736],
            "X": [0.4261307794360412, 0.014571356907755878, 0.76499858736],
            "B": [0.42330390805877954, -0.024945052629075825, 0.76499858736],
            "L": [0.42812488480606053, -0.06552775592208354, 0.76499858736],
            "I": [0.6512964488129009, 0.210711340903264, 0.76499858736],
        }
        for letter, position in observed.items():
            sample["positions"][sample["letters"].index(letter)] = position
        result = self.score(sample)
        self.assertEqual(result["reward"], 0.75)
        self.assertFalse(result["success"])
        self.assertEqual(result["correct_placements"], 3)

    def test_every_two_three_four_letter_permutation_has_independent_credit(self):
        for word in ("AB", "ABC", "ABCD"):
            for row in itertools.permutations(word):
                with self.subTest(word=word, row=row):
                    result = self.score(frame(row, word=word))
                    correct = sum(left == right for left, right in zip(word, row, strict=True))
                    self.assertEqual(result["reward"], (correct + 0.25 * (len(word) - correct)) / len(word))
                    self.assertEqual(result["success"], correct == len(word))

    def test_wrong_letter_missing_target_and_interior_gap(self):
        for row, expected in (("ZXBL", 0.75), ("ZXB", 0.75), ("Z_BI", 0.75), ("XZBI", 0.625), ("PQRS", 0)):
            with self.subTest(row=row):
                self.assertEqual(self.score(frame(row))["reward"], expected)
        self.assertEqual(self.score(frame("AB", word="ABC"))["reward"], 2 / 3)
        self.assertEqual(self.score(frame("BC", word="ABC"))["reward"], 0.5 / 3)

    def test_reset_grid_and_face_up_letters_without_placement_earn_zero(self):
        sample = frame("ZXBI")
        sample["placed"] = [False] * 26
        self.assertEqual(self.score(sample)["reward"], 0)
        sample["placed"][sample["letters"].index("I")] = True
        self.assertEqual(self.score(sample)["reward"], 0.25)

    def test_ineligible_or_untouched_occupant_keeps_its_slot(self):
        for flag in ("eligible", "placed"):
            sample = frame("ZXBI")
            sample[flag][sample["letters"].index("X")] = False
            result = self.score(sample)
            self.assertEqual(result["reward"], 0.75)
            self.assertFalse(result["success"])
        sample = frame("ZXBL")
        sample["eligible"][sample["letters"].index("L")] = False
        self.assertEqual(self.score(sample)["reward"], 0.75)

    def test_overlong_row_cannot_cherry_pick_the_target_substring(self):
        for row in ("AZXBI", "ZXBIA", "ZXABI"):
            result = self.score(frame(row))
            self.assertLessEqual(result["reward"], 0.25)
            self.assertFalse(result["success"])

    def test_collisions_wide_rows_and_scattered_letters_do_not_spell(self):
        for gap in (0, 0.01, 0.2):
            self.assertEqual(self.score(frame("ZXBI", gap=gap))["reward"], 0)
        sample = frame("ZXBI")
        for index, letter in enumerate("ZXBI"):
            sample["positions"][sample["letters"].index(letter)][0] += index * 0.015
        self.assertEqual(self.score(sample)["reward"], 0)

    def test_row_selection_uses_placement_count_without_consulting_goal(self):
        sample = frame("ZXBI")
        for index, letter in enumerate("ABC"):
            # B is already in the four-letter row.
            if letter == "B":
                continue
            j = sample["letters"].index(letter)
            sample["placed"][j] = True
            sample["positions"][j] = [0.2, 0.1 - 0.02 * index, 0.765]
        self.assertEqual(self.score(sample)["reward"], 1)
        for index, letter in enumerate("ACDE"):
            j = sample["letters"].index(letter)
            sample["placed"][j] = True
            sample["positions"][j] = [0.2, 0.1 - 0.04 * index, 0.765]
        self.assertEqual(self.score(sample)["reward"], 0)

    def test_discarded_stack_does_not_erase_complete_or_partial_spelling(self):
        for row, expected in (("ZXBI", 1), ("ZXBL", 0.75), ("ZXB", 0.75), ("ZX", 0.5)):
            for pile_size in (len(row), 5):
                with self.subTest(row=row, pile_size=pile_size):
                    sample = frame(row)
                    for height, letter in enumerate("ACDEF"[:pile_size]):
                        index = sample["letters"].index(letter)
                        sample["positions"][index] = [0.65, 0.2, 0.765 + 0.03 * height]
                        sample["placed"][index] = True
                        sample["eligible"][index] = height == 0
                    self.assertEqual(self.score(sample)["reward"], expected)

    def test_row_ranking_counts_admissible_slots_without_stacked_multiplicity(self):
        for row, expected in (("ZXBI", 1), ("ZXBL", 0.75), ("ZXB", 0.75)):
            for layout in ("two_slots", "too_wide", "too_close"):
                with self.subTest(row=row, layout=layout):
                    sample = frame(row)
                    for offset, letter in enumerate("ACDEF"):
                        index = sample["letters"].index(letter)
                        x, y, z = 0.65, 0.2, 0.765
                        if layout == "two_slots":
                            y -= 0.04 * (offset % 2)
                            z += 0.03 * (offset // 2)
                        elif layout == "too_wide":
                            x += 0.015 * offset
                            y -= 0.04 * offset
                        else:
                            y -= 0.01 * offset
                        sample["positions"][index] = [x, y, z]
                        sample["placed"][index] = True
                    self.assertEqual(self.score(sample)["reward"], expected)

    def test_credit_is_current_and_order_independent_of_inventory_or_translation(self):
        sample = frame("ZXBI")
        self.assertEqual(self.score(sample)["reward"], 1)
        index = sample["letters"].index("I")
        sample["positions"][index][0] += 0.5
        self.assertEqual(self.score(sample)["reward"], 0.75)
        for key in ("letters", "positions", "placed", "eligible"):
            sample[key] = sample[key][::-1]
        sample["positions"] = (np.asarray(sample["positions"]) + [0.1, -0.3, 0]).tolist()
        self.assertEqual(self.score(sample)["reward"], 0.75)

    def test_invalid_frames_and_configuration_fail(self):
        for change in ({"word": "ZZ"}, {"positions": [[float("nan"), 0, 0]] * 26}, {"placed": [1] * 26}):
            with self.assertRaises(ValueError):
                self.score(frame("ZXBI") | change)
        for kwargs in (
            {"revision": "v0"},
            {"force_dose_budget_ns": 0},
            {"force_allowance_n": -1},
            {"gentle_fraction": 1},
            {"misplaced_fraction": float("nan")},
            {"gap_min_m": 1},
        ):
            with self.assertRaises(ValueError):
                SpellingPlacementV2Cfg(**kwargs)


if __name__ == "__main__":
    unittest.main()
