"""Appearance records stay usable in processes without a simulator runtime."""

import hashlib
import json
import subprocess
import sys
import textwrap
import unittest
from copy import deepcopy
from pathlib import Path

from abc_sim.block_appearance import sample_appearance, validate_appearance


class BlockAppearanceRecordTests(unittest.TestCase):
    def test_seeded_records_keep_the_declared_distribution(self):
        digests = {
            0: "97c0d9c2bf0bd35e8de950de1a33946ef69c429ef76eb03727a9e8825dd6630d",
            12: "b31a64f7d65ff130ad69e5f55968293da65289a8c405b9177b2e873440eea4c3",
            4294967295: "b40f3ba5ab4317148e5824891c087ed997ddaf6df5c9eaccc984f71756ca3d6b",
        }
        for seed, expected in digests.items():
            with self.subTest(seed=seed):
                record = sample_appearance(seed)
                validate_appearance(record)
                encoded = json.dumps(record, sort_keys=True, separators=(",", ":"))
                self.assertEqual(hashlib.sha256(encoded.encode()).hexdigest(), expected)

    def test_sampling_rejects_invalid_seed(self):
        for seed in (-1, True, 1.5, "1", None):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                sample_appearance(seed)

    def test_validation_rejects_invalid_materials(self):
        original = sample_appearance(12)
        changes = {
            "family": "unknown",
            "finish": "unknown",
            "panel": "unknown",
            "border": "unknown",
            "stroke": True,
            "body_rgb": [1, 2],
            "panel_rgb": [0, 0, 256],
            "glyph_rgb": [0, 0, False],
            "border_rgb": [0, -1, 0],
            "texture_seed": 2**32,
        }
        for name, value in changes.items():
            with self.subTest(name=name):
                record = deepcopy(original)
                record["blocks"]["A"][name] = value
                with self.assertRaises(ValueError):
                    validate_appearance(record)
        record = deepcopy(original)
        style = record["blocks"]["A"]
        style["glyph_rgb"] = style["body_rgb"]
        style["panel"] = "none"
        with self.assertRaises(ValueError):
            validate_appearance(record)

    def test_validation_rejects_wrong_revision_and_missing_letter(self):
        record = sample_appearance(12)
        record["revision"] = "unknown"
        with self.assertRaises(ValueError):
            validate_appearance(record)
        record = sample_appearance(12)
        del record["blocks"]["Z"]
        with self.assertRaises(ValueError):
            validate_appearance(record)

    def test_record_import_and_validation_do_not_import_simulator(self):
        script = textwrap.dedent(
            """
            import importlib.abc
            import sys

            class RejectSimulator(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.split(".")[0] in {"mujoco", "warp", "gymnasium"}:
                        raise ModuleNotFoundError(fullname)
                    if fullname == "abc_sim.randomization":
                        raise AssertionError("Record validation imported randomizers")

            sys.meta_path.insert(0, RejectSimulator())
            from abc_sim.block_appearance import sample_appearance, validate_appearance
            validate_appearance(sample_appearance(12))
            assert "mujoco" not in sys.modules
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
