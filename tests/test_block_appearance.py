"""Seeded materials, replay, visual-only changes and legacy spelling behavior."""

import json
import unittest
from copy import deepcopy
from pathlib import Path

import mujoco
import numpy as np

from abc_sim import make_env
from abc_sim.batched_env import BatchedWarpYAMEnv
from abc_sim.randomization.assets.block_appearance import (
    FAMILIES,
    LETTERS,
    TILE,
    appearance_xml,
    apply_appearance,
    contrast,
    glyph_masks,
    sample_appearance,
    texture_tiles,
    validate_appearance,
)
from abc_sim.randomization.core import RandomizationState, _resolve_scene_xml_paths
from abc_sim.spelling_goal import SpellingGoal
from abc_sim.task_specs import SimTaskSpec, get_task_spec

SCENE = Path(__file__).resolve().parents[1] / "abc_sim/models/yam_blocks_scene.xml"


class AppearanceTextureTests(unittest.TestCase):
    def test_seed_roundtrip_variety_and_no_fixed_letter_color(self):
        records = [sample_appearance(seed) for seed in range(12)]
        self.assertEqual(records, [sample_appearance(seed) for seed in range(12)])
        self.assertEqual(records[0], json.loads(json.dumps(records[0])))
        for letter in LETTERS:
            styles = [record["blocks"][letter] for record in records]
            self.assertGreaterEqual(len({style["family"] for style in styles}), 2)
            self.assertGreater(len({tuple(style["body_rgb"]) for style in styles}), 5)
        self.assertEqual(
            {
                style["family"]
                for record in records
                for style in record["blocks"].values()
            },
            set(FAMILIES),
        )
        for record in records:
            validate_appearance(record)

    def test_glyphs_exclude_legacy_border_and_use_only_chosen_ink(self):
        masks = glyph_masks()
        self.assertEqual(set(masks), set(LETTERS))
        for letter, mask in masks.items():
            self.assertGreater(mask.sum(), 150)
            self.assertFalse(mask[:17].any() or mask[-17:].any())
            self.assertFalse(mask[:, :17].any() or mask[:, -17:].any())
            for seed in (0, 1, 2):
                style = sample_appearance(seed)["blocks"][letter]
                body, face = texture_tiles(letter, style)
                self.assertEqual(body.shape, (TILE, TILE, 3))
                np.testing.assert_array_equal(
                    face[mask], np.broadcast_to(style["glyph_rgb"], (mask.sum(), 3))
                )
                bg = (
                    style["body_rgb"]
                    if style["panel"] == "none"
                    else style["panel_rgb"]
                )
                self.assertGreaterEqual(contrast(style["glyph_rgb"], bg), 4.5)

    def test_corrupt_appearance_fails(self):
        record = sample_appearance(31)
        for key, value in (
            ("finish", "metal"),
            ("glyph_rgb", [240, 240, 240]),
            ("stroke", 9),
        ):
            bad = deepcopy(record)
            bad["blocks"]["A"][key] = value
            if key == "glyph_rgb":
                bad["blocks"]["A"]["panel"] = "square"
                bad["blocks"]["A"]["panel_rgb"] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_appearance(bad)


class AppearanceSceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = mujoco.MjModel.from_xml_path(str(SCENE))
        cls.varied = mujoco.MjModel.from_xml_string(
            _resolve_scene_xml_paths(appearance_xml(SCENE.read_text()), SCENE.parent)
        )

    def test_all_physical_parameters_are_identical(self):
        for name in (
            "body_mass",
            "body_inertia",
            "body_ipos",
            "body_iquat",
            "qpos0",
            "geom_contype",
            "geom_conaffinity",
            "geom_friction",
            "geom_size",
            "geom_solref",
            "geom_solimp",
            "geom_priority",
            "geom_condim",
            "jnt_type",
            "jnt_range",
            "dof_damping",
            "dof_armature",
            "actuator_gainprm",
            "actuator_biasprm",
        ):
            with self.subTest(parameter=name):
                np.testing.assert_array_equal(
                    getattr(self.base, name), getattr(self.varied, name)
                )
        # Only the compact per-cube tiles replace the original shared atlas.
        self.assertLess(self.varied.tex_data.nbytes, self.base.tex_data.nbytes)

    def test_replay_replaces_every_texture_and_finish(self):
        record = sample_appearance(31)
        apply_appearance(self.varied, record)
        expected = self.varied.tex_data.copy()
        expected_finish = self.varied.mat_shininess.copy()
        apply_appearance(self.varied, sample_appearance(32))
        self.assertFalse(np.array_equal(expected, self.varied.tex_data))
        apply_appearance(self.varied, json.loads(json.dumps(record)))
        np.testing.assert_array_equal(self.varied.tex_data, expected)
        np.testing.assert_array_equal(self.varied.mat_shininess, expected_finish)
        for letter, style in record["blocks"].items():
            body, face = texture_tiles(letter, style)
            start = int(
                self.varied.tex_adr[
                    self.varied.texture(f"block_appearance_{letter}").id
                ]
            )
            pixels = self.varied.tex_data[start : start + 2 * TILE * TILE * 3].reshape(
                2 * TILE, TILE, 3
            )
            np.testing.assert_array_equal(pixels[:TILE], body[::-1])
            np.testing.assert_array_equal(pixels[TILE:], face[::-1])

    def test_registered_tasks_preserve_goal_pose_and_legacy_reset(self):
        env = make_env(task="spell_cat", render_cameras=False)
        self.addCleanup(env.close)
        baseline_spec = SimTaskSpec(
            name="seeded_control",
            env_task="blocks",
            prompt="sampled spelling",
            evaluator_name="spell_word",
            evaluator_options=(("sequence_length", 3),),
            reset_request=(("sequence_length", 3),),
        )
        env.set_task(baseline_spec)
        _, before = env.reset(seed=31)
        baseline = before["randomization"]
        qpos = env.data.qpos.copy()
        for length in (2, 3, 4):
            env.set_task(get_task_spec(f"spell_random_{length}_appearance"))
            observation, info = env.reset(seed=31)
            state = info["randomization"]
            self.assertEqual(state.object_states, baseline.object_states)
            np.testing.assert_array_equal(env.data.qpos, qpos)
            self.assertEqual(
                SpellingGoal.from_metadata(state.metadata),
                SpellingGoal.sample(length, 31),
            )
            self.assertEqual(observation["prompt"], state.metadata["prompt"])
            pixels = env.model.tex_data.copy()
            restored = RandomizationState.from_dict(
                json.loads(json.dumps(state.to_dict()))
            )
            env._task_randomizer.apply(env.model, env.data, restored)
            np.testing.assert_array_equal(env.model.tex_data, pixels)
            np.testing.assert_array_equal(env.data.qpos, qpos)
        env.set_task("spell_cat")
        _, info = env.reset(seed=31)
        self.assertNotIn("block_appearance", info["randomization"].metadata)
        self.assertNotIn("spelling_goal", info["randomization"].metadata)
        self.assertEqual(env.prompt, "spell cat")
        np.testing.assert_array_equal(env.model.tex_data, self.base.tex_data)
        np.testing.assert_array_equal(env.data.qpos, qpos)

    def test_shared_texture_batch_fails_before_gpu_initialization(self):
        env = make_env(task="spell_random_2_appearance", render_cameras=False)
        self.addCleanup(env.close)
        batch = BatchedWarpYAMEnv(env, num_worlds=2, camera_height=64, camera_width=64)
        with self.assertRaisesRegex(NotImplementedError, "shares texture buffers"):
            batch.reset([31, 32])
        self.assertIsNone(batch._runtime)
        env.set_task("spell_cat")
        with self.assertRaisesRegex(NotImplementedError, "shares texture buffers"):
            batch.reset(
                [31, 32],
                options={
                    "randomization": {
                        "sequence_length": 2,
                        "randomize_appearance": True,
                    }
                },
            )
        self.assertIsNone(batch._runtime)


if __name__ == "__main__":
    unittest.main()
