"""Seeded materials, replay, visual-only changes and legacy spelling behavior."""

import gc
import json
import subprocess
import sys
import textwrap
import unittest
import weakref
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

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
from abc_sim.randomization.tasks.bottles import WaterBottleRandomizer
from abc_sim.randomization.tasks.chess import ChessRandomizer
from abc_sim.randomization.tasks.dishrack import DishRackRandomizer
from abc_sim.randomization.tasks.mugs import MugVariantRandomizer
from abc_sim.spelling_goal import SpellingGoal
from abc_sim.task_specs import SimTaskSpec, get_task_spec
from tests.test_spelling_goals import blocks_xml

SCENE = Path(__file__).resolve().parents[1] / "abc_sim/models/yam_blocks_scene.xml"


def sample_compiled_letter_faces(model, letter, texture_name):
    """Sample both physical letter faces using compiled geometry and UVs."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    geom = model.geom(f"block_{letter}_vis").id
    mesh = int(model.geom_dataid[geom])
    vertex_start = int(model.mesh_vertadr[mesh])
    vertices = model.mesh_vert[
        vertex_start : vertex_start + int(model.mesh_vertnum[mesh])
    ]
    vertices = vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
    body = int(model.geom_bodyid[geom])
    vertices = (vertices - data.xpos[body]) @ data.xmat[body].reshape(3, 3)
    face_start = int(model.mesh_faceadr[mesh])
    triangles = model.mesh_face[face_start : face_start + int(model.mesh_facenum[mesh])]
    uv_start = int(model.mesh_texcoordadr[mesh])
    face_uv = model.mesh_texcoord[
        uv_start
        + model.mesh_facetexcoord[
            face_start : face_start + int(model.mesh_facenum[mesh])
        ]
    ]
    texture = model.texture(texture_name).id
    height, width = int(model.tex_height[texture]), int(model.tex_width[texture])
    start = int(model.tex_adr[texture])
    pixels = model.tex_data[start : start + height * width * 3].reshape(
        height, width, 3
    )
    half_width = np.max(np.abs(vertices), axis=0)
    y, z = np.meshgrid(
        np.linspace(-0.67, 0.67, 96) * half_width[1],
        np.linspace(-0.67, 0.67, 96) * half_width[2],
    )
    points = np.c_[y.ravel(), z.ravel()]
    result = []
    for sign in (-1, 1):
        sampled = np.zeros((len(points), 3), dtype=np.uint8)
        covered = np.zeros(len(points), dtype=bool)
        for triangle, uv in zip(triangles, face_uv, strict=True):
            corners = vertices[triangle]
            if not np.allclose(corners[:, 0], sign * half_width[0], atol=1e-7):
                continue
            origin = corners[0, 1:]
            edges = (corners[1:, 1:] - origin).T
            barycentric = np.linalg.solve(edges, (points - origin).T).T
            inside = (barycentric >= -1e-6).all(axis=1) & (
                barycentric.sum(axis=1) <= 1 + 1e-6
            )
            coords = uv[0] + barycentric[inside] @ (uv[1:] - uv[0])
            rows = np.rint(coords[:, 1] * (height - 1)).astype(int)
            columns = np.rint(coords[:, 0] * (width - 1)).astype(int)
            sampled[inside] = pixels[rows, columns]
            covered |= inside
        if not covered.all():
            raise AssertionError(
                "Compiled letter face does not cover the sampled grid."
            )
        result.append(sampled.reshape(96, 96, 3))
    return result


class AppearanceTextureTests(unittest.TestCase):
    def test_simulator_randomizers_and_glyphs_do_not_require_imageio(self):
        probe = textwrap.dedent(
            """
            import importlib.abc
            import sys

            class UnavailableImagePackages(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.split(".", 1)[0] in {"imageio", "PIL"}:
                        raise ModuleNotFoundError(fullname)

            sys.meta_path.insert(0, UnavailableImagePackages())
            from abc_sim.task_registry import get_task_randomizer
            from abc_sim.randomization.assets.block_appearance import (
                LETTERS, TILE, glyph_masks, sample_appearance, texture_tiles,
            )

            assert get_task_randomizer("bottles") is not None
            masks = glyph_masks()
            assert set(masks) == set(LETTERS)
            assert all(mask.sum() > 150 for mask in masks.values())
            body, face = texture_tiles("B", sample_appearance(31)["blocks"]["B"])
            assert body.shape == face.shape == (TILE, TILE, 3)
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=SCENE.parents[2],
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

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

    def test_compiled_letter_faces_preserve_original_glyph_orientation(self):
        record = sample_appearance(31)
        for style in record["blocks"].values():
            style.update(
                family="white_plastic",
                body_rgb=[245, 245, 245],
                glyph_rgb=[0, 0, 0],
                panel="none",
                border="none",
                stroke=0,
            )
        apply_appearance(self.varied, record)
        for letter in ("F", "J", "R"):
            original = sample_compiled_letter_faces(self.base, letter, "block_tex")
            generated = sample_compiled_letter_faces(
                self.varied, letter, f"block_appearance_{letter}"
            )
            for side, (before, after) in enumerate(
                zip(original, generated, strict=True)
            ):
                with self.subTest(letter=letter, side=side):
                    # The original paint is dark and colored; new test ink is black.
                    original_ink = (
                        before.max(axis=2).astype(int) - before.min(axis=2) > 55
                    ) & (before.min(axis=2) < 140)
                    generated_ink = (after < 20).all(axis=2)
                    intersection = (original_ink & generated_ink).sum()
                    union = (original_ink | generated_ink).sum()
                    overlap = intersection / union
                    self.assertGreater(overlap, 0.6)
                    # Cleaned paint can fill atlas wear, but must retain the
                    # original physical orientation on either cube face.
                    for reflected in (
                        generated_ink[::-1],
                        generated_ink[:, ::-1],
                        generated_ink[::-1, ::-1],
                    ):
                        reflected_overlap = (original_ink & reflected).sum() / (
                            original_ink | reflected
                        ).sum()
                        self.assertGreater(overlap - reflected_overlap, 0.1)

    def test_registered_tasks_preserve_goal_pose_and_legacy_reset(self):
        env = make_env(task="spell_cat", render_cameras=False, wrist_fov=90)
        self.addCleanup(env.close)
        camera_ids = [env.model.camera(name).id for name in ("left", "right")]
        expected_fov = env.model.cam_fovy[camera_ids].copy()
        expected_positions = env.model.cam_pos[camera_ids].copy()
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
            np.testing.assert_array_equal(env.model.cam_fovy[camera_ids], expected_fov)
            np.testing.assert_array_equal(
                env.model.cam_pos[camera_ids], expected_positions
            )
            pixels = env.model.tex_data.copy()
            before_eval = env.evaluate_task()
            restored = RandomizationState.from_dict(
                json.loads(json.dumps(state.to_dict()))
            )
            env._task_randomizer.apply(env.model, env.data, restored)
            np.testing.assert_array_equal(env.model.tex_data, pixels)
            np.testing.assert_array_equal(env.data.qpos, qpos)
            after_eval = env.evaluate_task()
            self.assertEqual(before_eval.metrics["word"], after_eval.metrics["word"])
            self.assertEqual(before_eval.scalar_reward(), after_eval.scalar_reward())
            np.testing.assert_array_equal(env.model.cam_fovy[camera_ids], expected_fov)
            np.testing.assert_array_equal(
                env.model.cam_pos[camera_ids], expected_positions
            )
            _, _, _, _, step_info = env.step(env.get_state(), render_obs=False)
            self.assertEqual(step_info["episode_step"], 1)
        env.set_task("spell_cat")
        _, info = env.reset(seed=31)
        self.assertNotIn("block_appearance", info["randomization"].metadata)
        self.assertNotIn("spelling_goal", info["randomization"].metadata)
        self.assertEqual(env.prompt, "spell cat")
        np.testing.assert_array_equal(env.model.tex_data, self.base.tex_data)
        np.testing.assert_array_equal(env.data.qpos, qpos)
        np.testing.assert_array_equal(env.model.cam_fovy[camera_ids], expected_fov)
        np.testing.assert_array_equal(env.model.cam_pos[camera_ids], expected_positions)

    def test_replay_restores_an_earlier_episode_goal_prompt_and_metadata(self):
        env = make_env(task="spell_random_3_appearance", render_cameras=False)
        self.addCleanup(env.close)
        _, info = env.reset(seed=31)
        saved = RandomizationState.from_dict(
            json.loads(json.dumps(info["randomization"].to_dict()))
        )
        qpos, pixels = env.data.qpos.copy(), env.model.tex_data.copy()
        expected = env.evaluate_task()
        env.reset(seed=976)
        self.assertNotEqual(env.prompt, saved.metadata["prompt"])
        env._task_randomizer.apply(env.model, env.data, saved)
        self.assertEqual(env.prompt, saved.metadata["prompt"])
        self.assertEqual(env._last_randomization.to_dict(), saved.to_dict())
        np.testing.assert_array_equal(env.data.qpos, qpos)
        np.testing.assert_array_equal(env.model.tex_data, pixels)
        restored = env.evaluate_task()
        self.assertEqual(restored.metrics["word"], expected.metrics["word"])
        self.assertEqual(restored.scalar_reward(), expected.scalar_reward())
        _, _, _, _, step_info = env.step(env.get_state(), render_obs=False)
        self.assertEqual(step_info["episode_step"], 1)

    def test_initial_replay_after_stepping_repeats_the_first_action(self):
        env = make_env(task="spell_cat", render_cameras=False)
        self.addCleanup(env.close)
        for task in ("spell_cat", "spell_random_3_appearance"):
            with self.subTest(task=task):
                env.set_task(task)
                _, info = env.reset(seed=31)
                saved = info["randomization"]
                initial_qpos = env.data.qpos.copy()
                initial_qvel = env.data.qvel.copy()
                initial_ctrl = env.data.ctrl.copy()
                action = env.get_state()
                action[0] += 0.1
                env.step(action, render_obs=False)
                first_qpos = env.data.qpos.copy()
                first_qvel = env.data.qvel.copy()
                first_time = env.data.time
                self.assertGreater(first_time, 0)
                self.assertTrue(np.any(first_qvel))
                self.assertFalse(np.array_equal(env.data.ctrl, initial_ctrl))
                for _ in range(2):
                    env._task_randomizer.apply(env.model, env.data, saved)
                    np.testing.assert_array_equal(env.data.qpos, initial_qpos)
                    np.testing.assert_array_equal(env.data.qvel, initial_qvel)
                    np.testing.assert_array_equal(env.data.ctrl, initial_ctrl)
                    self.assertEqual(env.data.time, 0)
                    self.assertEqual(env.cur_step, 0)
                    env.step(action, render_obs=False)
                    np.testing.assert_array_equal(env.data.qpos, first_qpos)
                    np.testing.assert_array_equal(env.data.qvel, first_qvel)
                    self.assertEqual(env.data.time, first_time)

    def test_camera_provider_receives_materials_before_reset_and_closes_on_reload(self):
        events = []
        snapshots = []

        class RecordingProvider:
            def __init__(self, *, model, camera_names, height, width, **kwargs):
                self.index = len(snapshots)
                self.names = camera_names
                self.shape = (height, width, 3)
                offsets = np.linspace(0, model.tex_data.size - 1, 257, dtype=int)
                snapshots.append(
                    {
                        "textures": model.tex_data[offsets].copy(),
                        "shininess": model.mat_shininess.copy(),
                        "fov": model.cam_fovy.copy(),
                        "positions": model.cam_pos.copy(),
                    }
                )
                events.append(("create", self.index))

            def reset(self):
                events.append(("reset", self.index))

            def frames_for_step(self, step, time):
                events.append(("frame", self.index))
                return {
                    name: np.zeros(self.shape, dtype=np.uint8) for name in self.names
                }

            def close(self):
                events.append(("close", self.index))

        with patch(
            "abc_sim.env.create_live_camera_provider", side_effect=RecordingProvider
        ):
            env = make_env(
                task="spell_random_3_appearance",
                camera_backend="mjwarp",
                camera_height=4,
                camera_width=4,
                wrist_fov=90,
            )
            self.addCleanup(env.close)
            _, info = env.reset(seed=31)
            saved = info["randomization"]
            env.reset(seed=976)
            self.assertFalse(
                np.array_equal(snapshots[1]["textures"], snapshots[2]["textures"])
            )
            env._task_randomizer.apply(env.model, env.data, saved)
            env.get_obs()
            env.set_task("spell_cat")
            env.reset(randomize=False)
            env.close()
        for before, after in ((0, 4), (1, 3)):
            for key in snapshots[before]:
                np.testing.assert_array_equal(
                    snapshots[before][key], snapshots[after][key]
                )
        for snapshot in snapshots:
            np.testing.assert_array_equal(snapshot["fov"], snapshots[0]["fov"])
            np.testing.assert_array_equal(
                snapshot["positions"], snapshots[0]["positions"]
            )
        self.assertEqual(
            events,
            [("create", 0)]
            + [
                event
                for index in range(1, 5)
                for event in (
                    ("close", index - 1),
                    ("create", index),
                    ("reset", index),
                    ("frame", index),
                )
            ]
            + [("close", 4)],
        )

    def test_legacy_visual_restoration_is_independent_of_pose_sampling(self):
        env = make_env(task="spell_cat", render_cameras=False)
        self.addCleanup(env.close)
        for randomize in (False, True):
            with self.subTest(randomize=randomize):
                env.set_task("spell_random_2_appearance")
                env.reset(seed=31)
                env.set_task("spell_cat")
                observation, info = env.reset(seed=31, randomize=randomize)
                np.testing.assert_array_equal(env.model.tex_data, self.base.tex_data)
                self.assertEqual(observation["prompt"], "spell cat")
                self.assertEqual(env.evaluate_task().metrics["word"], "CAT")
                if randomize:
                    self.assertNotIn("block_appearance", info["randomization"].metadata)
                else:
                    self.assertNotIn("randomization", info)
                    self.assertIsNone(env._last_randomization)

    def test_nonrandomized_appearance_reset_requires_explicit_replay(self):
        env = make_env(task="spell_random_2_appearance", render_cameras=False)
        self.addCleanup(env.close)
        with self.assertRaisesRegex(ValueError, "saved-state replay"):
            env.reset(seed=31, randomize=False)

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


class CameraModelBindingTests(unittest.TestCase):
    def test_native_model_cache_cycles_preserve_camera_mounts(self):
        scene = blocks_xml().replace(
            "<worldbody>",
            '<worldbody><camera name="left" pos=".1 .2 .3" fovy="58"/>'
            '<camera name="right" pos=".4 .5 .6" fovy="58"/>',
        )
        replacement_scene = scene.replace('pos=".1 .2 .3"', 'pos=".7 .8 .9"')
        for wrist_fov in (58, 72):
            for randomizer_type in (
                WaterBottleRandomizer,
                ChessRandomizer,
                DishRackRandomizer,
                MugVariantRandomizer,
            ):
                with self.subTest(
                    randomizer=randomizer_type.__name__, wrist_fov=wrist_fov
                ):
                    env = make_env(
                        task="spell_cat",
                        scene_xml_string=scene,
                        render_cameras=False,
                        wrist_fov=wrist_fov,
                    )
                    self.addCleanup(env.close)
                    randomizer = randomizer_type()
                    randomizer.bind_env(env)
                    expected = {}
                    for key, xml in (
                        ("original", scene),
                        ("changed", replacement_scene),
                    ):
                        env.reload_from_xml(xml)
                        expected[key] = (
                            env.model.cam_pos.copy(),
                            env.model.cam_fovy.copy(),
                        )
                        randomizer._store_compiled_scene_model((key,))
                    for key in ("original", "changed") * 3:
                        self.assertTrue(
                            randomizer._try_reload_cached_scene_model((key,))
                        )
                        np.testing.assert_array_equal(
                            env.model.cam_pos, expected[key][0]
                        )
                        np.testing.assert_array_equal(
                            env.model.cam_fovy, expected[key][1]
                        )
                    templates = [
                        weakref.ref(model)
                        for model in randomizer._compiled_scene_model_cache.values()
                    ]
                    randomizer._compiled_scene_model_cache.clear()
                    gc.collect()
                    self.assertTrue(all(template() is None for template in templates))
                    self.assertEqual(
                        len(env._wrist_camera_positions), int(wrist_fov != 58)
                    )

    def test_cached_model_reloads_do_not_accumulate_camera_clearance(self):
        scene = blocks_xml().replace(
            "<worldbody>",
            '<worldbody><camera name="left" pos=".1 .2 .3" fovy="58"/>'
            '<camera name="right" pos=".4 .5 .6" fovy="58"/>',
        )
        env = make_env(
            task="spell_cat",
            scene_xml_string=scene,
            render_cameras=False,
            wrist_fov=72,
        )
        self.addCleanup(env.close)
        camera_ids = [env.model.camera(name).id for name in ("left", "right")]
        fov, positions = (
            env.model.cam_fovy[camera_ids].copy(),
            env.model.cam_pos[camera_ids].copy(),
        )
        self.assertTrue((positions[:, 1] < [0.2, 0.5]).all())
        for _ in range(3):
            env.reload_from_model(env.model)
            np.testing.assert_array_equal(env.model.cam_fovy[camera_ids], fov)
            np.testing.assert_array_equal(env.model.cam_pos[camera_ids], positions)

    def test_replacement_xml_keeps_its_mounts_without_retaining_discarded_models(self):
        scene = blocks_xml().replace(
            "<worldbody>",
            '<worldbody><camera name="left" pos=".1 .2 .3" fovy="58"/>',
        )
        env = make_env(
            task="spell_cat",
            scene_xml_string=scene,
            render_cameras=False,
            wrist_fov=72,
        )
        self.addCleanup(env.close)
        original_position = env.model.camera("left").pos.copy()
        old_model = weakref.ref(env.model)
        env.reload_from_xml(scene.replace('pos=".1 .2 .3"', 'pos=".7 .8 .9"'))
        expected_position = original_position + 0.6
        np.testing.assert_allclose(
            env.model.camera("left").pos, expected_position, rtol=0, atol=1e-15
        )
        for _ in range(3):
            env.reload_from_model(env.model)
            np.testing.assert_allclose(
                env.model.camera("left").pos, expected_position, rtol=0, atol=1e-15
            )
            self.assertEqual(env.model.camera("left").fovy[0], 72)
        gc.collect()
        self.assertIsNone(old_model())
        self.assertEqual(len(env._wrist_camera_positions), 1)

    def test_reset_arm_policy_is_explicit_and_preserves_only_requested_pose(self):
        env = make_env(
            task="spell_cat",
            scene_xml_string=blocks_xml(),
            render_cameras=False,
        )
        self.addCleanup(env.close)
        env.reset(seed=31)
        initial = env._get_reset_arm_state()
        action = env.get_state()
        action[0] += 0.1
        env.step(action, render_obs=False)
        carried = env._get_reset_arm_state()
        env.reset(seed=31, options={"keep_arm_state": True})
        np.testing.assert_array_equal(env._get_reset_arm_state(), carried)
        self.assertEqual(env.data.time, 0)
        np.testing.assert_array_equal(env.data.qvel, 0)
        np.testing.assert_array_equal(env.data.ctrl, 0)
        env.reset(seed=31)
        np.testing.assert_array_equal(env._get_reset_arm_state(), initial)


if __name__ == "__main__":
    unittest.main()
