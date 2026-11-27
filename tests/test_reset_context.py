"""Reset preflight receives actual seed and arm context before initialization."""

from __future__ import annotations

import pickle
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import abc_sim
from abc_sim.background import BackgroundConfig
from abc_sim.config import get_i2rt_sim_config
from abc_sim.env import MuJoCoYAMEnv
from abc_sim.randomization.assets.tabletop_distractors import DISTRACTOR_CATEGORIES
from abc_sim.randomization.core import SceneRandomizer
from abc_sim.randomization.tasks.blocks import BlocksRandomizer
from abc_sim.randomization.tasks.bottles_upright import BottlesUprightRandomizer
from abc_sim.randomization.tasks.multi_drawer_search import MultiDrawerSearchRandomizer
from abc_sim.randomization.tasks.pack_and_pour_fruit import PackAndPourFruitRandomizer
from abc_sim.randomization.tasks.pack_suitcase import PackSuitcaseRandomizer
from abc_sim.randomization.tasks.shape_matching import ShapeMatchingRandomizer
from tests.test_background import make_package


class RecordingRandomizer(SceneRandomizer):
    def __init__(self):
        super().__init__()
        self.calls = []

    def prepare_reset(
        self, *, request=None, randomize=True, seed=None, keep_arm_state=False
    ):
        self.calls.append(
            (request, randomize, seed, keep_arm_state, self._env_ref.data.time)
        )
        if request == {"reject": True}:
            raise ValueError("rejected before native initialization")


class ResetContextTest(unittest.TestCase):
    def setUp(self):
        self.env = MuJoCoYAMEnv(
            config=get_i2rt_sim_config(),
            scene_xml=Path(abc_sim.__file__).parent
            / "models"
            / "yam_bimanual_empty.xml",
            render_cameras=False,
            chunk_dim=1,
        )
        self.addCleanup(self.env.close)
        self.randomizer = RecordingRandomizer()
        self.randomizer.bind_env(self.env)
        self.env._task_randomizer = self.randomizer

    def test_actual_context_is_forwarded_before_physics_reset(self):
        request = {"selected": "exact"}
        for seed in (None, 0, 7, 913):
            for keep in (False, True):
                with self.subTest(seed=seed, keep_arm_state=keep):
                    self.env.data.time = 1.375
                    self.env.reset(
                        seed=seed,
                        options={
                            "randomization": request,
                            "keep_arm_state": keep,
                        },
                    )
                    self.assertEqual(
                        self.randomizer.calls[-1],
                        (request, True, seed, keep, 1.375),
                    )
                    self.assertEqual(self.env.data.time, 0)

    def test_rejection_preserves_native_physics_rng_and_episode(self):
        self.env.reset(seed=7)
        self.env.data.time = 2.25
        self.env.data.qvel[:] = 0.03125
        self.env.cur_step = 19
        before = pickle.dumps((self.env.model, self.env.data))
        rng = self.env.np_random
        rng_state = pickle.dumps(rng.bit_generator.state)
        state = self.env._last_randomization
        for keep in (False, True):
            with self.subTest(keep_arm_state=keep):
                with self.assertRaisesRegex(ValueError, "before native"):
                    self.env.reset(
                        seed=913,
                        options={
                            "randomization": {"reject": True},
                            "keep_arm_state": keep,
                        },
                    )
                self.assertEqual(pickle.dumps((self.env.model, self.env.data)), before)
                self.assertIs(self.env.np_random, rng)
                self.assertEqual(pickle.dumps(rng.bit_generator.state), rng_state)
                self.assertIs(self.env._last_randomization, state)
                self.assertEqual(self.env.cur_step, 19)


class BackgroundResetContextTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.package = make_package(Path(directory.name))

    def make_env(self):
        env = MuJoCoYAMEnv(
            config=get_i2rt_sim_config(),
            scene_xml=Path(abc_sim.__file__).parent
            / "models"
            / "yam_bimanual_empty.xml",
            render_cameras=False,
            chunk_dim=1,
            background=BackgroundConfig(
                asset_dir=str(self.package),
                seed=0,
                background_id=2,
                resample="episode",
                slot_scale=0.1,
            ),
        )
        self.addCleanup(env.close)
        randomizer = RecordingRandomizer()
        randomizer.bind_env(env)
        env._task_randomizer = randomizer
        env.reset(seed=3)
        return env

    @staticmethod
    def snapshot(env):
        background = env._background
        return pickle.dumps(
            (
                env.model,
                env.data,
                background.state,
                background.rng.bit_generator.state,
                background._images,
                background._unused,
                env._background_pending,
                env.np_random.bit_generator.state,
                env.cur_step,
            )
        )

    def test_rejection_preserves_background_rng_images_pixels_and_pending(self):
        env = self.make_env()
        background = env._background
        self.assertEqual(env.background_info["background_id"], 2)
        env.data.time = 1.375
        env.data.qvel[:] = 0.03125
        env.cur_step = 19
        for pinned in (None, 3):
            for pending in (False, True):
                for keep in (False, True):
                    with self.subTest(pinned=pinned, pending=pending, keep=keep):
                        env._background_pending = pending
                        before = self.snapshot(env)
                        state = background.state
                        images = background._images
                        randomization = env._last_randomization
                        options = {
                            "randomization": {"reject": True},
                            "keep_arm_state": keep,
                        }
                        if pinned is not None:
                            options["background_id"] = pinned
                        with self.assertRaisesRegex(ValueError, "before native"):
                            env.reset(seed=19, options=options)
                        self.assertTrue(
                            self.snapshot(env) == before,
                            "rejected reset changed the episode or background",
                        )
                        self.assertIs(background.state, state)
                        self.assertIs(background._images, images)
                        self.assertIs(env._last_randomization, randomization)

    def test_successful_background_stream_matches_an_uninterrupted_environment(self):
        env, control = self.make_env(), self.make_env()
        previous = deepcopy(env.background_info)
        observed = []
        prepare = env._task_randomizer.prepare_reset

        def record(**kwargs):
            observed.append(deepcopy(env.background_info))
            return prepare(**kwargs)

        env._task_randomizer.prepare_reset = record
        with self.assertRaises(ValueError):
            env.reset(
                seed=19, options={"randomization": {"reject": True}, "background_id": 3}
            )
        self.assertEqual(observed.pop(), previous)
        for pinned in (3, None, 5, None):
            options = {} if pinned is None else {"background_id": pinned}
            prior = deepcopy(env.background_info)
            _, info = env.reset(seed=23, options=options)
            _, expected = control.reset(seed=23, options=options)
            self.assertEqual(observed.pop(), prior)
            self.assertEqual(info["background"], expected["background"])
            if pinned is not None:
                self.assertEqual(info["background"]["background_id"], pinned)
            self.assertEqual(
                pickle.dumps(env._background.rng.bit_generator.state),
                pickle.dumps(control._background.rng.bit_generator.state),
            )
            self.assertEqual(
                pickle.dumps(env.model.tex_data), pickle.dumps(control.model.tex_data)
            )
            self.assertFalse(env._background_pending)


class ExistingHookCompatibilityTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        # Preflight discovers installed identities without loading their geometry.
        for category in DISTRACTOR_CATEGORIES:
            candidate = root / category / f"{category}_0"
            candidate.mkdir(parents=True)
            (candidate / "model.xml").write_text("<mujoco/>")
        lookup = patch(
            "abc_sim.randomization.assets.tabletop_distractors._CLUTTER_ROOT", root
        )
        lookup.start()
        self.addCleanup(lookup.stop)

    def test_base_and_all_existing_hooks_accept_default_and_new_context(self):
        classes = (
            SceneRandomizer,
            BlocksRandomizer,
            BottlesUprightRandomizer,
            MultiDrawerSearchRandomizer,
            PackAndPourFruitRandomizer,
            PackSuitcaseRandomizer,
            ShapeMatchingRandomizer,
        )
        for cls in classes:
            for context in (
                {},
                {"seed": None, "keep_arm_state": False},
                {"seed": 7, "keep_arm_state": True},
            ):
                with self.subTest(cls=cls.__name__, context=context):
                    cls().prepare_reset(**context)

    def test_context_preserves_each_existing_validation_error(self):
        cases = (
            (BlocksRandomizer, {"randomize_appearance": True}, False),
            (BottlesUprightRandomizer, {"unknown": 0}, True),
            (MultiDrawerSearchRandomizer, {"unknown": 0}, True),
            (PackAndPourFruitRandomizer, {"unknown": 0}, True),
            (PackSuitcaseRandomizer, {"unknown": 0}, True),
            (ShapeMatchingRandomizer, {"unknown": 0}, True),
        )
        for cls, request, randomize in cases:
            errors = []
            for context in ({}, {"seed": 913, "keep_arm_state": True}):
                with self.subTest(cls=cls.__name__, context=context):
                    with self.assertRaises((ValueError, TypeError)) as error:
                        cls().prepare_reset(
                            request=request, randomize=randomize, **context
                        )
                    self.assertNotIn("prepare_reset", str(error.exception))
                    errors.append((type(error.exception), str(error.exception)))
            self.assertEqual(errors[0], errors[1])


if __name__ == "__main__":
    unittest.main()
