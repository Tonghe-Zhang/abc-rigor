"""Dish-rack startup placement with the installed simulator assets."""

import unittest
from unittest.mock import patch

import abc_sim


class DishrackResetTests(unittest.TestCase):
    def setUp(self):
        self.env = abc_sim.make_env(
            task="load_plates_into_dish_rack",
            render_cameras=False,
            physics_profile="round2_low_friction_v1",
            chunk_dim=1,
            control_decimation=17,
        )
        self.addCleanup(self.env.close)

    def reset_and_check(self, seed):
        with patch(
            "mujoco.mj_step", side_effect=AssertionError("reset stepped physics")
        ):
            _, info = self.env.reset(seed=seed)
        state = info["randomization"]
        randomizer = self.env._task_randomizer
        self.assertEqual(self.env.data.time, 0)
        self.assertEqual(randomizer.max_tries, 200)
        self.assertTrue(randomizer._bounds_ok(state.object_states))
        self.assertTrue(randomizer._pairwise_ok(state.object_states))
        self.assertTrue(randomizer._contacts_ok(self.env.model, self.env.data))
        self.assertEqual(len(state.object_states), state.metadata["plate_count"] + 1)
        self.assertEqual(
            len(state.metadata["plate_variants"]), state.metadata["plate_count"]
        )
        return state

    def test_seed_72_preserves_four_plates_and_repeats_exactly(self):
        state = self.reset_and_check(72)
        self.assertEqual(state.metadata["dish_rack_variant"], "dish_rack_11")
        self.assertEqual(
            state.metadata["plate_variants"],
            ["plate_0", "plate_8", "plate_4", "plate_1"],
        )
        self.assertEqual(state.metadata["plate_count"], 4)
        self.assertAlmostEqual(state.scale_states["dishrack"], 1.021558330218739)
        self.assertEqual(self.reset_and_check(72).to_dict(), state.to_dict())

    def test_representative_seeds_have_valid_startup_placements(self):
        for seed in (0, 1, 2, 3, 7, 18, 32, 64, 70, 71, 73, 74):
            with self.subTest(seed=seed):
                self.reset_and_check(seed)


if __name__ == "__main__":
    unittest.main()
