"""Compiled-scene model caches: continuous random scales must not fill the cache with models no reset reuses."""

import unittest
from pathlib import Path

_ASSETS = Path(__file__).resolve().parents[1] / "abc_sim" / "models" / "assets"


@unittest.skipUnless((_ASSETS / "task_water_bottles").exists(), "water bottle assets are not provisioned")
class SceneModelCacheTest(unittest.TestCase):
    def make(self):
        import abc_sim

        return abc_sim.make_env(task="put_plastic_bottles_in_bin", render_cameras=False)

    def test_scaled_models_keep_one_entry(self):
        env = self.make()
        for seed in range(6):
            env.reset(seed=seed)
        cache = env._task_randomizer._compiled_scene_model_cache
        self.assertLessEqual(sum(1 for key in cache if key[-1]), 1)
        env.close()

    def test_unscaled_models_are_still_reused(self):
        env = self.make()
        randomizer = env._task_randomizer
        hits = []
        lookup = randomizer._try_reload_cached_scene_model
        randomizer._try_reload_cached_scene_model = lambda key: hits.append(lookup(key)) or hits[-1]
        options = {"randomization": {"bottle_count": 2, "randomize_variants": False, "randomize_scales": False}}
        for seed in range(3):
            env.reset(seed=seed, options=options)
        self.assertTrue(any(hits))
        env.close()


if __name__ == "__main__":
    unittest.main()
