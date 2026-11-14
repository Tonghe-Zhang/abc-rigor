"""Conveyor belt speed: 0.10-0.20 m/s by default; a reset request or the wide-speed task widens it."""

from __future__ import annotations

import unittest

from abc_sim import make_env
from abc_sim.randomization.tasks.conveyor_pick import _conveyor_belt_speed_range


class ConveyorBeltSpeedTest(unittest.TestCase):
    def speeds(self, task, seeds=range(12)):
        env = make_env(task, render_cameras=False)
        self.addCleanup(env.close)
        result = []
        for seed in seeds:
            _, info = env.reset(seed=seed)
            metadata = info["randomization"].metadata
            self.assertAlmostEqual(abs(env._task_runtime._belt_speed_y), metadata["belt_speed_m_s"])
            result.append((metadata["belt_speed_m_s"], tuple(metadata["belt_speed_range_m_s"])))
        return result

    def test_default_range_is_unchanged(self):
        for speed, speed_range in self.speeds("conveyor_pick"):
            self.assertEqual(speed_range, (0.10, 0.20))
            self.assertTrue(0.10 <= speed <= 0.20)

    def test_wide_task_samples_the_wider_range_with_the_same_draws(self):
        default = [speed for speed, _ in self.speeds("conveyor_pick")]
        wide = self.speeds("conveyor_pick_wide_belt_speed")
        for (speed, speed_range), base in zip(wide, default):
            self.assertEqual(speed_range, (0.05, 0.30))
            # Same uniform draw per seed, rescaled to the wider range.
            self.assertAlmostEqual((speed - 0.05) / 0.25, (base - 0.10) / 0.10, places=6)
        self.assertTrue(any(speed < 0.10 for speed, _ in wide) and any(speed > 0.20 for speed, _ in wide))

    def test_request_validation(self):
        self.assertEqual(_conveyor_belt_speed_range(None), (0.10, 0.20))
        self.assertEqual(_conveyor_belt_speed_range({"belt_speed_range_m_s": [0.02, 0.4]}), (0.02, 0.4))
        for bad in ({"belt_speed_range_m_s": [0.3, 0.1]}, {"belt_speed_range_m_s": [0.0, 0.2]}, {"belt_speed": 0.1}):
            with self.subTest(request=bad), self.assertRaises(ValueError):
                _conveyor_belt_speed_range(bad)


if __name__ == "__main__":
    unittest.main()
