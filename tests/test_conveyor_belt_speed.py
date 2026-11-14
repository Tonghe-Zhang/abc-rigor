"""Conveyor belt speed: 0.10-0.20 m/s by default; a reset request can move either bound."""

from __future__ import annotations

import unittest

from abc_sim import make_env
from abc_sim.randomization.tasks.conveyor_pick import _conveyor_belt_speed_range


class ConveyorBeltSpeedTest(unittest.TestCase):
    def speeds(self, request=None, seeds=range(12)):
        env = make_env("conveyor_pick", render_cameras=False)
        self.addCleanup(env.close)
        options = None if request is None else {"randomization": request}
        result = []
        for seed in seeds:
            _, info = env.reset(seed=seed, options=options)
            metadata = info["randomization"].metadata
            self.assertAlmostEqual(abs(env._task_runtime._belt_speed_y), metadata["belt_speed_m_s"])
            result.append((metadata["belt_speed_m_s"], tuple(metadata["belt_speed_range_m_s"])))
        return result

    def test_default_range_is_unchanged(self):
        for speed, speed_range in self.speeds():
            self.assertEqual(speed_range, (0.10, 0.20))
            self.assertTrue(0.10 <= speed <= 0.20)

    def test_requested_bounds_rescale_the_same_draw(self):
        default = [speed for speed, _ in self.speeds()]
        wide = self.speeds({"belt_speed_min_m_s": 0.05, "belt_speed_max_m_s": 0.30})
        for (speed, speed_range), base in zip(wide, default):
            self.assertEqual(speed_range, (0.05, 0.30))
            self.assertAlmostEqual((speed - 0.05) / 0.25, (base - 0.10) / 0.10, places=6)
        self.assertTrue(any(speed < 0.10 for speed, _ in wide) and any(speed > 0.20 for speed, _ in wide))

    def test_either_bound_alone_and_validation(self):
        self.assertEqual(_conveyor_belt_speed_range(None), (0.10, 0.20))
        self.assertEqual(_conveyor_belt_speed_range({"belt_speed_min_m_s": 0.05}), (0.05, 0.20))
        self.assertEqual(_conveyor_belt_speed_range({"belt_speed_max_m_s": 0.3}), (0.10, 0.3))
        for bad in ({"belt_speed_min_m_s": 0.3}, {"belt_speed_min_m_s": 0.0}, {"belt_speed_max_m_s": 0.6}, {"belt_speed": 0.1}):
            with self.subTest(request=bad), self.assertRaises(ValueError):
                _conveyor_belt_speed_range(bad)


if __name__ == "__main__":
    unittest.main()
