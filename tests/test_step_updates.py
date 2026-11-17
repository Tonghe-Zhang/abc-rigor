"""Task scoring must not add position refreshes to unrelated physics steps."""

import unittest
from types import SimpleNamespace
from unittest import mock

import mujoco
import numpy as np

from abc_sim.env import MuJoCoYAMEnv
from abc_sim.task_eval.ball_tray import BallTrayBalancingEvaluator


class StepUpdatesTest(unittest.TestCase):
    def make_env(self, *, requires_updates, control_rate=False):
        env = MuJoCoYAMEnv.__new__(MuJoCoYAMEnv)
        env.model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body pos="0 0 1"><freejoint/>'
            '<geom type="sphere" size=".01"/></body></worldbody></mujoco>'
        )
        env.data = mujoco.MjData(env.model)
        env._task_spec = None
        env._task_runtime = None
        env._control_decimation = 17
        evaluator = SimpleNamespace(
            requires_step_updates=requires_updates,
            evaluate_at_control_rate=control_rate,
            evaluate_model_data=mock.Mock(return_value=SimpleNamespace()),
        )
        with mock.patch("abc_sim.env.make_task_evaluator", return_value=evaluator):
            env._bind_task_evaluator(env._task_spec, env._task_runtime)
        env._task_runtime = SimpleNamespace(
            before_step=mock.Mock(),
            after_step=mock.Mock(),
            on_task_evaluation=mock.Mock(),
        )
        return env

    def test_unrelated_tasks_never_refresh_or_evaluate_during_physics(self):
        env = self.make_env(requires_updates=False)
        # The hot loop uses the cached requirement, rather than probing every step.
        del env._task_evaluator.requires_step_updates
        with mock.patch("abc_sim.env.mujoco.mj_fwdPosition") as refresh:
            env.advance_physics(36)
        refresh.assert_not_called()
        env._task_evaluator.evaluate_model_data.assert_not_called()
        self.assertEqual(env._task_runtime.before_step.call_count, 36)
        self.assertEqual(env._task_runtime.after_step.call_count, 36)
        self.assertAlmostEqual(env.data.time, 36 * env.model.opt.timestep)

    def test_control_rate_scoring_includes_the_final_partial_step(self):
        env = self.make_env(
            requires_updates=True,
            control_rate=BallTrayBalancingEvaluator.evaluate_at_control_rate,
        )
        times = []

        def evaluate(model, data):
            times.append(data.time)
            # Scoring sees geometry refreshed for the integrated qpos.
            np.testing.assert_allclose(data.xpos[1], data.qpos[:3])
            return SimpleNamespace()

        env._task_evaluator.evaluate_model_data.side_effect = evaluate
        with mock.patch(
            "abc_sim.env.mujoco.mj_fwdPosition", wraps=mujoco.mj_fwdPosition
        ) as refresh:
            env.advance_physics(36)
        np.testing.assert_allclose(
            times, np.array([17, 34, 36]) * env.model.opt.timestep
        )
        self.assertEqual(refresh.call_count, 3)
        self.assertEqual(env._task_runtime.after_step.call_count, 36)
        self.assertEqual(env._task_runtime.on_task_evaluation.call_count, 3)
        self.assertIs(
            env._task_runtime.on_task_evaluation.call_args.args[0],
            env._step_task_eval,
        )

    def test_other_temporal_evaluators_keep_physics_rate_updates(self):
        env = self.make_env(requires_updates=True)
        with mock.patch(
            "abc_sim.env.mujoco.mj_fwdPosition", wraps=mujoco.mj_fwdPosition
        ) as refresh:
            env.advance_physics(3)
        self.assertEqual(refresh.call_count, 3)
        self.assertEqual(env._task_evaluator.evaluate_model_data.call_count, 3)

    def test_native_observation_precedes_temporal_scoring_without_changing_cadence(
        self,
    ):
        env = self.make_env(requires_updates=True, control_rate=True)
        observed = []
        scored = []

        def observe(model, data):
            observed.append(float(data.time))

        def evaluate(model, data):
            # A score reads the observer's current integrated tick.
            self.assertEqual(observed[-1], float(data.time))
            scored.append(len(observed))
            return SimpleNamespace()

        env._physics_observer = observe
        env._task_evaluator.evaluate_model_data.side_effect = evaluate
        env.advance_physics(36)
        self.assertEqual(len(observed), 36)
        self.assertEqual(scored, [17, 34, 36])
        self.assertEqual(env._task_runtime.on_task_evaluation.call_count, 3)

    def test_rebinding_clears_the_previous_tasks_update_requirements(self):
        env = self.make_env(requires_updates=True, control_rate=True)
        with mock.patch("abc_sim.env.make_task_evaluator", return_value=None):
            env._bind_task_evaluator(env._task_spec, env._task_runtime)
        self.assertFalse(env._step_updates)
        self.assertEqual(env._step_update_interval, 1)
        self.assertIsNone(env._step_task_eval)
        with mock.patch("abc_sim.env.mujoco.mj_fwdPosition") as refresh:
            env.advance_physics()
        refresh.assert_not_called()


if __name__ == "__main__":
    unittest.main()
