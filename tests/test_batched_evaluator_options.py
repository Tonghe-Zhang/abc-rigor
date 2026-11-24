"""CPU regressions for runtime scoring options in native batched evaluators."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np

from abc_sim.batched_env import BatchedWarpYAMEnv, PerWorldEvaluators
from abc_sim.task_eval.base import TaskEvalResult
from abc_sim.task_eval.registry import _TASK_EVALUATORS, make_task_evaluator
from abc_sim.task_specs import SimTaskSpec


class _ScoringEvaluator:
    def __init__(self, *, model, spec, reward_scale=1.0, success_threshold=0.5):
        self.reward_scale = reward_scale
        self.success_threshold = success_threshold
        self.queries = 0

    def reset(self, *, nworld):
        self.queries = 0

    def evaluate_qpos_batch(self, qpos):
        self.queries += 1
        reward = np.asarray(qpos[:, 0] * self.reward_scale, dtype=np.float32)
        success = qpos[:, 0] >= self.success_threshold
        return TaskEvalResult(reward=reward, success=success, metrics={"scaled_position": reward.copy()})


class _RandomizedScoringEvaluator(_ScoringEvaluator):
    def configure_from_randomization(self, model, state):
        self.goal = state


class BatchedEvaluatorOptionsTest(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body><freejoint/><geom type="sphere" size="0.02"/></body></worldbody></mujoco>'
        )
        self.data = mujoco.MjData(self.model)
        self.qpos = np.repeat(self.data.qpos[None], 2, axis=0)
        self.qpos[:, 0] = [0.1, 0.3]
        self.spec = SimTaskSpec(
            name="test_batched_options",
            env_task="test_batched_options",
            prompt="",
            evaluator_name="test_batched_options",
            evaluator_options=(("reward_scale", 2.0), ("success_threshold", 0.2)),
        )
        self.runtime_options = {"reward_scale": 3.0, "success_threshold": 0.1}

    def _batch(self, options):
        base = SimpleNamespace(
            model=self.model,
            data=self.data,
            _task_spec=self.spec,
            _evaluator_options=options,
            _task_evaluator=make_task_evaluator(self.model, self.spec, options),
        )
        batch = BatchedWarpYAMEnv.__new__(BatchedWarpYAMEnv)
        batch.base_env = base
        batch.num_worlds = 2
        batch.camera_height = batch.camera_width = 8
        batch._gpu_id = None
        batch._signature = None
        batch._runtime = batch._renderer = batch._evaluator = None
        batch._bound_base_evaluator = None
        return batch

    def _assert_runtime_scoring(self, evaluator):
        result = evaluator.evaluate_qpos_batch(self.qpos)
        np.testing.assert_allclose(result.reward, [0.3, 0.9])
        np.testing.assert_array_equal(result.success, [True, True])

    def test_per_world_runtime_options_override_spec_for_worlds_and_probe(self):
        with patch.dict(_TASK_EVALUATORS, {self.spec.evaluator_name: _RandomizedScoringEvaluator}):
            evaluators = PerWorldEvaluators(self.model, self.spec, 2, evaluator_options=self.runtime_options)
        for _ in range(2):
            evaluators.reset(self.model, [{"world": 0}, {"world": 1}])
            self._assert_runtime_scoring(evaluators)
            self._assert_runtime_scoring(evaluators._metric_probe)
            counts = [e.queries for e in evaluators.evaluators]
            self.assertEqual(evaluators.per_world_keys(self.qpos), {"scaled_position"})
            self.assertEqual([e.queries for e in evaluators.evaluators], counts)
        self.assertEqual(self.spec.evaluator_kwargs(), {"reward_scale": 2.0, "success_threshold": 0.2})
        self.assertEqual(self.runtime_options, {"reward_scale": 3.0, "success_threshold": 0.1})

    def test_per_world_default_options_retain_spec(self):
        with patch.dict(_TASK_EVALUATORS, {self.spec.evaluator_name: _RandomizedScoringEvaluator}):
            evaluators = PerWorldEvaluators(self.model, self.spec, 2)
        result = evaluators.evaluate_qpos_batch(self.qpos)
        np.testing.assert_allclose(result.reward, [0.2, 0.6])
        np.testing.assert_array_equal(result.success, [False, True])

    def test_bind_model_forwards_runtime_options_to_both_evaluator_paths(self):
        for evaluator_cls in [_ScoringEvaluator, _RandomizedScoringEvaluator]:
            with (
                self.subTest(evaluator=evaluator_cls.__name__),
                patch.dict(_TASK_EVALUATORS, {self.spec.evaluator_name: evaluator_cls}),
                patch("abc_sim.batched_env.WarpReplayRuntime"),
                patch("abc_sim.batched_env.RendererWrapper"),
            ):
                batch = self._batch(self.runtime_options)
                batch._bind_model()
                self._assert_runtime_scoring(batch._evaluator)
                if isinstance(batch._evaluator, PerWorldEvaluators):
                    self._assert_runtime_scoring(batch._evaluator._metric_probe)

    def test_reconfigured_scoring_reuses_geometry_and_unchanged_binding_keeps_history(self):
        for evaluator_cls in [_ScoringEvaluator, _RandomizedScoringEvaluator]:
            with (
                self.subTest(evaluator=evaluator_cls.__name__),
                patch.dict(_TASK_EVALUATORS, {self.spec.evaluator_name: evaluator_cls}),
                patch("abc_sim.batched_env.WarpReplayRuntime") as runtime,
                patch("abc_sim.batched_env.RendererWrapper") as renderer,
            ):
                batch = self._batch({})
                batch._bind_model()
                first = batch._evaluator
                np.testing.assert_allclose(first.evaluate_qpos_batch(self.qpos).reward, [0.2, 0.6])
                batch._bind_model()
                self.assertIs(batch._evaluator, first)

                batch.base_env._evaluator_options = self.runtime_options
                batch.base_env._task_evaluator = make_task_evaluator(self.model, self.spec, self.runtime_options)
                batch._bind_model()
                self.assertIsNot(batch._evaluator, first)
                self._assert_runtime_scoring(batch._evaluator)
                runtime.assert_called_once()
                renderer.assert_called_once()


if __name__ == "__main__":
    unittest.main()
