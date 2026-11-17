"""Runtime hooks for tasks that need Python-side scene logic."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import mujoco

from abc_sim.task_eval.base import EpisodeTaskEvaluator, TaskEvalResult, TaskEvaluator
from abc_sim.task_specs import SimTaskSpec


class TaskRuntime(Protocol):
    """Python-side controller for task behavior that cannot live in XML alone.

    A runtime may implement ``on_task_evaluation(result)`` to consume the
    environment's initial and physics-step scores before its ``after_step`` hook.
    ``retained_randomization()`` may return a complete initial scene to restore
    through the randomizer on nonrandomized reset. An ``EpisodeTaskEvaluator``
    supplies the validated definition passed to ``after_reset(..., episode=...)``
    and ``configure_episode``.
    Optional ``initialize_robot_pose(model, data)`` applies task-specific default
    joint positions and controls on the final restored model. The environment
    calls it only when ``keep_arm_state`` is false. ``after_reset`` initializes
    episode bookkeeping without overriding the selected robot pose policy.
    """

    def bind(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Bind the runtime to the current MuJoCo model/data pair."""

    def after_reset(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        randomization: Any = None,
    ) -> None:
        """Run after env reset and randomization."""

    def before_step(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Run before each MuJoCo physics step."""

    def after_step(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Run after each MuJoCo physics step."""

    def debug_state(self) -> dict[str, Any]:
        """Return optional task runtime debug state."""


@runtime_checkable
class TaskProgressRuntime(TaskRuntime, Protocol):
    """Require chronological progress independently of reward exposure.

    The environment binds one source per model/task and observes it at every
    native physics tick. A configured reward evaluator must be compatible and
    returned as that same source, so scoring and runtime history advance once.
    Without a reward evaluator, the runtime creates its native progress source.
    """

    def make_progress_source(
        self,
        model: mujoco.MjModel,
        spec: SimTaskSpec,
        evaluator: TaskEvaluator | None,
    ) -> EpisodeTaskEvaluator:
        """Resolve progress or reject an incompatible evaluator at binding."""

    def after_reset(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        randomization: Any = None,
        episode: Any,
    ) -> None:
        """Start prompts from the same prepared definition as the progress source."""

    def on_task_evaluation(self, result: TaskEvalResult) -> None:
        """Consume the source's latest progress without observing physics again."""
