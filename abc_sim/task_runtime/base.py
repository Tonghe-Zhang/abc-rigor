"""Runtime hooks for tasks that need Python-side scene logic."""

from __future__ import annotations

from typing import Any, Protocol

import mujoco


class TaskRuntime(Protocol):
    """Python-side controller for task behavior that cannot live in XML alone.

    A runtime may implement ``on_task_evaluation(result)`` to consume the
    environment's initial and physics-step scores before its ``after_step`` hook.
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
