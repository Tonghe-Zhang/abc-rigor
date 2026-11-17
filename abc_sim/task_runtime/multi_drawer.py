"""Runtime prompt sequencing for the multi-drawer memory task."""

from __future__ import annotations

import copy
from typing import Any

import mujoco
from abc_sim.multi_drawer_episode import DrawerEpisode
from abc_sim.task_eval.base import TaskEvalResult, TaskEvaluator
from abc_sim.task_eval.multi_drawer import (
    DrawerProgressSource,
    MultiDrawerSearchProgress,
)
from abc_sim.task_specs import SimTaskSpec


class MultiDrawerSearchRuntime:
    """Advance through prompted objects as they are placed in the goal bin."""

    def __init__(self) -> None:
        self._targets: list[dict[str, str]] = []
        self._current_index = 0
        self._display_prompt = "find the target object"
        self._state = "waiting_for_target"
        self._episode_randomization = None

    def bind(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """The prompt sequence holds no compiled-model indices."""

    def make_progress_source(
        self,
        model: mujoco.MjModel,
        spec: SimTaskSpec,
        evaluator: TaskEvaluator | None,
    ) -> DrawerProgressSource:
        """Share compatible scoring or bind native progress for an unscored task."""
        if evaluator is None:
            return MultiDrawerSearchProgress(model=model, spec=spec)
        if not isinstance(evaluator, DrawerProgressSource):
            raise ValueError(
                "Multi-drawer runtime requires an evaluator with ordered drawer progress."
            )
        return evaluator

    def retained_randomization(self) -> Any:
        """Return the physical initial scene to restore on nonrandomized reset."""
        return copy.deepcopy(self._episode_randomization)

    def after_reset(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        randomization: Any = None,
        episode: DrawerEpisode,
    ) -> None:
        """Load the shared source's validated initial target sequence."""

        self._targets = [
            {"body": target.body, "joint": target.joint, "label": target.label}
            for target in episode.targets
        ]
        self._episode_randomization = copy.deepcopy(randomization)
        self._current_index = 0
        self._state = "waiting_for_target"
        self.bind(model, data)
        self._update_prompt()

    def before_step(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """No pre-step controls are needed."""

    def after_step(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """The environment has already observed the shared progress source."""

    def on_task_evaluation(self, result: TaskEvalResult) -> None:
        """Use the shared deposit counter as the prompt sequence authority."""
        if result.num_worlds != 1 or int(result.metrics["target_count"][0]) != len(
            self._targets
        ):
            raise ValueError("Runtime and evaluator target sequences must match.")
        self._current_index = int(result.metrics["completed_target_count"][0])
        self._state = (
            "finished"
            if self._current_index >= len(self._targets)
            else "waiting_for_target"
        )
        self._update_prompt()

    def observation_prompt(self) -> str:
        """Return the prompt that should be included in observations and recordings."""

        return self._display_prompt

    def debug_state(self) -> dict[str, Any]:
        """Return sequence progress for the VR streamer and task dashboard."""

        current_target = (
            None
            if self._current_index >= len(self._targets)
            else dict(self._targets[self._current_index])
        )
        return {
            "state": self._state,
            "display_prompt": self._display_prompt,
            "current_target_index": self._current_index,
            "completed_target_count": self._current_index,
            "target_count": len(self._targets),
            "current_target": current_target,
            "target_sequence": [dict(target) for target in self._targets],
            "complete": self._state == "finished",
        }

    def _update_prompt(self) -> None:
        if self._current_index >= len(self._targets):
            self._display_prompt = "finished"
            return
        self._display_prompt = f"find the {self._targets[self._current_index]['label']}"
