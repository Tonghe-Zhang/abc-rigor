"""Runtime prompt sequencing for the multi-drawer memory task."""

from __future__ import annotations

import copy
from typing import Any

import mujoco
from abc_sim.task_eval.base import TaskEvalResult


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

    def resolve_reset_randomization(self, randomization: Any) -> Any:
        """Preserve the current drawer goal when resetting without resampling.

        The environment passes this same episode definition to the runtime and
        evaluator. Counters and opening history are still reset for each episode.
        """
        if randomization is not None:
            self._episode_randomization = copy.deepcopy(randomization)
        return self._episode_randomization

    def after_reset(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        randomization: Any = None,
    ) -> None:
        """Load the reset's sampled target sequence."""

        metadata = getattr(randomization, "metadata", {})
        raw_targets = metadata.get("target_sequence", []) if isinstance(metadata, dict) else []
        if not raw_targets and isinstance(metadata, dict) and metadata.get("target_body"):
            raw_targets = [
                {
                    "body": metadata["target_body"],
                    "joint": metadata.get("target_joint", ""),
                    "label": metadata.get("target_label", "target object"),
                }
            ]
        self._targets = [
            {
                "body": str(target["body"]),
                "joint": str(target.get("joint", "")),
                "label": str(target["label"]),
            }
            for target in raw_targets
            if isinstance(target, dict) and target.get("body") and target.get("label")
        ]
        if not self._targets:
            self._targets = [{
                "body": "drawer_search_target",
                "joint": "drawer_search_target_joint",
                "label": "target object",
            }]
        self._current_index = 0
        self._state = "waiting_for_target"
        self.bind(model, data)
        self._update_prompt()

    def before_step(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """No pre-step controls are needed."""

    def after_step(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Progress arrives through the environment's task-evaluation hook."""

    def on_task_evaluation(self, result: TaskEvalResult) -> None:
        """Use the evaluator's deposit counter as the prompt sequence authority."""
        if result.num_worlds != 1 or int(result.metrics["target_count"][0]) != len(self._targets):
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
        self._display_prompt = f'find the {self._targets[self._current_index]["label"]}'
