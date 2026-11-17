from __future__ import annotations

from typing import Any

import mujoco

from abc_sim.spelling_goal import SpellingGoal

from ..assets.block_appearance import (
    appearance_xml,
    apply_appearance,
    sample_appearance,
)
from ..core import (
    PerturbRange,
    RandomizationState,
    ScalePerturbRange,
    SceneRandomizer,
    _resolve_scene_xml_paths,
)


class BlocksRandomizer(SceneRandomizer):
    """Per-block perturbation for the 26-letter blocks scene.

    Blocks are arranged in a 4-row grid with 55 mm spacing; perturbations
    scatter blocks across the table (±80 mm x, ±450 mm y, ±0.25 rad yaw).
    """

    min_clearance_m = 0.01
    # Dense 26-block placements need a larger retry budget.
    max_tries = 2000
    perturbations = [
        PerturbRange(
            f"block_{letter}_jnt",
            delta_x=(-0.08, 0.08),
            delta_y=(-0.45, 0.45),
            delta_yaw=(-0.25, 0.25),
        )
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ]

    def __init__(self) -> None:
        super().__init__()
        self._appearance_active = False
        self._appearance_model: mujoco.MjModel | None = None

    def _get_size_perturbations(self) -> list[ScalePerturbRange]:
        return []

    def prepare_reset(
        self, *, request: Any | None = None, randomize: bool = True
    ) -> None:
        """Restore native visuals when the requested task has no appearance."""
        appearance_requested = (
            isinstance(request, dict) and request.get("randomize_appearance") is True
        )
        if appearance_requested and not randomize:
            raise ValueError(
                "Appearance tasks require a randomized reset or saved-state replay."
            )
        if not appearance_requested:
            self._restore_visuals(None)

    def _restore_visuals(self, appearance: dict | None) -> None:
        """Reload complete textures before creating any camera backend."""
        if appearance is None and not self._appearance_active:
            return
        if self._env_ref is None:
            raise RuntimeError(
                "Randomized block appearance requires a bound environment."
            )
        env = self._env_ref
        arm_state = env._get_reset_arm_state()
        if appearance is None:
            env.reload_from_xml(
                _resolve_scene_xml_paths(
                    self._base_scene_xml_string, self._base_scene_xml_dir
                )
            )
        else:
            if self._appearance_model is None:
                xml = appearance_xml(self._base_scene_xml_string)
                self._appearance_model = mujoco.MjModel.from_xml_string(
                    _resolve_scene_xml_paths(xml, self._base_scene_xml_dir)
                )
            apply_appearance(self._appearance_model, appearance)
            env.reload_from_model(self._appearance_model)
        env._set_qpos_from_state(arm_state)
        self._appearance_active = appearance is not None

    def apply(self, model: Any, data: Any, state: RandomizationState) -> None:
        """Replay an initial object scene with fresh physics and arms at init_q."""
        if self._env_ref is not None:
            self._env_ref.initialize_randomization()
            model, data = self._env_ref.model, self._env_ref.data
        self._apply_scene(model, data, state)
        if self._env_ref is not None:
            self._env_ref.finalize_randomization(state)

    def _apply_scene(self, model: Any, data: Any, state: RandomizationState) -> None:
        """Restore the model and poses before episode finalization."""
        self._restore_visuals(state.metadata.get("block_appearance"))
        if self._env_ref is not None:
            model, data = self._env_ref.model, self._env_ref.data
        super().apply(model, data, state)

    def randomize(
        self,
        model: Any,
        data: Any,
        seed: int | None = None,
        request: Any | None = None,
    ) -> RandomizationState:
        """Sample block poses and, when requested, a seeded spelling goal."""
        appearance = None
        goal = None
        if isinstance(request, dict):
            if "sequence_length" in request:
                goal = SpellingGoal.sample(request["sequence_length"], seed)
            if "randomize_appearance" in request:
                if type(request["randomize_appearance"]) is not bool:
                    raise ValueError("randomize_appearance must be boolean.")
                if request["randomize_appearance"]:
                    if goal is None:
                        raise ValueError(
                            "Appearance tasks require a seeded spelling goal."
                        )
                    appearance = sample_appearance(seed)
            request = {
                key: value
                for key, value in request.items()
                if key not in ("sequence_length", "randomize_appearance")
            }
        state = super().randomize(
            model,
            data,
            seed=seed,
            request=request,
        )
        if goal is not None:
            state.metadata.update(goal.to_metadata())
        if appearance is not None:
            state.metadata["block_appearance"] = appearance
        if appearance is not None or self._appearance_active:
            self._apply_scene(model, data, state)
        return state


__all__ = ["BlocksRandomizer"]
