from __future__ import annotations

import numpy as np

from ..core import PerturbRange, SceneRandomizer


class DrawerRandomizer(SceneRandomizer):
    """Randomize the drawer frame position/yaw and marker positions.

    The drawer frame (drawer_body) is a fixed body moved via model.body_pos.
    Markers are free-jointed and sampled independently around the table, then
    shifted by the same (dx, dy) as the drawer so they remain nearby after reset.
    """

    min_clearance_m = 0.04
    perturbations = [
        PerturbRange("drawer_body", delta_x=(-0.08, 0.0), delta_y=(-0.10, 0.10),
                     delta_yaw=(-0.3, 0.3), fixed_body=True),
        *[
            PerturbRange(f"marker_{i}_joint", delta_x=(-0.08, 0.08), delta_y=(-0.10, 0.10))
            for i in range(1, 6)
        ],
    ]

    def randomize(self, model, data, seed=None, request=None):
        choices = dict(request or {})
        sampled = choices.pop("random_target_drawer", False)
        if not isinstance(sampled, bool):
            raise ValueError("random_target_drawer must be a boolean")
        state = super().randomize(model, data, seed=seed, request=choices or None)
        if sampled:
            state.metadata["random_target_drawer"] = True
            rng = np.random.default_rng(np.random.SeedSequence([seed or 0, 0x44524157]))
            target = str(rng.choice(("top", "middle", "bottom")))
            state.metadata["marker_target_drawer"] = target
            state.metadata["prompt"] = f"Put all the markers in the {target} drawer, then close that drawer."
        return state

    def _sample_once(
        self,
        nominals: dict[str, tuple[np.ndarray, np.ndarray]],
        rng: np.random.Generator,
    ) -> dict[str, dict[str, list[float]]]:
        states = super()._sample_once(nominals, rng)

        # Shift markers by the same (dx, dy) as the drawer so they stay near it.
        if "drawer_body" in states:
            drawer_nom_pos = nominals["drawer_body"][0]
            drawer_new_pos = np.array(states["drawer_body"]["pos"])
            dx = drawer_new_pos[0] - drawer_nom_pos[0]
            dy = drawer_new_pos[1] - drawer_nom_pos[1]
            for i in range(1, 6):
                key = f"marker_{i}_joint"
                if key in states:
                    states[key]["pos"][0] += dx
                    states[key]["pos"][1] += dy

        return states


__all__ = ["DrawerRandomizer"]
