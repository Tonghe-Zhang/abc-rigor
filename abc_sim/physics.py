"""Explicit physics profiles applied to compiled models before simulation starts."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import mujoco
import numpy as np


@dataclass(frozen=True)
class PhysicsProfile:
    """Serializable contact settings for a named simulation condition."""

    name: str
    sliding_friction_cap: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("A physics profile must have a name.")
        if (
            not math.isfinite(self.sliding_friction_cap)
            or self.sliding_friction_cap <= 0
        ):
            raise ValueError("sliding_friction_cap must be finite and positive.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "sliding_friction_cap": self.sliding_friction_cap,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PhysicsProfile:
        return cls(**value)


ROUND2_LOW_FRICTION_V1 = PhysicsProfile(
    name="round2_low_friction_v1", sliding_friction_cap=1.5
)


def resolve_physics_profile(
    profile: str | PhysicsProfile | None,
) -> PhysicsProfile | None:
    """Resolve a known profile name, preserving explicit custom definitions."""
    if profile is None or isinstance(profile, PhysicsProfile):
        return profile
    if profile == ROUND2_LOW_FRICTION_V1.name:
        return ROUND2_LOW_FRICTION_V1
    raise ValueError(f"Unknown physics profile: {profile!r}")


def apply_physics_profile(
    model: mujoco.MjModel, profile: str | PhysicsProfile | None
) -> None:
    """Cap sliding friction in place before creating data, renderers or model copies.

    MuJoCo selects geom/flex coefficients by priority or componentwise maximum.
    Explicit contact pairs and contact overrides have their own coefficients.
    Both tangent directions are capped, and torsional/rolling terms are preserved.
    Reapplying a profile is idempotent. Passing ``None`` leaves the model unchanged.
    """
    resolved = resolve_physics_profile(profile)
    if resolved is None:
        return
    cap = resolved.sliding_friction_cap
    for values in (
        model.geom_friction[:, :1],
        model.flex_friction[:, :1],
        model.pair_friction[:, :2],
        model.opt.o_friction[:2],
    ):
        np.minimum(values, cap, out=values)
