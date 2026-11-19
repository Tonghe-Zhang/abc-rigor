"""Versioned uniform surface-mass model for imported mugs."""

from __future__ import annotations

import numpy as np

MUG_INERTIA_REVISION = "whole_visual_shell_50g_v1"
MUG_MASS_KG = 0.05


def surface_inertia(
    triangles: np.ndarray, mass: float = MUG_MASS_KG
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a uniform thin shell's center and inertia about that center.

    Each visual triangle contributes in proportion to its area. Overlapping
    surfaces contribute independently. This physical model does not assume a
    watertight volume or a material density.

    Args:
        triangles: Triangle vertices in a common frame, with shape (N, 3, 3).
        mass: Total mass distributed over those triangles, in kilograms.

    Returns:
        Center of mass (3,) and symmetric inertia tensor (3, 3), in the input
        frame, with inertia measured about the returned center.
    """
    triangles = np.asarray(triangles, dtype=np.float64)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError("Surface triangles must have shape (N, 3, 3)")
    if not np.isfinite(triangles).all() or not np.isfinite(mass) or mass <= 0:
        raise ValueError("Surface vertices and positive mass must be finite")
    areas = (
        np.linalg.norm(
            np.cross(
                triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
            ),
            axis=1,
        )
        / 2
    )
    total_area = float(areas.sum())
    if not np.isfinite(total_area) or total_area <= 0:
        raise ValueError("Visual surfaces must have positive finite total area")
    sums = triangles.sum(axis=1)
    center = np.einsum("n,ni->i", areas, sums / 3) / total_area
    # Center before accumulating second moments to avoid subtracting large offsets.
    centered = triangles - center
    sums = centered.sum(axis=1)
    moments = (
        np.einsum("ni,nj->nij", sums, sums)
        + np.einsum("nki,nkj->nij", centered, centered)
    ) / 12
    covariance = np.einsum("n,nij->ij", areas, moments) / total_area
    inertia = mass * (np.trace(covariance) * np.eye(3) - covariance)
    if not np.isfinite(inertia).all() or np.any(np.linalg.eigvalsh(inertia) <= 0):
        raise ValueError("Visual surfaces must yield positive finite principal inertia")
    return center, inertia
