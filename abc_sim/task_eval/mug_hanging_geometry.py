"""Measure handle apertures and rack shafts from compiled collision meshes."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from abc_sim.task_eval.contacts import subtree_geom_ids


def _convex_hull(points):
    points = sorted(set(map(tuple, points)))

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    lower, upper = [], []
    for chain, sequence in ((lower, points), (upper, reversed(points))):
        for point in sequence:
            while len(chain) >= 2 and cross(chain[-2], chain[-1], point) <= 0:
                chain.pop()
            chain.append(point)
    hull = np.array(lower[:-1] + upper[:-1])
    if len(hull) < 3:
        raise ValueError("A mug collision mesh has a degenerate side projection.")
    return hull


def _collision_meshes(model, data, body):
    rotation = data.xmat[body].reshape(3, 3)
    result = []
    for geom in sorted(subtree_geom_ids(model, body)):
        if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
            continue
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            raise ValueError("Mug hanging geometry requires collision meshes.")
        mesh = int(model.geom_dataid[geom])
        start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        vertices = model.mesh_vert[start : start + count].astype(float)
        vertices = (vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom] - data.xpos[body]) @ rotation
        start, count = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
        faces = model.mesh_face[start : start + count].copy()
        result.append((geom, vertices, faces))
    if not result:
        raise ValueError("Mug hanging body has no collision meshes.")
    return result


@dataclass(frozen=True)
class RackPeg:
    """One finite convex rack branch, expressed in the rack body's frame."""

    geom_id: int
    endpoints: np.ndarray
    radius: float
    vertices: np.ndarray
    edges: np.ndarray


def measure_rack_pegs(model, data, tree_body_id):
    """Resolve the six branch colliders and fit their longitudinal centerlines."""
    names = {f"mug_tree_col_{index}" for index in (1, 2, 5, 6, 8, 9)}
    pegs = []
    for geom, vertices, faces in _collision_meshes(model, data, tree_body_id):
        name = model.mesh(int(model.geom_dataid[geom])).name
        if name not in names:
            continue
        triangles = vertices[faces]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        areas = np.linalg.norm(normals, axis=1)
        nondegenerate = areas > 1e-12
        normals, areas = normals[nondegenerate] / areas[nondegenerate, None], areas[nondegenerate]
        axis = np.linalg.eigh(np.cov(vertices.T))[1][:, -1]
        # Lateral face normals fit the shaft axis without weighting densely
        # tessellated caps more heavily than its long faces.
        for _ in range(5):
            lateral = np.abs(normals @ axis) < 0.3
            if np.count_nonzero(lateral) < 3:
                raise ValueError(f"Rack branch {name} has no measurable shaft.")
            axis = np.linalg.eigh((normals[lateral].T * areas[lateral]) @ normals[lateral])[1][:, 0]
        basis = np.linalg.svd(axis[None, :], full_matrices=True)[2][1:].T
        transverse = vertices @ basis
        midpoint = (transverse.min(axis=0) + transverse.max(axis=0)) / 2
        center = basis @ midpoint
        longitudinal = vertices @ axis
        endpoints = center + np.array([longitudinal.min(), longitudinal.max()])[:, None] * axis
        radius = float(np.linalg.norm(transverse - midpoint, axis=1).max())
        edges = np.unique(
            np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0
        )
        pegs.append(RackPeg(geom, endpoints, radius, vertices, edges))
    if len(pegs) != 6:
        raise ValueError("The mug rack must contain all six measured branch meshes.")
    return pegs


@dataclass(frozen=True)
class HandleAperture:
    """A closed YZ aperture and its measured, possibly asymmetric X thickness.

    All mug-tree assets orient their handle normal along local X. The aperture
    is the enclosed empty component in the union of projected collision hulls.
    The material immediately bordering that component defines its two faces.
    """

    origin: np.ndarray
    resolution: float
    opening: np.ndarray
    contact_opening: np.ndarray
    x_faces: np.ndarray

    def _contains(self, points, *, contact_tolerance=False):
        pixels = np.rint((points - self.origin) / self.resolution).astype(int)
        mask = self.contact_opening if contact_tolerance else self.opening
        if np.any(pixels < 0) or np.any(pixels >= np.array(mask.shape[::-1])):
            return False
        return bool(mask[pixels[:, 1], pixels[:, 0]].all())

    def threaded_by(self, endpoints, radius, *, vertices, edges):
        """Require a shaft through both handle faces with its section in the hole.

        Actual convex mesh sections supply the branch shape. A 0.75 mm boundary
        tolerance allows the solved contact indentation and raster resolution.
        Each endpoint must extend at least 1 mm beyond the measured handle face.
        """
        order = np.argsort(endpoints[:, 0])
        start, end = endpoints[order]
        front, back = self.x_faces
        if start[0] > front - 0.001 or end[0] < back + 0.001:
            return False
        delta = end - start
        planes = np.linspace(front, back, 3)
        centers = start + ((planes - start[0]) / delta[0])[:, None] * delta
        if not self._contains(centers[:, 1:]):
            return False
        first, second = vertices[edges[:, 0]], vertices[edges[:, 1]]
        dx = second[:, 0] - first[:, 0]
        # The shaft crosses the interior of the handle at its middle plane.
        # Its endpoints clear the whole thickness above; projecting a shaft
        # section at an outer face would reject ordinary tilted hangs where
        # that section is already outside the handle material.
        for plane in [float(self.x_faces.mean())]:
            crossing = (
                (np.minimum(first[:, 0], second[:, 0]) <= plane)
                & (np.maximum(first[:, 0], second[:, 0]) >= plane)
                & (np.abs(dx) > 1e-12)
            )
            if np.count_nonzero(crossing) < 3:
                return False
            fraction = (plane - first[crossing, 0]) / dx[crossing]
            section = first[crossing] + fraction[:, None] * (second[crossing] - first[crossing])
            if not self._contains(section[:, 1:], contact_tolerance=True):
                return False
            # Inspect polygon edges too: a convex section must not bridge an
            # indentation in a non-convex handle opening.
            hull = _convex_hull(section[:, 1:])
            for a, b in zip(hull, np.roll(hull, -1, axis=0), strict=True):
                steps = int(np.ceil(np.linalg.norm(b - a) / self.resolution)) + 1
                if not self._contains(np.linspace(a, b, steps), contact_tolerance=True):
                    return False
        return True


def measure_handle_aperture(model, data, mug_body_id):
    """Measure the enclosed handle opening from the realized scaled mug meshes."""
    meshes = _collision_meshes(model, data, mug_body_id)
    vertices = np.concatenate([vertices for _, vertices, _ in meshes])
    resolution = 0.00025
    origin = vertices[:, 1:].min(axis=0) - 3 * resolution
    shape = np.ceil((vertices[:, 1:].max(axis=0) - origin) / resolution).astype(int) + 4
    occupied = np.zeros(tuple(shape[::-1]), dtype=bool)
    for _, points, _ in meshes:
        hull = _convex_hull(points[:, 1:])
        lo = np.floor((hull.min(axis=0) - origin) / resolution).astype(int)
        hi = np.ceil((hull.max(axis=0) - origin) / resolution).astype(int)
        ii, jj = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
        samples = np.stack([ii, jj], axis=-1) * resolution + origin
        inside = np.ones(ii.shape, dtype=bool)
        for a, b in zip(hull, np.roll(hull, -1, axis=0), strict=True):
            edge = b - a
            inside &= edge[0] * (samples[..., 1] - a[1]) - edge[1] * (samples[..., 0] - a[0]) >= -1e-12
        occupied[jj, ii] |= inside
    # Close sub-pixel seams between convex collision parts before locating the
    # enclosed hole. Depths below are still measured on actual material.
    silhouette = (
        Image.fromarray(occupied.astype(np.uint8)).filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3))
    )
    image = silhouette.copy()
    ImageDraw.floodfill(image, (0, 0), 2)
    opening = np.asarray(image) == 0
    cells = np.argwhere(opening)
    if len(cells) == 0:
        raise ValueError(f"Mug {model.body(mug_body_id).name} has no closed handle aperture.")
    # Convex decomposition can leave sub-millimeter seams at hull junctions.
    # Keep the handle-sized connected component; a second substantial cavity
    # is an unsupported asset rather than another valid threading target.
    connected = Image.fromarray(opening.astype(np.uint8)).copy()
    components = []
    while len(cells):
        ImageDraw.floodfill(connected, tuple(int(v) for v in cells[0, ::-1]), 2)
        component = np.asarray(connected) == 2
        components.append(component)
        connected = Image.fromarray((np.asarray(connected) == 1).astype(np.uint8)).copy()
        cells = np.argwhere(np.asarray(connected) == 1)
    components.sort(key=lambda component: int(component.sum()), reverse=True)
    if len(components) > 1 and components[1].sum() * resolution**2 > 1e-6:
        raise ValueError("Mug hanging expects exactly one closed handle aperture.")
    opening = components[0]
    dilated = np.asarray(Image.fromarray(opening).filter(ImageFilter.MaxFilter(3)))
    boundary = occupied & dilated
    samples = np.argwhere(boundary)[:, ::-1] * resolution + origin
    front, back = np.full(len(samples), np.inf), np.full(len(samples), -np.inf)
    for _, points, faces in meshes:
        for triangle in points[faces]:
            a, b, c = triangle[:, 1:]
            ab, ac = b - a, c - a
            determinant = ab[0] * ac[1] - ab[1] * ac[0]
            if abs(determinant) < 1e-14:
                continue
            relative = samples - a
            u = (relative[:, 0] * ac[1] - relative[:, 1] * ac[0]) / determinant
            v = (ab[0] * relative[:, 1] - ab[1] * relative[:, 0]) / determinant
            inside = (u >= -1e-7) & (v >= -1e-7) & (u + v <= 1 + 1e-7)
            depth = (
                triangle[0, 0]
                + u[inside] * (triangle[1, 0] - triangle[0, 0])
                + v[inside] * (triangle[2, 0] - triangle[0, 0])
            )
            front[inside] = np.minimum(front[inside], depth)
            back[inside] = np.maximum(back[inside], depth)
    if not np.isfinite(front).all() or not np.isfinite(back).all():
        raise ValueError("Mug collision faces do not cover the measured aperture boundary.")
    contact_opening = np.asarray(Image.fromarray(opening).filter(ImageFilter.MaxFilter(7)))
    return HandleAperture(origin, resolution, opening, contact_opening, np.array([front.min(), back.max()]))
