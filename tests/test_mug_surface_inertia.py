"""Analytic checks for the mug surface-mass model; no simulator imports."""

import unittest

import numpy as np

from abc_sim.mug_inertia import MUG_MASS_KG, surface_inertia


def square(side=2.0, center=(0.0, 0.0, 0.0)):
    half = side / 2
    vertices = (
        np.array(
            [[-half, -half, 0], [half, -half, 0], [half, half, 0], [-half, half, 0]]
        )
        + center
    )
    return vertices[[[0, 1, 2], [0, 2, 3]]]


class SurfaceInertiaTest(unittest.TestCase):
    def test_uniform_square_lamina(self):
        center, tensor = surface_inertia(square(), mass=3)
        np.testing.assert_allclose(center, 0, atol=1e-15)
        np.testing.assert_allclose(tensor, np.diag([1, 1, 2]), atol=1e-15)

    def test_triangle_known_moments(self):
        triangles = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]])
        center, tensor = surface_inertia(triangles, mass=1)
        np.testing.assert_allclose(center, [1 / 3, 1 / 3, 0])
        np.testing.assert_allclose(
            tensor, [[1 / 18, 1 / 36, 0], [1 / 36, 1 / 18, 0], [0, 0, 1 / 9]]
        )

    def test_small_first_fragment_receives_area_weight(self):
        small = square(side=1, center=(-4, 0, 0))
        large = square(side=2, center=(1, 0, 0))
        surfaces = np.concatenate([small, large])
        center, tensor = surface_inertia(surfaces)
        np.testing.assert_allclose(center, 0, atol=1e-15)
        # Fractions are 1/5 and 4/5. Parallel-axis terms use the same weights.
        expected = MUG_MASS_KG * np.diag(
            [
                0.2 / 12 + 0.8 / 3,
                0.2 * (1 / 12 + 16) + 0.8 * (1 / 3 + 1),
                0.2 * (1 / 6 + 16) + 0.8 * (2 / 3 + 1),
            ]
        )
        np.testing.assert_allclose(tensor, expected)
        reverse = surface_inertia(np.concatenate([large, small]))
        np.testing.assert_allclose(reverse[0], center, atol=1e-15)
        np.testing.assert_allclose(reverse[1], tensor)

    def test_rotation_translation_and_fixed_mass_scaling(self):
        triangles = np.array([[[0, 0, 0], [2, 0, 0], [0, 1, 1]]], dtype=float)
        center, tensor = surface_inertia(triangles)
        angle = 0.7
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        offset = np.array([3.5, -4.25, 2])
        for factor in (0.9, 1.0, 1.1):
            with self.subTest(factor=factor):
                transformed = factor * triangles @ rotation.T + offset
                actual_center, actual_tensor = surface_inertia(transformed)
                np.testing.assert_allclose(
                    actual_center, factor * center @ rotation.T + offset
                )
                np.testing.assert_allclose(
                    actual_tensor,
                    factor**2 * rotation @ tensor @ rotation.T,
                    atol=1e-15,
                )

    def test_large_coordinate_offset_does_not_destroy_inertia(self):
        center, tensor = surface_inertia(square(center=(1e8, -1e8, 1e8)), mass=3)
        np.testing.assert_allclose(center, [1e8, -1e8, 1e8])
        np.testing.assert_allclose(tensor, np.diag([1, 1, 2]), atol=1e-14)

    def test_triangle_winding_and_subdivision_preserve_mass(self):
        triangle = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=float)
        a, b, c = triangle[0]
        midpoint = (a + b) / 2
        split = np.array([[a, midpoint, c], [midpoint, b, c]])
        expected = surface_inertia(triangle)
        for equivalent in (triangle[:, ::-1], split):
            actual = surface_inertia(equivalent)
            np.testing.assert_allclose(actual[0], expected[0])
            np.testing.assert_allclose(actual[1], expected[1])

    def test_invalid_surface_and_mass_fail(self):
        for triangles in (
            np.empty((0, 3, 3)),
            np.zeros((2, 3, 3)),
            np.full((1, 3, 3), np.nan),
        ):
            with self.subTest(triangles=triangles):
                with self.assertRaises(ValueError):
                    surface_inertia(triangles)
        for mass in (0, -1, float("inf")):
            with self.subTest(mass=mass):
                with self.assertRaises(ValueError):
                    surface_inertia(square(), mass)


if __name__ == "__main__":
    unittest.main()
