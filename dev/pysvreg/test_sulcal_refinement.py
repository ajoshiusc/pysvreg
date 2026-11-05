from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
from scipy.sparse import csr_matrix

from .sulcal_refinement import _shortest_path, _zero_level_set, refine_curves_on_surface


class SulcalRefinementTests(unittest.TestCase):
    def test_shortest_path_uses_edges_and_returns_source_to_target(self) -> None:
        graph = csr_matrix(
            (
                np.ones(8),
                (
                    [0, 1, 1, 2, 1, 0, 2, 1],
                    [1, 0, 2, 1, 3, 3, 3, 1],
                ),
            ),
            shape=(4, 4),
        )
        np.testing.assert_array_equal(_shortest_path(graph, 0, 2), [0, 1, 2])

    def test_zero_level_set_interpolates_crossing_edges(self) -> None:
        vertices = np.asarray([[0, 0, 0], [2, 0, 0], [0, 2, 0]], dtype=np.float64)
        faces = np.asarray([[0, 1, 2]], dtype=np.int32)
        points, crossing_faces = _zero_level_set(vertices, faces, np.asarray([-1, 1, 1]))
        self.assertEqual(len(points), 2)
        self.assertEqual(len(crossing_faces), 1)
        np.testing.assert_allclose(points, [[1, 0, 0], [0, 1, 0]])

    def test_geodesic_flow_returns_finite_curve_on_small_mesh(self) -> None:
        n = 12
        x, y = np.meshgrid(np.linspace(-5, 5, n), np.linspace(-5, 5, n))
        z = 0.3 * np.sin(x / 3) * np.cos(y / 4)
        vertices = np.c_[x.ravel(), y.ravel(), z.ravel()].astype(np.float32)
        faces = []
        for row in range(n - 1):
            for col in range(n - 1):
                a = row * n + col
                b, c, d = a + 1, a + n, a + n + 1
                faces.extend(((a, b, d), (a, d, c)))
        surface = SimpleNamespace(vertices=vertices, faces=np.asarray(faces, dtype=np.int32))
        curve = vertices[[row * n + n // 2 for row in range(2, 10)]]
        result = refine_curves_on_surface(surface, [curve])[0]
        self.assertGreaterEqual(len(result), 4)
        self.assertTrue(np.isfinite(result).all())


if __name__ == "__main__":
    unittest.main()
