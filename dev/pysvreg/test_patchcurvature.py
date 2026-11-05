from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
from scipy.spatial import ConvexHull

from . import native


class PatchCurvatureTests(unittest.TestCase):
    def test_outward_sphere_has_positive_local_mean_curvature(self):
        rng = np.random.default_rng(2604)
        points = rng.normal(size=(180, 3))
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        hull = ConvexHull(points)
        faces = hull.simplices.copy()
        tri = points[faces]
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        inward = np.einsum("ij,ij->i", normals, tri.mean(axis=1)) < 0
        faces[inward, 1], faces[inward, 2] = faces[inward, 2].copy(), faces[inward, 1].copy()
        surface = SimpleNamespace(vertices=points.astype(np.float32), faces=faces.astype(np.int32))

        curvature = native.patchcurvature_mean(surface)
        self.assertTrue(np.all(np.isfinite(curvature)))
        self.assertGreater(float(np.mean(curvature)), 0.0)
        self.assertGreater(float(np.mean(curvature > 0.0)), 0.9)
        self.assertLess(abs(float(np.mean(curvature)) - 1.0), 0.3)

        larger = SimpleNamespace(vertices=2.0 * surface.vertices, faces=surface.faces)
        curvature_larger = native.patchcurvature_mean(larger)
        self.assertLess(abs(float(np.mean(curvature_larger)) * 2 - float(np.mean(curvature))), 0.1)


if __name__ == "__main__":
    unittest.main()
