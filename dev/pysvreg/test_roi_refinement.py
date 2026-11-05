from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from . import roi_refinement as roi


class ZeroIterationParityTests(unittest.TestCase):
    def test_zero_iterations_do_not_relabel_pair(self):
        # Two large, adjacent regions meet the pair face-count threshold.
        n = 12
        vertices = np.array([(x, y, 0.0) for y in range(n) for x in range(n)], dtype=np.float32)
        faces = []
        for y in range(n - 1):
            for x in range(n - 1):
                a = y * n + x
                faces.extend(((a, a + 1, a + n), (a + 1, a + n + 1, a + n)))
        labels = np.array([100 if x < n // 2 else 200 for y in range(n) for x in range(n)], dtype=np.uint16)
        surface = SimpleNamespace(vertices=vertices, faces=np.asarray(faces, dtype=np.int32), labels=labels.copy())
        before = surface.labels.copy()
        with patch.object(roi, "_flow_pair", return_value=(np.ones(len(vertices)), {"nit": 0, "accepted": True})):
            roi.refine_surface(surface, min_common_faces=1)
        np.testing.assert_array_equal(surface.labels, before)

    def test_patch_cleanup_does_not_join_islands_touching_at_one_vertex(self):
        # The two triangle fans share vertex zero but no edge. MATLAB's
        # faces2faces_connectivity keeps them as separate face components.
        vertices = np.array([
            (0, 0, 0), (1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0),
            (1, 1, 0), (2, 1, 0), (1, 2, 0), (0, 2, 0), (-1, 2, 0),
        ], dtype=np.float32)
        faces = np.array([
            (0, 1, 2), (0, 3, 4),
            (5, 6, 7), (5, 7, 8), (5, 8, 9),
        ], dtype=np.int32)
        labels = np.array([100, 200, 200, 100, 200, 100, 200, 100, 200, 100], dtype=np.uint16)
        surface = SimpleNamespace(vertices=vertices, faces=faces, labels=labels)
        patch_surface, _ = roi._compact_pair(surface, (100, 200))
        self.assertEqual(len(patch_surface.faces), 3)


if __name__ == "__main__":
    unittest.main()
