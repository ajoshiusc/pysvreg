from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
from scipy import ndimage as ndi

from .svreg_refinements import _matlab_label_component_cleanup, _matlab_pial_surface_mask


class SVRegRefinementTests(unittest.TestCase):
    def test_label_cleanup_fills_holes_and_removes_small_components(self) -> None:
        image = np.zeros((17, 17, 17), dtype=np.int16)
        image[2:15, 2:15, 2:15] = 10
        image[7:10, 7:10, 7:10] = 0
        image[0, 0, 0] = 10

        output = _matlab_label_component_cleanup(image)

        self.assertTrue(np.all(output[7:10, 7:10, 7:10] == 10))
        self.assertEqual(int(output[0, 0, 0]), 0)
        self.assertEqual(int(np.count_nonzero(output == 10)), 13**3)

    def test_pial_mask_matches_four_centroid_refinement_and_morphology(self) -> None:
        vertices = np.asarray(
            ((5.0, 5.0, 5.0), (20.0, 5.0, 5.0), (5.0, 20.0, 5.0), (5.0, 5.0, 20.0)),
            dtype=np.float64,
        )
        faces = np.asarray(((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)), dtype=np.int32)
        surface = SimpleNamespace(vertices=vertices, faces=faces)
        reference = SimpleNamespace(shape=(32, 32, 32), zooms=(1.0, 1.0, 1.0))

        refined_vertices = vertices.copy()
        refined_faces = faces.copy()
        for _ in range(4):
            centroids = refined_vertices[refined_faces].mean(axis=1)
            old_count = refined_vertices.shape[0]
            centroid_index = np.arange(centroids.shape[0], dtype=np.int32) + old_count
            refined_faces = np.concatenate(
                (
                    np.column_stack((refined_faces[:, 1], centroid_index, refined_faces[:, 0])),
                    np.column_stack((refined_faces[:, 2], centroid_index, refined_faces[:, 1])),
                    np.column_stack((refined_faces[:, 0], centroid_index, refined_faces[:, 2])),
                ),
                axis=0,
            )
            refined_vertices = np.vstack((refined_vertices, centroids))

        expected = np.zeros(reference.shape, dtype=bool)
        voxel = np.floor(refined_vertices + 0.5).astype(np.int64)
        expected[voxel[:, 0], voxel[:, 1], voxel[:, 2]] = True
        expected = ndi.binary_dilation(expected, structure=np.ones((4, 4, 4), dtype=bool))
        expected = ndi.binary_erosion(expected, structure=np.ones((3, 3, 3), dtype=bool), border_value=0)
        expected = ndi.binary_fill_holes(expected, structure=ndi.generate_binary_structure(3, 1))

        actual = _matlab_pial_surface_mask(surface, reference)

        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
