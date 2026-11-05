from __future__ import annotations

import unittest

import numpy as np
from scipy.sparse import csr_matrix

from .svreg_elastic import ElasticIntensityRefiner, second_difference_laplacian


class ElasticRefinementTests(unittest.TestCase):
    def test_sampled_target_uses_matlab_rounding_at_half_voxels(self) -> None:
        source = np.zeros((4, 4, 4), dtype=np.float64)
        target = np.arange(source.size, dtype=np.float64).reshape(source.shape)
        coordinates = np.asarray([[0.5, 1.5, 1.49]], dtype=np.float64)
        refiner = ElasticIntensityRefiner(
            source,
            target,
            coordinates,
            csr_matrix((1, 1), dtype=np.float64),
            np.asarray([0]),
            alpha=0.0,
        )
        np.testing.assert_array_equal(refiner.target_samples, [target[1, 2, 1]])

    def test_second_difference_laplacian_annihilates_constants_away_from_mask_edge(self) -> None:
        shape = (7, 7, 7)
        mask = np.zeros(shape, dtype=bool)
        mask[1:6, 1:6, 1:6] = True
        flat = np.flatnonzero(mask.ravel(order="F"))
        operator, boundary = second_difference_laplacian(flat, shape)
        constant = np.ones(flat.size)

        result = operator @ constant
        self.assertGreater(boundary.size, 0)
        self.assertTrue(np.isfinite(result).all())
        self.assertLess(float(np.max(np.abs(result[np.setdiff1d(np.arange(flat.size), boundary)]))), 1e-12)

    def test_intensity_refinement_reduces_a_small_translation_mismatch(self) -> None:
        shape = (11, 11, 11)
        grid = np.indices(shape, dtype=np.float64).transpose(1, 2, 3, 0)
        center = np.asarray((5.0, 5.0, 5.0))
        source = np.exp(-np.sum((grid - center) ** 2, axis=-1) / 5.0)
        target = np.exp(-np.sum((grid - (center + (0.4, 0.0, 0.0))) ** 2, axis=-1) / 5.0)
        mask = np.zeros(shape, dtype=bool)
        mask[2:9, 2:9, 2:9] = True
        flat = np.flatnonzero(mask.ravel(order="F"))
        coordinates = np.column_stack(np.unravel_index(flat, shape, order="F")).astype(np.float64)
        operator, boundary = second_difference_laplacian(flat, shape)
        refiner = ElasticIntensityRefiner(source, target, coordinates, operator, boundary, alpha=0.05)

        refined = refiner.optimize(iterations=4)

        self.assertTrue(np.isfinite(refined).all())
        self.assertLessEqual(refiner.cost(refined), refiner.cost(coordinates) + 1e-10)
        np.testing.assert_array_equal(refined[boundary], coordinates[boundary])


if __name__ == "__main__":
    unittest.main()
