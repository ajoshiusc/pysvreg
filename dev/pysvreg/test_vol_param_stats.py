from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dev.pysvreg.statistics import generate_vol_param_stats_xls


def _weighted_reference(weights: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    weight_sum = float(np.sum(weights))
    count = int(np.count_nonzero(weights))
    if weight_sum == 0 or count == 0:
        return 0.0, 0.0
    mean = float(np.sum(weights * values) / weight_sum)
    if count < 2:
        return mean, 0.0
    denominator = ((count - 1) / count) * weight_sum
    deviation = float(np.sqrt(np.sum(weights * (values - mean) ** 2) / denominator))
    return mean, deviation


class TestVolParamStatistics(unittest.TestCase):
    def test_tissue_rows_and_roi_rows_follow_matlab_formula(self) -> None:
        shape = (6, 7, 8)
        grid = np.indices(shape)
        labels = np.where(grid[0] % 2 == 0, 120, 130).astype(np.int16)
        labels[grid[1] == 0] = 0
        labels[0, 0, 0] = 1
        labels[1, 0, 0] = 2
        labels[2, 0, 0] = 3
        fraction = np.choose(
            (grid[0] + grid[1] + grid[2]) % 5,
            [1.0, 1.5, 2.0, 2.5, 3.0],
        ).astype(np.float32)
        values = (0.25 * grid[0] + 0.5 * grid[1] - 0.75 * grid[2] + ((grid[0] * grid[1]) % 3)).astype(np.float32)
        gm = (fraction > 1) * (1 - np.abs(fraction - 2))
        csf = (fraction < 2) * (1 - np.abs(fraction - 1))
        wm = (fraction > 2) * (1 - np.abs(fraction - 3))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prefix = root / "subject"
            affine = np.diag([1.2, 0.8, 2.0, 1.0])
            nib.save(nib.Nifti1Image(labels, affine), str(prefix) + ".svreg.label.nii.gz")
            nib.save(nib.Nifti1Image(fraction, affine), str(prefix) + ".pvc.frac.nii.gz")
            parameter_file = root / "param.nii.gz"
            nib.save(nib.Nifti1Image(values, affine), parameter_file)

            self.assertEqual(generate_vol_param_stats_xls([str(prefix), str(parameter_file)]), 0)

            rows = {}
            for line in Path(str(prefix) + ".param.roiwise.stats.txt").read_text().splitlines()[1:]:
                fields = line.split("\t")
                rows[int(fields[0])] = np.asarray([float(value) for value in fields[1:] if value])

        expected_all = (float(np.mean(values)), float(np.std(values, ddof=1)))
        for label, weights, column_start in ((1, csf, 4), (2, gm, 0), (3, wm, 2)):
            expected = np.zeros(8, dtype=float)
            expected[column_start : column_start + 2] = _weighted_reference(weights.ravel(), values.ravel())
            expected[6:8] = expected_all
            np.testing.assert_allclose(rows[label], expected, atol=5e-6, rtol=0.0)

        for label in (120, 130):
            roi = labels == label
            expected = np.concatenate(
                [
                    _weighted_reference(gm[roi], values[roi]),
                    _weighted_reference(wm[roi], values[roi]),
                    _weighted_reference(csf[roi], values[roi]),
                    [float(np.mean(values[roi])), float(np.std(values[roi], ddof=1))],
                ]
            )
            np.testing.assert_allclose(rows[label], expected, atol=5e-6, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
