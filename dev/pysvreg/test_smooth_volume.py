from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import nibabel as nib
import numpy as np

from . import standalone


def _finite_replicate_gaussian(data: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    result = np.asarray(data, dtype=np.float64)
    for axis, axis_sigma in enumerate(sigma):
        radius = int(np.ceil(2.0 * axis_sigma))
        positions = np.arange(-radius, radius + 1, dtype=np.float64)
        weights = np.exp(-0.5 * (positions / axis_sigma) ** 2)
        weights /= np.sum(weights)
        padding = [(0, 0)] * result.ndim
        padding[axis] = (radius, radius)
        padded = np.pad(result, padding, mode="edge")
        result = np.apply_along_axis(
            lambda line: np.convolve(line, weights, mode="valid"), axis, padded
        )
    return result


class SmoothVolumeTests(unittest.TestCase):
    def test_matches_matlab_filter_size_and_replicated_padding(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "input.nii.gz"
            output = root / "smoothed.nii.gz"
            data = np.zeros((7, 8, 6), dtype=np.float32)
            data[0, 0, 0] = 1.0
            data[3, 4, 2] = -0.25
            image = nib.Nifti1Image(data, np.diag([2.0, 1.5, 2.5, 1.0]))
            image.header.set_zooms((2.0, 1.5, 2.5))
            nib.save(image, source)

            std_mm = np.asarray([1.2, 1.0, 1.7])
            standalone.smooth_volume(
                str(source), *(str(value) for value in std_mm), str(output)
            )

            sigma = (std_mm + 1e-4) / np.asarray([2.0, 1.5, 2.5])
            expected = _finite_replicate_gaussian(data, sigma).astype(np.float32)
            actual = np.asarray(nib.load(output).dataobj)
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)

    def test_four_dimensional_input_is_filtered_per_frame(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "input.nii.gz"
            output = root / "smoothed.nii.gz"
            first = np.zeros((5, 5, 5), dtype=np.float32)
            first[0, 0, 0] = 1.0
            data = np.stack((first, first + 10.0), axis=3)
            image = nib.Nifti1Image(data, np.eye(4))
            image.header.set_zooms((1.0, 1.0, 1.0, 1.0))
            nib.save(image, source)

            standalone.smooth_volume(str(source), "0.8", "0.8", "0.8", str(output))

            actual = np.asarray(nib.load(output).dataobj)
            self.assertEqual(actual.shape, data.shape)
            np.testing.assert_allclose(actual[..., 1] - actual[..., 0], 10.0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
