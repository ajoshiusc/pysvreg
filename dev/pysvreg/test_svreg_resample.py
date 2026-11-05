from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import nibabel as nib
import numpy as np

from . import standalone


class SvregResampleTests(unittest.TestCase):
    def test_matlab_voxel_rounding_sends_ties_away_from_zero(self):
        values = np.asarray([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, -1.49, 1.49])
        expected = np.asarray([-3, -2, -1, 1, 2, 3, -1, 1])
        np.testing.assert_array_equal(standalone._matlab_round(values), expected)

    def test_mni_to_tal_matches_bundled_spm_matrix_above_and_below_ac(self):
        theta = 0.05
        rotation_x = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(theta), np.sin(theta)],
                [0.0, -np.sin(theta), np.cos(theta)],
            ]
        )
        for point in (
            np.asarray([12.0, -18.0, 30.0]),
            np.asarray([-8.0, 6.0, -24.0]),
        ):
            scales = np.asarray([0.99, 0.97, 0.92 if point[2] >= 0 else 0.84])
            expected = rotation_x @ (point * scales)
            np.testing.assert_allclose(standalone._mni_to_tal(point), expected, rtol=0, atol=1e-14)

    def _write_input(self, path: Path, *, four_dimensional: bool = False) -> np.ndarray:
        grid = np.indices((5, 6, 7), dtype=np.float32)
        volume = (grid[0] * 17.0 + grid[1] * 3.0 + grid[2] * 0.5).astype(np.float32)
        if four_dimensional:
            volume = np.stack([volume, volume + 1000.0], axis=3)
        image = nib.Nifti1Image(volume, np.diag([2.0, 1.5, 2.5, 1.0]))
        image.header.set_zooms((2.0, 1.5, 2.5) + ((1.0,) if four_dimensional else ()))
        nib.save(image, str(path))
        return volume

    def test_cubic_and_spline_use_matlab_samplers_and_explicit_extrapolation(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "input.nii.gz"
            self._write_input(source)

            for method, helper_name in (
                ("cubic", "_cubic_convolution_volume"),
                ("spline", "_spline_interpolate_volume"),
            ):
                with self.subTest(method=method):
                    output = root / f"{method}.nii.gz"
                    helper = getattr(standalone, helper_name)
                    with mock.patch.object(standalone, helper_name, wraps=helper) as sampler:
                        standalone.resample_volume(
                            str(source), str(output), "-res", "1", "1", "1", method, "123"
                        )
                    self.assertEqual(sampler.call_count, 1)
                    self.assertEqual(sampler.call_args.args[2], 123.0)
                    result = np.asarray(nib.load(str(output)).dataobj)
                    self.assertEqual(result[0, 0, 0], 123.0)
                    x = (2.0 + 0.5) * 1.0 / 2.0 - 0.5
                    y = (2.0 + 0.5) * 1.0 / 1.5 - 0.5
                    z = (2.0 + 0.5) * 1.0 / 2.5 - 0.5
                    expected = 17.0 * x + 3.0 * y + 0.5 * z
                    self.assertAlmostEqual(float(result[2, 2, 2]), expected, places=4)

    def test_identity_resampling_preserves_volume_for_each_interpolator(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "input.nii.gz"
            data = self._write_input(source)
            for method in ("nearest", "linear", "cubic", "spline"):
                with self.subTest(method=method):
                    output = root / f"identity-{method}.nii.gz"
                    standalone.resample_volume(
                        str(source), str(output), "-size", "5", "6", "7", method
                    )
                    np.testing.assert_allclose(
                        np.asarray(nib.load(str(output)).dataobj), data, atol=1e-5, rtol=0.0
                    )

    def test_default_extrapolation_uses_corner_median(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "input.nii.gz"
            data = self._write_input(source)
            output = root / "output.nii.gz"
            standalone.resample_volume(
                str(source), str(output), "-res", "1", "1", "1", "cubic", ""
            )
            corners = np.asarray(
                [data[x, y, z] for x in (0, -1) for y in (0, -1) for z in (0, -1)]
            )
            self.assertEqual(np.asarray(nib.load(str(output)).dataobj)[0, 0, 0], np.median(corners))

    def test_four_dimensional_resample_runs_each_frame_through_matched_sampler(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "input.nii.gz"
            self._write_input(source, four_dimensional=True)
            output = root / "output.nii.gz"
            helper = standalone._cubic_convolution_volume
            with mock.patch.object(
                standalone, "_cubic_convolution_volume", wraps=helper
            ) as sampler:
                standalone.resample_volume(
                    str(source), str(output), "-res", "1", "1", "1", "cubic", "123"
                )
            self.assertEqual(sampler.call_count, 2)
            result = np.asarray(nib.load(str(output)).dataobj)
            self.assertEqual(result.shape[3], 2)
            self.assertEqual(result[0, 0, 0, 0], 123.0)
            self.assertEqual(result[0, 0, 0, 1], 123.0)


if __name__ == "__main__":
    unittest.main()
