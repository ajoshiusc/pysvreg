from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import nibabel as nib
import numpy as np

from .multiparc import _volume_transfer


class MultiParcVolumeTransferTests(unittest.TestCase):
    def test_volume_labels_use_matlab_interp3_axis_order_and_zero_extrapolation(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            subject = root / "subject"
            tmp_dir = root / "subject.svreg.tmp"
            tmp_dir.mkdir()

            atlas_indices = np.indices((5, 6, 4))
            atlas_labels = (
                100 * atlas_indices[0] + 10 * atlas_indices[1] + atlas_indices[2]
            ).astype(np.int16)
            atlas_prefix = root / "atlas"
            nib.save(nib.Nifti1Image(atlas_labels, np.eye(4)), f"{atlas_prefix}.label.nii.gz")

            subject_shape = (3, 4, 2)
            row, column, slice_index = np.indices(subject_shape)
            coordinate_map = np.empty((*subject_shape, 3), dtype=np.float32)
            coordinate_map[..., 0] = column + 1.0  # MATLAB interp3 X: column
            coordinate_map[..., 1] = row + 1.0     # MATLAB interp3 Y: row
            coordinate_map[..., 2] = slice_index + 1.0
            coordinate_map[0, 0, 0, 0] = 7.0      # outside atlas columns -> cval 0
            nib.save(nib.Nifti1Image(coordinate_map, np.eye(4)), f"{subject}.svreg.map.nii.gz")
            subject_header = nib.Nifti1Header()
            subject_header.set_data_dtype(np.int16)
            nib.save(
                nib.Nifti1Image(np.zeros(subject_shape, dtype=np.int16), np.eye(4), subject_header),
                f"{subject}.svreg.label.nii.gz",
            )

            self.assertTrue(_volume_transfer(subject, atlas_prefix, "test"))

            output = nib.load(f"{tmp_dir / 'subject'}.svreg.testlabel.nii.gz")
            actual = np.asanyarray(output.dataobj)
            expected = atlas_labels[row, column, slice_index].copy()
            expected[0, 0, 0] = 0
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(actual.dtype, np.dtype(np.int16))


if __name__ == "__main__":
    unittest.main()
