from __future__ import annotations

import gzip
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dev.pysvreg.standalone import _read_brainsuite_eig, apply_map


def _write_eig(path: Path, fields: np.ndarray) -> None:
    dims = fields.shape[:3]
    header = bytearray(348)
    struct.pack_into("<i", header, 0, 348)
    struct.pack_into("<8h", header, 40, 4, *dims, 12, 1, 1, 1)
    struct.pack_into("<h", header, 70, 0)
    struct.pack_into("<h", header, 72, 384)
    struct.pack_into("<f", header, 108, 352.0)
    header[344:348] = b"n+1\x00"
    with gzip.open(path, "wb") as stream:
        stream.write(header)
        stream.write(b"\x00" * 4)
        stream.write(np.asarray(fields, dtype="<f4").ravel(order="F").tobytes())


def _identity_map(shape: tuple[int, int, int]) -> np.ndarray:
    coordinates = np.indices(shape, dtype=np.float32)
    result = np.empty((*shape, 3), dtype=np.float32)
    for axis in range(3):
        result[..., axis] = coordinates[axis] + 1.0
    return result


def _basis_eig(shape: tuple[int, int, int]) -> np.ndarray:
    fields = np.zeros((*shape, 12), dtype=np.float32)
    fields[..., 0] = 1.0
    fields[..., 4] = 1.0
    fields[..., 8] = 1.0
    fields[..., 9] = 3.0
    fields[..., 10] = 2.0
    fields[..., 11] = 1.0
    return fields


class TestBrainSuiteEigMap(unittest.TestCase):
    def test_identity_map_preserves_all_twelve_fields_and_header(self) -> None:
        shape = (5, 6, 7)
        source_fields = _basis_eig(shape)
        coordinates = np.indices(shape, dtype=np.float32)
        source_fields[..., 9] += coordinates[0]
        source_fields[..., 10] += coordinates[1]
        source_fields[..., 11] += coordinates[2]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            map_file = root / "identity.map.nii.gz"
            eig_file = root / "subject.eig.nii.gz"
            out_file = root / "warped.eig.nii.gz"
            nib.save(nib.Nifti1Image(_identity_map(shape), np.eye(4)), map_file)
            _write_eig(eig_file, source_fields)

            apply_map(str(map_file), str(eig_file), str(out_file), "", "0")

            output, header_bytes, _ = _read_brainsuite_eig(str(out_file))
            np.testing.assert_array_equal(output, source_fields)
            self.assertEqual(struct.unpack_from("<h", header_bytes, 70)[0], 0)
            self.assertEqual(struct.unpack_from("<h", header_bytes, 72)[0], 384)
            self.assertEqual(struct.unpack_from("<8h", header_bytes, 40)[0:5], (4, *shape, 12))

    def test_rigid_rotation_applies_ppd_to_eigenvectors(self) -> None:
        shape = (16, 16, 16)
        angle = 0.2
        cosine, sine = np.cos(angle), np.sin(angle)
        grid = np.indices(shape, dtype=np.float32)
        x = grid[0] - 7.5
        y = grid[1] - 7.5
        mapping = np.empty((*shape, 3), dtype=np.float32)
        mapping[..., 0] = cosine * x - sine * y + 8.5
        mapping[..., 1] = sine * x + cosine * y + 8.5
        mapping[..., 2] = grid[2] + 1.0

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            map_file = root / "rotation.map.nii.gz"
            eig_file = root / "subject.eig.nii.gz"
            out_file = root / "warped.eig.nii.gz"
            nib.save(nib.Nifti1Image(mapping, np.eye(4)), map_file)
            _write_eig(eig_file, _basis_eig(shape))

            apply_map(str(map_file), str(eig_file), str(out_file), "", "0")

            output, _, _ = _read_brainsuite_eig(str(out_file))
            np.testing.assert_allclose(output[8, 8, 8, 0:3], [cosine, -sine, 0.0], atol=1e-6)
            np.testing.assert_allclose(output[8, 8, 8, 3:6], [sine, cosine, 0.0], atol=1e-6)
            np.testing.assert_allclose(output[8, 8, 8, 6:9], [0.0, 0.0, 1.0], atol=1e-6)
            np.testing.assert_array_equal(output[8, 8, 8, 9:12], [3.0, 2.0, 1.0])


if __name__ == "__main__":
    unittest.main()
