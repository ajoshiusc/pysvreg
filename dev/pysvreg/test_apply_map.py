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

from dev.pysvreg.standalone import apply_map


class TestApplyMapInterpolation(unittest.TestCase):
    def test_interpolation_modes_match_matlab_reference_samples(self) -> None:
        shape = (11, 12, 13)
        rng = np.random.default_rng(20261104)
        source = rng.normal(size=shape).astype(np.float32)
        grid = np.indices(shape, dtype=np.float32)
        mapping = np.empty((*shape, 3), dtype=np.float32)
        mapping[..., 0] = grid[0] + 1.25
        mapping[..., 1] = grid[1] + 1.5
        mapping[..., 2] = grid[2] + 1.0
        samples = ((0, 0, 0), (0, 4, 5), (4, 5, 5), (5, 5, 5), (8, 9, 9), (10, 11, 12))
        expected = {
            "nearest": [0.0004963828, -1.3507514, -0.96810395, 0.21814279, -1.20854735, 0.0],
            "linear": [0.24201272, 0.028502874, 0.012189973, 0.30420804, -0.38855565, 0.0],
            "cubic": [-0.15671334, -0.032136764, -0.029048538, 0.42248315, -0.34532607, 0.0],
            "spline": [-1.10343838, -0.27470458, -0.09758037, 0.52317166, -0.16864663, 1.16172171],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            map_file = root / "map.nii.gz"
            source_file = root / "source.nii.gz"
            target_file = root / "target.nii.gz"
            nib.save(nib.Nifti1Image(mapping, np.eye(4)), map_file)
            nib.save(nib.Nifti1Image(source, np.eye(4)), source_file)
            nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.float32), np.eye(4)), target_file)

            for method, expected_values in expected.items():
                with self.subTest(method=method):
                    out_file = root / f"{method}.nii.gz"
                    apply_map(
                        str(map_file),
                        str(source_file),
                        str(out_file),
                        str(target_file),
                        "0",
                        "16",
                        "32",
                        method,
                    )
                    output = np.asarray(nib.load(out_file).dataobj)
                    values = [float(output[index]) for index in samples]
                    np.testing.assert_allclose(values, expected_values, atol=4e-6, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
