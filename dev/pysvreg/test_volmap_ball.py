from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import nibabel as nib
import numpy as np
from scipy.io import loadmat

from .native import make_unitball_map


class UnitBallMapTests(unittest.TestCase):
    def test_writes_matlab_brain_struct_and_harmonic_volume_map(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            mask = np.zeros((9, 9, 9), dtype=np.uint8)
            mask[1:8, 1:8, 1:8] = 255
            mask_path = root / "subject.cortex.dewisp.mask.nii.gz"
            nib.save(nib.Nifti1Image(mask, np.eye(4)), mask_path)

            vertices = np.asarray(
                ((1, 1, 1), (7, 1, 1), (1, 7, 1), (1, 1, 7)), dtype=np.float32
            )
            faces = np.asarray(((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)), dtype=np.int32)

            def surface() -> SimpleNamespace:
                return SimpleNamespace(
                    vertices=vertices.copy(),
                    faces=faces.copy(),
                    u=np.asarray((0.2, 0.8, 0.2, 0.2)),
                    v=np.asarray((0.2, 0.2, 0.8, 0.2)),
                )

            output_base = str(root / "subject.svreg.tmp" / "subject")
            Path(output_base).parent.mkdir()
            make_unitball_map(
                output_base,
                str(root / "subject"),
                (1.0, 1.0, 1.0),
                surface(),
                surface(),
                surface(),
                surface(),
            )

            brain = loadmat(f"{output_base}_unitball_map.mat", simplify_cells=True)["Brain"]
            self.assertEqual(np.asarray(brain["Msize"]).tolist(), [9, 9, 9])
            self.assertEqual(np.asarray(brain["volmap"]).shape, (343, 3))
            self.assertEqual(np.asarray(brain["mask_indx"]).size, 343)
            self.assertEqual(np.asarray(brain["surf_map"]).shape, (8, 3))
            self.assertTrue(np.isfinite(brain["volmap"]).all())
            self.assertLessEqual(float(np.linalg.norm(brain["volmap"], axis=1).max()), 1.00001)


if __name__ == "__main__":
    unittest.main()
