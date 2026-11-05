from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import nibabel as nib
import numpy as np

from . import make_atlas, native


class AtlasERoiIntegrationTests(unittest.TestCase):
    def test_edit_mode_invokes_roi_refinement_for_all_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / "atlas"
            labels = np.zeros((20, 20, 20), dtype=np.int16)
            labels[:10] = 2
            labels[10:] = 4
            image = nib.Nifti1Image(labels, np.eye(4))
            nib.save(image, f"{prefix}.label.nii.gz")
            nib.save(nib.Nifti1Image(np.ones_like(labels, dtype=np.float32), np.eye(4)), f"{prefix}.bfc.nii.gz")

            vertices = np.asarray(
                [(x, y, 10) for y in range(20) for x in range(20)], dtype=np.float32
            )
            faces = []
            for y in range(19):
                for x in range(19):
                    a = y * 20 + x
                    faces.extend(((a, a + 1, a + 20), (a + 1, a + 21, a + 20)))
            surface = SimpleNamespace(
                vertices=vertices,
                faces=np.asarray(faces, dtype=np.int32),
                labels=np.zeros(len(vertices), dtype=np.uint16),
            )
            for hemi in ("left", "right"):
                native.writedfs(f"{prefix}.{hemi}.mid.cortex.dfs", surface)

            with patch.object(make_atlas, "refine_surface") as refine:
                make_atlas._label_surfaces_from_volume(prefix)

            self.assertEqual(refine.call_count, 2)
            for call in refine.call_args_list:
                self.assertTrue(call.kwargs["include_zero"])
                self.assertIsNone(call.kwargs["min_label"])
                self.assertIsNone(call.kwargs["max_label"])
                self.assertEqual(call.kwargs["excluded_labels"], ())
                self.assertEqual(call.kwargs["excluded_pairs"], set())

            self.assertEqual(set(np.unique(native.readdfs(f"{prefix}.left.mid.cortex.dfs").labels)), {3, 5})
            self.assertEqual(set(np.unique(native.readdfs(f"{prefix}.right.mid.cortex.dfs").labels)), {2, 4})

    def test_edit_mode_executes_nonzero_level_set_flow(self):
        size = 120
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / "atlas"
            labels = np.zeros((size, size, 20), dtype=np.int16)
            labels[size // 3:2 * size // 3] = 2
            labels[2 * size // 3:] = 4
            nib.save(nib.Nifti1Image(labels, np.eye(4)), f"{prefix}.label.nii.gz")
            nib.save(
                nib.Nifti1Image(np.ones_like(labels, dtype=np.float32), np.eye(4)),
                f"{prefix}.bfc.nii.gz",
            )
            vertices = np.asarray(
                [(x, y, 10) for y in range(size) for x in range(size)], dtype=np.float32
            )
            faces = []
            for y in range(size - 1):
                for x in range(size - 1):
                    a = y * size + x
                    faces.extend(((a, a + 1, a + size), (a + 1, a + size + 1, a + size)))
            surface = SimpleNamespace(
                vertices=vertices,
                faces=np.asarray(faces, dtype=np.int32),
                labels=np.zeros(len(vertices), dtype=np.uint16),
            )
            for hemi in ("left", "right"):
                native.writedfs(f"{prefix}.{hemi}.mid.cortex.dfs", surface)

            actual_refine = make_atlas.refine_surface
            observed = []

            def track_refinement(surface, **kwargs):
                results = actual_refine(surface, **kwargs)
                observed.extend(results)
                return results

            with patch.object(make_atlas, "refine_surface", side_effect=track_refinement):
                make_atlas._label_surfaces_from_volume(prefix)

            self.assertTrue(
                any(stats.get("nit", 0) > 0 for _, stats in observed),
                "atlas -E did not run any nonzero level-set flow",
            )


if __name__ == "__main__":
    unittest.main()
