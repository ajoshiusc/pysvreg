from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from .dfsio import readdfs, writedfs
from .native import surface_jacobian_hemi


class SurfaceJacobianTests(unittest.TestCase):
    def test_affine_surface_map_has_expected_area_determinant(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            subject_prefix = root / "subject"
            atlas_vertices = np.asarray(
                ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)), dtype=np.float32
            )
            faces = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32)
            uv = atlas_vertices[:, :2]

            for hemi in ("left", "right"):
                atlas = SimpleNamespace(
                    vertices=atlas_vertices.copy(),
                    faces=faces.copy(),
                    u=uv[:, 0].copy(),
                    v=uv[:, 1].copy(),
                )
                subject = SimpleNamespace(
                    vertices=(atlas_vertices * (2, 3, 1)).astype(np.float32),
                    faces=faces.copy(),
                    u=uv[:, 0].copy(),
                    v=uv[:, 1].copy(),
                )
                writedfs(str(root / f"atlas.{hemi}.mid.cortex.svreg.dfs"), atlas)
                writedfs(str(root / f"subject.{hemi}.mid.cortex.svreg.dfs"), subject)

                result = surface_jacobian_hemi(str(subject_prefix), hemi)
                output = root / f"atlas.{hemi}.mid.cortex.jacobian.dfs"
                written = readdfs(str(output))

                self.assertEqual(len(result.vertices), 4)
                self.assertTrue(output.is_file())
                np.testing.assert_allclose(written.attributes, 6.0, rtol=0, atol=1e-6)
                self.assertTrue(np.isfinite(written.vColor).all())


if __name__ == "__main__":
    unittest.main()
