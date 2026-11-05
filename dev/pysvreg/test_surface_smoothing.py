from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from . import standalone
from .dfsio import readdfs, writedfs


class SurfaceSmoothingTests(unittest.TestCase):
    def test_matches_matlab_weighted_stiffness_normal_equations(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            surface_path = root / "geometry.dfs"
            function_path = root / "function.dfs"
            output_path = root / "smoothed.dfs"
            n = 8
            x, y = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n), indexing="ij")
            vertices = np.column_stack(
                [x.ravel(), y.ravel(), 0.1 * np.sin(2 * x.ravel()) * np.cos(2 * y.ravel())]
            ).astype(np.float32)
            faces = []
            for i in range(n - 1):
                for j in range(n - 1):
                    a = i * n + j
                    b = (i + 1) * n + j
                    c = (i + 1) * n + j + 1
                    d = i * n + j + 1
                    faces.extend(((a, b, c), (a, c, d)))
            faces = np.asarray(faces, dtype=np.int32)
            values = (np.sin(3 * x.ravel()) + 0.3 * np.cos(5 * y.ravel())).astype(np.float32)
            geometry = SimpleNamespace(vertices=vertices, faces=faces)
            function = SimpleNamespace(vertices=vertices, faces=faces, attributes=values)
            writedfs(surface_path, geometry)
            writedfs(function_path, function)

            parameter = 2.5
            standalone.smooth_surface_function(
                str(surface_path), str(function_path), str(output_path), str(parameter)
            )

            loaded = readdfs(surface_path)
            stiffness, dx, dy = standalone.get_stiffness_matrix_tri_wt(
                loaded, np.ones(vertices.shape[0], dtype=np.float64)
            )
            a = np.sqrt(parameter)
            system = (
                diags(np.ones(vertices.shape[0]))
                + a**2 * (stiffness.T @ stiffness)
                + a**2 * (dx.T @ dx)
                + a**2 * (dy.T @ dy)
            ).tocsr()
            expected = spsolve(system, values.astype(np.float64))
            actual = readdfs(output_path).attributes
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
