from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import nibabel as nib
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay, cKDTree

from .dfsio import writedfs
from .native import NiftiVolume, _solve_coupled_square_system, dws_write_nii
from .svreg_volreg import (
    _local_affine_scattered_interpolation,
    _interpolate_unit_sphere_surface_values,
    _refine_with_local_delaunay_simplices,
    extend_deformation_laplacian_hippo,
    fix_surfreg_map,
    inv_svreg_map,
    make_map_better,
    _map_jacobian_determinant,
)


class VolumetricMapInterpolationTests(unittest.TestCase):
    def test_unit_sphere_surface_transfer_matches_matlab_nearest_extrapolation(self) -> None:
        rng = np.random.default_rng(27)
        source = rng.normal(size=(1200, 3))
        source = (source / np.linalg.norm(source, axis=1)[:, None]).astype(np.float32).astype(np.float64)
        query = rng.normal(size=(600, 3))
        query = (query / np.linalg.norm(query, axis=1)[:, None]).astype(np.float32).astype(np.float64)
        values = np.column_stack(
            (
                source[:, 0] ** 2 + source[:, 1],
                np.sin(3.0 * source[:, 1]),
                source[:, 2] * source[:, 0],
            )
        )

        triangulation = Delaunay(source)
        expected = LinearNDInterpolator(triangulation, values)(query)
        outside = np.isnan(expected[:, 0])
        nearest_index = cKDTree(source).query(query, k=1)[1]
        expected[outside] = values[nearest_index[outside]]

        mapped = _interpolate_unit_sphere_surface_values(source, values, query)

        self.assertIsNotNone(mapped)
        self.assertGreater(float(np.mean(outside)), 0.99)
        np.testing.assert_allclose(mapped, expected, rtol=0.0, atol=1e-12)

    def test_local_interpolation_is_exact_for_affine_fields(self) -> None:
        rng = np.random.default_rng(4)
        source = rng.uniform(-1.0, 1.0, size=(100, 3))
        source = np.vstack(
            [
                source,
                np.asarray(
                    [
                        (-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
                        (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1),
                    ],
                    dtype=np.float64,
                ),
            ]
        )
        matrix = np.asarray(((2.0, 0.2, 0.3), (-0.4, 1.5, 0.1), (0.2, 0.5, 3.0)))
        offset = np.asarray((10.0, -4.0, 2.0))
        values = source @ matrix + offset
        query = rng.uniform(-0.9, 0.9, size=(250, 3))

        mapped = _local_affine_scattered_interpolation(source, values, query, neighbors=16, chunk_size=37)

        np.testing.assert_allclose(mapped, query @ matrix + offset, rtol=0.0, atol=1e-9)

    def test_local_empty_simplex_refinement_matches_global_delaunay_where_found(self) -> None:
        rng = np.random.default_rng(12)
        source = rng.uniform(-1.0, 1.0, size=(600, 3))
        values = np.column_stack(
            (
                np.sin(2 * source[:, 0]) + source[:, 1] ** 2,
                np.cos(2 * source[:, 1]) + source[:, 2] ** 2,
                source[:, 0] * source[:, 2] + np.sin(source[:, 1]),
            )
        )
        triangulation = Delaunay(source)
        query = rng.uniform(-0.8, 0.8, size=(2000, 3))
        query = query[triangulation.find_simplex(query) >= 0]
        expected = LinearNDInterpolator(triangulation, values)(query)
        affine = _local_affine_scattered_interpolation(
            source, values, query, hull_fallback=False
        )

        refined, used = _refine_with_local_delaunay_simplices(source, values, query, affine)

        self.assertGreater(float(np.mean(used)), 0.4)
        np.testing.assert_allclose(refined[used], expected[used], rtol=0.0, atol=1e-10)
        affine_error = np.linalg.norm(affine - expected, axis=1).mean()
        refined_error = np.linalg.norm(refined - expected, axis=1).mean()
        self.assertLess(refined_error, 0.6 * affine_error)

    def test_scattered_interpolation_uses_nearest_extrapolation_outside_hull(self) -> None:
        source = np.asarray(
            [
                (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
                (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1),
            ],
            dtype=np.float64,
        )
        values = source @ np.asarray(((2.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 4.0)))
        queries = np.asarray(((0.25, 0.5, 0.75), (2.2, 0.2, 0.2)))

        mapped = _local_affine_scattered_interpolation(source, values, queries, neighbors=8)

        np.testing.assert_allclose(mapped[0], queries[0] @ np.diag((2.0, 3.0, 4.0)), atol=1e-9)
        nearest = int(np.argmin(np.linalg.norm(source - queries[1], axis=1)))
        np.testing.assert_array_equal(mapped[1], values[nearest])

    def test_large_cloud_hull_fallback_catches_near_boundary_extrapolation(self) -> None:
        axis = np.linspace(-1.0, 1.0, 51)
        x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
        source = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
        values = (x * x + 0.3 * y * y + 0.2 * z * z).ravel()[:, None]
        query = np.asarray(((1.001, 0.0, 0.0),))

        mapped = _local_affine_scattered_interpolation(source, values, query)

        nearest = int(np.argmin(np.linalg.norm(source - query[0], axis=1)))
        np.testing.assert_array_equal(mapped[0], values[nearest])

    def test_coupled_square_solver_preserves_fixed_boundary_values(self) -> None:
        n = 6
        x, y = np.meshgrid(np.arange(n, dtype=np.float64), np.arange(n, dtype=np.float64), indexing="ij")
        vertices = np.column_stack((x.ravel(), y.ravel(), 0.04 * x.ravel() * y.ravel()))
        faces = []
        for i in range(n - 1):
            for j in range(n - 1):
                a = i * n + j
                b = (i + 1) * n + j
                c = (i + 1) * n + j + 1
                d = i * n + j + 1
                faces.extend(((a, b, c), (a, c, d)))
        surface = SimpleNamespace(vertices=vertices, faces=np.asarray(faces, dtype=np.int32))
        boundary = np.unique(
            np.concatenate(
                [
                    np.arange(n),
                    np.arange(n, n * n, n),
                    np.arange(n * n - n, n * n),
                    np.arange(n - 1, n * n, n),
                ]
            )
        )
        boundary_values = np.zeros(n * n, dtype=np.float64)
        boundary_values[boundary] = np.linspace(-1.0, 1.0, boundary.size)
        zeros = np.zeros(n * n, dtype=np.float64)
        mapped = _solve_coupled_square_system(
            surface,
            surface,
            boundary,
            boundary,
            boundary_values[boundary],
            zeros[boundary],
            boundary_values[boundary] * 0.8,
            zeros[boundary],
            np.asarray([n + 2, 2 * n + 3]),
            np.asarray([n + 2, 2 * n + 3]),
            np.asarray([10.0, 10.0]),
            zeros,
            zeros,
            zeros,
            zeros,
            maxiter=100,
        )

        for component, boundary_component in zip(mapped, (boundary_values[boundary], zeros[boundary], boundary_values[boundary] * 0.8, zeros[boundary])):
            self.assertTrue(np.isfinite(component).all())
            self.assertEqual(component.size, n * n)
            # Boundary values are prescribed exactly in the MATLAB system.
            np.testing.assert_allclose(component[boundary], boundary_component, rtol=0.0, atol=0.0)

    def test_laplacian_extension_preserves_affine_outside_brain_mask(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            tmp = root / "subject.svreg.tmp"
            tmp.mkdir()
            sub = str(root / "subject")
            tmp_base = str(tmp / "subject")
            shape = (9, 9, 9)
            affine = np.eye(4)
            header = nib.Nifti1Header()
            header.set_data_shape(shape)
            header.set_data_dtype(np.uint8)
            subject = NiftiVolume(np.zeros(shape, dtype=np.uint8), affine, header, None)

            brain_mask = np.zeros(shape, dtype=np.uint8)
            brain_mask[1:8, 1:8, 1:8] = 1
            nib.save(nib.Nifti1Image(brain_mask, affine), f"{sub}.cerebrum.mask.nii.gz")
            cortex = np.zeros(shape, dtype=np.uint8)
            cortex[2:7, 2:7, 2:7] = 1
            nib.save(nib.Nifti1Image(cortex, affine), f"{sub}.cortex.dewisp.mask.nii.gz")
            nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.uint8), affine), f"{tmp_base}.target.hippo_carved.nii.gz")

            vertices = np.asarray(((2, 2, 2), (6, 2, 2), (2, 6, 2), (2, 2, 6)), dtype=np.float32)
            faces = np.asarray(((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)), dtype=np.int32)
            surf = SimpleNamespace(vertices=vertices, faces=faces)
            writedfs(f"{sub}.pial.cortex.dfs", surf)
            writedfs(f"{sub}.inner.cortex.dfs", surf)

            coords = np.indices(shape, dtype=np.float32).transpose(1, 2, 3, 0) + 1.0
            affine_map = coords.copy()
            surfreg_map = coords.copy()
            surfreg_map[1:8, 1:8, 1:8, 0] += 0.25
            dws_write_nii(f"{tmp_base}.affine.map.nii.gz", affine_map, subject)

            extended = extend_deformation_laplacian_hippo(
                sub, tmp_base, surfreg_map, affine_map, subject, data_fit=10.0
            )

            self.assertTrue(np.isfinite(extended).all())
            np.testing.assert_array_equal(extended[0, :, :, :], affine_map[0, :, :, :])
            np.testing.assert_array_equal(extended[-1, :, :, :], affine_map[-1, :, :, :])
            self.assertGreater(float(np.max(np.abs(extended - affine_map))), 0.0)
            self.assertTrue(Path(f"{tmp_base}.warped.hippo_carved.nii.gz").is_file())
            self.assertTrue(Path(f"{tmp_base}.dilated.mask.nii.gz").is_file())

    def test_inverse_consistency_cycles_preserve_identity_map(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            tmp = root / "subject.svreg.tmp"
            tmp.mkdir()
            shape = (9, 9, 9)
            affine = np.eye(4)
            header = nib.Nifti1Header()
            header.set_data_shape(shape)
            header.set_data_dtype(np.uint8)
            mask = np.zeros(shape, dtype=np.uint8)
            mask[2:7, 2:7, 2:7] = 1
            volume = NiftiVolume(mask, affine, header, None)
            coords = np.indices(shape, dtype=np.float32).transpose(1, 2, 3, 0) + 1.0
            map_path = str(tmp / "subject.surfreg.map.nii.gz")
            dws_write_nii(map_path, coords, volume)

            repaired = fix_surfreg_map(
                str(root / "subject"), str(tmp / "subject"), coords, volume, volume
            )

            np.testing.assert_allclose(repaired, coords, rtol=0.0, atol=0.0)
            self.assertTrue(Path(str(tmp / "subject.orig.surfreg.map.nii.gz")).is_file())

    def test_map_jacobian_and_nonfolded_outputs(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            shape = (8, 8, 8)
            affine = np.eye(4)
            header = nib.Nifti1Header()
            header.set_data_shape(shape)
            header.set_data_dtype(np.float32)
            reference = NiftiVolume(np.zeros(shape, dtype=np.float32), affine, header, None)
            coords = np.indices(shape, dtype=np.float32).transpose(1, 2, 3, 0)
            mapping = np.stack((2 * coords[..., 0], 3 * coords[..., 1], 4 * coords[..., 2]), axis=-1)

            np.testing.assert_allclose(_map_jacobian_determinant(mapping), 24.0, rtol=0, atol=0)
            make_map_better(str(root / "case"), mapping, reference, reference)
            self.assertTrue(Path(f"{root}/case.surfreg.map2.corr.nii.gz").is_file())
            self.assertTrue(Path(f"{root}/case.surfreg.jacobian.nii.gz").is_file())
            self.assertEqual(nib.load(f"{root}/case.surfreg.jacobian.nii.gz").get_data_dtype(), np.dtype(np.float32))

    def test_final_inverse_map_identity_and_output_headers(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            tmp = root / "subject.svreg.tmp"
            tmp.mkdir()
            shape = (9, 9, 9)
            affine = np.eye(4)
            header = nib.Nifti1Header()
            header.set_data_shape(shape)
            header.set_data_dtype(np.float32)
            mask = np.zeros(shape, dtype=np.float32)
            mask[2:7, 2:7, 2:7] = 1.0
            source = NiftiVolume(mask.copy(), affine, header.copy(), None)
            target = NiftiVolume(mask.copy(), affine, header.copy(), None)
            pvc = NiftiVolume(mask.copy(), affine, header.copy(), None)
            coords = np.indices(shape, dtype=np.float32).transpose(1, 2, 3, 0) + 1.0
            dws_write_nii(str(tmp / "subject.svreg.map.nii.gz"), coords, source)

            inverse = inv_svreg_map(str(root / "subject"), str(tmp / "subject"), source, target, pvc)

            np.testing.assert_allclose(inverse, coords, rtol=0.0, atol=0.0)
            self.assertEqual(nib.load(f"{root}/subject.svreg.inv.map.nii.gz").get_data_dtype(), np.dtype(np.float32))
            self.assertEqual(nib.load(f"{tmp}/subject.invsvreg.nii.gz").get_data_dtype(), np.dtype(np.float32))
            self.assertEqual(nib.load(f"{root}/subject.svreg.inv.jacobian.nii.gz").get_data_dtype(), np.dtype(np.float32))


if __name__ == "__main__":
    unittest.main()
