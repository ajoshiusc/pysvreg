from __future__ import annotations

import sys
from pathlib import Path
import shutil
import re
from itertools import combinations

try:
    from . import _deps  # noqa: F401
except ImportError:
    import _deps  # noqa: F401

import numpy as np
from scipy.io import loadmat
from scipy.ndimage import binary_dilation, gaussian_filter, map_coordinates
from scipy.sparse import diags
from scipy.sparse.linalg import LinearOperator, cg
from scipy.spatial import ConvexHull, Delaunay, QhullError, cKDTree

try:
    from .native import (
        NiftiVolume,
        _masked_graph_laplacian,
        build_affine_map,
        dws_write_nii,
        estimate_affine_from_pial_surfaces,
        load_nii_z,
        overwrite_map_with_surface_correspondence,
        parameterize_surface_uv,
        readdfs,
        save_untouch_nii_gz,
        warp_volume_from_map,
    )
    from .runtime import normalize_fileprefix
    from .svreg_elastic import svreg_elastic_vol_reg_nonlin_cg
except ImportError:
    from native import NiftiVolume, _masked_graph_laplacian, build_affine_map, dws_write_nii, estimate_affine_from_pial_surfaces, load_nii_z, overwrite_map_with_surface_correspondence, parameterize_surface_uv, readdfs, save_untouch_nii_gz, warp_volume_from_map
    from runtime import normalize_fileprefix
    from svreg_elastic import svreg_elastic_vol_reg_nonlin_cg


def _local_affine_scattered_interpolation(
    source_points: np.ndarray,
    source_values: np.ndarray,
    query_points: np.ndarray,
    neighbors: int = 16,
    chunk_size: int = 12000,
    hull_fallback: bool = True,
    refine_with_simplex: bool = False,
) -> np.ndarray:
    """Approximate MATLAB linear scattered interpolation without global Delaunay.

    SVReg's MATLAB helper builds a 3D Delaunay triangulation over every voxel
    in the target cortex. That triangulation can exceed practical memory at
    full subject resolution. A local weighted affine fit to nearby samples is
    exact for affine fields and second-order accurate for smooth harmonic maps.
    MATLAB uses nearest-neighbor extrapolation outside the source convex hull;
    those queries are detected from negative local affine weights and confirmed
    against the hull before substituting the nearest sample.
    """
    source_points = np.asarray(source_points, dtype=np.float64)
    source_values = np.asarray(source_values, dtype=np.float64)
    query_points = np.asarray(query_points, dtype=np.float64)
    if source_points.ndim != 2 or source_points.shape[1] != 3:
        raise ValueError("scattered interpolation source points must be N x 3")
    if source_values.ndim == 1:
        source_values = source_values[:, None]
    if source_values.shape[0] != source_points.shape[0]:
        raise ValueError("scattered interpolation points and values differ in length")
    if source_points.shape[0] == 0:
        raise ValueError("scattered interpolation requires source samples")

    k = min(max(1, int(neighbors)), source_points.shape[0])
    tree = cKDTree(source_points)
    output = np.empty((query_points.shape[0], source_values.shape[1]), dtype=np.float64)
    hull_triangulation = None
    hull_failed = False
    for first in range(0, query_points.shape[0], chunk_size):
        last = min(first + chunk_size, query_points.shape[0])
        q = query_points[first:last]
        distance, index = tree.query(q, k=k, workers=-1)
        if k == 1:
            output[first:last] = source_values[index]
            continue
        p = source_points[index]
        y = source_values[index]
        exact = distance[:, 0] <= 1e-10
        result = np.empty((q.shape[0], source_values.shape[1]), dtype=np.float64)
        if np.any(exact):
            result[exact] = y[exact, 0]
        fit = ~exact
        if np.any(fit):
            d = distance[fit]
            pfit = p[fit]
            yfit = y[fit]
            weight = 1.0 / np.maximum(d, 1e-10) ** 2
            weight /= np.sum(weight, axis=1, keepdims=True)
            pmean = np.einsum("bk,bki->bi", weight, pfit)
            ymean = np.einsum("bk,bko->bo", weight, yfit)
            dp = pfit - pmean[:, None, :]
            dy = yfit - ymean[:, None, :]
            covariance = np.einsum("bk,bki,bkj->bij", weight, dp, dp)
            cross = np.einsum("bk,bki,bko->bio", weight, dp, dy)
            scale = np.maximum(np.trace(covariance, axis1=1, axis2=2), 1.0)
            covariance += (scale * 1e-12)[:, None, None] * np.eye(3)[None, :, :]
            try:
                slopes = np.linalg.solve(covariance, cross)
                result[fit] = ymean + np.einsum("bi,bio->bo", q[fit] - pmean, slopes)
                # With nonnegative affine weights, the query is inside the
                # local convex hull and therefore cannot need MATLAB's outside
                # hull nearest-neighbor fallback.
                affine_direction = np.linalg.solve(
                    covariance, (q[fit] - pmean)[..., None]
                )[..., 0]
                affine_weights = weight * (
                    1.0 + np.einsum("bki,bi->bk", dp, affine_direction)
                )
                # Large brain-mask clouds generate many weakly negative local
                # weights for queries that are still inside the global hull.
                # A -0.005 threshold captures near-boundary points missed by
                # the wider cutoff while skipping weak negatives.
                candidate_threshold = -0.005 if source_points.shape[0] >= 100_000 else -1e-8
                outside_candidates = np.min(affine_weights, axis=1) < candidate_threshold
                candidate_rows = np.flatnonzero(outside_candidates) if hull_fallback else np.empty(0, dtype=np.int64)
                if candidate_rows.size:
                    if hull_triangulation is None and not hull_failed:
                        try:
                            hull_vertices = ConvexHull(source_points).vertices
                            hull_triangulation = Delaunay(source_points[hull_vertices])
                        except QhullError:
                            hull_failed = True
                    if hull_triangulation is not None:
                        query_rows = np.flatnonzero(fit)[candidate_rows]
                        is_outside = hull_triangulation.find_simplex(q[query_rows], tol=1e-8) < 0
                        nearest_rows = query_rows[is_outside]
                        if nearest_rows.size:
                            _, nearest = tree.query(q[nearest_rows], k=1, workers=-1)
                            result[nearest_rows] = source_values[nearest]
            except np.linalg.LinAlgError:
                # Degenerate local neighborhoods occur only at thin mask tips;
                # their nearest sample is the stable extrapolation fallback.
                result[fit] = yfit[:, 0]
        output[first:last] = result
    if refine_with_simplex:
        output, _ = _refine_with_local_delaunay_simplices(
            source_points, source_values, query_points, output
        )
    return output


def _refine_with_local_delaunay_simplices(
    source_points: np.ndarray,
    source_values: np.ndarray,
    query_points: np.ndarray,
    affine_values: np.ndarray,
    *,
    neighbors: int = 8,
    emptiness_neighbors: int = 32,
    chunk_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace local affine estimates when a nearby empty Delaunay tetrahedron exists.

    The candidate tetrahedra use the nearest ``neighbors`` source samples. A
    candidate is accepted only when it contains the query and its circumsphere
    is empty with respect to a larger nearby sample. Accepted values are true
    barycentric linear interpolation over that tetrahedron; queries for which
    the local test cannot identify a simplex retain the affine estimate.
    """
    source_points = np.asarray(source_points, dtype=np.float64)
    source_values = np.asarray(source_values, dtype=np.float64)
    query_points = np.asarray(query_points, dtype=np.float64)
    if source_values.ndim == 1:
        source_values = source_values[:, None]
    output = np.asarray(affine_values, dtype=np.float64)
    if output.ndim == 1:
        output = output[:, None]
    output = output.copy()
    if source_values.shape[0] != source_points.shape[0]:
        raise ValueError("scattered interpolation points and values differ in length")
    if output.shape != (query_points.shape[0], source_values.shape[1]):
        raise ValueError("affine fallback values have the wrong shape")
    k = min(max(4, int(neighbors)), source_points.shape[0])
    check_k = min(max(k, int(emptiness_neighbors)), source_points.shape[0])
    if source_points.shape[0] < 4:
        return output, np.zeros(query_points.shape[0], dtype=bool)

    tetrahedra = np.asarray(list(combinations(range(k), 4)), dtype=np.int32)
    n_tetra = tetrahedra.shape[0]
    vertex_ranks = np.zeros((n_tetra, check_k), dtype=bool)
    for corner in range(4):
        vertex_ranks[np.arange(n_tetra), tetrahedra[:, corner]] = True

    tree = cKDTree(source_points)
    used = np.zeros(query_points.shape[0], dtype=bool)
    for first in range(0, query_points.shape[0], chunk_size):
        last = min(first + chunk_size, query_points.shape[0])
        query = query_points[first:last]
        _, neighbor_index = tree.query(query, k=check_k, workers=-1)
        if check_k == 1:
            neighbor_index = neighbor_index[:, None]
        local_index = neighbor_index[:, :k]
        points = source_points[local_index[:, tetrahedra]]
        origin = points[:, :, 0, :]
        edges = points[:, :, 1:, :] - origin[:, :, None, :]

        # Solve barycentric coordinates for every local candidate simplex.
        matrix = np.swapaxes(edges, -1, -2)
        rhs = query[:, None, :] - origin
        determinant = np.linalg.det(matrix)
        scale = np.maximum(np.max(np.linalg.norm(edges, axis=-1), axis=-1), 1.0)
        nonsingular = np.abs(determinant) > (1e-12 * scale**3)
        barycentric_tail = np.zeros((*determinant.shape, 3), dtype=np.float64)
        if np.any(nonsingular):
            barycentric_tail[nonsingular] = np.linalg.solve(
                matrix[nonsingular], rhs[..., None][nonsingular]
            )[..., 0]
        barycentric = np.concatenate(
            (1.0 - barycentric_tail.sum(axis=-1, keepdims=True), barycentric_tail), axis=-1
        )
        contains_query = nonsingular & (barycentric.min(axis=-1) >= -1e-9) & (
            barycentric.max(axis=-1) <= 1.0 + 1e-9
        )

        # A Delaunay tetrahedron's circumsphere contains no other source point.
        squared_edges = np.sum(edges * edges, axis=-1)
        center_relative = np.zeros_like(squared_edges)
        if np.any(nonsingular):
            center_relative[nonsingular] = np.linalg.solve(
                (2.0 * edges)[nonsingular], squared_edges[..., None][nonsingular]
            )[..., 0]
        radius_squared = np.sum(center_relative * center_relative, axis=-1)
        center = origin + center_relative
        nearby_points = source_points[neighbor_index]
        delta = nearby_points[:, None, :, :] - center[:, :, None, :]
        distance_squared = np.sum(delta * delta, axis=-1)
        sphere_tolerance = np.maximum(radius_squared[..., None] * 1e-9, 1e-13)
        contains_other = np.any(
            (distance_squared < radius_squared[..., None] - sphere_tolerance)
            & ~vertex_ranks[None, :, :],
            axis=-1,
        )
        accepted = contains_query & ~contains_other

        rows = np.arange(query.shape[0])
        selected = np.argmax(accepted, axis=1)
        has_simplex = accepted[rows, selected]
        local_rows = rows[has_simplex]
        tetra_rows = selected[has_simplex]
        if local_rows.size:
            selected_ranks = tetrahedra[tetra_rows]
            selected_indices = local_index[local_rows[:, None], selected_ranks]
            weights = barycentric[local_rows, tetra_rows]
            output[first + local_rows] = np.einsum(
                "bi,bij->bj", weights, source_values[selected_indices]
            )
            used[first + local_rows] = True
    return output, used


def _interpolate_unit_sphere_surface_values(
    source_points: np.ndarray,
    source_values: np.ndarray,
    query_points: np.ndarray,
    *,
    sphere_tolerance: float = 2e-6,
) -> np.ndarray | None:
    """Apply MATLAB's nearest extrapolation for points on a sampled unit sphere.

    ``volmap_ball`` places every cortical surface vertex on the unit sphere.
    A point on the sphere is outside the convex hull of a finite set of other
    sphere points, so ``TriScatteredInterp(..., 'linear')`` returns NaN there
    and SVReg's ``mygriddata3`` replaces it with the nearest source value.
    Return ``None`` when inputs do not satisfy that geometric invariant so
    callers can use the general scattered interpolation fallback.
    """
    source_points = np.asarray(source_points, dtype=np.float64)
    source_values = np.asarray(source_values)
    query_points = np.asarray(query_points, dtype=np.float64)
    if source_points.ndim != 2 or source_points.shape[1] != 3:
        return None
    if query_points.ndim != 2 or query_points.shape[1] != 3:
        return None
    if source_points.shape[0] == 0 or source_values.shape[0] != source_points.shape[0]:
        return None
    source_radius = np.linalg.norm(source_points, axis=1)
    query_radius = np.linalg.norm(query_points, axis=1)
    if (
        np.max(np.abs(source_radius - 1.0), initial=0.0) > sphere_tolerance
        or np.max(np.abs(query_radius - 1.0), initial=0.0) > sphere_tolerance
    ):
        return None

    tree = cKDTree(source_points)
    _, nearest = tree.query(query_points, k=1, workers=-1)
    return source_values[nearest]


def _load_unitball_brain(path: str) -> dict:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"missing unit-ball map: {file_path}")
    brain = loadmat(file_path, simplify_cells=True).get("Brain")
    if not isinstance(brain, dict):
        raise ValueError(f"invalid unit-ball MAT structure: {file_path}")
    for field in ("mask_indx", "Msize", "resolution", "volmap", "surf_map"):
        if field not in brain:
            raise ValueError(f"unit-ball MAT file {file_path} is missing Brain.{field}")
    return brain


def extend_deformation_laplacian_hippo(
    subbasename: str,
    tmp_base: str,
    surfreg_map: np.ndarray,
    affine_map: np.ndarray,
    reference: NiftiVolume,
    data_fit: float = 50.0,
) -> np.ndarray:
    """Extend surface displacement through the cerebrum with a Laplacian fit.

    This follows ``extend_deformation_laplacian_hippo.m``: solve for a smooth
    displacement over the cerebrum mask, fit it to hippo-supported samples and
    surface/boundary constraints, then add it to the affine map.
    """
    brain = load_nii_z(f"{subbasename}.cerebrum.mask.nii.gz")
    brain_mask = np.asarray(brain.img) > 0
    if brain_mask.shape != reference.shape[:3]:
        raise ValueError("cerebrum mask dimensions do not match subject image")
    # MATLAB clears the outermost voxel planes before constructing periodic
    # finite differences, so the periodic wrap does not cross the brain mask.
    brain_mask[[0, -1], :, :] = False
    brain_mask[:, [0, -1], :] = False
    brain_mask[:, :, [0, -1]] = False
    mask_flat = np.flatnonzero(brain_mask.ravel(order="F"))
    if mask_flat.size == 0:
        raise ValueError("cerebrum mask is empty after clearing image borders")

    cortex_volume = load_nii_z(f"{subbasename}.cortex.dewisp.mask.nii.gz")
    cortex = np.asarray(cortex_volume.img, dtype=np.float64).copy()
    resolution = np.asarray(brain.zooms[:3], dtype=np.float64)

    def _mark_surface(vertices: np.ndarray) -> None:
        voxel = np.floor(np.asarray(vertices, dtype=np.float64) / resolution + 0.5).astype(np.int64)
        valid = np.all((voxel >= 0) & (voxel < np.asarray(cortex.shape[:3])), axis=1)
        voxel = voxel[valid]
        if voxel.size:
            cortex[voxel[:, 0], voxel[:, 1], voxel[:, 2]] = 200.0

    pial = readdfs(f"{subbasename}.pial.cortex.dfs")
    inner = readdfs(f"{subbasename}.inner.cortex.dfs")
    _mark_surface(pial.vertices)
    _mark_surface(inner.vertices)
    _mark_surface((np.asarray(pial.vertices) + np.asarray(inner.vertices)) * 0.5)

    mask_coords = np.column_stack(np.unravel_index(mask_flat, brain_mask.shape, order="F"))
    cortex_on_mask = cortex[tuple(mask_coords.T)]
    surf_indices = np.flatnonzero(cortex_on_mask == 200.0)
    lap = _masked_graph_laplacian(mask_flat, brain_mask.shape)
    boundary_indices = np.flatnonzero(lap.diagonal() < 6.0)

    hippo = load_nii_z(f"{tmp_base}.target.hippo_carved.nii.gz")
    affine_nifti = load_nii_z(f"{tmp_base}.affine.map.nii.gz")
    warped_hippo_img = warp_volume_from_map(hippo.img, affine_nifti.img, order=0, cval=0.0)
    save_untouch_nii_gz(
        NiftiVolume(warped_hippo_img.astype(brain.img.dtype), brain.affine, brain.header.copy(), brain.path),
        f"{tmp_base}.warped.hippo_carved.nii.gz",
    )

    msk = cortex * warped_hippo_img
    gm_indices = np.flatnonzero(msk[tuple(mask_coords.T)] > 0)
    known = np.union1d(np.union1d(gm_indices, boundary_indices), surf_indices)
    weights = np.zeros(mask_flat.size, dtype=np.float64)
    weights[known] = 1.0
    weights[gm_indices] = float(data_fit)

    map_flat = np.stack(
        [surfreg_map[..., axis].ravel(order="F")[mask_flat] for axis in range(3)], axis=1
    ).astype(np.float64)
    affine_flat = np.stack(
        [affine_map[..., axis].ravel(order="F")[mask_flat] for axis in range(3)], axis=1
    ).astype(np.float64)
    displacement = map_flat - affine_flat
    carve_out = (cortex - warped_hippo_img) > 0
    carve_out_on_mask = carve_out.ravel(order="F")[mask_flat]
    displacement[carve_out_on_mask] = 0.0

    resolution_volume = float(np.prod(resolution))
    alpha = 11.0 / np.sqrt(resolution_volume)
    normal = (alpha * alpha) * lap + diags(weights * weights, format="csr")
    diagonal = normal.diagonal()
    preconditioner = LinearOperator(normal.shape, matvec=lambda x: x / np.maximum(diagonal, 1e-12))
    updated = np.asarray(affine_map, dtype=np.float32).copy()
    for axis in range(3):
        rhs = np.zeros(mask_flat.size, dtype=np.float64)
        rhs[known] = weights[known] * float(data_fit) * displacement[known, axis]
        solution, info = cg(normal, rhs, rtol=1e-8, atol=0.0, maxiter=500, M=preconditioner)
        if info < 0:
            raise RuntimeError(f"Laplacian deformation solver failed for coordinate {axis}: {info}")
        values = affine_flat[:, axis] + solution
        updated[mask_coords[:, 0], mask_coords[:, 1], mask_coords[:, 2], axis] = values.astype(np.float32)

    brain_mask_volume = NiftiVolume(brain_mask.astype(brain.img.dtype), brain.affine, brain.header.copy(), brain.path)
    save_untouch_nii_gz(brain_mask_volume, f"{tmp_base}.dilated.mask.nii.gz")
    return updated


def _smoothed_mask_coordinates(image: np.ndarray, radius: int = 6) -> np.ndarray:
    """MATLAB smooth3(...,'gaussian',[13 13 13],5) followed by find(img>0)."""
    smooth = gaussian_filter(np.asarray(image, dtype=np.float64), sigma=5.0, radius=radius)
    flat = np.flatnonzero(smooth.ravel(order="F") > 0)
    return np.column_stack(np.unravel_index(flat, smooth.shape[:3], order="F"))


def _invert_deformation_once(
    deformation: np.ndarray,
    source_coordinates: np.ndarray,
    target_coordinates: np.ndarray,
    target_shape: tuple[int, int, int],
) -> np.ndarray:
    """Invert a voxel map by scattered interpolation over the source mask."""
    mapped = np.asarray(deformation, dtype=np.float32)
    source_map_values = np.column_stack(
        [mapped[source_coordinates[:, 0], source_coordinates[:, 1], source_coordinates[:, 2], axis] for axis in range(3)]
    ).astype(np.float64)
    source_values = np.asarray(source_coordinates, dtype=np.float64) + 1.0
    queries = np.asarray(target_coordinates, dtype=np.float64) + 1.0
    inverse_values = _local_affine_scattered_interpolation(source_map_values, source_values, queries)
    inverse = np.zeros((*target_shape, 3), dtype=np.float32)
    inverse[target_coordinates[:, 0], target_coordinates[:, 1], target_coordinates[:, 2], :] = inverse_values.astype(np.float32)
    return inverse


def fix_surfreg_map(
    subbasename: str,
    tmp_base: str,
    surfreg_map: np.ndarray,
    source_bfc: NiftiVolume,
    target_bfc: NiftiVolume,
) -> np.ndarray:
    """Run MATLAB fix_surfreg_map's four forward/inverse consistency cycles."""
    shutil.copyfile(f"{tmp_base}.surfreg.map.nii.gz", f"{tmp_base}.orig.surfreg.map.nii.gz")
    source_coordinates = _smoothed_mask_coordinates(source_bfc.img)
    target_coordinates = _smoothed_mask_coordinates(target_bfc.img)
    if source_coordinates.size == 0 or target_coordinates.size == 0:
        raise ValueError("cannot repair deformation with an empty smoothed source or target mask")

    original = np.asarray(surfreg_map, dtype=np.float32)
    forward = original.copy()
    for _ in range(4):
        inverse = _invert_deformation_once(
            forward, source_coordinates, target_coordinates, target_bfc.shape[:3]
        )
        forward = _invert_deformation_once(
            inverse, target_coordinates, source_coordinates, source_bfc.shape[:3]
        )

    # MATLAB preserves the original map outside src_mask after the final inverse.
    repaired = original.copy()
    repaired[source_coordinates[:, 0], source_coordinates[:, 1], source_coordinates[:, 2], :] = forward[
        source_coordinates[:, 0], source_coordinates[:, 1], source_coordinates[:, 2], :
    ]
    return repaired


def _map_jacobian_determinant(map_values: np.ndarray) -> np.ndarray:
    """Port myjacobian3dmap's determinant using MATLAB-compatible axes."""
    coordinate_fields = [np.asarray(map_values[..., axis], dtype=np.float32) for axis in range(3)]
    derivatives = [
        [np.gradient(field, axis=axis) for axis in range(3)]
        for field in coordinate_fields
    ]
    dx0, dx1, dx2 = derivatives[0]
    dy0, dy1, dy2 = derivatives[1]
    dz0, dz1, dz2 = derivatives[2]
    return (
        dx0 * (dy1 * dz2 - dy2 * dz1)
        - dy0 * (dx1 * dz2 - dx2 * dz1)
        + dz0 * (dx1 * dy2 - dx2 * dy1)
    )


def make_map_better(tmp_base: str, surfreg_map: np.ndarray, reference: NiftiVolume, pvc_volume: NiftiVolume) -> np.ndarray:
    """Repair negative-Jacobian voxels and write MATLAB make_map_better outputs."""
    corrected = np.asarray(surfreg_map, dtype=np.float32).copy()
    jacobian = _map_jacobian_determinant(corrected)
    negative = jacobian < 0
    sphere = np.indices((7, 7, 7), dtype=np.int8) - 3
    structure = np.sum(sphere * sphere, axis=0) <= 9
    surrounding = binary_dilation(negative, structure=structure)
    surrounding[negative] = False
    bad = np.argwhere(negative)
    good = np.argwhere(surrounding)
    if bad.size:
        if not good.size:
            raise ValueError("cannot repair negative Jacobians: no surrounding non-folded voxels")
        tree = cKDTree(good.astype(np.float64))
        _, nearest = tree.query(bad.astype(np.float64) + np.asarray((0.753, 0.352, 0.551)), k=1, workers=-1)
        corrected[bad[:, 0], bad[:, 1], bad[:, 2], :] = corrected[
            good[nearest, 0], good[nearest, 1], good[nearest, 2], :
        ]
    jacobian = _map_jacobian_determinant(corrected)
    dws_write_nii(f"{tmp_base}.surfreg.map2.corr.nii.gz", corrected, reference)
    jacobian_header = pvc_volume.header.copy()
    jacobian_header.set_data_dtype(np.float32)
    jacobian_volume = NiftiVolume(jacobian.astype(np.float32), pvc_volume.affine, jacobian_header, pvc_volume.path)
    save_untouch_nii_gz(jacobian_volume, f"{tmp_base}.surfreg.jacobian.nii.gz")
    return corrected


def inv_svreg_map(
    subbasename: str,
    tmp_base: str,
    source_bfc: NiftiVolume,
    target_bfc: NiftiVolume,
    source_pvc: NiftiVolume,
) -> np.ndarray:
    """Generate MATLAB inv_svreg_map outputs and Jacobians."""
    forward_volume = load_nii_z(f"{tmp_base}.svreg.map.nii.gz")
    source_coordinates = _smoothed_mask_coordinates(source_bfc.img)
    target_coordinates = _smoothed_mask_coordinates(target_bfc.img)
    inverse_map = _invert_deformation_once(
        forward_volume.img,
        source_coordinates,
        target_coordinates,
        target_bfc.shape[:3],
    )
    dws_write_nii(f"{subbasename}.svreg.inv.map.nii.gz", inverse_map, target_bfc)

    inverse_pvc = map_coordinates(
        np.asarray(source_bfc.img, dtype=np.float64),
        [inverse_map[..., axis] - 1.0 for axis in range(3)],
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    inverse_pvc *= np.asarray(target_bfc.img > 0, dtype=np.float64)
    inverse_pvc_volume = NiftiVolume(
        inverse_pvc.astype(np.float32), target_bfc.affine, target_bfc.header.copy(), target_bfc.path
    )
    inverse_pvc_volume.header.set_data_shape(inverse_pvc_volume.img.shape)
    inverse_pvc_volume.header.set_data_dtype(np.float32)
    save_untouch_nii_gz(inverse_pvc_volume, f"{tmp_base}.invsvreg.nii.gz")

    inverse_jacobian = _map_jacobian_determinant(inverse_map)
    inverse_jacobian_header = target_bfc.header.copy()
    inverse_jacobian_header.set_data_shape(target_bfc.shape[:3])
    inverse_jacobian_header.set_data_dtype(np.float32)
    save_untouch_nii_gz(
        NiftiVolume(inverse_jacobian.astype(np.float32), target_bfc.affine, inverse_jacobian_header, target_bfc.path),
        f"{subbasename}.svreg.inv.jacobian.nii.gz",
    )

    forward_jacobian = _map_jacobian_determinant(forward_volume.img)
    forward_jacobian_header = source_pvc.header.copy()
    forward_jacobian_header.set_data_dtype(np.float32)
    save_untouch_nii_gz(
        NiftiVolume(forward_jacobian.astype(np.float32), source_pvc.affine, forward_jacobian_header, source_pvc.path),
        f"{subbasename}.svreg.jacobian.nii.gz",
    )
    return inverse_map


def svreg_volreg(subbasename: str, atlas_name: str, *varargin: str) -> None:
    subbasename = normalize_fileprefix(subbasename)
    atlas_name = normalize_fileprefix(atlas_name)
    tmp_base = f"{subbasename}.svreg.tmp/{Path(subbasename).name}"

    subj_bfc = load_nii_z(f"{subbasename}.bfc.nii.gz")
    subj_pvc = load_nii_z(f"{subbasename}.pvc.frac.nii.gz")
    atlas_bfc = load_nii_z(f"{tmp_base}.target.bfc.nii.gz")
    atlas_label = load_nii_z(f"{tmp_base}.target.label.nii.gz")
    atlas_pvc = load_nii_z(f"{tmp_base}.target.pvc.frac.nii.gz")

    def _read_surface(preferred: str, fallback: str):
        path = preferred if Path(preferred).is_file() else fallback
        return parameterize_surface_uv(readdfs(path))

    sub_left = _read_surface(f"{tmp_base}.left.pial.cortex.reg.dfs", f"{subbasename}.left.pial.cortex.dfs")
    sub_right = _read_surface(f"{tmp_base}.right.pial.cortex.reg.dfs", f"{subbasename}.right.pial.cortex.dfs")
    atlas_left = _read_surface(f"{tmp_base}.target.left.pial.cortex.reg.dfs", f"{atlas_name}.left.pial.cortex.dfs")
    atlas_right = _read_surface(f"{tmp_base}.target.right.pial.cortex.reg.dfs", f"{atlas_name}.right.pial.cortex.dfs")
    sub_left_inner = _read_surface(f"{tmp_base}.left.inner.cortex.reg.dfs", f"{subbasename}.left.inner.cortex.dfs")
    sub_right_inner = _read_surface(f"{tmp_base}.right.inner.cortex.reg.dfs", f"{subbasename}.right.inner.cortex.dfs")
    atlas_left_inner = _read_surface(f"{tmp_base}.target.left.inner.cortex.reg.dfs", f"{atlas_name}.left.inner.cortex.dfs")
    atlas_right_inner = _read_surface(f"{tmp_base}.target.right.inner.cortex.reg.dfs", f"{atlas_name}.right.inner.cortex.dfs")
    sub_left_mid = _read_surface(f"{tmp_base}.left.mid.cortex.reg.dfs", f"{subbasename}.left.mid.cortex.svreg.init.dfs")
    sub_right_mid = _read_surface(f"{tmp_base}.right.mid.cortex.reg.dfs", f"{subbasename}.right.mid.cortex.svreg.init.dfs")
    atlas_left_mid = _read_surface(f"{tmp_base}.target.left.mid.cortex.reg.dfs", f"{atlas_name}.left.mid.cortex.dfs")
    atlas_right_mid = _read_surface(f"{tmp_base}.target.right.mid.cortex.reg.dfs", f"{atlas_name}.right.mid.cortex.dfs")

    transform = estimate_affine_from_pial_surfaces(sub_left, sub_right, atlas_left, atlas_right)

    affine_map = build_affine_map(subj_bfc, atlas_bfc, transform)
    dws_write_nii(f"{tmp_base}.affine.map.nii.gz", affine_map, subj_bfc)
    surfreg_map = np.asarray(affine_map, dtype=np.float32).copy()

    sub_brain = _load_unitball_brain(f"{tmp_base}_unitball_map.mat")
    atlas_brain = _load_unitball_brain(f"{tmp_base}.target_unitball_map.mat")
    subject_mask_index = np.asarray(sub_brain["mask_indx"], dtype=np.int64).reshape(-1) - 1
    atlas_mask_index = np.asarray(atlas_brain["mask_indx"], dtype=np.int64).reshape(-1) - 1
    sub_size = tuple(int(v) for v in np.asarray(sub_brain["Msize"]).reshape(-1))
    atlas_size = tuple(int(v) for v in np.asarray(atlas_brain["Msize"]).reshape(-1))
    if sub_size != subj_bfc.shape[:3] or atlas_size != atlas_bfc.shape[:3]:
        raise ValueError("unit-ball mask dimensions do not match subject/atlas NIfTI dimensions")
    sub_coordinates = np.column_stack(np.unravel_index(subject_mask_index, sub_size, order="F"))
    atlas_coordinates = np.column_stack(np.unravel_index(atlas_mask_index, atlas_size, order="F")) + 1.0
    subject_unitball = np.asarray(sub_brain["volmap"], dtype=np.float64).reshape(-1, 3)
    atlas_unitball = np.asarray(atlas_brain["volmap"], dtype=np.float64).reshape(-1, 3)
    if subject_unitball.shape[0] != subject_mask_index.size or atlas_unitball.shape[0] != atlas_mask_index.size:
        raise ValueError("unit-ball coordinate count does not match its cortex mask")
    mapped_volume_coordinates = _local_affine_scattered_interpolation(
        atlas_unitball, atlas_coordinates, subject_unitball, refine_with_simplex=True
    )
    surfreg_map[tuple(sub_coordinates.T)] = mapped_volume_coordinates.astype(np.float32)

    subject_surface_map = np.asarray(sub_brain["surf_map"], dtype=np.float64).reshape(-1, 3)
    atlas_surface_map = np.asarray(atlas_brain["surf_map"], dtype=np.float64).reshape(-1, 3)
    left_count = len(sub_left_inner.vertices)
    right_count = len(sub_right_inner.vertices)
    if subject_surface_map.shape[0] != left_count + right_count:
        raise ValueError("subject Brain.surf_map vertex count does not match inner surfaces")
    atlas_left_count = len(atlas_left_inner.vertices)
    atlas_right_count = len(atlas_right_inner.vertices)
    if atlas_surface_map.shape[0] != atlas_left_count + atlas_right_count:
        raise ValueError("target Brain.surf_map vertex count does not match inner surfaces")

    def _mapped_surface_values(left_values: np.ndarray, right_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        atlas_values = np.vstack((left_values, right_values))
        if atlas_values.shape[0] != atlas_surface_map.shape[0]:
            raise ValueError("target surface value count does not match target unit-ball map")
        # For the unit-sphere surface clouds, MATLAB's scattered interpolant
        # extrapolates to the nearest target vertex at subject query points.
        # This avoids both a global Delaunay triangulation and a slower local
        # affine fit that would not match mygriddata3's nearest fallback.
        mapped = _interpolate_unit_sphere_surface_values(
            atlas_surface_map, atlas_values, subject_surface_map
        )
        if mapped is None:
            mapped = _local_affine_scattered_interpolation(
                atlas_surface_map,
                atlas_values,
                subject_surface_map,
                hull_fallback=True,
                refine_with_simplex=True,
            )
        return mapped[:left_count], mapped[left_count:]

    atlas_pial_values = _mapped_surface_values(atlas_left.vertices, atlas_right.vertices)
    atlas_inner_values = _mapped_surface_values(atlas_left_inner.vertices, atlas_right_inner.vertices)
    atlas_mid_values = _mapped_surface_values(
        0.5 * (atlas_left.vertices + atlas_left_inner.vertices),
        0.5 * (atlas_right.vertices + atlas_right_inner.vertices),
    )
    surface_correspondences = (
        (sub_left, atlas_pial_values[0]),
        (sub_right, atlas_pial_values[1]),
        (sub_left_inner, atlas_inner_values[0]),
        (sub_right_inner, atlas_inner_values[1]),
        (sub_left_mid, atlas_mid_values[0]),
        (sub_right_mid, atlas_mid_values[1]),
    )
    for subject_surface, mapped_vertices in surface_correspondences:
        surfreg_map = overwrite_map_with_surface_correspondence(
            surfreg_map,
            np.asarray(subject_surface.vertices, dtype=np.float64),
            mapped_vertices,
            subj_bfc.zooms[:3],
            atlas_bfc.zooms[:3],
        )

    dws_write_nii(f"{tmp_base}.surfreg.map.nii.gz", surfreg_map, subj_bfc)
    surfreg_labels = warp_volume_from_map(atlas_label.img, surfreg_map, order=0, cval=0).astype(np.int16)
    label_header = subj_pvc.header.copy()
    label_header.set_data_dtype(np.int16)
    label_volume = NiftiVolume(surfreg_labels, subj_pvc.affine, label_header, subj_pvc.path)
    save_untouch_nii_gz(label_volume, f"{tmp_base}.label.surfreg.nii.gz")
    save_untouch_nii_gz(label_volume, f"{tmp_base}.svreg.label.nii.gz")
    shutil.copyfile(f"{tmp_base}.svreg.label.nii.gz", f"{subbasename}.svreg.init.label.nii.gz")

    flag_string = "".join(str(flag) for flag in varargin)
    data_fit_match = re.search(r"-A([0-9]+(?:\.[0-9]*)?)", flag_string)
    data_fit = float(data_fit_match.group(1)) if data_fit_match else 50.0
    surfreg_map = extend_deformation_laplacian_hippo(
        subbasename, tmp_base, surfreg_map, affine_map, subj_bfc, data_fit=data_fit
    )
    initial_surfreg_pvc = warp_volume_from_map(atlas_pvc.img, surfreg_map, order=1, cval=0.0).astype(np.float32)
    initial_surfreg_pvc_volume = NiftiVolume(
        initial_surfreg_pvc, subj_pvc.affine, subj_pvc.header.copy(), subj_pvc.path
    )
    save_untouch_nii_gz(initial_surfreg_pvc_volume, f"{tmp_base}.surfreg.nii.gz")
    dws_write_nii(f"{tmp_base}.surfreg.map.nii.gz", surfreg_map, subj_bfc)
    surfreg_map = fix_surfreg_map(subbasename, tmp_base, surfreg_map, subj_bfc, atlas_bfc)
    dws_write_nii(f"{tmp_base}.surfreg.map.nii.gz", surfreg_map, subj_bfc)
    make_map_better(tmp_base, surfreg_map, subj_bfc, initial_surfreg_pvc_volume)
    dws_write_nii(f"{tmp_base}.svreg.map.nii.gz", surfreg_map, subj_bfc)
    shutil.copyfile(f"{tmp_base}.svreg.map.nii.gz", f"{subbasename}.svreg.map.nii.gz")

    surfreg_pvc = warp_volume_from_map(atlas_pvc.img, surfreg_map, order=1, cval=0.0).astype(np.float32)
    surfreg_pvc_volume = NiftiVolume(surfreg_pvc, subj_pvc.affine, subj_pvc.header.copy(), subj_pvc.path)
    save_untouch_nii_gz(surfreg_pvc_volume, f"{tmp_base}.surfreg2.nii.gz")
    nonlinear_flag = re.search(r"-H([0-9]+(?:\.[0-9]*)?)?", flag_string)
    iterations = int(float(nonlinear_flag.group(1))) if nonlinear_flag and nonlinear_flag.group(1) else (600 if nonlinear_flag else 100)
    svreg_elastic_vol_reg_nonlin_cg(subbasename, tmp_base, iterations=iterations)
    inv_svreg_map(subbasename, tmp_base, subj_bfc, atlas_bfc, subj_pvc)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2:
        print("usage: svreg_volreg.py SUBJECT_PREFIX ATLAS_PREFIX [flags ...]")
        return 1
    svreg_volreg(argv[0], argv[1], *argv[2:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
