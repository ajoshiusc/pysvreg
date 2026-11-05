"""Python replacement for the SVReg partial-volume cortical thickness solver."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
from scipy.io import savemat
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter, map_coordinates
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import LinearOperator, cg
from scipy.spatial import cKDTree

from . import native
from .runtime import normalize_fileprefix


def _sphere(radius: int) -> np.ndarray:
    axes = np.arange(-radius, radius + 1)
    x, y, z = np.meshgrid(axes, axes, axes, indexing="ij")
    return x * x + y * y + z * z <= radius * radius


def _upsample_surface(surface, area_threshold: float, iterations: int):
    out = native._surface_copy(surface)
    for _ in range(iterations):
        faces = np.asarray(out.faces, dtype=np.int32)
        vertices = np.asarray(out.vertices, dtype=np.float32)
        if not len(faces):
            break
        tri = vertices[faces]
        area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        split = area > area_threshold
        if not np.any(split):
            break
        old_faces = faces[~split]
        split_faces = faces[split]
        centers = vertices[split_faces].mean(axis=1)
        new_indices = np.arange(len(vertices), len(vertices) + len(centers), dtype=np.int32)
        a, b, c = split_faces.T
        new_faces = np.vstack((np.column_stack((b, new_indices, a)),
                               np.column_stack((c, new_indices, b)),
                               np.column_stack((a, new_indices, c))))
        out.vertices = np.vstack((vertices, centers)).astype(np.float32)
        out.faces = np.vstack((old_faces, new_faces)).astype(np.int32)
        if hasattr(out, "labels"):
            labels = np.asarray(out.labels)
            new_labels = np.median(labels[split_faces], axis=1).astype(labels.dtype)
            out.labels = np.concatenate((labels, new_labels))
    return out


def _surface_mask(vertices: np.ndarray, shape: tuple[int, int, int], resolution: float, grid_min: np.ndarray) -> np.ndarray:
    scaled = np.asarray(vertices, dtype=np.float64) / resolution
    rounded = np.sign(scaled) * np.floor(np.abs(scaled) + 0.5)
    vox = rounded.astype(np.int64) - grid_min[None, :]
    valid = np.all((vox >= 0) & (vox < np.asarray(shape)[None, :]), axis=1)
    vox = vox[valid]
    mask = np.zeros(shape, dtype=bool)
    if len(vox):
        mask[vox[:, 0], vox[:, 1], vox[:, 2]] = True
    return mask


def _sample_comp_grid(
    volume: np.ndarray,
    volume_zooms: np.ndarray,
    grid_min: np.ndarray,
    grid_shape: tuple[int, int, int],
    resolution: float,
    *,
    cval: float,
    order: int,
    chunk_depth: int = 16,
) -> np.ndarray:
    # MATLAB's `interpn` promotes its single precision and integer inputs to
    # double.  Keep the heat-solver grid in double precision: rounding the
    # potential to float32 creates artificial zero gradients that explode the
    # derived thickness values.
    out = np.empty(grid_shape, dtype=np.float64)
    x = (grid_min[0] + np.arange(grid_shape[0])) * resolution / volume_zooms[0]
    y = (grid_min[1] + np.arange(grid_shape[1])) * resolution / volume_zooms[1]
    xx, yy = np.meshgrid(x, y, indexing="ij")
    for z0 in range(0, grid_shape[2], chunk_depth):
        z1 = min(z0 + chunk_depth, grid_shape[2])
        z = (grid_min[2] + np.arange(z0, z1)) * resolution / volume_zooms[2]
        zz = np.broadcast_to(z[None, None, :], (grid_shape[0], grid_shape[1], len(z)))
        coords = np.stack((np.broadcast_to(xx[:, :, None], zz.shape),
                           np.broadcast_to(yy[:, :, None], zz.shape), zz))
        out[:, :, z0:z1] = map_coordinates(
            np.asarray(volume, dtype=np.float64), coords, order=order,
            mode="constant", cval=cval, prefilter=False,
        )
    return out


def _difference_matrix(mask: np.ndarray, weights: np.ndarray, resolution: float, *, central: bool,
                       old_forward: bool = False):
    shape = tuple(int(v) for v in mask.shape)
    total = int(np.prod(shape))
    mask_f = np.asarray(mask, dtype=bool).ravel(order="F")
    weights_f = np.asarray(weights, dtype=np.float64).ravel(order="F")
    increments = (1, shape[0], shape[0] * shape[1])
    rows_all: list[np.ndarray] = []
    cols_all: list[np.ndarray] = []
    vals_all: list[np.ndarray] = []
    row_offset = 0
    for axis, increment in enumerate(increments):
        if central:
            valid_grid = np.zeros(shape, dtype=bool)
            center = [slice(None)] * 3
            before = [slice(None)] * 3
            after = [slice(None)] * 3
            center[axis] = slice(1, -1)
            before[axis] = slice(0, -2)
            after[axis] = slice(2, None)
            valid_grid[tuple(center)] = mask[tuple(center)] & mask[tuple(before)] & mask[tuple(after)]
            centers = np.flatnonzero(valid_grid.ravel(order="F"))
            if not len(centers):
                continue
            coeff = weights_f[centers] * (0.5 / resolution)
            row = row_offset + np.arange(len(centers), dtype=np.int64)
            rows_all.extend((row, row))
            cols_all.extend((centers + increment, centers - increment))
            vals_all.extend((coeff, -coeff))
            row_offset += len(centers)
        else:
            valid_grid = np.zeros(shape, dtype=bool)
            start = [slice(None)] * 3
            stop = [slice(None)] * 3
            start[axis] = slice(0, -1)
            stop[axis] = slice(1, None)
            valid_grid[tuple(start)] = mask[tuple(start)] & mask[tuple(stop)]
            centers = np.flatnonzero(valid_grid.ravel(order="F"))
            if not len(centers):
                continue
            if old_forward:
                coeff = weights_f[centers] / resolution
            else:
                coeff = (weights_f[centers] + weights_f[centers + increment]) * (0.5 / resolution)
            row = row_offset + np.arange(len(centers), dtype=np.int64)
            rows_all.extend((row, row))
            cols_all.extend((centers, centers + increment))
            vals_all.extend((coeff, -coeff))
            row_offset += len(centers)
    if not row_offset:
        return coo_matrix((0, total), dtype=np.float64).tocsr()
    rows = np.concatenate(rows_all)
    cols = np.concatenate(cols_all)
    values = np.concatenate(vals_all).astype(np.float64, copy=False)
    return coo_matrix((values, (rows, cols)), shape=(row_offset, total)).tocsr()


def _heat_solve(mask: np.ndarray, heat: np.ndarray, gm_fraction: np.ndarray, *, resolution: float,
                central: bool, central_weight: float, max_iterations: int) -> tuple[np.ndarray, int]:
    shape = tuple(int(v) for v in mask.shape)
    nvox = int(np.prod(shape))
    known = (mask & (heat > 0)).ravel(order="F")
    unknown = (mask & (heat == 0)).ravel(order="F")
    known_ids = np.flatnonzero(known)
    unknown_ids = np.flatnonzero(unknown)
    heat_f = np.asarray(heat, dtype=np.float64).ravel(order="F")
    result = np.zeros(nvox, dtype=np.float64)
    result[known_ids] = heat_f[known_ids]
    if not len(unknown_ids):
        return result.reshape(shape, order="F"), 0

    weights = np.sqrt(1.0 / np.maximum(gm_fraction, 1e-4)).astype(np.float64, copy=False)
    if central:
        dcentral = _difference_matrix(mask, weights, resolution, central=True)
        if central_weight < 1e10:
            dforward = _difference_matrix(mask, weights, resolution, central=False)
            dcentral = dcentral * central_weight
            derivative = __import__("scipy.sparse", fromlist=["vstack"]).vstack((dcentral, dforward), format="csr")
        else:
            derivative = dcentral
    else:
        derivative = _difference_matrix(mask, weights, resolution, central=False, old_forward=True)
    if derivative.shape[0] == 0:
        result[unknown_ids] = 1.5
        return result.reshape(shape, order="F"), 0

    d_unknown = derivative[:, unknown_ids].tocsr()
    d_known = derivative[:, known_ids]
    rhs = -(d_known @ heat_f[known_ids])
    del d_known, derivative
    ata = (d_unknown.T @ d_unknown).tocsr()
    atb = np.asarray(d_unknown.T @ rhs, dtype=np.float64).ravel()
    del d_unknown, rhs
    diag = np.asarray(ata.diagonal(), dtype=np.float64)
    diag[~np.isfinite(diag) | (diag <= 1e-20)] = 1.0
    preconditioner = LinearOperator(ata.shape, matvec=lambda x: x / diag, dtype=np.float64)
    count = [0]
    def count_iteration(_):
        count[0] += 1
    initial = np.full(len(unknown_ids), 1.5, dtype=np.float64)
    solution, info = cg(
        ata.astype(np.float64), atb, x0=initial, rtol=1e-9, atol=0.0,
        maxiter=max_iterations, M=preconditioner, callback=count_iteration,
    )
    if info > 0:
        print(f"PCG reached the iteration limit ({info})", file=sys.stderr)
    elif info < 0:
        raise RuntimeError("PCG failed while solving the heat equation")
    result[unknown_ids] = solution
    return result.reshape(shape, order="F"), count[0]


def _cube_triangles(case: int, face_choices: int) -> tuple[tuple[int, int, int], ...]:
    """Build edge-triangle topology for one marching-cubes case.

    Face contour connectivity is determined from the face-center value for
    checkerboard faces, avoiding the fixed diagonal artifacts of tetrahedral
    decomposition. Polygon loops are triangulated as fans.
    """
    faces = (
        ((0, 1, 2, 3), (0, 1, 2, 3)),
        ((4, 5, 6, 7), (4, 5, 6, 7)),
        ((0, 1, 5, 4), (0, 9, 4, 8)),
        ((1, 2, 6, 5), (1, 10, 5, 9)),
        ((3, 2, 6, 7), (2, 10, 6, 11)),
        ((0, 3, 7, 4), (3, 11, 7, 8)),
    )
    graph: dict[int, list[int]] = {}
    for face_id, (corners, edges) in enumerate(faces):
        active = [k for k in range(4) if ((case >> corners[k]) & 1) != ((case >> corners[(k + 1) % 4]) & 1)]
        pairs: tuple[tuple[int, int], ...]
        if len(active) == 2:
            pairs = ((edges[active[0]], edges[active[1]]),)
        elif len(active) == 4:
            if (face_choices >> face_id) & 1:
                pairs = ((edges[3], edges[0]), (edges[1], edges[2]))
            else:
                pairs = ((edges[0], edges[1]), (edges[2], edges[3]))
        else:
            continue
        for first, second in pairs:
            graph.setdefault(first, []).append(second)
            graph.setdefault(second, []).append(first)

    triangles: list[tuple[int, int, int]] = []
    unused = set(graph)
    while unused:
        start = min(unused)
        loop = [start]
        previous = -1
        current = start
        for _ in range(14):
            neighbors = graph[current]
            following = neighbors[0] if neighbors[0] != previous else neighbors[1]
            if following == start:
                break
            loop.append(following)
            previous, current = current, following
        unused.difference_update(loop)
        for k in range(1, len(loop) - 1):
            triangles.append((loop[0], loop[k], loop[k + 1]))
    return tuple(triangles)


def _isosurface(volume: np.ndarray, level: float, grid_min: np.ndarray, resolution: float,
                thickness: np.ndarray):
    """Extract a marching-cubes style isosurface in bounded z chunks."""
    # These offsets participate in Python slice-bound arithmetic below. NumPy
    # 2.x rejects converting dimensions larger than 127 to int8 there.
    corners = np.asarray(((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                          (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)), dtype=np.int16)
    edge_corners = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
                    (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
    face_corners = ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
                    (1, 2, 6, 5), (3, 2, 6, 7), (0, 3, 7, 4))
    vertices_parts: list[np.ndarray] = []
    faces_parts: list[np.ndarray] = []
    vertex_offset = 0
    n0, n1, n2 = volume.shape
    for z0 in range(0, n2 - 1, 12):
        z1 = min(z0 + 12, n2 - 1)
        cube_values = np.stack([
            volume[o[0]:n0 - 1 + o[0], o[1]:n1 - 1 + o[1], z0 + o[2]:z1 + o[2]]
            for o in corners
        ], axis=-1)
        inside = cube_values < level
        case = np.sum(inside.astype(np.uint16) << np.arange(8, dtype=np.uint16), axis=-1)
        crossing = (case != 0) & (case != 255)
        if not np.any(crossing):
            continue
        topology_key = case.astype(np.uint16)
        for face_id, face in enumerate(face_corners):
            signs = inside[..., face]
            ambiguous = (signs[..., 0] == signs[..., 2]) & (signs[..., 1] == signs[..., 3]) & (signs[..., 0] != signs[..., 1])
            center_inside = cube_values[..., face].mean(axis=-1) < level
            choose_alt = ambiguous & (center_inside != signs[..., 0])
            topology_key |= choose_alt.astype(np.uint16) << (8 + face_id)
        cell_coords_all = np.argwhere(crossing)
        cell_coords_all[:, 2] += z0
        keys = topology_key[crossing]
        values_all = cube_values[crossing].astype(np.float64)
        for key in np.unique(keys):
            case_id = int(key & 0xFF)
            choice_bits = int(key >> 8)
            triangles = _cube_triangles(case_id, choice_bits)
            if not triangles:
                continue
            selected = np.flatnonzero(keys == key)
            coords = cell_coords_all[selected].astype(np.float64)
            values = values_all[selected]
            edge_points: dict[int, np.ndarray] = {}
            for edge_id in set(edge for tri in triangles for edge in tri):
                a, b = edge_corners[edge_id]
                va, vb = values[:, a], values[:, b]
                t = np.divide(level - va, vb - va, out=np.zeros_like(va), where=vb != va)
                edge_points[edge_id] = coords + corners[a] + t[:, None] * (corners[b] - corners[a])
            mesh_points = np.stack([
                np.stack([edge_points[edge] for edge in tri], axis=1)
                for tri in triangles
            ], axis=1).reshape(-1, 3, 3)
            mesh_points = (mesh_points + grid_min[None, None, :]) * resolution
            ntri = mesh_points.shape[0] * mesh_points.shape[1]
            mesh_faces = np.arange(ntri, dtype=np.int64).reshape(-1, 3) + vertex_offset
            vertices_parts.append(mesh_points.reshape(-1, 3))
            faces_parts.append(mesh_faces)
            vertex_offset += ntri
    if not vertices_parts:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32), np.zeros(0, dtype=np.float32)
    raw_vertices = np.vstack(vertices_parts)
    raw_faces = np.vstack(faces_parts)
    rounded = np.round(raw_vertices, decimals=6)
    points, inverse = np.unique(rounded, axis=0, return_inverse=True)
    faces = inverse[raw_faces].astype(np.int32)
    nondegenerate = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    faces = faces[nondegenerate]
    idx = points / resolution - grid_min[None, :]
    attrs = map_coordinates(thickness.astype(np.float64), idx.T, order=1, mode="nearest", prefilter=False)
    return points.astype(np.float32), faces, attrs.astype(np.float32)


def _jet_colors(values: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(values, dtype=np.float64) / 6.0, 0, 1)
    # MATLAB jet(256), as used by colorDFS, samples half a colormap step in
    # from each endpoint and quantizes its ramps to 64 levels.
    t = np.clip((np.arange(256, dtype=np.float64) + 0.5) / 255.0, 0, 1)
    lut = np.column_stack((np.clip(1.5 - np.abs(4 * t - 3), 0, 1),
                           np.clip(1.5 - np.abs(4 * t - 2), 0, 1),
                           np.clip(1.5 - np.abs(4 * t - 1), 0, 1)))
    lut = np.rint(lut * 64.0) / 64.0
    pos = x * 255.0
    lower = np.floor(pos).astype(np.int32)
    upper = np.minimum(lower + 1, 255)
    frac = (pos - lower)[:, None]
    return (lut[lower] * (1.0 - frac) + lut[upper] * frac).astype(np.float32)


def _save_surface_with_thickness(path: Path, surface, values: np.ndarray) -> None:
    surface.attributes = np.asarray(values, dtype=np.float32).reshape(-1)
    surface.vColor = _jet_colors(surface.attributes)
    native.writedfs(str(path), surface)


def _write_thickness_meshes(prefix: Path, points: np.ndarray, faces: np.ndarray, point_values: np.ndarray) -> None:
    iso = type("Surface", (), {})()
    iso.vertices = points
    iso.faces = faces
    iso.attributes = point_values
    iso.vColor = _jet_colors(point_values)
    native.writedfs(f"{prefix}.pvc-thickness_0-6mm.isosurface.dfs", iso)

    inner = native.readdfs(f"{prefix}.inner.cortex.dfs")
    pial = native.readdfs(f"{prefix}.pial.cortex.dfs")
    if inner.vertices.shape != pial.vertices.shape:
        print("Inner and pial surface sizes differ; skipping cortical thickness surface output", file=sys.stderr)
        return
    mid = native._surface_copy(inner)
    mid.vertices = (np.asarray(inner.vertices) + np.asarray(pial.vertices)) * 0.5
    if len(points):
        nearest = cKDTree(points)
        _, idx = nearest.query(mid.vertices, k=1, workers=-1)
        values = point_values[idx]
    else:
        values = np.zeros(len(mid.vertices), dtype=np.float32)
    _save_surface_with_thickness(Path(f"{prefix}.pvc-thickness_0-6mm.mid.cortex.dfs"), mid, values)

    whole_tree = cKDTree(np.asarray(inner.vertices, dtype=np.float64))
    for hemi in ("left", "right"):
        inner_h = native.readdfs(f"{prefix}.{hemi}.inner.cortex.dfs")
        pial_h = native.readdfs(f"{prefix}.{hemi}.pial.cortex.dfs")
        hemi_mid = native._surface_copy(inner_h)
        hemi_mid.vertices = (np.asarray(inner_h.vertices) + np.asarray(pial_h.vertices)) * 0.5
        _, idx = whole_tree.query(np.asarray(inner_h.vertices, dtype=np.float64), k=1, workers=-1)
        hemi_values = values[idx]
        hemi_mid = native.smooth_cortex_fast(hemi_mid, 0.5, 2000)[0]
        _save_surface_with_thickness(
            Path(f"{prefix}.pvc-thickness_0-6mm.{hemi}.mid.cortex.dfs"), hemi_mid, hemi_values
        )


def thickness_pvc(args: list[str]) -> int:
    if not args:
        print("usage: thicknessPVC SUBJECT_PREFIX [MAX_PCG_ITER] [VOXEL_RES_MM] [INNER_ERODE_MM] [-cN|-f]", file=sys.stderr)
        return 2
    prefix = Path(normalize_fileprefix(args[0]))
    max_iterations = int(args[1]) if len(args) > 1 else 6000
    resolution = float(args[2]) if len(args) > 2 else 0.48
    erode_mm = float(args[3]) if len(args) > 3 else 1.0
    flag = args[4] if len(args) > 4 else "-c"
    central = not flag.startswith("-f")
    central_weight = float(flag[2:]) if flag.startswith("-c") and len(flag) > 2 else 1.7
    if resolution <= 0 or max_iterations < 1:
        raise ValueError("voxel resolution and PCG iteration count must be positive")

    with open(f"{prefix}.thickness.log", "a", encoding="utf-8") as log:
        log.write(f"{datetime.now().isoformat(timespec='seconds')}:thicknessPVC {' '.join(args)}\n")
    print("Loading surfaces and tissue fractions")
    inner = native.readdfs(f"{prefix}.inner.cortex.dfs")
    pial = native.readdfs(f"{prefix}.pial.cortex.dfs")
    inner_mask = native.load_nii_z(f"{prefix}.cortex.dewisp.mask.nii.gz")
    cerebrum = native.load_nii_z(f"{prefix}.cerebrum.mask.nii.gz")
    pvc = native.load_nii_z(f"{prefix}.pvc.frac.nii.gz")
    pvc_data = np.asarray(pvc.img, dtype=np.float64)
    gm_fraction = np.zeros_like(pvc_data, dtype=np.float64)
    low = (pvc_data > 1) & (pvc_data <= 2)
    high = (pvc_data > 2) & (pvc_data <= 3)
    gm_fraction[low] = pvc_data[low] - 1
    gm_fraction[high] = 3 - pvc_data[high]
    native.save_untouch_nii_gz(
        native.NiftiVolume(gm_fraction, pvc.affine, pvc.header.copy(), pvc.path),
        f"{prefix}.GM_frac.nii.gz",
    )
    gm_fraction = np.maximum(gm_fraction, 1e-4)

    pial = _upsample_surface(pial, (resolution * resolution) / 3.0, 6)
    all_vertices = np.vstack((np.asarray(inner.vertices), np.asarray(pial.vertices)))
    grid_min = np.floor(np.min(all_vertices, axis=0) / resolution - 10).astype(np.int64)
    grid_max = np.ceil(np.max(all_vertices, axis=0) / resolution + 10).astype(np.int64)
    grid_shape = tuple(int(x) for x in grid_max - grid_min + 1)
    if min(grid_shape) < 3:
        raise ValueError(f"computation volume too small: {grid_shape}")
    print(f"Computational grid: {grid_shape[0]} x {grid_shape[1]} x {grid_shape[2]} at {resolution:g} mm")
    pial_volume = _surface_mask(pial.vertices, grid_shape, resolution, grid_min)

    volume_zooms = np.asarray(pvc.zooms[:3], dtype=np.float64)
    gm_vol = _sample_comp_grid(gm_fraction, volume_zooms, grid_min, grid_shape, resolution, cval=1e-4, order=1)
    gm_vol = np.maximum(gm_vol, 1e-4)
    se1 = _sphere(1)
    inner_smooth = gaussian_filter((inner_mask.img > 0).astype(np.float64), sigma=0.7, mode="nearest", radius=3)
    inner_vol = _sample_comp_grid(inner_smooth, volume_zooms, grid_min, grid_shape, resolution, cval=0, order=1) > 0.5
    erode_vox = int(np.ceil(erode_mm / resolution))
    for _ in range(erode_vox):
        eroded = binary_erosion(inner_vol, structure=se1, border_value=0)
        inner_vol = ((inner_vol & ~eroded) & (gm_vol < 1e-3)) | eroded
    cerebrum_smooth = gaussian_filter((cerebrum.img > 0).astype(np.float64), sigma=0.7, mode="nearest", radius=3)
    cerebrum_vol = _sample_comp_grid(cerebrum_smooth, volume_zooms, grid_min, grid_shape, resolution, cval=0, order=1) > 0.5
    cerebrum_vol = binary_dilation(cerebrum_vol, structure=se1)
    compute = binary_dilation(cerebrum_vol & ~inner_vol, structure=_sphere(3))
    heat = np.zeros(grid_shape, dtype=np.float64)
    heat[~cerebrum_vol] = 2
    heat[inner_vol] = 1
    heat[pial_volume] = 2

    print("Solving the heat equation")
    heat_map, iterations = _heat_solve(
        compute, heat, gm_vol, resolution=resolution, central=central,
        central_weight=central_weight, max_iterations=max_iterations,
    )
    print(f"PCG iterations: {iterations}")
    heat_map[~cerebrum_vol] = 2
    heat_map[inner_vol] = 1

    mask_ref = native.load_nii_z(f"{prefix}.cortex.dewisp.mask.nii.gz")
    back = np.empty(mask_ref.shape, dtype=np.float64)
    in_shape = mask_ref.shape
    axes = [np.arange(in_shape[i], dtype=np.float64) * volume_zooms[i] / resolution - grid_min[i] for i in range(3)]
    xx, yy = np.meshgrid(axes[0], axes[1], indexing="ij")
    for z0 in range(0, in_shape[2], 16):
        z1 = min(z0 + 16, in_shape[2])
        zz = np.broadcast_to(axes[2][None, None, z0:z1], (in_shape[0], in_shape[1], z1 - z0))
        coords = np.stack((np.broadcast_to(xx[:, :, None], zz.shape), np.broadcast_to(yy[:, :, None], zz.shape), zz))
        back[:, :, z0:z1] = map_coordinates(heat_map, coords, order=1, mode="constant", cval=2, prefilter=False)
    heat_header = mask_ref.header.copy()
    heat_header.set_data_dtype(np.float64)
    native.save_untouch_nii_gz(
        native.NiftiVolume(back, mask_ref.affine, heat_header, mask_ref.path),
        f"{prefix}.heat_map_sol.nii.gz",
    )

    grads = np.gradient(heat_map, resolution, resolution, resolution)
    grad_mag = np.sqrt(sum(g * g for g in grads))
    thickness = np.divide(gm_vol, grad_mag, out=np.zeros_like(gm_vol), where=grad_mag > 0)
    thickness[~np.isfinite(thickness)] = 0
    points, faces, point_values = _isosurface(heat_map, 1.5, grid_min, resolution, thickness)
    _write_thickness_meshes(prefix, points, faces, point_values)
    axes = [(grid_min[i] + np.arange(grid_shape[i])) * resolution for i in range(3)]
    savemat(
        f"{prefix}.heat_sol_comp.mat",
        {
            "volComp_x": np.broadcast_to(axes[0][:, None, None], grid_shape),
            "volComp_y": np.broadcast_to(axes[1][None, :, None], grid_shape),
            "volComp_z": np.broadcast_to(axes[2][None, None, :], grid_shape),
            "heat_map": heat_map.astype(np.float64),
            "thickness_vol": thickness.astype(np.float64),
            "vox_res_comp": resolution,
            "gopts": {"central": central, "centralWt": central_weight},
        },
        do_compression=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    return thickness_pvc(list(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    raise SystemExit(main())

