from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import colorsys
import math
import shutil
import xml.etree.ElementTree as ET

try:
    from . import _deps  # noqa: F401
    from .dfsio import readdfs, writedfs
except ImportError:
    import _deps  # noqa: F401
    from dfsio import readdfs, writedfs

import nibabel as nib
import numpy as np
from scipy.io import savemat
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.ndimage import (
    binary_dilation,
    binary_fill_holes,
    gaussian_filter,
    map_coordinates,
    median_filter,
)
from scipy.sparse import csr_matrix, diags, hstack, vstack
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.sparse.linalg import LinearOperator, cg, factorized, spsolve, splu
from scipy.spatial import cKDTree


@dataclass
class NiftiVolume:
    img: np.ndarray
    affine: np.ndarray
    header: nib.nifti1.Nifti1Header
    path: str | None = None

    @property
    def zooms(self) -> tuple[float, ...]:
        return tuple(float(v) for v in self.header.get_zooms()[: self.img.ndim])

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(v) for v in self.img.shape)


def matlab_round(values: np.ndarray | float) -> np.ndarray:
    """Round like MATLAB, sending exact half values away from zero."""
    values = np.asarray(values, dtype=np.float64)
    return np.copysign(np.floor(np.abs(values) + 0.5), values).astype(np.int64)


def _surface_copy(surface):
    out = SimpleNamespace()
    for key, value in vars(surface).items():
        if isinstance(value, np.ndarray):
            setattr(out, key, value.copy())
        else:
            setattr(out, key, value)
    return out


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_air_warp(warp_file: str | Path):
    warp_path = Path(warp_file)
    if not warp_path.is_file():
        raise FileNotFoundError(f"missing AIR warp file: {warp_path}")
    raw = warp_path.read_bytes()
    if len(raw) < 368:
        raise ValueError(f"invalid AIR warp file (short header): {warp_path}")
    order, n_coefficients, dimension = np.frombuffer(raw, dtype="<i4", count=3)
    order, n_coefficients, dimension = int(order), int(n_coefficients), int(dimension)
    if order < 0 or dimension != 3 or n_coefficients != math.comb(order + 3, 3):
        raise ValueError(
            f"unsupported AIR polynomial header in {warp_path}: "
            f"order={order}, coefficients={n_coefficients}, dimension={dimension}"
        )
    coefficients_size = dimension * n_coefficients * 8
    if len(raw) < 12 + coefficients_size:
        raise ValueError(f"invalid AIR warp file (truncated coefficients): {warp_path}")
    coefficients = np.frombuffer(
        raw,
        dtype="<f8",
        count=dimension * n_coefficients,
        offset=len(raw) - coefficients_size,
    ).reshape(dimension, n_coefficients)
    standard_dimensions = np.frombuffer(raw, dtype="<i4", count=3, offset=148).copy()
    standard_voxel_size = np.frombuffer(raw, dtype="<f8", count=3, offset=160).copy()
    reslice_dimensions = np.frombuffer(raw, dtype="<i4", count=3, offset=316).copy()
    reslice_voxel_size = np.frombuffer(raw, dtype="<f8", count=3, offset=328).copy()
    if (
        np.any(standard_dimensions <= 0)
        or np.any(reslice_dimensions <= 0)
        or not np.all(np.isfinite(standard_voxel_size))
        or np.any(standard_voxel_size <= 0)
        or not np.all(np.isfinite(reslice_voxel_size))
        or np.any(reslice_voxel_size <= 0)
    ):
        raise ValueError(f"invalid AIR image geometry in {warp_path}")
    return (
        order,
        coefficients,
        standard_dimensions,
        standard_voxel_size,
        reslice_dimensions,
        reslice_voxel_size,
    )


def _air_polynomial_basis(voxel_coordinates: np.ndarray, order: int) -> np.ndarray:
    terms = []
    for degree in range(order + 1):
        for z_power in range(degree + 1):
            for y_power in range(degree - z_power + 1):
                x_power = degree - z_power - y_power
                terms.append((x_power, y_power, z_power))
    basis = np.empty((len(voxel_coordinates), len(terms)), dtype=np.float64)
    for index, (xp, yp, zp) in enumerate(terms):
        basis[:, index] = (
            np.power(voxel_coordinates[:, 0], xp)
            * np.power(voxel_coordinates[:, 1], yp)
            * np.power(voxel_coordinates[:, 2], zp)
        )
    return basis


def air_warp_points(
    points: np.ndarray,
    warp_file: str | Path,
    *,
    input_ucf: str | Path | None = None,
    output_ucf: str | Path | None = None,
    already_rounded: bool = False,
) -> np.ndarray:
    """Apply an AIR polynomial warp to physical XYZ points.

    AIR stores polynomial coefficients at the end of the binary ``.warp``
    file. Coefficients act on standard-file voxel coordinates, while UCF
    points use real units. The standard voxel sizes are applied before
    evaluating the polynomial and the reslice voxel sizes are applied to its
    result. Terms are ordered by increasing total degree, then increasing Z
    exponent and decreasing X exponent. SVReg rounds UCF inputs to six decimal
    places; this is retained unless the caller has already read a serialized
    UCF file. ``input_ucf`` and ``output_ucf`` remain accepted for source
    compatibility, but no temporary files or AIR executable are needed.
    """
    warp_path = Path(warp_file)
    (
        order,
        coefficients,
        _,
        standard_voxel_size,
        _,
        reslice_voxel_size,
    ) = _read_air_warp(warp_path)

    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {pts.shape}")
    if not already_rounded:
        pts = np.floor(pts * 1_000_000.0) / 1_000_000.0
    pts = pts / standard_voxel_size

    basis = _air_polynomial_basis(pts, order)
    return (basis @ coefficients.T) * reslice_voxel_size


def air_warp_coord_volume(data: np.ndarray, warp_file: str | Path) -> np.ndarray:
    """Reslice a scalar volume through an AIR polynomial warp.

    The input grid is the warp's reslice file and the output grid is its
    standard file. This is the trilinear AIR behavior used by SVReg's
    ``warp_coord_vol`` calls for coordinate-valued images.
    """
    (
        order,
        coefficients,
        standard_dimensions,
        _,
        reslice_dimensions,
        _,
    ) = _read_air_warp(warp_file)
    values = np.asarray(data)
    if values.ndim != 3 or tuple(values.shape) != tuple(reslice_dimensions):
        raise ValueError(
            f"AIR reslice input shape must be {tuple(reslice_dimensions)}, got {values.shape}"
        )

    from scipy.ndimage import map_coordinates

    output = np.empty(tuple(standard_dimensions), dtype=np.int16)
    nx, ny, nz = (int(v) for v in standard_dimensions)
    chunk_depth = max(1, min(nz, 1_000_000 // (nx * ny)))
    x_axis = np.arange(nx, dtype=np.float64)
    y_axis = np.arange(ny, dtype=np.float64)
    terms = []
    for degree in range(order + 1):
        for z_power in range(degree + 1):
            for y_power in range(degree - z_power + 1):
                terms.append((degree - z_power - y_power, y_power, z_power))
    x_powers = [np.power(x_axis, p) for p in range(order + 1)]
    y_powers = [np.power(y_axis, p) for p in range(order + 1)]
    polynomial_xy = np.zeros((nx, ny, order + 1, 3), dtype=np.float64)
    for index, (x_power, y_power, z_power) in enumerate(terms):
        xy_term = x_powers[x_power][:, None] * y_powers[y_power][None, :]
        polynomial_xy[:, :, z_power, :] += xy_term[:, :, None] * coefficients[:, index]
    z_exponents = np.arange(order + 1, dtype=np.int32)
    # AIR's warp_coord_vol wrapper passes scale=1 to AIR_do_reslice_warp,
    # whose output convention is twice the sampled voxel value.  Ask SciPy to
    # retain the interpolation result in float64 so rounding occurs only after
    # this scale, matching AIR's signed-short conversion (floor(x + 0.5)).
    # AIR treats this helper's voxel intensities as nonnegative values (the
    # SVReg caller supplies uint16 coordinate fields). Preserve uint16 inputs
    # and clamp negative signed-short values as the AIR reslicer does.
    source = np.maximum(np.asarray(values), 0)
    for z_start in range(0, nz, chunk_depth):
        z_stop = min(z_start + chunk_depth, nz)
        target_shape = (nx, ny, z_stop - z_start)
        z_values = np.arange(z_start, z_stop, dtype=np.float64)
        z_powers = np.power(z_values[:, None], z_exponents[None, :])
        warped_coordinates = np.einsum(
            "xyka,zk->xyza", polynomial_xy, z_powers, optimize=True
        )
        sampled = map_coordinates(
            source,
            warped_coordinates.reshape(-1, 3).T,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
            output=np.float64,
        )
        output[:, :, z_start:z_stop] = (
            sampled.reshape(target_shape) * 2.0 + 0.5
        ).astype(np.int16, copy=False)
    return output


def air_warp_surface_guidance(
    surface,
    warp_file: str | Path,
    *,
    input_ucf: str | Path | None = None,
    output_ucf: str | Path | None = None,
) -> SimpleNamespace:
    warped = _surface_copy(surface)
    warped.vertices = air_warp_points(
        np.asarray(surface.vertices, dtype=np.float64),
        warp_file,
        input_ucf=input_ucf,
        output_ucf=output_ucf,
    ).astype(np.float32)
    return warped


def load_nii_z(fname: str) -> NiftiVolume:
    path = Path(fname)
    if path.suffix == ".gz":
        path = path.with_suffix("")

    candidate = path if path.is_file() else path.with_suffix(path.suffix + ".gz")
    if not candidate.is_file():
        raise FileNotFoundError(f"file:{path}.gz doesn't exist. exiting")

    img = nib.load(str(candidate))
    return NiftiVolume(
        img=np.asanyarray(img.dataobj),
        affine=img.affine.copy(),
        header=img.header.copy(),
        path=str(candidate),
    )


def _dtype_from_datatype_code(datatype_code: int | None, default: np.dtype) -> np.dtype:
    mapping = {
        2: np.uint8,
        4: np.int16,
        8: np.int32,
        16: np.float32,
        32: np.complex64,
        64: np.float64,
    }
    if datatype_code is None:
        return np.dtype(default)
    return np.dtype(mapping.get(int(datatype_code), default))


def save_untouch_nii_gz(volume: NiftiVolume, saved_file_name: str) -> str:
    out_path = Path(saved_file_name)
    datatype = volume.header.get_data_dtype()
    image = np.asarray(volume.img, dtype=datatype)
    header = volume.header.copy()
    header.set_data_dtype(image.dtype)
    header.set_data_shape(image.shape)
    nib.save(nib.Nifti1Image(image, volume.affine, header=header), str(out_path))
    return str(out_path)


def dws_write_nii(ofname: str, image: np.ndarray, reference: NiftiVolume | nib.nifti1.Nifti1Image) -> str:
    if isinstance(reference, NiftiVolume):
        affine = reference.affine
        header = reference.header.copy()
    else:
        affine = reference.affine.copy()
        header = reference.header.copy()

    data = np.asarray(image)
    header.set_data_dtype(data.dtype)
    header.set_data_shape(data.shape)
    nib.save(nib.Nifti1Image(data, affine, header=header), ofname)
    return ofname


def tri_area(faces: np.ndarray, verts: np.ndarray) -> np.ndarray:
    r12 = verts[faces[:, 0], :]
    r13 = verts[faces[:, 2], :] - r12
    r12 = verts[faces[:, 1], :] - r12
    return 0.5 * np.linalg.norm(np.cross(r12, r13), axis=1)


def euler_number_mesh(surface) -> int:
    nv = surface.vertices.shape[0]
    edges = np.vstack(
        [
            surface.faces[:, [0, 1]],
            surface.faces[:, [1, 2]],
            surface.faces[:, [0, 2]],
        ]
    )
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    ne = edges.shape[0]
    nf = surface.faces.shape[0]
    return int(nf - ne + nv)


def combine_surf(surfaces: list) -> SimpleNamespace:
    out = SimpleNamespace(faces=[], vertices=[], vColor=[], labels=[])
    vnum = 0
    for surface in surfaces:
        out.vertices.append(surface.vertices)
        if hasattr(surface, "labels"):
            out.labels.append(np.asarray(surface.labels))
        if hasattr(surface, "vColor"):
            out.vColor.append(np.asarray(surface.vColor))
        out.faces.append(np.asarray(surface.faces) + vnum)
        vnum += surface.vertices.shape[0]

    out.vertices = np.vstack(out.vertices) if out.vertices else np.zeros((0, 3), dtype=np.float32)
    out.faces = np.vstack(out.faces).astype(np.int32) if out.faces else np.zeros((0, 3), dtype=np.int32)
    if out.labels:
        out.labels = np.concatenate(out.labels).astype(np.uint16)
    else:
        delattr(out, "labels")
    if out.vColor:
        out.vColor = np.vstack(out.vColor).astype(np.float32)
    else:
        delattr(out, "vColor")
    return out


def calc_mid_surface(inner_surface, pial_surface):
    out = SimpleNamespace()
    out.faces = np.asarray(inner_surface.faces).copy()
    out.vertices = 0.5 * (np.asarray(inner_surface.vertices) + np.asarray(pial_surface.vertices))
    for field_name in ("vColor", "attributes", "labels", "u", "v", "normals"):
        if hasattr(inner_surface, field_name):
            value = getattr(inner_surface, field_name)
            setattr(out, field_name, np.asarray(value).copy())
    return out


def calc_mid_surf(inner: str, pial: str, mid: str) -> None:
    inner_surface = readdfs(inner)
    pial_surface = readdfs(pial)
    writedfs(mid, calc_mid_surface(inner_surface, pial_surface))


def vertices_connectivity_fast(surface) -> tuple[list[np.ndarray], csr_matrix]:
    faces = np.asarray(surface.faces, dtype=np.int32)
    n_vertices = int(surface.vertices.shape[0])
    rowno = np.concatenate(
        [faces[:, 0], faces[:, 0], faces[:, 1], faces[:, 1], faces[:, 2], faces[:, 2]]
    )
    colno = np.concatenate(
        [faces[:, 1], faces[:, 2], faces[:, 0], faces[:, 2], faces[:, 0], faces[:, 1]]
    )
    data = np.ones(rowno.shape[0], dtype=np.uint8)
    adjacency = csr_matrix((data, (rowno, colno)), shape=(n_vertices, n_vertices))
    adjacency.data[:] = 1
    adjacency.eliminate_zeros()
    adjacency = adjacency.maximum(adjacency.T)
    conn = [adjacency.indices[adjacency.indptr[i] : adjacency.indptr[i + 1]] for i in range(n_vertices)]
    return conn, adjacency


def face_v_conn(surface) -> csr_matrix:
    n_faces = surface.faces.shape[0]
    n_vertices = surface.vertices.shape[0]
    rows = np.concatenate([surface.faces[:, 0], surface.faces[:, 1], surface.faces[:, 2]])
    cols = np.concatenate([np.arange(n_faces), np.arange(n_faces), np.arange(n_faces)])
    data = np.ones(rows.shape[0], dtype=np.uint8)
    return csr_matrix((data, (rows, cols)), shape=(n_vertices, n_faces))


def delete_unused_vertices(surface):
    faces = np.asarray(surface.faces, dtype=np.int64)
    if faces.size == 0:
        return _surface_copy(surface), np.zeros(0, dtype=np.int64)
    used = np.unique(faces.reshape(-1))
    remap = np.full(int(surface.vertices.shape[0]), -1, dtype=np.int64)
    remap[used] = np.arange(used.shape[0], dtype=np.int64)
    out = _surface_copy(surface)
    out.vertices = np.asarray(surface.vertices)[used].copy()
    out.faces = remap[faces].astype(np.int32)
    for field_name in ("vColor", "attributes", "labels", "u", "v", "normals"):
        if hasattr(surface, field_name):
            value = np.asarray(getattr(surface, field_name))
            if value.shape[0] == surface.vertices.shape[0]:
                setattr(out, field_name, value[used].copy())
    return out, used


def keep_largest_surface_component(surface):
    out = _surface_copy(surface)
    if out.faces.size == 0:
        return out, np.arange(out.vertices.shape[0], dtype=np.int64)
    _, adjacency = vertices_connectivity_fast(out)
    n_comp, labels = connected_components(adjacency, directed=False)
    if n_comp <= 1:
        return out, np.arange(out.vertices.shape[0], dtype=np.int64)
    counts = np.bincount(labels)
    keep_label = int(np.argmax(counts))
    keep_vertices = np.flatnonzero(labels == keep_label)
    keep_mask = np.zeros(out.vertices.shape[0], dtype=bool)
    keep_mask[keep_vertices] = True
    face_mask = np.all(keep_mask[np.asarray(out.faces, dtype=np.int64)], axis=1)
    out.faces = np.asarray(out.faces, dtype=np.int32)[face_mask]
    out, used = delete_unused_vertices(out)
    return out, used


def tri2nodes(surface) -> csr_matrix:
    area = tri_area(surface.faces, surface.vertices)
    conn = face_v_conn(surface)
    weights = np.asarray(conn @ area[:, None]).ravel()
    weights[weights == 0] = 1.0
    return diags(1.0 / weights) @ conn @ diags(area)


def smooth_cortex_fast(surface, a: float, n_iterations: int, A: csr_matrix | None = None, size_preserve: bool = False):
    if A is None:
        _, adjacency = vertices_connectivity_fast(surface)
        degree = np.asarray(adjacency.sum(axis=1)).ravel()
        degree[degree == 0] = 1.0
        A = diags(a / degree) @ adjacency
        A = A + diags(np.full(surface.vertices.shape[0], 1.0 - a))

    out = _surface_copy(surface)
    vertices = np.asarray(out.vertices, dtype=np.float64)
    for _ in range(int(n_iterations)):
        vertices = A @ vertices

    if size_preserve:
        source = np.asarray(surface.vertices, dtype=np.float64)
        for axis in range(3):
            src_min = float(source[:, axis].min())
            src_max = float(source[:, axis].max())
            dst_min = float(vertices[:, axis].min())
            dst_max = float(vertices[:, axis].max())
            if dst_max > dst_min:
                vertices[:, axis] = (vertices[:, axis] - dst_min) * ((src_max - src_min) / (dst_max - dst_min)) + src_min

    out.vertices = vertices.astype(np.float32)
    return out, A


def _compute_vertex_normals(surface) -> np.ndarray:
    vertices = np.asarray(surface.vertices, dtype=np.float64)
    faces = np.asarray(surface.faces, dtype=np.int32)
    normals = np.zeros_like(vertices)
    tri = vertices[faces]
    face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for i in range(3):
        np.add.at(normals, faces[:, i], face_normals)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return -(normals / norms)


def curvature_cortex_fast(surface, sigmoid_const: float, show_sigmoid: int = 0, adjacency: csr_matrix | None = None):
    if adjacency is None:
        _, adjacency = vertices_connectivity_fast(surface)
    vertices = np.asarray(surface.vertices, dtype=np.float64)
    normals = _compute_vertex_normals(surface)

    rows, cols = adjacency.nonzero()
    edge_vectors = vertices[cols] - vertices[rows]
    edge_norms = np.linalg.norm(edge_vectors, axis=1, keepdims=True)
    edge_norms[edge_norms == 0] = 1.0
    edge_vectors = edge_vectors / edge_norms
    dots = np.einsum("ij,ij->i", normals[rows], edge_vectors)
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)

    counts = np.asarray(adjacency.sum(axis=1)).ravel()
    counts[counts == 0] = 1.0
    curvature = np.bincount(rows, weights=angles, minlength=vertices.shape[0]) / counts - (math.pi / 2.0)
    cs = 1.0 / (1.0 + np.exp(-curvature * sigmoid_const)) - 0.5
    return cs.astype(np.float32), curvature.astype(np.float32)


def patchcurvature_mean(surface) -> np.ndarray:
    """Port patchcurvature.m's two-ring local quadratic mean curvature."""
    vertices = np.asarray(surface.vertices, dtype=np.float64)
    faces = np.asarray(surface.faces, dtype=np.int64)
    n_vertices = len(vertices)
    if faces.size == 0:
        return np.zeros(n_vertices, dtype=np.float64)

    # patchnormals weights each oriented face normal by the corner angle.
    tri = vertices[faces]
    e1 = tri[:, 0] - tri[:, 1]
    e2 = tri[:, 1] - tri[:, 2]
    e3 = tri[:, 2] - tri[:, 0]

    def unit_edges(edges):
        lengths = np.linalg.norm(edges, axis=1, keepdims=True)
        lengths[lengths == 0] = 1.0
        return edges / lengths

    u1, u2, u3 = unit_edges(e1), unit_edges(e2), unit_edges(e3)
    angle_a = np.arccos(np.clip(np.einsum("ij,ij->i", u1, -u3), -1.0, 1.0))
    angle_b = np.arccos(np.clip(np.einsum("ij,ij->i", u2, -u1), -1.0, 1.0))
    angle_c = np.arccos(np.clip(np.einsum("ij,ij->i", u3, -u2), -1.0, 1.0))
    face_normal = np.cross(e1, e3)
    normals = np.zeros_like(vertices)
    for column, angle in enumerate((angle_a, angle_b, angle_c)):
        np.add.at(normals, faces[:, column], face_normal * angle[:, None])
    norm = np.linalg.norm(normals, axis=1) + np.finfo(np.float64).eps
    normals /= norm[:, None]

    _, adjacency = vertices_connectivity_fast(surface)
    neighborhoods: list[np.ndarray] = []
    for vertex in range(n_vertices):
        start, stop = adjacency.indptr[vertex : vertex + 2]
        first_ring = adjacency.indices[start:stop]
        # MATLAB uses unique([Ne{Ne{i}}]), i.e. the union of neighbors of
        # every one-ring neighbor (which includes the one-ring itself).
        neighborhoods.append(np.unique(adjacency[first_ring].indices) if first_ring.size else np.empty(0, dtype=np.int64))

    # Batch the six-term least-squares fits by vertex block. Scaling each local
    # patch before forming its normal equations keeps the small systems well
    # conditioned while avoiding one LAPACK call per cortical vertex.
    identity = np.eye(3, dtype=np.float64)
    axis = identity[np.argmin(np.abs(normals), axis=1)]
    tangent_1 = np.cross(axis, normals)
    tangent_norm = np.linalg.norm(tangent_1, axis=1)
    valid = tangent_norm > 0
    tangent_1[valid] /= tangent_norm[valid, None]
    tangent_2 = np.cross(tangent_1, normals)
    basis = np.stack((normals, tangent_1, tangent_2), axis=2)
    result = np.zeros(n_vertices, dtype=np.float64)
    batch_size = 1024
    for batch_start in range(0, n_vertices, batch_size):
        ids = np.arange(batch_start, min(batch_start + batch_size, n_vertices))
        lengths = np.fromiter((len(neighborhoods[i]) for i in ids), dtype=np.int32, count=len(ids))
        max_length = int(lengths.max(initial=0))
        if max_length < 6:
            continue
        local = np.zeros((len(ids), max_length, 3), dtype=np.float64)
        mask = np.arange(max_length)[None, :] < lengths[:, None]
        for row, vertex in enumerate(ids):
            count = int(lengths[row])
            if count == 0 or not valid[vertex]:
                continue
            offsets = vertices[neighborhoods[vertex]] - vertices[vertex]
            local[row, :count] = offsets @ basis[vertex]
        f = local[:, :, 0]
        x = local[:, :, 1]
        y = local[:, :, 2]
        scale = np.sqrt(np.sum(np.where(mask, x * x + y * y, 0.0), axis=1) / np.maximum(lengths, 1))
        scale[scale == 0] = 1.0
        f, x, y = f / scale[:, None], x / scale[:, None], y / scale[:, None]
        design = np.stack((x * x, y * y, x * y, x, y, np.ones_like(x)), axis=2)
        design *= mask[:, :, None]
        gram = np.einsum("bni,bnj->bij", design, design)
        rhs = np.einsum("bni,bn->bi", design, f * mask)
        try:
            coefficients = np.linalg.solve(gram, rhs[..., None])[..., 0]
        except np.linalg.LinAlgError:
            coefficients = np.zeros((len(ids), 6), dtype=np.float64)
            for row in range(len(ids)):
                if lengths[row] >= 6 and valid[ids[row]]:
                    coefficients[row] = np.linalg.lstsq(
                        design[row, :lengths[row]], f[row, :lengths[row]], rcond=None
                    )[0]
        # For H=[[2a,c],[c,2b]], trace(H)/2 = a+b = Cmean.
        result[ids] = (coefficients[:, 0] + coefficients[:, 1]) / scale
    return result


def smooth_surf_hierarchy(surf_name: str, num_scales: int, flags: str = "") -> None:
    surface = readdfs(surf_name)
    _, adjacency = vertices_connectivity_fast(surface)
    nit = max(1, round(5000 / max(1, num_scales)))
    surface, A = smooth_cortex_fast(surface, 0.8, 50)
    surface.attributes = curvature_cortex_fast(surface, 50, 0, adjacency)[0]
    writedfs(f"{surf_name[:-4]}_smooth1.dfs", surface)
    for jj in range(2, num_scales + 1):
        surface, _ = smooth_cortex_fast(surface, 0.4, max(1, round(nit / 4)), A)
        surface.attributes = curvature_cortex_fast(surface, 50, 0, adjacency)[0]
        writedfs(f"{surf_name[:-4]}_smooth{jj}.dfs", surface)


def corr_topology_labels(surfname: str, sub_out: str | None = None):
    surface = readdfs(surfname)
    if not hasattr(surface, "labels"):
        if sub_out and sub_out != surfname:
            writedfs(sub_out, surface)
        return surface

    adjacency = vertices_connectivity_fast(surface)[1]
    labels = np.unique(surface.labels)
    out_labels = np.asarray(surface.labels, dtype=np.int32).copy()

    for label in labels:
        label_mask = out_labels == label
        if not np.any(label_mask):
            continue
        subgraph = adjacency[label_mask][:, label_mask]
        n_comp, components = connected_components(subgraph, directed=False)
        if n_comp <= 1:
            continue
        counts = np.bincount(components)
        keep = int(np.argmax(counts))
        vertex_ids = np.flatnonzero(label_mask)
        out_labels[vertex_ids[components != keep]] = -1

    for label in labels:
        keep_mask = (out_labels == label) | (out_labels == -1)
        if not np.any(keep_mask):
            continue
        subgraph = adjacency[keep_mask][:, keep_mask]
        n_comp, components = connected_components(subgraph, directed=False)
        vertex_ids = np.flatnonzero(keep_mask)
        scores = np.zeros(n_comp, dtype=np.int32)
        for comp_idx in range(n_comp):
            scores[comp_idx] = int(np.sum(out_labels[vertex_ids[components == comp_idx]] == label))
        keep = int(np.argmax(scores))
        out_labels[vertex_ids[components == keep]] = int(label)

    out_labels[out_labels == -1] = 0
    surface.labels = out_labels.astype(np.uint16)
    out_path = sub_out or surfname
    writedfs(out_path, surface)
    return surface


def _label_color_map(xmlf: str | Path | None) -> dict[int, np.ndarray]:
    if not xmlf:
        return {}
    path = Path(xmlf)
    if not path.is_file():
        return {}

    root = ET.parse(path).getroot()
    colors: dict[int, np.ndarray] = {}
    for label in root.findall(".//label"):
        attrs = label.attrib
        if "id" not in attrs or "color" not in attrs:
            continue
        raw = attrs["color"]
        raw = raw[2:] if raw.lower().startswith("0x") else raw
        raw = raw.zfill(6)
        color = np.array(
            [int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16)],
            dtype=np.float32,
        )
        colors[int(attrs["id"])] = color / 256.0
    return colors


def get_hippo_label(xmlf: str | Path) -> list[int]:
    path = Path(xmlf)
    if not path.is_file():
        return []
    root = ET.parse(path).getroot()
    labels: list[int] = []
    for label in root.findall(".//label"):
        attrs = label.attrib
        name = attrs.get("fullname", "").lower()
        if ("hippo" in name or "amygd" in name) and ("sulc" not in name and "gyru" not in name):
            try:
                labels.append(int(attrs["id"]))
            except (KeyError, ValueError):
                pass
    return labels


def recolor_by_label(surfn: str, atlas_name: str | None = None, xmlf: str | None = None) -> None:
    surface = readdfs(surfn)
    if xmlf is None and atlas_name is not None:
        atlas_dir = Path(atlas_name).parent
        candidate = atlas_dir / "brainsuite_labeldescription.xml"
        if candidate.is_file():
            xmlf = str(candidate)
    colors = _label_color_map(xmlf)
    labels = np.unique(getattr(surface, "labels", np.array([], dtype=np.uint16)))
    labels = labels[labels != 0]

    vcolor = np.full((surface.vertices.shape[0], 3), 0.5, dtype=np.float32)
    for index, label in enumerate(labels):
        if int(label) in colors:
            color = colors[int(label)]
        else:
            color = np.array(colorsys.hsv_to_rgb(index / max(1, len(labels)), 0.75, 0.95), dtype=np.float32)
        vcolor[surface.labels == label] = color
    surface.vColor = vcolor
    writedfs(surfn, surface)


def copy_attrib_colors(sub_base: str, hemi: str | None = None, ext: str = "svreg.dfs") -> None:
    hemis = [hemi] if hemi else ["left", "right"]
    for hemisphere in hemis:
        mid_path = f"{sub_base}.{hemisphere}.mid.cortex.{ext}"
        if not Path(mid_path).is_file():
            continue
        source = readdfs(mid_path)
        for surf_type in ("inner", "pial"):
            base_path = f"{sub_base}.{hemisphere}.{surf_type}.cortex.dfs"
            if not Path(base_path).is_file():
                continue
            target = _surface_copy(source)
            target.vertices = readdfs(base_path).vertices
            writedfs(f"{sub_base}.{hemisphere}.{surf_type}.cortex.{ext}", target)


def readdfc_sipi(fname: str):
    path = Path(fname)
    with path.open("rb") as stream:
        magic = stream.read(8)
        version = np.frombuffer(stream.read(4), dtype=np.uint8)
        hdrsize = int(np.frombuffer(stream.read(4), dtype=np.int32)[0])
        data_start = int(np.frombuffer(stream.read(4), dtype=np.int32)[0])
        mdoffset = int(np.frombuffer(stream.read(4), dtype=np.int32)[0])
        pdoffset = int(np.frombuffer(stream.read(4), dtype=np.int32)[0])
        n_contours = int(np.frombuffer(stream.read(4), dtype=np.int32)[0])
        hdr = SimpleNamespace(
            magic=magic,
            version=version,
            hdrsize=hdrsize,
            dataStart=data_start,
            mdoffset=mdoffset,
            pdoffset=pdoffset,
            nContours=n_contours,
        )
        stream.seek(data_start)
        curves = []
        for _ in range(n_contours):
            nopts = int(np.frombuffer(stream.read(4), dtype=np.int32)[0])
            xyz = np.frombuffer(stream.read(12 * nopts), dtype=np.float32).reshape(nopts, 3)
            curves.append(xyz.copy())
    return curves, hdr


def writedfc(fname: str, curves: list[np.ndarray], fprotocol: str) -> None:
    protocol_data = Path(fprotocol).read_bytes()
    with Path(fname).open("wb") as stream:
        stream.write(b"DFCLE\\00")
        stream.write(bytes([0, 0, 0, 2]))
        stream.write(np.array(32, dtype=np.int32).tobytes())
        stream.write(np.array(len(protocol_data) + 36, dtype=np.int32).tobytes())
        stream.write(np.array(36, dtype=np.int32).tobytes())
        stream.write(np.array(-1, dtype=np.int32).tobytes())
        stream.write(np.array(len(curves), dtype=np.int32).tobytes())
        stream.write(np.array(len(curves), dtype=np.int32).tobytes())
        stream.write(protocol_data)
        for curve in curves:
            curve = np.asarray(curve, dtype=np.float32)
            stream.write(np.array(curve.shape[0], dtype=np.int32).tobytes())
            stream.write(curve.astype(np.float32).tobytes())


def remove_curve_defects(curve: np.ndarray) -> np.ndarray:
    """Port of Remove_Curve_Defects.m for a curve stored as N x 3 points."""
    curve = np.asarray(curve, dtype=np.float32)
    if curve.ndim != 2 or curve.shape[1] != 3:
        raise ValueError("curve must have shape (n_points, 3)")
    n_points = curve.shape[0]
    if n_points == 0:
        return curve

    # RemoveDuplicatePoints keeps the first occurrence of each coordinate,
    # including non-adjacent repeats.
    _, first = np.unique(curve, axis=0, return_index=True)
    curve = curve[np.sort(first)]
    if curve.shape[0] < 2:
        return curve

    # RemoveTraceOverDefects.m: MATLAB's gradient uses one-sided differences
    # at endpoints and centered differences in the interior.
    pdiff = np.gradient(curve, axis=0).T
    normpdiff = np.linalg.norm(pdiff, axis=0)
    bad = np.flatnonzero(normpdiff == 0)
    if bad.size:
        remove = set(int(i) for i in bad)
        n = normpdiff.size
        for center in bad:
            radius = max(0, min(int(center), n - int(center) - 2))
            for offset in range(1, radius + 1):
                if normpdiff[center + offset] == normpdiff[center - offset]:
                    remove.add(int(center + offset))
                    remove.add(int(center - offset))
        curve = curve[[i for i in range(n) if i not in remove]]
        if curve.size == 0:
            return curve.reshape(0, 3)
        _, first = np.unique(curve, axis=0, return_index=True)
        curve = curve[np.sort(first)]

    # MATLAB chooses, among the next three points, the closest one to the
    # current point, jumping over skipped points when the winner is farther
    # than the immediate successor.
    selected = [0]
    i = 0
    while i < curve.shape[0] - 1:
        end = min(3, curve.shape[0] - i - 1)
        distances = np.linalg.norm(curve[i + 1:i + 1 + end] - curve[i], axis=1)
        step = int(np.argmin(distances)) + 1
        selected.append(i + step)
        i += step if step > 1 else 1
    return curve[np.asarray(selected, dtype=np.int64)]


def _coords_from_voxels(shape: tuple[int, int, int], zooms: tuple[float, float, float]) -> np.ndarray:
    grid = np.indices(shape, dtype=np.float32)
    coords = np.stack(
        [
            grid[0] * zooms[0],
            grid[1] * zooms[1],
            grid[2] * zooms[2],
        ],
        axis=-1,
    )
    return coords


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    homo = np.c_[points, np.ones(points.shape[0], dtype=np.float64)]
    mapped = homo @ transform.T
    return mapped[:, :3]


def estimate_similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    cov = (dst_centered.T @ src_centered) / max(1, src.shape[0])
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[-1, -1] = -1
    R = U @ D @ Vt
    var = np.mean(np.sum(src_centered**2, axis=1))
    scale = 1.0 if var == 0 else float(np.trace(np.diag(S) @ D) / var)
    t = dst_mean - scale * (R @ src_mean)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = scale * R
    transform[:3, 3] = t
    return transform


def estimate_affine_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape[0] == 0 or dst.shape[0] == 0:
        return np.eye(4, dtype=np.float64)
    src_h = np.c_[src, np.ones(src.shape[0], dtype=np.float64)]
    coeffs, _, _, _ = np.linalg.lstsq(src_h, dst, rcond=None)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :] = coeffs.T
    return transform


def icp_similarity(src: np.ndarray, dst: np.ndarray, iterations: int = 12, sample_size: int = 5000) -> np.ndarray:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape[0] == 0 or dst.shape[0] == 0:
        return np.eye(4, dtype=np.float64)
    if src.shape[0] > sample_size:
        step = max(1, src.shape[0] // sample_size)
        src_sample = src[::step]
    else:
        src_sample = src
    if dst.shape[0] > sample_size:
        step = max(1, dst.shape[0] // sample_size)
        dst_sample = dst[::step]
    else:
        dst_sample = dst
    tree = cKDTree(dst_sample)
    transform = np.eye(4, dtype=np.float64)
    for _ in range(iterations):
        moved = apply_transform(src_sample, transform)
        _, idx = tree.query(moved, k=1)
        delta = estimate_similarity_transform(moved, dst_sample[idx])
        transform = delta @ transform
    return transform


def transfer_labels_by_nearest(subject_surface, atlas_surface, transform_subject_to_atlas: np.ndarray):
    out = _surface_copy(subject_surface)
    atlas_points = np.asarray(atlas_surface.vertices, dtype=np.float64)
    subject_points = apply_transform(np.asarray(subject_surface.vertices, dtype=np.float64), transform_subject_to_atlas)
    tree = cKDTree(atlas_points)
    _, idx = tree.query(subject_points, k=1)
    if hasattr(atlas_surface, "labels"):
        out.labels = np.asarray(atlas_surface.labels)[idx].astype(np.uint16)
    if hasattr(atlas_surface, "vColor"):
        out.vColor = np.asarray(atlas_surface.vColor)[idx].astype(np.float32)
    if hasattr(atlas_surface, "u"):
        out.u = np.asarray(atlas_surface.u)[idx].astype(np.float32)
        out.v = np.asarray(atlas_surface.v)[idx].astype(np.float32)
    return out


def boundary_loop(surface) -> np.ndarray:
    loops = boundary_loops(surface)
    if not loops:
        return np.zeros(0, dtype=np.int32)
    return loops[0]


def boundary_loops(surface) -> list[np.ndarray]:
    faces = np.asarray(surface.faces, dtype=np.int64)
    directed = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    undirected = np.sort(directed, axis=1)
    _, inverse, counts = np.unique(undirected, axis=0, return_inverse=True, return_counts=True)
    boundary_directed = directed[counts[inverse] == 1]
    if boundary_directed.shape[0] == 0:
        return []

    neighbors: dict[int, list[int]] = {}
    remaining: set[tuple[int, int]] = set()
    for a, b in boundary_directed:
        ia = int(a)
        ib = int(b)
        neighbors.setdefault(ia, []).append(ib)
        neighbors.setdefault(ib, []).append(ia)
        remaining.add(tuple(sorted((ia, ib))))

    loops: list[np.ndarray] = []
    while remaining:
        start_a, start_b = next(iter(remaining))
        remaining.remove((start_a, start_b))
        loop = [start_a, start_b]
        prev = start_a
        current = start_b
        while True:
            nxt = None
            for nb in neighbors.get(current, []):
                edge = tuple(sorted((current, nb)))
                if nb != prev and edge in remaining:
                    nxt = nb
                    break
            if nxt is None:
                break
            loop.append(nxt)
            remaining.remove(tuple(sorted((current, nxt))))
            prev, current = current, nxt
            if current == loop[0]:
                loop.pop()
                break
        loops.append(np.asarray(loop, dtype=np.int32))

    loops.sort(key=len, reverse=True)
    return loops


def para_curve_circle_segmt(points: np.ndarray, fix_pts: np.ndarray | list[int], p_fix: np.ndarray | list[float]) -> np.ndarray:
    fix_pts = np.asarray(fix_pts, dtype=np.int64).reshape(-1)
    p_fix = np.asarray(p_fix, dtype=np.float64).reshape(-1)
    if points.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)

    order = np.argsort(fix_pts)
    fix_pts = fix_pts[order]
    p_fix = p_fix[order]

    start = int(fix_pts[0])
    reorder = np.concatenate([np.arange(start, points.shape[0]), np.arange(0, start)])
    pts = np.asarray(points, dtype=np.float64)[reorder]
    fix_pts = fix_pts - start
    fix_pts[fix_pts < 0] += points.shape[0]

    fix_ext = np.concatenate([fix_pts, [points.shape[0]]])
    p_ext = np.concatenate([p_fix, [p_fix[0]]])
    pts_ext = np.vstack([pts, pts[:1]])
    par = np.zeros(points.shape[0] + 1, dtype=np.float64)

    for idx in range(fix_ext.shape[0] - 1):
        start_i = int(fix_ext[idx])
        stop_i = int(fix_ext[idx + 1])
        arc = pts_ext[start_i : stop_i + 1]
        seg = np.linalg.norm(arc[1:] - arc[:-1], axis=1)
        total = float(np.sum(seg))
        s1 = float(p_ext[idx])
        s2 = float(p_ext[idx + 1])
        if s2 < s1:
            s2 += 2.0 * math.pi
        if total <= 0:
            values = np.linspace(s1, s2, arc.shape[0])
        else:
            values = np.concatenate([[s1], s1 + np.cumsum(seg) * ((s2 - s1) / total)])
        par[start_i : stop_i + 1] = values

    out = np.zeros(points.shape[0], dtype=np.float64)
    out[reorder] = par[:-1]
    return out


def matlab_boundary_square_values(surface, transform: np.ndarray | None = None, orient: str = "RAS") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loop = boundary_loop(surface)
    if loop.size == 0:
        raise ValueError("surface has no boundary loop")

    points = np.asarray(surface.vertices, dtype=np.float64)
    if transform is not None:
        points = apply_transform(points, transform)
    boundary_points = points[loop]

    if "A" in orient:
        anterior_dim = orient.index("A")
        p1 = int(np.argmax(boundary_points[:, anterior_dim]))
        p2 = int(np.argmin(boundary_points[:, anterior_dim]))
    else:
        posterior_dim = orient.index("P")
        p1 = int(np.argmin(boundary_points[:, posterior_dim]))
        p2 = int(np.argmax(boundary_points[:, posterior_dim]))

    par = para_curve_circle_segmt(boundary_points, [p1, p2], [0.0, math.pi])

    if "S" in orient:
        superior_dim = orient.index("S")
        zvals = boundary_points[:, superior_dim]
        if np.mean(zvals[par < math.pi]) < np.mean(zvals[par >= math.pi]):
            par = 2.0 * math.pi - par
    else:
        inferior_dim = orient.index("I")
        zvals = boundary_points[:, inferior_dim]
        if np.mean(zvals[par < math.pi]) > np.mean(zvals[par >= math.pi]):
            par = 2.0 * math.pi - par

    xbdr = np.cos(par)
    ybdr = np.sin(par)
    xbdr = np.sign(xbdr) * (xbdr**2)
    ybdr = np.sign(ybdr) * (ybdr**2)
    square = np.column_stack([xbdr + ybdr, -xbdr + ybdr])
    return loop, square[:, 0], square[:, 1]


def harmonic_scalars_from_boundary(
    surface, boundary_vertices: np.ndarray, boundary_values: np.ndarray
) -> np.ndarray:
    """Solve one or more harmonic fields with a shared sparse factorization."""
    n_vertices = int(surface.vertices.shape[0])
    boundary_vertices = np.asarray(boundary_vertices, dtype=np.int64).reshape(-1)
    boundary_values = np.asarray(boundary_values, dtype=np.float64)
    if boundary_values.ndim == 1:
        boundary_values = boundary_values[:, None]
    values = np.zeros((n_vertices, boundary_values.shape[1]), dtype=np.float64)
    if n_vertices == 0:
        return values

    values[boundary_vertices] = boundary_values
    boundary_mask = np.zeros(n_vertices, dtype=bool)
    boundary_mask[boundary_vertices] = True
    interior = np.flatnonzero(~boundary_mask)
    if interior.size == 0:
        return values

    _, adjacency = vertices_connectivity_fast(surface)
    adjacency = adjacency.astype(np.float64)
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    lap = diags(degree) - adjacency
    lii = lap[interior][:, interior]
    lib = lap[interior][:, boundary_vertices]
    rhs = -(lib @ values[boundary_vertices])
    values[interior] = splu(lii.tocsc()).solve(np.asarray(rhs, dtype=np.float64))
    return values


def harmonic_scalar_from_boundary(surface, boundary_vertices: np.ndarray, boundary_values: np.ndarray) -> np.ndarray:
    return harmonic_scalars_from_boundary(surface, boundary_vertices, boundary_values)[:, 0]


def matlab_style_boundary_harmonic_map(
    atlas_surface,
    subject_surface,
    *,
    atlas_guidance=None,
    subject_guidance=None,
    atlas_warp_file: str | Path | None = None,
    subject_warp_file: str | Path | None = None,
    atlas_input_ucf: str | Path | None = None,
    atlas_output_ucf: str | Path | None = None,
    subject_input_ucf: str | Path | None = None,
    subject_output_ucf: str | Path | None = None,
):
    atlas_geom = _surface_copy(atlas_guidance if atlas_guidance is not None else atlas_surface)
    subject_geom = _surface_copy(subject_guidance if subject_guidance is not None else subject_surface)
    atlas_geom.faces = np.asarray(atlas_surface.faces, dtype=np.int32).copy()
    subject_geom.faces = np.asarray(subject_surface.faces, dtype=np.int32).copy()

    if atlas_warp_file is not None:
        atlas_geom = air_warp_surface_guidance(
            atlas_geom,
            atlas_warp_file,
            input_ucf=atlas_input_ucf,
            output_ucf=atlas_output_ucf,
        )
        atlas_geom.faces = np.asarray(atlas_surface.faces, dtype=np.int32).copy()
    if subject_warp_file is not None:
        subject_geom = air_warp_surface_guidance(
            subject_geom,
            subject_warp_file,
            input_ucf=subject_input_ucf,
            output_ucf=subject_output_ucf,
        )
        subject_geom.faces = np.asarray(subject_surface.faces, dtype=np.int32).copy()

    atlas_boundary, atlas_xb, atlas_yb = matlab_boundary_square_values(atlas_geom)
    subject_boundary = boundary_loop(subject_geom)
    if subject_boundary.size == 0:
        raise ValueError("subject surface has no boundary loop")
    subject_xb, subject_yb = register_cc_boundary(
        atlas_geom,
        subject_geom,
        atlas_boundary,
        subject_boundary,
        atlas_xb,
        atlas_yb,
        transform_subject_to_atlas=None,
    )

    atlas_out = _surface_copy(atlas_surface)
    subject_out = _surface_copy(subject_surface)
    atlas_uv = harmonic_scalars_from_boundary(atlas_surface, atlas_boundary, np.column_stack([atlas_xb, atlas_yb]))
    subject_uv = harmonic_scalars_from_boundary(subject_surface, subject_boundary, np.column_stack([subject_xb, subject_yb]))
    atlas_out.u = (0.5 * (1.0 + atlas_uv[:, 0])).astype(np.float32)
    atlas_out.v = (0.5 * (1.0 + atlas_uv[:, 1])).astype(np.float32)
    subject_out.u = (0.5 * (1.0 + subject_uv[:, 0])).astype(np.float32)
    subject_out.v = (0.5 * (1.0 + subject_uv[:, 1])).astype(np.float32)
    return atlas_out, subject_out


def build_elastic_map_operator(surface, p_norm: float = 2.0, mu: float = 100.0, lam: float = 0.1) -> csr_matrix:
    vertices = np.asarray(surface.vertices, dtype=np.float64)
    faces = np.asarray(surface.faces, dtype=np.int64)
    n_tri = int(faces.shape[0])
    n_vertices = int(vertices.shape[0])

    v1 = vertices[faces[:, 0]]
    v2 = vertices[faces[:, 1]]
    v3 = vertices[faces[:, 2]]
    v21 = v2 - v1
    x2 = np.linalg.norm(v21, axis=1)
    x2[x2 == 0] = 1.0
    x3 = np.einsum("ij,ij->i", v3 - v1, v21 / x2[:, None])
    normals = np.cross(v21, v3 - v1)
    yunit = np.cross(normals, v21)
    ynorm = np.linalg.norm(yunit, axis=1)
    ynorm[ynorm == 0] = 1.0
    y3 = np.einsum("ij,ij->i", yunit, v3 - v1) / ynorm

    x1 = np.zeros(n_tri, dtype=np.float64)
    y1 = np.zeros(n_tri, dtype=np.float64)
    y2 = np.zeros(n_tri, dtype=np.float64)
    sqrt_dt = 2.0 * np.abs((x1 * y2 - y1 * x2) + (x2 * y3 - y2 * x3) + (x3 * y1 - y3 * x1)) ** ((p_norm - 1.0) / p_norm)
    sqrt_dt[sqrt_dt < np.finfo(np.float64).eps] = np.finfo(np.float64).eps

    y1 = y1 / sqrt_dt
    y2 = y2 / sqrt_dt
    y3 = y3 / sqrt_dt
    x1 = x1 / sqrt_dt
    x2 = x2 / sqrt_dt
    x3 = x3 / sqrt_dt

    tmp_a = np.concatenate([y2 - y3, y3 - y1, y1 - y2])
    tmp_b = np.concatenate([x3 - x2, x1 - x3, x2 - x1])
    rows = np.concatenate([np.arange(n_tri), np.arange(n_tri), np.arange(n_tri)])
    cols = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    dx = csr_matrix((tmp_a, (rows, cols)), shape=(n_tri, n_vertices))
    dy = csr_matrix((tmp_b, (rows, cols)), shape=(n_tri, n_vertices))

    if mu != 0 or lam != 0:
        return vstack(
            [
                hstack([math.sqrt(lam) * dx, math.sqrt(lam) * dy]),
                hstack([math.sqrt(mu) * dy, math.sqrt(mu) * dx]),
                hstack([math.sqrt(2.0 * mu) * dx, csr_matrix(dx.shape)]),
                hstack([csr_matrix(dy.shape), math.sqrt(2.0 * mu) * dy]),
            ]
        ).tocsr()
    return vstack(
        [
            hstack([dx, csr_matrix(dx.shape)]),
            hstack([csr_matrix(dx.shape), dx]),
            hstack([dy, csr_matrix(dy.shape)]),
            hstack([csr_matrix(dy.shape), dy]),
        ]
    ).tocsr()


def orientation_tri(faces: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    faces = np.asarray(faces, dtype=np.int64)
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    ori = np.zeros(faces.shape[0], dtype=np.int8)
    cross_z = (
        (x[faces[:, 1]] - x[faces[:, 0]]) * (y[faces[:, 2]] - y[faces[:, 1]])
        - (y[faces[:, 1]] - y[faces[:, 0]]) * (x[faces[:, 2]] - x[faces[:, 1]])
    )
    ori[cross_z < 0] = -1
    ori[cross_z > 0] = 1
    return ori


def cpd_normalize(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, SimpleNamespace]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    normal = SimpleNamespace()
    normal.xd = np.mean(x, axis=0)
    normal.yd = np.mean(y, axis=0)
    x0 = x - normal.xd[None, :]
    y0 = y - normal.yd[None, :]
    normal.xscale = math.sqrt(float(np.sum(np.sum(x0**2, axis=1)) / max(1, x.shape[0])))
    normal.yscale = math.sqrt(float(np.sum(np.sum(y0**2, axis=1)) / max(1, y.shape[0])))
    if normal.xscale == 0:
        normal.xscale = 1.0
    if normal.yscale == 0:
        normal.yscale = 1.0
    return x0 / normal.xscale, y0 / normal.yscale, normal


def cpd_G(x: np.ndarray, y: np.ndarray, beta: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    diff = x[:, None, :] - y[None, :, :]
    return np.exp(np.sum(diff**2, axis=2) / (-2.0 * (beta**2)))


def cpd_P(x: np.ndarray, y: np.ndarray, sigma2: float, outliers: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, d = x.shape
    m = y.shape[0]
    ksig = -2.0 * max(float(sigma2), np.finfo(np.float64).eps)
    outlier_term = (outliers * m * ((-ksig * math.pi) ** (0.5 * d))) / max((1.0 - outliers) * n, np.finfo(np.float64).eps)
    diff = x[:, None, :] - y[None, :, :]
    p = np.exp(np.sum(diff**2, axis=2) / ksig)
    sp = np.sum(p, axis=1) + outlier_term
    sp = np.maximum(sp, np.finfo(np.float64).eps)
    pt1 = 1.0 - outlier_term / sp
    weights = p / sp[:, None]
    p1 = np.sum(weights, axis=0)
    px = weights.T @ x
    energy = float(-np.sum(np.log(sp)) + (d * n * math.log(max(float(sigma2), np.finfo(np.float64).eps)) / 2.0))
    return p1, pt1, px, energy


def cpd_Pcorrespondence(x: np.ndarray, y: np.ndarray, sigma2: float, outliers: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ksig = -2.0 * (max(float(sigma2), np.finfo(np.float64).eps) + 1e-3)
    diff = x[:, None, :] - y[None, :, :]
    p = np.exp(np.sum(diff**2, axis=2) / ksig)
    denom = np.sum(p, axis=1, keepdims=True)
    d = x.shape[1]
    outlier_term = (outliers * y.shape[0] * ((-ksig * math.pi) ** (0.5 * d))) / max((1.0 - outliers) * x.shape[0], np.finfo(np.float64).eps)
    denom = np.maximum(denom + outlier_term, np.finfo(np.float64).eps)
    weights = p / denom
    return np.argmax(weights, axis=0).astype(np.int64)


def cpd_rigid(
    x: np.ndarray,
    y: np.ndarray,
    *,
    rot: bool = True,
    scale: bool = True,
    max_it: int = 150,
    tol: float = 1e-5,
    outliers: float = 0.1,
    sigma2: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, int, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, d = x.shape
    m = y.shape[0]
    if sigma2 <= 0:
        sigma2 = float((m * np.trace(x.T @ x) + n * np.trace(y.T @ y) - 2.0 * np.sum(x, axis=0) @ np.sum(y, axis=0)) / max(m * n * d, 1))
    sigma2_init = sigma2
    T = y.copy()
    s = 1.0
    R = np.eye(d, dtype=np.float64)
    t = np.zeros(d, dtype=np.float64)
    iteration = 0
    ntol = tol + 10.0
    L = 0.0
    sigma2_save = sigma2
    while iteration < max_it and ntol > tol and sigma2 > 10.0 * np.finfo(np.float64).eps:
        L_old = L
        p1, pt1, px, L = cpd_P(x, T, sigma2, outliers)
        ntol = abs((L - L_old) / L) if L != 0 else 0.0
        np_sum = float(np.sum(pt1))
        if np_sum <= 0:
            break
        mu_x = (x.T @ pt1) / np_sum
        mu_y = (y.T @ p1) / np_sum
        A = px.T @ y - np_sum * np.outer(mu_x, mu_y)
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        C = np.eye(d, dtype=np.float64)
        if rot:
            C[-1, -1] = np.linalg.det(U @ Vt)
        R = U @ C @ Vt
        sigma2_save = sigma2
        if scale:
            denom = np.sum(np.sum((y**2) * p1[:, None], axis=1)) - np_sum * float(mu_y.T @ mu_y)
            denom = denom if abs(denom) > np.finfo(np.float64).eps else np.finfo(np.float64).eps
            s = float(np.trace(np.diag(S) @ C) / denom)
            sigma2 = abs(
                (
                    np.sum(np.sum((x**2) * pt1[:, None], axis=1))
                    - np_sum * float(mu_x.T @ mu_x)
                    - s * np.trace(np.diag(S) @ C)
                )
                / max(np_sum * d, np.finfo(np.float64).eps)
            )
        else:
            sigma2 = abs(
                (
                    np.sum(np.sum((x**2) * pt1[:, None], axis=1))
                    - np_sum * float(mu_x.T @ mu_x)
                    + np.sum(np.sum((y**2) * p1[:, None], axis=1))
                    - np_sum * float(mu_y.T @ mu_y)
                    - 2.0 * np.trace(np.diag(S) @ C)
                )
                / max(np_sum * d, np.finfo(np.float64).eps)
            )
        t = mu_x - s * (R @ mu_y)
        T = s * y @ R.T + t[None, :]
        iteration += 1
    C = cpd_Pcorrespondence(x, T, sigma2_save, outliers)
    return C, R, t, s, sigma2_init, iteration, T


def cpd_GRBF(
    x: np.ndarray,
    y: np.ndarray,
    *,
    beta: float = 2.0,
    lambda_: float = 3.0,
    max_it: int = 150,
    tol: float = 1e-5,
    outliers: float = 0.1,
    sigma2: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, float, int, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, d = x.shape
    m = y.shape[0]
    T = y.copy()
    W = np.zeros((m, d), dtype=np.float64)
    if sigma2 <= 0:
        sigma2 = float((m * np.trace(x.T @ x) + n * np.trace(y.T @ y) - 2.0 * np.sum(x, axis=0) @ np.sum(y, axis=0)) / max(m * n * d, 1))
    G = cpd_G(y, y, beta)
    iteration = 0
    ntol = tol + 10.0
    L = 1.0
    sigma2_save = sigma2
    eye = np.eye(m, dtype=np.float64)
    while iteration < max_it and ntol > tol and sigma2 > 1e-8:
        L_old = L
        p1, pt1, px, L = cpd_P(x, T, sigma2, outliers)
        L = float(L + (lambda_ / 2.0) * np.trace(W.T @ G @ W))
        ntol = abs((L - L_old) / L) if L != 0 else 0.0
        dP = np.diag(p1)
        W = np.linalg.solve(dP @ G + lambda_ * sigma2 * eye, px - dP @ y)
        T = y + G @ W
        np_sum = float(np.sum(p1))
        sigma2_save = sigma2
        sigma2 = abs(
            (
                np.sum(np.sum((x**2) * pt1[:, None], axis=1))
                + np.sum(np.sum((T**2) * p1[:, None], axis=1))
                - 2.0 * np.trace(px.T @ T)
            )
            / max(np_sum * d, np.finfo(np.float64).eps)
        )
        iteration += 1
    C = cpd_Pcorrespondence(x, T, sigma2_save, outliers)
    return C, W, sigma2, iteration, T


def cpd_register(
    x: np.ndarray,
    y: np.ndarray,
    *,
    method: str = "rigid",
    normalize: bool = True,
    max_it: int = 150,
    tol: float = 1e-5,
    corresp: bool = False,
    outliers: float = 0.1,
    rot: bool = True,
    scale: bool = True,
    beta: float = 2.0,
    lambda_: float = 3.0,
    sigma2: float = 0.0,
) -> tuple[SimpleNamespace, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    normal = SimpleNamespace(xd=np.zeros(x.shape[1]), yd=np.zeros(y.shape[1]), xscale=1.0, yscale=1.0)
    if normalize:
        x_norm, y_norm, normal = cpd_normalize(x, y)
    else:
        x_norm = x
        y_norm = y
    if method == "rigid":
        C, R, t, s, sigma2_out, iteration, T = cpd_rigid(
            x_norm,
            y_norm,
            rot=rot,
            scale=scale,
            max_it=max_it,
            tol=tol,
            outliers=outliers,
            sigma2=sigma2,
        )
        transform = SimpleNamespace(iter=iteration, method=method, Y=T, normal=normal, R=R, t=t, s=s)
        if normalize:
            transform.s = transform.s * (normal.xscale / normal.yscale)
            transform.t = normal.xscale * transform.t + normal.xd - transform.s * (transform.R @ normal.yd)
            transform.Y = T * normal.xscale + normal.xd[None, :]
    elif method == "nonrigid":
        C, W, sigma2_out, iteration, T = cpd_GRBF(
            x_norm,
            y_norm,
            beta=beta,
            lambda_=lambda_,
            max_it=max_it,
            tol=tol,
            outliers=outliers,
            sigma2=sigma2,
        )
        transform = SimpleNamespace(
            iter=iteration,
            method=method,
            Y=T,
            normal=normal,
            W=W,
            beta=beta,
            s=1.0,
            t=np.zeros(x.shape[1], dtype=np.float64),
            Yorig=y.copy(),
        )
        if normalize:
            transform.Y = T * normal.xscale + normal.xd[None, :]
    else:
        raise ValueError(f"unsupported CPD method: {method}")
    if not corresp:
        C = np.zeros(y.shape[0], dtype=np.int64)
    return transform, np.asarray(C, dtype=np.int64)


def mypcg_lp(A, b, tol: float, maxit: int, M_diag: np.ndarray, x: np.ndarray | None = None) -> np.ndarray:
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if x is None:
        x = np.zeros_like(b, dtype=np.float64)
    else:
        x = np.asarray(x, dtype=np.float64).reshape(-1).copy()

    M_diag = np.asarray(M_diag, dtype=np.float64).reshape(-1)
    M_diag[M_diag == 0] = 1.0
    Minv = 1.0 / M_diag

    def cast32(value: np.ndarray | float) -> np.ndarray | float:
        arr = np.asarray(value, dtype=np.float32).astype(np.float64)
        if np.isscalar(value):
            return float(arr)
        return arr

    r = b - (A @ x)
    p = Minv * r
    z = p.copy()
    p = cast32(p)
    r = cast32(r)
    rtz = float(cast32(r @ z))

    for _ in range(int(maxit)):
        if rtz < tol:
            break
        Ap = cast32(A @ p)
        denom = float(p @ Ap)
        if abs(denom) <= np.finfo(np.float64).eps:
            break
        alpha = float(cast32(rtz / denom))
        x = cast32(x + alpha * p)
        r = cast32(r - alpha * Ap)
        z = cast32(Minv * r)
        newrtz = float(cast32(r @ z))
        if abs(rtz) <= np.finfo(np.float64).eps:
            break
        beta = float(cast32(newrtz / rtz))
        rtz = float(cast32(newrtz))
        p = cast32(z + beta * p)
    return np.asarray(x, dtype=np.float64)


def _grid_sample_indices(points: np.ndarray, fraction: float = 0.05) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    n_points = points.shape[0]
    if n_points == 0:
        return np.zeros(0, dtype=np.int64)
    target = max(512, int(round(n_points * float(fraction))))
    if target >= n_points:
        return np.arange(n_points, dtype=np.int64)
    mins = np.min(points, axis=0)
    spans = np.ptp(points, axis=0)
    spans[spans == 0] = 1.0
    volume = float(np.prod(spans))
    cell = (volume / target) ** (1.0 / 3.0) if volume > 0 else 1.0
    best = np.arange(0, n_points, max(1, n_points // target), dtype=np.int64)
    best_err = abs(best.shape[0] - target)
    for scale in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
        cell_size = max(cell * scale, np.finfo(np.float64).eps)
        keys = np.floor((points - mins[None, :]) / cell_size).astype(np.int64)
        _, idx = np.unique(keys, axis=0, return_index=True)
        idx = np.sort(idx.astype(np.int64))
        err = abs(idx.shape[0] - target)
        if err < best_err:
            best = idx
            best_err = err
    if best.shape[0] > target:
        step = max(1, best.shape[0] // target)
        best = best[::step][:target]
    return np.asarray(best, dtype=np.int64)


def _knn_graph_laplacian(points: np.ndarray, k: int = 8) -> csr_matrix:
    points = np.asarray(points, dtype=np.float64)
    n_points = points.shape[0]
    if n_points == 0:
        return csr_matrix((0, 0), dtype=np.float64)
    tree = cKDTree(points)
    distances, neighbors = tree.query(points, k=min(k + 1, n_points))
    rows = np.repeat(np.arange(n_points), neighbors.shape[1] - 1)
    cols = neighbors[:, 1:].reshape(-1)
    dst = distances[:, 1:].reshape(-1)
    sigma = float(np.median(dst[dst > 0])) if np.any(dst > 0) else 1.0
    sigma = max(sigma, np.finfo(np.float64).eps)
    weights = np.exp(-(dst**2) / (2.0 * sigma * sigma))
    W = csr_matrix((weights, (rows, cols)), shape=(n_points, n_points))
    W = 0.5 * (W + W.T)
    degree = np.asarray(W.sum(axis=1)).ravel()
    return diags(degree) - W


def reducepatch_subset(surface, fraction: float = 0.05):
    surface = _surface_copy(surface)
    vertices = np.asarray(surface.vertices, dtype=np.float64)
    faces = np.asarray(surface.faces, dtype=np.int64)
    if vertices.shape[0] == 0 or faces.shape[0] == 0:
        surface.faces = np.zeros((0, 3), dtype=np.int32)
        return surface, np.zeros(0, dtype=np.int64)

    rep_idx = _grid_sample_indices(vertices, fraction=fraction)
    rep_vertices = vertices[rep_idx]
    tree = cKDTree(rep_vertices)
    _, cluster = tree.query(vertices, k=1)
    reduced_faces = cluster[faces]
    keep = (
        (reduced_faces[:, 0] != reduced_faces[:, 1])
        & (reduced_faces[:, 0] != reduced_faces[:, 2])
        & (reduced_faces[:, 1] != reduced_faces[:, 2])
    )
    reduced_faces = reduced_faces[keep]
    if reduced_faces.shape[0] == 0:
        step = max(1, int(round(1.0 / max(float(fraction), np.finfo(np.float64).eps))))
        rep_idx = np.arange(0, vertices.shape[0], step, dtype=np.int64)
        rep_vertices = vertices[rep_idx]
        remap = np.full(vertices.shape[0], -1, dtype=np.int64)
        remap[rep_idx] = np.arange(rep_idx.shape[0], dtype=np.int64)
        keep = np.all(remap[faces] >= 0, axis=1)
        reduced_faces = remap[faces[keep]]

    if reduced_faces.shape[0] == 0:
        surface.vertices = rep_vertices.astype(np.float32)
        surface.faces = np.zeros((0, 3), dtype=np.int32)
        return surface, rep_idx

    face_keys = np.sort(reduced_faces, axis=1)
    _, unique_idx = np.unique(face_keys, axis=0, return_index=True)
    reduced_faces = reduced_faces[np.sort(unique_idx)].astype(np.int32)

    out = _surface_copy(surface)
    out.vertices = rep_vertices.astype(np.float32)
    out.faces = reduced_faces
    out, used = delete_unused_vertices(out)
    rep_idx = rep_idx[used]
    out, used2 = keep_largest_surface_component(out)
    rep_idx = rep_idx[used2]
    return out, rep_idx


def get_stiffness_matrix_tri_wt(surface, weights: np.ndarray | None = None):
    vertices = np.asarray(surface.vertices, dtype=np.float64)
    faces = np.asarray(surface.faces, dtype=np.int64)
    n_tri = int(faces.shape[0])
    n_vertices = int(vertices.shape[0])
    if weights is None:
        weights = np.ones(n_vertices, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)

    v1 = vertices[faces[:, 0]]
    v2 = vertices[faces[:, 1]]
    v3 = vertices[faces[:, 2]]
    v21 = v2 - v1
    x2 = np.linalg.norm(v21, axis=1)
    x2[x2 == 0] = 1.0
    x3 = np.einsum("ij,ij->i", v3 - v1, v21 / x2[:, None])
    normals = np.cross(v21, v3 - v1)
    yunit = np.cross(normals, v21)
    ynorm = np.linalg.norm(yunit, axis=1)
    ynorm[ynorm == 0] = 1.0
    y3 = np.einsum("ij,ij->i", yunit, v3 - v1) / ynorm

    x1 = np.zeros(n_tri, dtype=np.float64)
    y1 = np.zeros(n_tri, dtype=np.float64)
    y2 = np.zeros(n_tri, dtype=np.float64)
    sqrt_dt = np.abs((x1 * y2 - y1 * x2) + (x2 * y3 - y2 * x3) + (x3 * y1 - y3 * x1))
    sqrt_dt[sqrt_dt < np.finfo(np.float64).eps] = np.finfo(np.float64).eps
    areas = 0.5 * sqrt_dt

    y1 = y1 / sqrt_dt
    y2 = y2 / sqrt_dt
    y3 = y3 / sqrt_dt
    x1 = x1 / sqrt_dt
    x2 = x2 / sqrt_dt
    x3 = x3 / sqrt_dt

    tmp_a = np.concatenate([y2 - y3, y3 - y1, y1 - y2])
    tmp_b = np.concatenate([x3 - x2, x1 - x3, x2 - x1])
    rows = np.concatenate([np.arange(n_tri), np.arange(n_tri), np.arange(n_tri)])
    cols = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    Dx = csr_matrix((tmp_a, (rows, cols)), shape=(n_tri, n_vertices))
    Dy = csr_matrix((tmp_b, (rows, cols)), shape=(n_tri, n_vertices))

    tc = face_v_conn(surface)
    tri_weights = np.asarray((tc.T @ weights[:, None])).ravel() / 3.0
    Wt = diags(tri_weights * areas)
    S = (Dx.T @ Wt @ Dx) + (Dy.T @ Wt @ Dy)
    return S.tocsr(), Dx.tocsr(), Dy.tocsr()


def loreta(surface) -> csr_matrix:
    S, _, _ = get_stiffness_matrix_tri_wt(surface, np.ones(surface.vertices.shape[0], dtype=np.float64))
    return S


def l2_surface_matching(
    atlas_surface,
    subject_surface,
    *,
    fraction: float = 0.05,
    niter: int = 62,
    step_size: float = 0.6,
    lap_reg: float = 25.0,
    corr_step: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    atlas_vertices = np.asarray(atlas_surface.vertices, dtype=np.float64)
    subject_vertices = np.asarray(subject_surface.vertices, dtype=np.float64)
    if atlas_vertices.shape[0] == 0 or subject_vertices.shape[0] == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

    atlas_reduced, atlas_orig_idx = reducepatch_subset(atlas_surface, fraction=fraction)
    if atlas_reduced.faces.shape[0] == 0 or atlas_reduced.vertices.shape[0] < 3:
        atlas_idx = _grid_sample_indices(atlas_vertices, fraction=fraction)
        moving = atlas_vertices[atlas_idx].copy()
        subject_tree = cKDTree(subject_vertices)
        _, nearest = subject_tree.query(moving, k=1)
        atlas_corr = atlas_idx[:: max(1, int(corr_step))]
        subject_corr = nearest[:: max(1, int(corr_step))].astype(np.int64)
        return np.asarray(atlas_corr, dtype=np.int64), subject_corr

    moving = np.asarray(atlas_reduced.vertices, dtype=np.float64).copy()
    smoothness = loreta(atlas_reduced)
    n_points = moving.shape[0]
    identity = diags(np.ones(n_points, dtype=np.float64))
    system = vstack([lap_reg * smoothness, identity]).tocsr()
    normal_matrix = (system.T @ system).tocsr()
    M_diag = np.asarray(normal_matrix.diagonal()).ravel() + np.finfo(np.float64).eps
    rhs_prefix = np.zeros(smoothness.shape[0], dtype=np.float64)
    subject_tree = cKDTree(subject_vertices)

    for _ in range(int(niter)):
        _, nearest = subject_tree.query(moving, k=1)
        vec = subject_vertices[nearest] - moving
        bx = system.T @ np.concatenate([rhs_prefix, vec[:, 0]])
        by = system.T @ np.concatenate([rhs_prefix, vec[:, 1]])
        bz = system.T @ np.concatenate([rhs_prefix, vec[:, 2]])
        vx = mypcg_lp(normal_matrix, bx, 1e-100, 300, M_diag)
        vy = mypcg_lp(normal_matrix, by, 1e-100, 300, M_diag)
        vz = mypcg_lp(normal_matrix, bz, 1e-100, 300, M_diag)
        moving += float(step_size) * np.column_stack([vx, vy, vz])

    _, nearest = subject_tree.query(moving, k=1)
    atlas_corr = np.asarray(atlas_orig_idx, dtype=np.int64)[:: max(1, int(corr_step))]
    subject_corr = nearest[:: max(1, int(corr_step))].astype(np.int64)
    return atlas_corr, subject_corr


def register_cc_boundary(
    atlas_surface,
    subject_surface,
    atlas_boundary: np.ndarray,
    subject_boundary: np.ndarray,
    atlas_xb: np.ndarray,
    atlas_yb: np.ndarray,
    transform_subject_to_atlas: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    atlas_vertices = np.asarray(atlas_surface.vertices, dtype=np.float64)
    subject_vertices = np.asarray(subject_surface.vertices, dtype=np.float64)
    if transform_subject_to_atlas is not None:
        subject_vertices = apply_transform(subject_vertices, transform_subject_to_atlas)
    atlas_yz = atlas_vertices[np.asarray(atlas_boundary, dtype=np.int64)][:, [1, 2]]
    subject_yz = subject_vertices[np.asarray(subject_boundary, dtype=np.int64)][:, [1, 2]]
    rigid_tr, _ = cpd_register(atlas_yz, subject_yz, method="rigid", corresp=False, normalize=True, max_it=150, tol=1e-5, outliers=0.1)
    rigid_y = np.asarray(rigid_tr.Y, dtype=np.float32).astype(np.float64)
    nonrigid_tr, _ = cpd_register(
        atlas_yz,
        rigid_y,
        method="nonrigid",
        corresp=True,
        normalize=True,
        max_it=150,
        tol=1e-5,
        outliers=0.1,
        beta=2.0,
        lambda_=3.0,
    )
    subject_reg = np.asarray(nonrigid_tr.Y, dtype=np.float32).astype(np.float64)
    xbdr2 = interpolate_points_to_values(atlas_yz, np.asarray(atlas_xb, dtype=np.float64), subject_reg, method="nearest")
    ybdr2 = interpolate_points_to_values(atlas_yz, np.asarray(atlas_yb, dtype=np.float64), subject_reg, method="nearest")
    return np.asarray(xbdr2, dtype=np.float64), np.asarray(ybdr2, dtype=np.float64)


def _solve_coupled_square_system(
    surface1,
    surface2,
    boundary1: np.ndarray,
    boundary2: np.ndarray,
    xbdr1: np.ndarray,
    ybdr1: np.ndarray,
    xbdr2: np.ndarray,
    ybdr2: np.ndarray,
    corr1: np.ndarray,
    corr2: np.ndarray,
    sigma: np.ndarray,
    xinit1: np.ndarray,
    yinit1: np.ndarray,
    xinit2: np.ndarray,
    yinit2: np.ndarray,
    *,
    mu: float = 100.0,
    lam: float = 0.1,
    maxiter: int = 2600,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n1 = int(surface1.vertices.shape[0])
    n2 = int(surface2.vertices.shape[0])
    a1 = build_elastic_map_operator(surface1, p_norm=2.0, mu=mu, lam=lam)
    a2 = build_elastic_map_operator(surface2, p_norm=2.0, mu=mu, lam=lam)
    system = vstack(
        [
            hstack([a1, csr_matrix((a1.shape[0], 2 * n2))]),
            hstack([csr_matrix((a2.shape[0], 2 * n1)), a2]),
        ]
    ).tocsr()

    corr1 = np.asarray(corr1, dtype=np.int64).reshape(-1)
    corr2 = np.asarray(corr2, dtype=np.int64).reshape(-1)
    sigma = np.asarray(sigma, dtype=np.float64).reshape(-1)
    if corr1.size:
        rows = np.concatenate(
            [
                np.arange(corr1.shape[0]),
                np.arange(corr1.shape[0]),
                np.arange(corr1.shape[0], 2 * corr1.shape[0]),
                np.arange(corr1.shape[0], 2 * corr1.shape[0]),
            ]
        )
        cols = np.concatenate([corr1, 2 * n1 + corr2, n1 + corr1, 2 * n1 + n2 + corr2])
        data = np.concatenate([sigma, -sigma, sigma, -sigma])
        penalty = csr_matrix((data, (rows, cols)), shape=(2 * corr1.shape[0], 2 * n1 + 2 * n2))
        system = vstack([system, penalty]).tocsr()

    pinned = np.concatenate([boundary1, n1 + boundary1, 2 * n1 + boundary2, 2 * n1 + n2 + boundary2]).astype(np.int64)
    pinned_values = np.concatenate([xbdr1, ybdr1, xbdr2, ybdr2]).astype(np.float64)
    free_mask = np.ones(2 * n1 + 2 * n2, dtype=bool)
    free_mask[pinned] = False
    system_free = system[:, free_mask].tocsr()
    rhs = -(system[:, pinned].tocsr() @ pinned_values)
    target = np.asarray(system_free.T @ rhs).ravel()
    diag = np.asarray(system_free.multiply(system_free).sum(axis=0)).ravel()
    diag[diag == 0] = 1.0
    initial = np.concatenate([xinit1, yinit1, xinit2, yinit2]).astype(np.float64)[free_mask]
    lhs = LinearOperator(
        (target.shape[0], target.shape[0]),
        matvec=lambda x: np.asarray(system_free.T @ (system_free @ x)).ravel(),
        dtype=np.float64,
    )
    # MATLAB uses mypcg_lp on the normal equations with the harmonic map as
    # the initial iterate. Reuse the same ported routine here instead of
    # SciPy CG so the solve behavior matches more closely.
    solution = mypcg_lp(lhs, target, 1e-112, maxiter, diag, initial)
    full = np.zeros(2 * n1 + 2 * n2, dtype=np.float64)
    full[free_mask] = solution
    full[pinned] = pinned_values
    return full[:n1], full[n1 : 2 * n1], full[2 * n1 : 2 * n1 + n2], full[2 * n1 + n2 :]


def coupled_square_map_from_guidance(
    atlas_surface,
    subject_surface,
    atlas_guidance=None,
    subject_guidance=None,
    transform_subject_to_atlas: np.ndarray | None = None,
    corr_step: int = 4,
    sigma: float = 10.0,
    mu: float = 100.0,
    lam: float = 0.1,
    maxiter: int = 1800,
    solve_cycles: int = 1,
    initial_map: str = "harmonic",
    l2_iterations: int = 80,
):
    atlas_geom = _surface_copy(atlas_guidance if atlas_guidance is not None else atlas_surface)
    subject_geom = _surface_copy(subject_guidance if subject_guidance is not None else subject_surface)
    atlas_geom.faces = np.asarray(atlas_surface.faces, dtype=np.int32).copy()
    subject_geom.faces = np.asarray(subject_surface.faces, dtype=np.int32).copy()

    if atlas_geom.vertices.shape[0] != atlas_surface.vertices.shape[0]:
        atlas_geom = _surface_copy(atlas_surface)
    if subject_geom.vertices.shape[0] != subject_surface.vertices.shape[0]:
        subject_geom = _surface_copy(subject_surface)

    if transform_subject_to_atlas is None:
        transform_subject_to_atlas = icp_similarity(
            np.asarray(subject_geom.vertices, dtype=np.float64),
            np.asarray(atlas_geom.vertices, dtype=np.float64),
        )

    atlas_boundary, atlas_xb, atlas_yb = matlab_boundary_square_values(atlas_geom)
    subject_boundary = boundary_loop(subject_geom)
    if subject_boundary.size == 0:
        raise ValueError("subject surface has no boundary loop")
    subject_xb, subject_yb = register_cc_boundary(
        atlas_geom,
        subject_geom,
        atlas_boundary,
        subject_boundary,
        atlas_xb,
        atlas_yb,
        transform_subject_to_atlas=transform_subject_to_atlas,
    )

    atlas_uv0 = harmonic_scalars_from_boundary(atlas_geom, atlas_boundary, np.column_stack([atlas_xb, atlas_yb]))
    subject_uv0 = harmonic_scalars_from_boundary(subject_geom, subject_boundary, np.column_stack([subject_xb, subject_yb]))
    atlas_x0, atlas_y0 = atlas_uv0.T
    subject_x0, subject_y0 = subject_uv0.T

    subject_match = _surface_copy(subject_geom)
    subject_match.vertices = apply_transform(np.asarray(subject_geom.vertices, dtype=np.float64), transform_subject_to_atlas).astype(np.float32)
    corr_atlas, corr_subject = l2_surface_matching(
        atlas_geom,
        subject_match,
        fraction=0.05,
        niter=int(l2_iterations),
        step_size=0.6,
        lap_reg=25.0,
        corr_step=max(1, int(corr_step)),
    )
    sigma_vec = np.full(corr_atlas.shape[0], float(sigma), dtype=np.float64)
    if initial_map == "harmonic":
        x1, y1, x2, y2 = atlas_x0, atlas_y0, subject_x0, subject_y0
    elif initial_map == "zero":
        x1 = np.zeros(atlas_geom.vertices.shape[0], dtype=np.float64)
        y1 = np.zeros(atlas_geom.vertices.shape[0], dtype=np.float64)
        x2 = np.zeros(subject_geom.vertices.shape[0], dtype=np.float64)
        y2 = np.zeros(subject_geom.vertices.shape[0], dtype=np.float64)
    else:
        raise ValueError(f"unsupported coupled-map initialization: {initial_map}")

    for _ in range(max(1, int(solve_cycles))):
        x1, y1, x2, y2 = _solve_coupled_square_system(
            atlas_geom,
            subject_geom,
            atlas_boundary,
            subject_boundary,
            atlas_xb,
            atlas_yb,
            subject_xb,
            subject_yb,
            corr_atlas,
            corr_subject,
            sigma_vec,
            x1,
            y1,
            x2,
            y2,
            mu=mu,
            lam=lam,
            maxiter=maxiter,
        )
        if corr_atlas.size == 0:
            continue
        ori1 = orientation_tri(np.asarray(atlas_surface.faces, dtype=np.int64), x1, y1)
        wrong1 = 1 if np.count_nonzero(ori1 == 1) < np.count_nonzero(ori1 == -1) else -1
        bad_vertices1 = np.unique(np.asarray(atlas_surface.faces, dtype=np.int64)[ori1 == wrong1].reshape(-1))
        ori2 = orientation_tri(np.asarray(subject_surface.faces, dtype=np.int64), x2, y2)
        wrong2 = 1 if np.count_nonzero(ori2 == 1) < np.count_nonzero(ori2 == -1) else -1
        bad_vertices2 = np.unique(np.asarray(subject_surface.faces, dtype=np.int64)[ori2 == wrong2].reshape(-1))
        if bad_vertices1.size == 0 and bad_vertices2.size == 0:
            continue
        bad_corr = np.zeros(corr_atlas.shape[0], dtype=bool)
        if bad_vertices1.size:
            bad_corr |= np.isin(corr_atlas, bad_vertices1)
        if bad_vertices2.size:
            bad_corr |= np.isin(corr_subject, bad_vertices2)
        sigma_vec[bad_corr] *= 0.5

    atlas_out = _surface_copy(atlas_surface)
    subject_out = _surface_copy(subject_surface)
    atlas_out.u = (0.5 * (1.0 + x1)).astype(np.float32)
    atlas_out.v = (0.5 * (1.0 + y1)).astype(np.float32)
    subject_out.u = (0.5 * (1.0 + x2)).astype(np.float32)
    subject_out.v = (0.5 * (1.0 + y2)).astype(np.float32)

    return atlas_out, subject_out



def _stabilized_pca_basis(points: np.ndarray) -> np.ndarray:
    centered = np.asarray(points, dtype=np.float64) - np.mean(points, axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:2].copy()
    for row in range(basis.shape[0]):
        major = int(np.argmax(np.abs(basis[row])))
        if basis[row, major] < 0:
            basis[row] *= -1.0
    if np.linalg.det(np.stack([basis[0], basis[1], np.cross(basis[0], basis[1])], axis=0)) < 0:
        basis[1] *= -1.0
    return basis


def _square_perimeter_points(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.mod(np.asarray(t, dtype=np.float64), 1.0)
    u = np.zeros_like(t)
    v = np.zeros_like(t)

    m0 = t < 0.25
    u[m0] = 4.0 * t[m0]
    v[m0] = 0.0

    m1 = (t >= 0.25) & (t < 0.5)
    u[m1] = 1.0
    v[m1] = 4.0 * (t[m1] - 0.25)

    m2 = (t >= 0.5) & (t < 0.75)
    u[m2] = 1.0 - 4.0 * (t[m2] - 0.5)
    v[m2] = 1.0

    m3 = t >= 0.75
    u[m3] = 0.0
    v[m3] = 1.0 - 4.0 * (t[m3] - 0.75)
    return u, v


def _ordered_boundary_uv(points: np.ndarray, loop: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boundary_points = np.asarray(points, dtype=np.float64)[loop]
    basis = _stabilized_pca_basis(boundary_points)
    proj = (boundary_points - boundary_points.mean(axis=0, keepdims=True)) @ basis.T
    angles = np.arctan2(proj[:, 1], proj[:, 0])
    start = int(np.argmin(angles))
    order = np.roll(loop, -start)
    ordered_points = np.asarray(points, dtype=np.float64)[order]
    seg = np.linalg.norm(np.roll(ordered_points, -1, axis=0) - ordered_points, axis=1)
    perimeter = float(np.sum(seg))
    if perimeter <= 0:
        perimeter = float(len(order))
        seg = np.ones(len(order), dtype=np.float64)
    cumulative = np.concatenate([[0.0], np.cumsum(seg[:-1])]) / perimeter
    u, v = _square_perimeter_points(cumulative)
    return order, u.astype(np.float64), v.astype(np.float64)


def _signed_uv_areas(faces: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    tri_u = u[faces]
    tri_v = v[faces]
    return 0.5 * (
        (tri_u[:, 1] - tri_u[:, 0]) * (tri_v[:, 2] - tri_v[:, 0])
        - (tri_v[:, 1] - tri_v[:, 0]) * (tri_u[:, 2] - tri_u[:, 0])
    )


def harmonic_flatten_to_square(surface, transform: np.ndarray | None = None):
    out = _surface_copy(surface)
    points = np.asarray(surface.vertices, dtype=np.float64)
    if transform is not None:
        points = apply_transform(points, transform)

    loop = boundary_loop(surface)
    if loop.size == 0:
        raise ValueError("surface has no boundary; harmonic square flattening requires a disk-like surface")

    _, adjacency = vertices_connectivity_fast(surface)
    adjacency = adjacency.astype(np.float64)
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    lap = diags(degree) - adjacency
    n_vertices = int(surface.vertices.shape[0])

    candidates = []
    for reverse in (False, True):
        current_loop = loop[::-1].copy() if reverse else loop
        order, ub, vb = _ordered_boundary_uv(points, current_loop)
        boundary_mask = np.zeros(n_vertices, dtype=bool)
        boundary_mask[order] = True
        interior = np.flatnonzero(~boundary_mask)
        boundary = np.flatnonzero(boundary_mask)
        u = np.zeros(n_vertices, dtype=np.float64)
        v = np.zeros(n_vertices, dtype=np.float64)
        u[order] = ub
        v[order] = vb
        if interior.size:
            lii = lap[interior][:, interior]
            lib = lap[interior][:, boundary]
            u[interior] = spsolve(lii, -(lib @ u[boundary]))
            v[interior] = spsolve(lii, -(lib @ v[boundary]))
        areas = _signed_uv_areas(np.asarray(surface.faces, dtype=np.int64), u, v)
        negative = int(np.sum(areas < 0))
        candidates.append((negative, u, v))

    _, u_best, v_best = min(candidates, key=lambda item: item[0])
    areas = _signed_uv_areas(np.asarray(surface.faces, dtype=np.int64), u_best, v_best)
    if np.mean(areas) < 0:
        v_best = 1.0 - v_best
        areas = _signed_uv_areas(np.asarray(surface.faces, dtype=np.int64), u_best, v_best)
    if np.sum(areas < 0) > np.sum(areas > 0):
        u_best = 1.0 - u_best

    out.u = np.clip(u_best, 0.0, 1.0).astype(np.float32)
    out.v = np.clip(v_best, 0.0, 1.0).astype(np.float32)
    return out


def _square_symmetries(uv: np.ndarray) -> list[np.ndarray]:
    xy = np.asarray(uv, dtype=np.float64) - 0.5
    matrices = [
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([[0.0, -1.0], [1.0, 0.0]]),
        np.array([[-1.0, 0.0], [0.0, -1.0]]),
        np.array([[0.0, 1.0], [-1.0, 0.0]]),
        np.array([[-1.0, 0.0], [0.0, 1.0]]),
        np.array([[1.0, 0.0], [0.0, -1.0]]),
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        np.array([[0.0, -1.0], [-1.0, 0.0]]),
    ]
    return [np.clip((xy @ mat.T) + 0.5, 0.0, 1.0) for mat in matrices]


def canonicalize_square_orientation(surface, symmetry_index: int = 3):
    out = _surface_copy(surface)
    matrices = [
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([[0.0, -1.0], [1.0, 0.0]]),
        np.array([[-1.0, 0.0], [0.0, -1.0]]),
        np.array([[0.0, 1.0], [-1.0, 0.0]]),
        np.array([[-1.0, 0.0], [0.0, 1.0]]),
        np.array([[1.0, 0.0], [0.0, -1.0]]),
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        np.array([[0.0, -1.0], [-1.0, 0.0]]),
    ]
    uv = np.asarray(uv_points(out), dtype=np.float64) - 0.5
    uv = (uv @ matrices[int(symmetry_index)].T) + 0.5
    out.u = uv[:, 0].astype(np.float32)
    out.v = uv[:, 1].astype(np.float32)
    return out


def align_uv_to_reference(subject_surface, atlas_surface, transform_subject_to_atlas: np.ndarray):
    out = _surface_copy(subject_surface)
    subject_uv = uv_points(subject_surface)
    atlas_uv = uv_points(atlas_surface)
    atlas_vertices = np.asarray(atlas_surface.vertices, dtype=np.float64)
    subject_aligned = apply_transform(np.asarray(subject_surface.vertices, dtype=np.float64), transform_subject_to_atlas)
    atlas_tree = cKDTree(atlas_vertices)
    _, atlas_idx = atlas_tree.query(subject_aligned, k=1)

    best_error = float("inf")
    best_uv = subject_uv
    target = atlas_uv[atlas_idx]
    for candidate in _square_symmetries(subject_uv):
        error = float(np.mean(np.sum((candidate - target) ** 2, axis=1)))
        if error < best_error:
            best_error = error
            best_uv = candidate

    out.u = best_uv[:, 0].astype(np.float32)
    out.v = best_uv[:, 1].astype(np.float32)
    return out


def parameterize_surface_uv(surface, transform: np.ndarray | None = None):
    out = _surface_copy(surface)
    if hasattr(surface, "u") and hasattr(surface, "v"):
        u = np.asarray(surface.u, dtype=np.float64).reshape(-1)
        v = np.asarray(surface.v, dtype=np.float64).reshape(-1)
        if (
            u.shape[0] == surface.vertices.shape[0]
            and v.shape[0] == surface.vertices.shape[0]
            and np.all(np.isfinite(u))
            and np.all(np.isfinite(v))
            and np.ptp(u) > 1e-6
            and np.ptp(v) > 1e-6
        ):
            out.u = u.astype(np.float32)
            out.v = v.astype(np.float32)
            return out

    return harmonic_flatten_to_square(surface, transform)


def uv_points(surface) -> np.ndarray:
    if not hasattr(surface, "u") or not hasattr(surface, "v"):
        surface = parameterize_surface_uv(surface)
    return np.column_stack([np.asarray(surface.u, dtype=np.float64), np.asarray(surface.v, dtype=np.float64)])


def interpolate_surface_values_by_uv(source_surface, values: np.ndarray, target_surface, method: str = "linear") -> np.ndarray:
    points = uv_points(source_surface)
    queries = uv_points(target_surface)
    return interpolate_points_to_values(points, values, queries, method=method)


def transfer_surface_fields_by_uv(subject_surface, atlas_surface):
    out = _surface_copy(subject_surface)
    subject_uv = uv_points(subject_surface)
    atlas_uv = uv_points(atlas_surface)

    if hasattr(atlas_surface, "labels"):
        labels = interpolate_points_to_values(atlas_uv, np.asarray(atlas_surface.labels), subject_uv, method="nearest")
        out.labels = np.asarray(np.rint(labels), dtype=np.uint16)
    if hasattr(atlas_surface, "vColor"):
        colors = np.column_stack(
            [
                interpolate_points_to_values(atlas_uv, np.asarray(atlas_surface.vColor)[:, channel], subject_uv, method="nearest")
                for channel in range(3)
            ]
        )
        out.vColor = np.asarray(colors, dtype=np.float32)
    out.u = np.asarray(subject_uv[:, 0], dtype=np.float32)
    out.v = np.asarray(subject_uv[:, 1], dtype=np.float32)
    return out


def map_surface_vertices_by_uv(subject_surface, atlas_surface, method: str = "linear") -> np.ndarray:
    atlas_vertices = np.asarray(atlas_surface.vertices, dtype=np.float64)
    return np.column_stack(
        [
            interpolate_surface_values_by_uv(atlas_surface, atlas_vertices[:, axis], subject_surface, method=method)
            for axis in range(3)
        ]
    )


def surface_jacobian_hemi(subbasename: str, hemi: str):
    """Write the smoothed atlas-space surface Jacobian for one hemisphere."""
    subject = readdfs(f"{subbasename}.{hemi}.mid.cortex.svreg.dfs")
    atlas_path = Path(subbasename).parent / f"atlas.{hemi}.mid.cortex.svreg.dfs"
    atlas = readdfs(str(atlas_path))
    faces = np.asarray(atlas.faces, dtype=np.int64)

    subject_vertices = np.asarray(subject.vertices, dtype=np.float64)
    mapped = np.column_stack(
        [
            interpolate_surface_values_by_uv(
                subject, subject_vertices[:, axis], atlas, method="linear"
            )
            for axis in range(3)
        ]
    ).astype(np.float64, copy=False)
    atlas_vertices = np.asarray(atlas.vertices, dtype=np.float64)
    mapped_v1 = mapped[faces[:, 1]] - mapped[faces[:, 0]]
    mapped_v2 = mapped[faces[:, 2]] - mapped[faces[:, 0]]
    atlas_v1 = atlas_vertices[faces[:, 1]] - atlas_vertices[faces[:, 0]]
    atlas_v2 = atlas_vertices[faces[:, 2]] - atlas_vertices[faces[:, 0]]
    mapped_area2 = np.linalg.norm(np.cross(mapped_v1, mapped_v2), axis=1)
    atlas_area2 = np.linalg.norm(np.cross(atlas_v1, atlas_v2), axis=1)
    # Preserve undefined per-triangle Jacobians so the weighted vertex map
    # propagates NaNs the same way the MATLAB determinant and sparse multiply do.
    triangle_jacobian = np.divide(
        mapped_area2,
        atlas_area2,
        out=np.full_like(mapped_area2, np.nan),
        where=(atlas_area2 > 0) & (mapped_area2 > 0),
    )
    vertex_jacobian = np.asarray(tri2nodes(atlas) @ triangle_jacobian).reshape(-1)

    result, _ = smooth_cortex_fast(atlas, 0.5, 2000)
    result.attributes = vertex_jacobian.astype(np.float32)
    ramp = np.linspace(0.0, 1.0, 1000, dtype=np.float64)
    hot = np.column_stack(
        [np.minimum(3.0 * ramp, 1.0), np.clip(3.0 * ramp - 1.0, 0.0, 1.0), np.clip(3.0 * ramp - 2.0, 0.0, 1.0)]
    )
    color_index = np.floor(999.0 * np.minimum(vertex_jacobian, 3.0) / 3.0 + 0.5)
    color_index = np.nan_to_num(color_index, nan=0.0, posinf=999.0, neginf=0.0).astype(np.int64)
    result.vColor = hot[np.clip(color_index, 0, 999)].astype(np.float32)
    output_path = Path(subbasename).parent / f"atlas.{hemi}.mid.cortex.jacobian.dfs"
    writedfs(str(output_path), result)
    return result


def map_thickness_to_atlas(subject_mid, subject_inner, subject_pial, atlas_mid, transform_subject_to_atlas: np.ndarray):
    out_subject = _surface_copy(subject_mid)
    thickness = np.linalg.norm(np.asarray(subject_pial.vertices) - np.asarray(subject_inner.vertices), axis=1)
    out_subject.attributes = thickness.astype(np.float32)
    atlas_out = _surface_copy(atlas_mid)
    atlas_out.attributes = np.asarray(
        interpolate_surface_values_by_uv(out_subject, thickness, atlas_out, method="linear"),
        dtype=np.float32,
    )
    return out_subject, atlas_out


def interpolate_points_to_values(points: np.ndarray, values: np.ndarray, queries: np.ndarray, method: str = "linear", fill_value: float = 0.0) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    queries = np.asarray(queries, dtype=np.float64)
    values = np.asarray(values)
    if points.shape[0] == 0:
        return np.full(queries.shape[0], fill_value, dtype=values.dtype if values.size else np.float32)
    if method == "nearest":
        nearest = NearestNDInterpolator(points, values)
        return np.asarray(nearest(queries))

    linear = LinearNDInterpolator(points, values, fill_value=np.nan if method == "linear" else fill_value)
    result = linear(queries)
    if method == "linear" and np.any(np.isnan(result)):
        nearest = NearestNDInterpolator(points, values)
        nan_mask = np.isnan(result)
        result[nan_mask] = nearest(queries[nan_mask])
    return np.asarray(result)


def nearest_fill(mask_points: np.ndarray, mask_values: np.ndarray, query_points: np.ndarray) -> np.ndarray:
    tree = cKDTree(mask_points)
    _, idx = tree.query(query_points, k=1)
    return mask_values[idx]


def surface_to_mask(surface, reference: NiftiVolume, dilation: int = 1) -> np.ndarray:
    coords = matlab_round(np.asarray(surface.vertices) / np.asarray(reference.zooms[:3])).astype(np.int32)
    coords = np.clip(coords, 0, np.asarray(reference.shape[:3]) - 1)
    mask = np.zeros(reference.shape[:3], dtype=bool)
    mask[coords[:, 0], coords[:, 1], coords[:, 2]] = True
    if dilation > 0:
        mask = binary_dilation(mask, iterations=dilation)
    return mask


def save_surface(path: str, surface) -> None:
    writedfs(path, surface)


def ensure_target_metadata_copy(subbasename_tmp: str, atlasbasename: str) -> None:
    atlas_path = Path(atlasbasename)
    atlas_dir = atlas_path.parent
    tmp_path = Path(subbasename_tmp).parent
    for name in ("svregmanifest.xml", "brainsuite_labeldescription.xml", "roi_list_svreg.txt", "sulcal_protocol_HD.xml"):
        src = atlas_dir.parent / name if name == "svregmanifest.xml" else atlas_dir / name
        if src.is_file():
            shutil.copyfile(src, tmp_path / name)
            if name == "sulcal_protocol_HD.xml":
                shutil.copyfile(src, tmp_path / f"{Path(subbasename_tmp).name}.{name}")


def _unitball_surface_map(surface, hemisphere_sign: float) -> np.ndarray:
    """Reproduce volmap_ball's square-to-unit-sphere surface coordinates."""
    uv = np.column_stack((2.0 * np.asarray(surface.u) - 1.0, 2.0 * np.asarray(surface.v) - 1.0))
    uv = np.clip(uv, -1.0, 1.0)
    uv = 0.5 * np.column_stack((uv[:, 0] + uv[:, 1], uv[:, 0] - uv[:, 1]))
    l1 = np.sum(np.abs(uv), axis=1)
    l2 = np.maximum(np.linalg.norm(uv, axis=1), 1e-6)
    uv = uv / l2[:, None] * l1[:, None]
    radius = np.linalg.norm(uv, axis=1)
    z = hemisphere_sign * np.sqrt(np.maximum(0.0, 1.0 - radius * radius))
    return np.column_stack((uv, z)).astype(np.float32)


def _masked_graph_laplacian(mask_flat: np.ndarray, shape: tuple[int, int, int]) -> csr_matrix:
    """Six-neighbor periodic graph Laplacian in MATLAB find() voxel order."""
    count = mask_flat.size
    edges = []
    for axis, stride in enumerate((1, shape[0], shape[0] * shape[1])):
        coord = (mask_flat // stride) % shape[axis]
        next_flat = mask_flat + stride
        next_flat[coord == shape[axis] - 1] -= stride * (shape[axis] - 1)
        locations = np.searchsorted(mask_flat, next_flat)
        valid = locations < count
        valid[valid] &= mask_flat[locations[valid]] == next_flat[valid]
        src = np.flatnonzero(valid)
        dst = locations[valid]
        if src.size:
            edges.append(np.column_stack((src, dst)))
    if not edges:
        return csr_matrix((count, count), dtype=np.float64)
    edge = np.vstack(edges)
    rows = np.concatenate((edge[:, 0], edge[:, 1]))
    cols = np.concatenate((edge[:, 1], edge[:, 0]))
    data = np.full(rows.size, -1.0, dtype=np.float64)
    degree = np.bincount(rows, minlength=count).astype(np.float64)
    rows = np.concatenate((rows, np.arange(count)))
    cols = np.concatenate((cols, np.arange(count)))
    data = np.concatenate((data, degree))
    return csr_matrix((data, (rows, cols)), shape=(count, count))


def make_unitball_map(basename: str, mask_basename: str, resolution: tuple[float, float, float], left_inner, right_inner, left_pial, right_pial) -> None:
    """Create BrainSuite SVReg's surface-constrained harmonic unit-ball map.

    The boundary coordinates are linearly interpolated from the combined inner
    cortex mesh, then the six-neighbor Laplace equation is solved over the
    cortex mask with those surface values fixed. This replaces the former
    center-and-scale placeholder, which could not initialize SVReg volume maps.
    """
    mask_volume = load_nii_z(f"{mask_basename}.cortex.dewisp.mask.nii.gz")
    mask_array = np.asarray(mask_volume.img) >= 200
    shape = tuple(int(v) for v in mask_array.shape[:3])
    mask_flat = np.flatnonzero(mask_array.ravel(order="F"))
    if not mask_flat.size:
        raise ValueError(f"empty cortex mask: {basename}.cortex.dewisp.mask.nii.gz")
    coords = np.column_stack(np.unravel_index(mask_flat, shape, order="F"))

    # MATLAB find_mask_bdr uses the six axial neighbors; the center voxel is
    # counted three times, so a boundary voxel has fewer than nine samples.
    mask3 = mask_array
    neighbor_count = np.zeros(shape, dtype=np.uint8)
    for axis in range(3):
        for shift in (-1, 1):
            neighbor_count += np.roll(mask3, shift, axis=axis)
    is_boundary = neighbor_count.ravel(order="F")[mask_flat] < 6
    boundary = np.flatnonzero(is_boundary)
    interior = np.flatnonzero(~is_boundary)
    if boundary.size == 0:
        raise ValueError("cortex mask has no boundary voxels")

    surf = combine_surf([left_inner, right_inner])
    surf_map = np.vstack((_unitball_surface_map(left_inner, 1.0), _unitball_surface_map(right_inner, -1.0)))
    surf_points = np.asarray(surf.vertices, dtype=np.float64)
    resolution_arr = np.asarray(resolution, dtype=np.float64)
    boundary_mm = coords[boundary].astype(np.float64) * resolution_arr
    try:
        interp = LinearNDInterpolator(surf_points, surf_map, fill_value=np.nan)
        bvalues = np.asarray(interp(boundary_mm), dtype=np.float64)
    except Exception as exc:
        raise RuntimeError(f"could not interpolate unit-ball boundary values: {exc}") from exc
    missing = ~np.isfinite(bvalues).all(axis=1)
    if missing.any():
        nearest = NearestNDInterpolator(surf_points, surf_map)
        bvalues[missing] = nearest(boundary_mm[missing])

    lap = _masked_graph_laplacian(mask_flat, shape)
    A = lap[interior][:, interior].tocsr()
    coupling = lap[interior][:, boundary].tocsr()
    boundary_values = bvalues.astype(np.float64)
    volmap = np.empty((mask_flat.size, 3), dtype=np.float32)
    volmap[boundary] = boundary_values.astype(np.float32)
    diagonal = A.diagonal()
    preconditioner = LinearOperator(A.shape, matvec=lambda x: x / np.maximum(diagonal, 1.0))
    if interior.size:
        for axis in range(3):
            rhs = -(coupling @ boundary_values[:, axis])
            solution, info = cg(A, rhs, rtol=1e-8, atol=0.0, maxiter=16000, M=preconditioner)
            if info < 0:
                raise RuntimeError(f"unit-ball harmonic solver failed for coordinate {axis}: {info}")
            volmap[interior, axis] = solution.astype(np.float32)

    def _surface_struct(surface):
        return {"vertices": np.asarray(surface.vertices), "faces": np.asarray(surface.faces) + 1,
                "u": np.asarray(getattr(surface, "u", [])), "v": np.asarray(getattr(surface, "v", []))}

    # Preserve fields and the historical surf_mid assignment used by MATLAB.
    brain = {
        "mask_indx": mask_flat.astype(np.int32) + 1,
        "Msize": np.asarray(shape, dtype=np.int32),
        "resolution": resolution_arr.astype(np.float32),
        "surf": _surface_struct(surf),
        "surfmap": surf_map,
        "surf_cortex": _surface_struct(surf),
        "surf_mid": _surface_struct(combine_surf([left_pial, right_pial])),
        "surf_pial": _surface_struct(combine_surf([left_pial, right_pial])),
        "surf_map": surf_map,
        "volmap": volmap,
    }
    savemat(f"{basename}_unitball_map.mat", {"Brain": brain}, do_compression=True)


def estimate_affine_from_surface_uv(subject_surface, atlas_surface) -> np.ndarray:
    mapped_subject = np.column_stack(
        [
            interpolate_surface_values_by_uv(subject_surface, np.asarray(subject_surface.vertices)[:, axis], atlas_surface, method="linear")
            for axis in range(3)
        ]
    )
    return estimate_affine_transform(mapped_subject, np.asarray(atlas_surface.vertices, dtype=np.float64))


def estimate_affine_from_pial_surfaces(sub_left, sub_right, atlas_left, atlas_right) -> np.ndarray:
    src = []
    dst = []
    for sub_surface, atlas_surface in ((sub_left, atlas_left), (sub_right, atlas_right)):
        mapped_subject = np.column_stack(
            [
                interpolate_surface_values_by_uv(sub_surface, np.asarray(sub_surface.vertices)[:, axis], atlas_surface, method="linear")
                for axis in range(3)
            ]
        )
        src.append(mapped_subject)
        dst.append(np.asarray(atlas_surface.vertices, dtype=np.float64))
    return estimate_affine_transform(np.vstack(src), np.vstack(dst))


def map_curves_via_uv(atlas_curves: list[np.ndarray], atlas_surface, subject_surface) -> list[np.ndarray]:
    atlas_tree = cKDTree(np.asarray(atlas_surface.vertices, dtype=np.float64))
    subject_uv_tree = cKDTree(uv_points(subject_surface))
    atlas_uv = uv_points(atlas_surface)
    mapped: list[np.ndarray] = []
    for curve in atlas_curves:
        curve = remove_curve_defects(curve)
        if curve.size == 0:
            mapped.append(curve)
            continue
        _, atlas_idx = atlas_tree.query(np.asarray(curve, dtype=np.float64), k=1)
        curve_uv = atlas_uv[atlas_idx]
        _, subject_idx = subject_uv_tree.query(curve_uv, k=1)
        mapped.append(np.asarray(subject_surface.vertices)[subject_idx].astype(np.float32))
    return mapped


def overwrite_map_with_surface_correspondence(
    voxel_map_1based: np.ndarray,
    subject_vertices_mm: np.ndarray,
    mapped_vertices_atlas_mm: np.ndarray,
    subject_zooms: tuple[float, float, float] | np.ndarray,
    atlas_zooms: tuple[float, float, float] | np.ndarray,
) -> np.ndarray:
    out = np.asarray(voxel_map_1based, dtype=np.float32).copy()
    shape = np.asarray(out.shape[:3], dtype=np.int32)
    subject_vertices_mm = np.asarray(subject_vertices_mm, dtype=np.float64)
    mapped_vertices_atlas_mm = np.asarray(mapped_vertices_atlas_mm, dtype=np.float64)
    subject_zooms = np.asarray(subject_zooms[:3], dtype=np.float64)
    atlas_zooms = np.asarray(atlas_zooms[:3], dtype=np.float64)

    subject_vox = subject_vertices_mm / subject_zooms[None, :]
    subject_idx = matlab_round(subject_vox).astype(np.int32)
    valid = np.all((subject_idx >= 0) & (subject_idx < shape[None, :]), axis=1)
    if not np.any(valid):
        return out

    subject_idx = subject_idx[valid]
    subject_vox = subject_vox[valid]
    mapped_vertices_atlas_mm = mapped_vertices_atlas_mm[valid]

    fractional_offset = matlab_round(subject_vox) - subject_vox
    atlas_vox = mapped_vertices_atlas_mm / atlas_zooms[None, :] + 1.0 + fractional_offset * (subject_zooms / atlas_zooms)[None, :]

    flat_idx = np.ravel_multi_index(subject_idx.T, out.shape[:3])
    unique_idx, inverse = np.unique(flat_idx, return_inverse=True)
    collapsed = np.zeros((unique_idx.shape[0], 3), dtype=np.float64)
    counts = np.bincount(inverse).astype(np.float64)
    counts[counts == 0] = 1.0
    for axis in range(3):
        collapsed[:, axis] = np.bincount(inverse, weights=atlas_vox[:, axis], minlength=unique_idx.shape[0]) / counts

    out.reshape(-1, 3)[unique_idx] = collapsed.astype(np.float32)
    return out


def _index_coords(shape: tuple[int, int, int]) -> np.ndarray:
    return np.indices(shape, dtype=np.float32).transpose(1, 2, 3, 0)


def build_affine_map(subject: NiftiVolume, atlas: NiftiVolume, transform_subject_to_atlas_mm: np.ndarray) -> np.ndarray:
    coords = _coords_from_voxels(subject.shape[:3], subject.zooms[:3])
    coords_flat = coords.reshape(-1, 3)
    atlas_mm = apply_transform(coords_flat, transform_subject_to_atlas_mm)
    atlas_idx = atlas_mm / np.asarray(atlas.zooms[:3])[None, :]
    atlas_idx = atlas_idx.reshape(*subject.shape[:3], 3) + 1.0
    return atlas_idx.astype(np.float32)


def warp_volume_from_map(atlas_data: np.ndarray, voxel_map_1based: np.ndarray, order: int = 1, cval: float = 0.0) -> np.ndarray:
    coords = [
        np.asarray(voxel_map_1based[..., axis] - 1.0, dtype=np.float64)
        for axis in range(3)
    ]
    return map_coordinates(np.asarray(atlas_data, dtype=np.float64), coords, order=order, mode="constant", cval=cval)
