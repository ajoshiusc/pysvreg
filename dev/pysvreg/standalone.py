"""Native Python ports of small standalone SVReg volume utilities."""

from __future__ import annotations

import gzip
from itertools import product
import struct
import sys
from pathlib import Path
import shutil

try:
    from . import _deps  # noqa: F401
    from .dfsio import readdfs, writedfs
    from .native import (
        NiftiVolume,
        get_stiffness_matrix_tri_wt,
        interpolate_surface_values_by_uv,
        load_nii_z,
        matlab_round,
        mypcg_lp,
        cpd_register,
        recolor_by_label,
        save_untouch_nii_gz,
        smooth_cortex_fast,
        warp_volume_from_map,
    )
    from .runtime import remove_extn_basename
    from .refine_ROIs2 import refine_ROIs2
    from .svreg_refinements import svreg_refinements
except ImportError:
    import _deps  # noqa: F401
    from dfsio import readdfs, writedfs
    from native import NiftiVolume, cpd_register, get_stiffness_matrix_tri_wt, interpolate_surface_values_by_uv, load_nii_z, matlab_round, mypcg_lp, save_untouch_nii_gz, smooth_cortex_fast, warp_volume_from_map
    from runtime import remove_extn_basename
    from refine_ROIs2 import refine_ROIs2
    from svreg_refinements import svreg_refinements

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.interpolate import RegularGridInterpolator
from scipy.io import loadmat, savemat
from scipy.sparse import diags


def _save_like(source: NiftiVolume, data: np.ndarray, filename: str, *, zooms=None) -> None:
    header = source.header.copy()
    data = np.asarray(data, dtype=source.header.get_data_dtype())
    header.set_data_shape(data.shape)
    header.set_data_dtype(data.dtype)
    if zooms is not None:
        old = list(header.get_zooms())
        for axis, value in enumerate(zooms[:3]):
            old[axis] = float(value)
        header.set_zooms(tuple(old[: data.ndim]))
        header["scl_slope"] = 0
    affine = source.affine.copy()
    if zooms is not None:
        affine[:3, :3] = np.diag(np.asarray(zooms[:3], dtype=float))
    image = nib.Nifti1Image(data, affine, header=header)
    image.set_sform(affine, code=1)
    nib.save(image, filename)


def _matlab_gaussian_filter3(data: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Apply MATLAB ``imgaussfilt3``'s default finite Gaussian kernel."""
    sigma = np.asarray(sigma, dtype=np.float64)
    radius = tuple(int(value) for value in np.ceil(2.0 * sigma))
    return gaussian_filter(
        np.asarray(data, dtype=np.float64),
        sigma=tuple(float(value) for value in sigma),
        mode="nearest",
        radius=radius,
    )


def smooth_volume(infile: str, stdx: str, stdy: str, stdz: str, outfile: str) -> None:
    """Match MATLAB imgaussfilt3 defaults; standard deviations are in mm."""
    volume = load_nii_z(infile)
    if volume.img.ndim not in (3, 4):
        raise ValueError(f"expected a 3D or 4D NIfTI image, got {volume.img.ndim}D")
    spacing = np.asarray(volume.zooms[:3], dtype=float)
    sigma = (np.asarray([float(stdx), float(stdy), float(stdz)]) + 1e-4) / spacing
    if volume.img.ndim == 3:
        result = _matlab_gaussian_filter3(volume.img, sigma)
    else:
        result = np.empty(volume.img.shape, dtype=float)
        for frame in range(volume.img.shape[3]):
            result[..., frame] = _matlab_gaussian_filter3(volume.img[..., frame], sigma)
    save_untouch_nii_gz(
        NiftiVolume(result, volume.affine, volume.header, volume.path), outfile
    )


def _corner_median(data: np.ndarray) -> float:
    corners = np.asarray(
        [
            data[x, y, z]
            for x in (0, data.shape[0] - 1)
            for y in (0, data.shape[1] - 1)
            for z in (0, data.shape[2] - 1)
        ],
        dtype=float,
    )
    return float(np.median(corners))


def _cubic_convolution_volume(
    data: np.ndarray,
    sampling_map_1based: np.ndarray,
    extrapval: float | None = None,
) -> np.ndarray:
    """Match MATLAB interp3's separable cubic-convolution interpolation."""
    source = np.asarray(data, dtype=np.float64)
    if source.ndim != 3 or min(source.shape) < 3:
        raise ValueError("cubic interpolation requires at least three samples on each axis")
    coordinates = [
        np.asarray(sampling_map_1based[..., axis], dtype=np.float64) - 1.0
        for axis in range(3)
    ]
    padded = source
    for axis in range(3):
        first = [np.take(padded, index, axis=axis) for index in range(3)]
        last = [np.take(padded, -index - 1, axis=axis) for index in range(3)]
        # MATLAB's cubic-convolution edge stencil extends the first/last three
        # samples with a quadratic polynomial before applying the four-point
        # convolution kernel.
        left = 3.0 * first[0] - 3.0 * first[1] + first[2]
        right = 3.0 * last[0] - 3.0 * last[1] + last[2]
        padded = np.concatenate(
            [np.expand_dims(left, axis), padded, np.expand_dims(right, axis)], axis=axis
        )

    shape = coordinates[0].shape
    output = np.zeros(shape, dtype=np.float64)
    chunk_depth = max(1, min(shape[2], 500_000 // max(1, shape[0] * shape[1])))
    for start in range(0, shape[2], chunk_depth):
        stop = min(start + chunk_depth, shape[2])
        query = [axis[..., start:stop] for axis in coordinates]
        valid = np.ones(query[0].shape, dtype=bool)
        bases = []
        weights = []
        for axis in range(3):
            finite = np.isfinite(query[axis])
            valid &= finite & (query[axis] >= 0.0) & (query[axis] <= source.shape[axis] - 1)
            safe_query = np.where(finite, query[axis], 0.0)
            base = np.floor(safe_query).astype(np.int64)
            bases.append(base)
            axis_weights = []
            for offset in (-1, 0, 1, 2):
                distance = np.abs(safe_query - (base + offset))
                weight = np.where(
                    distance <= 1.0,
                    1.0 - 2.5 * distance**2 + 1.5 * distance**3,
                    np.where(
                        distance < 2.0,
                        -0.5 * distance**3 + 2.5 * distance**2 - 4.0 * distance + 2.0,
                        0.0,
                    ),
                )
                axis_weights.append(weight)
            weights.append(axis_weights)

        values = np.zeros(query[0].shape, dtype=np.float64)
        for tap_indices in product(range(4), repeat=3):
            source_indices = []
            weight = np.ones(query[0].shape, dtype=np.float64)
            for axis, tap in enumerate(tap_indices):
                offset = tap - 1
                index = bases[axis] + offset + 1
                source_indices.append(np.clip(index, 0, padded.shape[axis] - 1))
                weight *= weights[axis][tap]
            values += padded[tuple(source_indices)] * weight
        values[~valid] = 0.0 if extrapval is None else float(extrapval)
        output[..., start:stop] = values
    return output


def _spline_interpolate_volume(
    data: np.ndarray,
    sampling_map_1based: np.ndarray,
    extrapval: float | None = None,
) -> np.ndarray:
    """Match MATLAB interp3's tensor-product cubic spline and extrapolation."""
    source = np.asarray(data, dtype=np.float64)
    if source.ndim != 3 or min(source.shape) < 4:
        raise ValueError("spline interpolation requires at least four samples on each axis")
    interpolator = RegularGridInterpolator(
        tuple(np.arange(size, dtype=np.float64) for size in source.shape),
        source,
        method="cubic",
        bounds_error=False,
        fill_value=None,
        solver_args={"rtol": 1e-8},
    )
    shape = sampling_map_1based.shape[:3]
    output = np.empty(shape, dtype=np.float64)
    chunk_depth = max(1, min(shape[2], 500_000 // max(1, shape[0] * shape[1])))
    for start in range(0, shape[2], chunk_depth):
        stop = min(start + chunk_depth, shape[2])
        coordinates = np.stack(
            [np.asarray(sampling_map_1based[..., start:stop, axis], dtype=np.float64) - 1.0
             for axis in range(3)],
            axis=-1,
        )
        values = interpolator(coordinates.reshape(-1, 3)).reshape(
            shape[0], shape[1], stop - start
        )
        if extrapval is not None:
            valid = np.ones(coordinates.shape[:-1], dtype=bool)
            for axis, size in enumerate(source.shape):
                axis_coordinates = coordinates[..., axis]
                valid &= (
                    np.isfinite(axis_coordinates)
                    & (axis_coordinates >= 0.0)
                    & (axis_coordinates <= size - 1)
                )
            values[~valid] = float(extrapval)
        output[..., start:stop] = values
    return output


def resample_volume(
    infile: str,
    outfile: str,
    dim: str = "-res",
    dx: str = "1",
    dy: str = "1",
    dz: str = "1",
    method: str = "linear",
    extrapval: str = "",
) -> None:
    """Resample 3D/4D NIfTI data using MATLAB SVReg voxel-center conventions."""
    volume = load_nii_z(infile)
    if volume.img.ndim not in (3, 4):
        raise ValueError(f"expected a 3D or 4D NIfTI image, got {volume.img.ndim}D")
    old_shape = np.asarray(volume.img.shape[:3], dtype=int)
    old_spacing = np.asarray(volume.zooms[:3], dtype=float)
    values = np.asarray([float(dx), float(dy), float(dz)], dtype=float)
    if np.any(values <= 0) or np.any(old_spacing <= 0):
        raise ValueError("voxel dimensions and spacings must be positive")
    if dim == "-size":
        out_shape = values.astype(int)
        if not np.allclose(values, out_shape) or np.any(out_shape < 1):
            raise ValueError("-size dimensions must be positive integers")
        out_spacing = old_spacing * old_shape / out_shape
    elif dim == "-res":
        out_spacing = values
        out_shape = np.floor(old_shape * old_spacing / out_spacing).astype(int)
        if np.any(out_shape < 1):
            raise ValueError("requested resolution produces an empty output volume")
    else:
        raise ValueError("dim must be '-size' or '-res'")

    method_name = method.lower()
    orders = {"nearest": 0, "linear": 1, "cubic": 3, "spline": 3}
    if method_name not in orders:
        raise ValueError(f"unsupported interpolation method: {method}")
    order = orders[method_name]
    try:
        fill = float(extrapval)
        if not np.isfinite(fill):
            fill = _corner_median(volume.img)
    except (TypeError, ValueError):
        fill = _corner_median(volume.img)

    # MATLAB samples voxel centers: (j + .5) * output_spacing, converted
    # back to zero-based input voxel coordinates.
    out_indices = np.indices(tuple(out_shape), dtype=float)
    coordinates = np.stack(
        [
            (out_indices[axis] + 0.5) * out_spacing[axis] / old_spacing[axis] - 0.5
            for axis in range(3)
        ],
        axis=0,
    )
    sampling_map = (
        np.moveaxis(coordinates, 0, -1) + 1.0
        if method_name in {"cubic", "spline"}
        else None
    )

    def resample_frame(frame: np.ndarray) -> np.ndarray:
        if method_name == "cubic":
            return _cubic_convolution_volume(frame, sampling_map, fill)
        if method_name == "spline":
            return _spline_interpolate_volume(frame, sampling_map, fill)
        return map_coordinates(
            np.asarray(frame, dtype=float),
            coordinates,
            order=order,
            mode="constant",
            cval=fill,
            prefilter=order > 1,
        )

    if volume.img.ndim == 3:
        result = resample_frame(volume.img)
    else:
        result = np.stack(
            [resample_frame(volume.img[..., frame]) for frame in range(volume.img.shape[3])],
            axis=3,
        )
    _save_like(volume, result, outfile, zooms=out_spacing)


def _map_at(map_volume: NiftiVolume, xyz1: tuple[int, int, int]) -> np.ndarray:
    idx = np.asarray(xyz1, dtype=int) - 1
    if np.any(idx < 0) or np.any(idx >= np.asarray(map_volume.img.shape[:3])):
        raise IndexError(f"voxel coordinate outside map volume: {xyz1}")
    mapped = np.asarray(map_volume.img[tuple(idx)], dtype=float).squeeze()
    if mapped.shape != (3,):
        raise ValueError("SVReg coordinate map must have three values at each voxel")
    return mapped


_matlab_round = matlab_round


def _mni_to_tal(point: np.ndarray) -> np.ndarray:
    # Equivalent to the two SPM matrices used by the bundled mni2tal.m.
    theta = 0.05
    rotation = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, np.cos(theta), np.sin(theta)], [0.0, -np.sin(theta), np.cos(theta)]]
    )
    scales = np.asarray([0.99, 0.97, 0.92 if point[2] >= 0 else 0.84])
    return rotation @ (point * scales)


def get_mni_tal(subject_prefix: str, atlas_prefix: str, sx: str, sy: str, sz: str) -> tuple[np.ndarray, np.ndarray]:
    subject_prefix = remove_extn_basename(subject_prefix)
    atlas_prefix = remove_extn_basename(atlas_prefix)
    xyz = (int(float(sx)), int(float(sy)), int(float(sz)))
    mapped = _map_at(load_nii_z(f"{subject_prefix}.svreg.map.nii.gz"), xyz)
    map2mni = Path(atlas_prefix).parent / "map2mni.nii.gz"
    if map2mni.is_file():
        coords = tuple(int(v) for v in _matlab_round(mapped))
        mapped = _map_at(load_nii_z(str(map2mni)), coords)
    mni = mapped - np.asarray([79.0, 113.0, 51.0])
    tal = _mni_to_tal(mni)
    print(f"\nMNI coordinates: {mni[0]:g}, {mni[1]:g}, {mni[2]:g}")
    print(f"\nTalairach coordinates: {tal[0]:g}, {tal[1]:g}, {tal[2]:g}")
    return mni, tal


def mni_to_subject(subject_prefix: str, map_or_atlas: str, mx: str, my: str, mz: str) -> np.ndarray:
    subject_prefix = remove_extn_basename(subject_prefix)
    x, y, z = np.asarray([float(mx), float(my), float(mz)]) + [79.0, 113.0, 51.0]
    map_path = Path(map_or_atlas)
    if not map_path.is_file():
        map_path = map_path.parent / "mni2bci.nii.gz"
    if map_path.is_file():
        mapped = _map_at(load_nii_z(str(map_path)), tuple(int(v) for v in _matlab_round((x, y, z))))
        x, y, z = mapped
    inverse = load_nii_z(f"{subject_prefix}.svreg.inv.map.nii.gz")
    subject_xyz = _map_at(inverse, tuple(int(v) for v in _matlab_round((x, y, z))))
    print(f"Subject voxel coordinates: {subject_xyz[0]:g}, {subject_xyz[1]:g}, {subject_xyz[2]:g}")
    return subject_xyz


def apply_map(
    map_file: str,
    data_file: str,
    out_file: str,
    target_file: str,
    smoothness: str = "1",
    datatype: str = "16",
    bitpix: str = "32",
    interp_type: str = "nearest",
) -> None:
    """Apply a 3D SVReg coordinate map to scalar or BrainSuite tensor data."""
    coordinate_map = load_nii_z(map_file)
    coords_img = np.asarray(coordinate_map.img, dtype=float)
    if coords_img.ndim != 4 or coords_img.shape[3] < 3:
        raise ValueError("map NIfTI must have a fourth dimension with x, y, z coordinates")
    coords = np.stack([coords_img[..., axis] - 1.0 for axis in range(3)], axis=0)
    smooth = float(smoothness) if str(smoothness).strip() else 1.0
    if smooth > 0:
        for axis in range(3):
            coords[axis] = gaussian_filter(coords[axis], sigma=smooth, truncate=1.5)
    sampling_map = np.stack([coords[axis] + 1.0 for axis in range(3)], axis=3)
    if ".eig." in data_file:
        _apply_eig_map(
            coordinate_map,
            sampling_map,
            data_file,
            out_file,
        )
        return

    source = load_nii_z(data_file)
    target = load_nii_z(target_file)
    order_by_method = {"nearest": 0, "linear": 1, "cubic": 3, "spline": 3}
    method = str(interp_type).strip().lower() if str(interp_type).strip() else "nearest"
    if method not in order_by_method:
        raise ValueError(f"unsupported interpolation method: {interp_type}")
    order = order_by_method[method]
    if source.img.ndim == 3:
        if method == "cubic":
            data = _cubic_convolution_volume(source.img, sampling_map)
        elif method == "spline":
            data = _spline_interpolate_volume(source.img, sampling_map)
        else:
            data = warp_volume_from_map(source.img, sampling_map, order=order, cval=0.0)
    elif source.img.ndim == 4:
        if method == "cubic":
            data = np.stack(
                [_cubic_convolution_volume(source.img[..., t], sampling_map)
                 for t in range(source.img.shape[3])],
                axis=3,
            )
        elif method == "spline":
            data = np.stack(
                [_spline_interpolate_volume(source.img[..., t], sampling_map)
                 for t in range(source.img.shape[3])],
                axis=3,
            )
        else:
            data = np.stack(
                [warp_volume_from_map(source.img[..., t], sampling_map, order=order, cval=0.0)
                 for t in range(source.img.shape[3])],
                axis=3,
            )
    else:
        raise ValueError(f"expected scalar 3D or 4D data, got {source.img.ndim}D")
    data = np.nan_to_num(data, nan=0.0)
    dtype_map = {2: np.uint8, 4: np.int16, 8: np.int32, 16: np.float32, 64: np.float64}
    requested_datatype = str(datatype).strip()
    output_dtype = (
        dtype_map.get(int(requested_datatype), source.header.get_data_dtype())
        if requested_datatype
        else source.header.get_data_dtype()
    )
    data = np.asarray(data, dtype=output_dtype)
    header = target.header.copy()
    header.set_data_dtype(output_dtype)
    header.set_data_shape(data.shape)
    out_img = nib.Nifti1Image(data, target.affine, header=header)
    out_img.set_sform(target.affine, code=1)
    nib.save(out_img, out_file)


def _read_brainsuite_eig(path: str) -> tuple[np.ndarray, bytes, str]:
    """Read the 12-float-per-voxel NIfTI variant used by BrainSuite BDP."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as stream:
        header_bytes = stream.read(348)
        if len(header_bytes) != 348:
            raise ValueError(f"{path}: truncated NIfTI header")
        little = struct.unpack_from("<i", header_bytes, 0)[0]
        big = struct.unpack_from(">i", header_bytes, 0)[0]
        if little == 348:
            endian = "<"
        elif big == 348:
            endian = ">"
        else:
            raise ValueError(f"{path}: not a NIfTI-1 file")
        dims = struct.unpack_from(endian + "8h", header_bytes, 40)
        if dims[0] != 4 or dims[4] != 12 or dims[1] < 1 or dims[2] < 1 or dims[3] < 1:
            raise ValueError(
                f"{path}: expected BrainSuite .eig NIfTI with dimensions X,Y,Z,12; "
                f"got dim={dims}"
            )
        datatype = struct.unpack_from(endian + "h", header_bytes, 70)[0]
        if datatype != 0:
            raise ValueError(f"{path}: expected BrainSuite .eig datatype 0, got {datatype}")
        vox_offset = max(352, int(round(struct.unpack_from(endian + "f", header_bytes, 108)[0])))
        stream.seek(vox_offset)
        count = int(dims[1]) * int(dims[2]) * int(dims[3]) * 12
        raw = stream.read(count * 4)
        if len(raw) != count * 4:
            raise ValueError(f"{path}: truncated BrainSuite .eig voxel data")
    flat = np.frombuffer(raw, dtype=np.dtype(endian + "f4"), count=count)
    spatial_size = int(dims[1]) * int(dims[2]) * int(dims[3])
    fields = np.stack(
        [
            flat[i * spatial_size : (i + 1) * spatial_size].reshape(
                (int(dims[1]), int(dims[2]), int(dims[3])), order="F"
            )
            for i in range(12)
        ],
        axis=-1,
    )
    return fields.astype(np.float32, copy=False), header_bytes, endian


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return np.divide(vectors, norm, out=np.zeros_like(vectors), where=norm > 0)


def _rotate_vectors(vectors: np.ndarray, axis: np.ndarray, cosine: np.ndarray) -> np.ndarray:
    axis = _normalize_vectors(axis)
    cosine = np.clip(cosine, -1.0, 1.0)[..., None]
    sine = np.sqrt(np.maximum(0.0, 1.0 - cosine * cosine))
    return (
        vectors * cosine
        + np.cross(axis, vectors) * sine
        + axis * np.sum(axis * vectors, axis=-1, keepdims=True) * (1.0 - cosine)
    )


def _ppd_reorient(
    vectors: np.ndarray,
    gradients: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    """Port the three-vector Preservation of Principal Direction transform."""
    v1, v2, v3 = (_normalize_vectors(vectors[..., i, :].astype(np.float64)) for i in range(3))
    g1, g2, g3 = gradients
    n1 = _normalize_vectors(np.stack([np.sum(g * v1, axis=-1) for g in (g1, g2, g3)], axis=-1))
    n2 = _normalize_vectors(np.stack([np.sum(g * v2, axis=-1) for g in (g1, g2, g3)], axis=-1))
    cosine1 = np.clip(np.sum(v1 * n1, axis=-1), -1.0, 1.0)
    axis1 = np.cross(v1, n1)
    projected_n2 = _normalize_vectors(n2 - n1 * np.sum(n1 * n2, axis=-1, keepdims=True))
    v2rot = _rotate_vectors(v2, axis1, cosine1)
    cosine2 = np.clip(np.sum(v2rot * projected_n2, axis=-1), -1.0, 1.0)
    axis2 = np.cross(v2rot, projected_n2)
    w2 = _rotate_vectors(v2rot, axis2, cosine2)
    v3rot = _rotate_vectors(v3, axis1, cosine1)
    w3 = _rotate_vectors(v3rot, axis2, cosine2)
    return np.nan_to_num(np.stack([n1, w2, w3], axis=-2), nan=0.0, posinf=0.0, neginf=0.0)


def _write_brainsuite_eig(
    out_file: str,
    fields: np.ndarray,
    template_header: nib.Nifti1Header,
) -> None:
    """Write BrainSuite's nonstandard datatype-0, 12-float NIfTI encoding."""
    header = template_header.copy()
    header.set_data_shape(fields.shape)
    header.set_data_dtype(np.float32)
    header["vox_offset"] = 352.0
    header["descrip"] = b".eig file for BrainSuite"
    raw_header = bytearray(header.binaryblock)
    endian = header.endianness
    struct.pack_into(endian + "h", raw_header, 70, 0)  # BrainSuite .eig datatype
    struct.pack_into(endian + "h", raw_header, 72, 384)  # 12 float32 values per voxel
    struct.pack_into(endian + "f", raw_header, 108, 352.0)
    raw_header[344:348] = b"n+1\x00"
    payload = np.asarray(fields, dtype=np.dtype(endian + "f4")).ravel(order="F").tobytes()
    opener = gzip.open if out_file.endswith(".gz") else open
    with opener(out_file, "wb") as stream:
        stream.write(raw_header)
        stream.write(b"\x00\x00\x00\x00")
        stream.write(payload)


def _apply_eig_map(
    coordinate_map: NiftiVolume,
    sampling_map: np.ndarray,
    data_file: str,
    out_file: str,
) -> None:
    eig, _, _ = _read_brainsuite_eig(data_file)
    if coordinate_map.img.ndim != 4 or coordinate_map.img.shape[3] < 3:
        raise ValueError("map NIfTI must have a fourth dimension with x, y, z coordinates")

    # MATLAB's warp_DTI interpolates eigenvalues and eigenvectors with nearest
    # neighbors, regardless of the scalar interpolation option.
    warped = np.empty((*sampling_map.shape[:3], 12), dtype=np.float32)
    for component in range(12):
        warped[..., component] = warp_volume_from_map(
            eig[..., component], sampling_map, order=0, cval=0.0
        ).astype(np.float32, copy=False)

    spacing = np.asarray(coordinate_map.zooms[:3], dtype=float)
    if np.any(spacing <= 0):
        raise ValueError("map NIfTI voxel spacings must be positive")
    smooth_coords = np.asarray(sampling_map[..., :3], dtype=float) - 1.0
    map_gradients = []
    for input_axis in range(3):
        components = [
            np.gradient(smooth_coords[..., component], spacing[input_axis], axis=input_axis)
            for component in range(3)
        ]
        map_gradients.append(np.stack(components, axis=-1))
    vector_fields = np.stack([warped[..., 0:3], warped[..., 3:6], warped[..., 6:9]], axis=-2)
    rotated = _ppd_reorient(vector_fields, tuple(map_gradients))
    out_fields = np.concatenate([rotated.reshape(*rotated.shape[:3], 9), warped[..., 9:12]], axis=-1)
    _write_brainsuite_eig(out_file, out_fields, coordinate_map.header)


def smooth_surface_function(insurf: str, function_surface: str, outsurf: str, param: str = "10") -> None:
    """Smooth DFS per-vertex attributes using SVReg's sparse stiffness system."""
    surface = readdfs(insurf)
    values = np.asarray(readdfs(function_surface).attributes, dtype=float).reshape(-1)
    if values.size != surface.vertices.shape[0]:
        raise ValueError("function surface attribute count does not match surface vertex count")
    parameter = float(param)
    if parameter < 0:
        raise ValueError("smoothing parameter must be nonnegative")
    a = np.sqrt(parameter)
    stiffness, dx, dy = get_stiffness_matrix_tri_wt(surface, np.ones(values.size))
    system = (
        diags(np.ones(values.size, dtype=float))
        + (a * stiffness).T @ (a * stiffness)
        + (a * dx).T @ (a * dx)
        + (a * dy).T @ (a * dy)
    ).tocsr()
    diag = np.asarray(system.diagonal()).ravel()
    smoothed = mypcg_lp(system, values, 1e-100, 3000, diag)
    surface.attributes = np.asarray(smoothed, dtype=np.float32)
    writedfs(outsurf, surface)


def _jet_colors(values: np.ndarray, low: float = 0.0, high: float = 6.0) -> np.ndarray:
    scaled = np.clip((np.asarray(values, dtype=float) - low) / (high - low), 0.0, 1.0)
    return np.clip(
        np.minimum(4.0 * scaled[:, None] - np.asarray([1.5, 0.5, -0.5]),
                   -4.0 * scaled[:, None] + np.asarray([4.5, 3.5, 2.5])),
        0.0,
        1.0,
    ).astype(np.float32)


def thickness_to_atlas(subject_prefix: str) -> None:
    """Transfer PVC thickness to the atlas surfaces and write colored DFS files."""
    subject_prefix = remove_extn_basename(subject_prefix)
    parent = Path(subject_prefix).parent
    for hemi in ("left", "right"):
        inner = readdfs(f"{subject_prefix}.{hemi}.inner.cortex.svreg.dfs")
        thickness = readdfs(f"{subject_prefix}.pvc-thickness_0-6mm.{hemi}.mid.cortex.dfs")
        atlas_path = parent / f"atlas.{hemi}.mid.cortex.svreg.dfs"
        atlas = readdfs(str(atlas_path))
        values = interpolate_surface_values_by_uv(
            inner, np.asarray(thickness.attributes), atlas, method="linear"
        )
        atlas.attributes = np.asarray(values, dtype=np.float32)
        atlas.vColor = _jet_colors(np.asarray(values).reshape(-1))
        atlas, _ = smooth_cortex_fast(atlas, 0.5, 2000)
        writedfs(str(parent / f"atlas.pvc-thickness_0-6mm.{hemi}.mid.cortex.dfs"), atlas)


def label_with_atlas(subject_prefix: str, atlas_prefix: str, postfix: str) -> None:
    """Transfer an alternate atlas' surface and volume labels into subject space."""
    subject_prefix = remove_extn_basename(subject_prefix)
    atlas_prefix = remove_extn_basename(atlas_prefix)
    suffix = postfix if postfix.endswith(".") else f"{postfix}."
    subject = Path(subject_prefix)
    tmp_dir = subject.parent / f"{subject.name}.svreg.tmp"
    tmp_base = tmp_dir / subject.name
    tmp_dir.mkdir(parents=True, exist_ok=True)
    atlas_xml = Path(atlas_prefix).parent / "brainsuite_labeldescription.xml"
    if atlas_xml.is_file():
        shutil.copyfile(atlas_xml, tmp_dir / "brainsuite_labeldescription.xml")
        shutil.copyfile(atlas_xml, subject.parent / f"brainsuite_labeldescription.{postfix}.xml")

    for hemi in ("left", "right"):
        subject_mid = readdfs(f"{subject_prefix}.{hemi}.mid.cortex.svreg.dfs")
        atlas_mid = readdfs(str(subject.parent / f"atlas.{hemi}.mid.cortex.svreg.dfs"))
        new_atlas_mid = readdfs(f"{atlas_prefix}.{hemi}.mid.cortex.dfs")
        labels = interpolate_surface_values_by_uv(
            atlas_mid,
            np.asarray(new_atlas_mid.labels),
            subject_mid,
            method="nearest",
        )
        subject_mid.labels = np.rint(labels).astype(np.uint16)
        writedfs(f"{tmp_base}.{hemi}.mid.cortex.reg.{suffix}dfs", subject_mid)

        for surface_type in ("inner", "pial"):
            surface = readdfs(f"{subject_prefix}.{hemi}.{surface_type}.cortex.svreg.dfs")
            surface.labels = subject_mid.labels.copy()
            writedfs(f"{tmp_base}.{hemi}.{surface_type}.cortex.svreg.{suffix}dfs", surface)
        refine_ROIs2(subject_prefix, hemi, suffix)
        for surface_type in ("mid", "inner", "pial"):
            colored = Path(f"{subject_prefix}.{hemi}.{surface_type}.cortex.svreg.{suffix}dfs")
            recolor_by_label(str(colored), atlas_prefix)

    map_volume = load_nii_z(f"{subject_prefix}.svreg.map.nii.gz")
    atlas_labels = load_nii_z(f"{atlas_prefix}.label.nii.gz")
    subject_labels = warp_volume_from_map(atlas_labels.img, map_volume.img, order=0, cval=0.0)
    target = load_nii_z(f"{subject_prefix}.svreg.label.nii.gz")
    out_name = f"{tmp_base}.svreg.{suffix}label.nii.gz"
    save_untouch_nii_gz(
        NiftiVolume(subject_labels.astype(np.int16), target.affine, target.header, target.path),
        out_name,
    )
    svreg_refinements(subject_prefix, atlas_prefix, suffix)


def _mat_field(value, key: str):
    if isinstance(value, dict):
        return value[key]
    if hasattr(value, key):
        return getattr(value, key)
    if getattr(value, "dtype", None) is not None and value.dtype.names and key in value.dtype.names:
        return value[key].item()
    raise KeyError(f"MAT file struct is missing field {key!r}")


def register_cc_curve(surface_prefix: str) -> None:
    """Register corpus-callosum boundary curves with the native Python CPD port."""
    prefix = surface_prefix.removesuffix(".right.mid.cortex_smooth10.dfs").removesuffix(
        ".left.mid.cortex_smooth10.dfs"
    )
    source = loadmat(f"{prefix}_in_register_cc.mat", simplify_cells=True)
    surface1 = _mat_field(source, "surf1")
    surface2 = _mat_field(source, "surf2")
    vertices1 = np.asarray(_mat_field(surface1, "vertices"), dtype=float)
    vertices2 = np.asarray(_mat_field(surface2, "vertices"), dtype=float)
    bdr1 = np.asarray(_mat_field(source, "bdr1"), dtype=int).reshape(-1) - 1
    bdr2 = np.asarray(_mat_field(source, "bdr2"), dtype=int).reshape(-1) - 1
    x = vertices1[bdr1][:, [1, 2]]
    y = vertices2[bdr2][:, [1, 2]]
    rigid, _ = cpd_register(x, y, method="rigid", corresp=False)
    rigid.Y = np.asarray(rigid.Y, dtype=np.float32).astype(np.float64)
    nonrigid, _ = cpd_register(x, rigid.Y, method="nonrigid", corresp=True)
    nonrigid.Y = np.asarray(nonrigid.Y, dtype=np.float32).astype(np.float64)
    result = dict(vars(nonrigid))
    result["normal"] = dict(vars(nonrigid.normal))
    savemat(f"{prefix}_out_register_cc.mat", {"Tr": result})


def main(argv: list[str] | None = None, command: str | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    command = command or Path(sys.argv[0]).stem
    try:
        if command == "svreg_smooth_vol_function":
            if len(argv) != 5:
                raise ValueError("usage: svreg_smooth_vol_function INFILE STDX STDY STDZ OUTFILE")
            smooth_volume(*argv)
        elif command == "svreg_resample":
            if len(argv) < 2 or len(argv) > 8:
                raise ValueError("usage: svreg_resample INFILE OUTFILE [DIM DX DY DZ METHOD EXTRAPVAL]")
            resample_volume(*argv)
        elif command == "svreg_get_mni_tal":
            if len(argv) != 5:
                raise ValueError("usage: svreg_get_mni_tal SUBJECT_PREFIX ATLAS_PREFIX SX SY SZ")
            get_mni_tal(*argv)
        elif command == "svreg_mni2sub":
            if len(argv) != 5:
                raise ValueError("usage: svreg_mni2sub SUBJECT_PREFIX MAP_OR_ATLAS MX MY MZ")
            mni_to_subject(*argv)
        elif command == "svreg_apply_map":
            if len(argv) < 4 or len(argv) > 8:
                raise ValueError("usage: svreg_apply_map MAP DATA OUT TARGET [SMOOTHNESS DATATYPE BITPIX INTERP]")
            apply_map(*argv)
        elif command == "svreg_smooth_surf_function":
            if len(argv) < 3 or len(argv) > 4:
                raise ValueError("usage: svreg_smooth_surf_function IN_SURFACE FUNCTION_SURFACE OUT_SURFACE [PARAM]")
            smooth_surface_function(*argv)
        elif command == "svreg_thickness2atlas":
            if len(argv) != 1:
                raise ValueError("usage: svreg_thickness2atlas SUBJECT_PREFIX")
            thickness_to_atlas(argv[0])
        elif command == "svreg_labelwith_atlas":
            if len(argv) != 3:
                raise ValueError("usage: svreg_labelwith_atlas SUBJECT_PREFIX ATLAS_PREFIX POSTFIX")
            label_with_atlas(*argv)
        elif command == "register_cc_curve":
            if len(argv) != 1:
                raise ValueError("usage: register_cc_curve SURFACE_PREFIX")
            register_cc_curve(argv[0])
        else:
            raise ValueError(f"no standalone Python command is implemented for {command}")
    except (OSError, ValueError, IndexError, NotImplementedError, KeyError) as exc:
        print(f"{command}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
