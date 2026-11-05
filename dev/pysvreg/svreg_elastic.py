from __future__ import annotations

import shutil

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.sparse import csr_matrix

try:
    from .native import NiftiVolume, dws_write_nii, load_nii_z, matlab_round, readdfs, save_untouch_nii_gz
    from .runtime import normalize_fileprefix
except ImportError:
    from native import NiftiVolume, dws_write_nii, load_nii_z, matlab_round, readdfs, save_untouch_nii_gz
    from runtime import normalize_fileprefix


def second_difference_laplacian(mask_flat: np.ndarray, shape: tuple[int, int, int]) -> tuple[csr_matrix, np.ndarray]:
    """Build createDDWithDBoundary3Dmsk's summed, masked second differences."""
    mask_flat = np.asarray(mask_flat, dtype=np.int64)
    count = mask_flat.size
    coords = np.column_stack(np.unravel_index(mask_flat, shape, order="F"))
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    degree = np.zeros(count, dtype=np.uint8)
    diagonal = np.full(count, -6.0, dtype=np.float64)
    for axis, stride in enumerate((1, shape[0], shape[0] * shape[1])):
        edge = (coords[:, axis] == 0) | (coords[:, axis] == shape[axis] - 1)
        diagonal[edge] += 1.0  # source matrix has -1 rather than -2 at volume edges
        for direction in (-1, 1):
            neighbor_flat = mask_flat + direction * stride
            valid = (coords[:, axis] + direction >= 0) & (coords[:, axis] + direction < shape[axis])
            positions = np.searchsorted(mask_flat, neighbor_flat)
            candidate = np.flatnonzero(valid & (positions < count))
            if candidate.size:
                matched = mask_flat[positions[candidate]] == neighbor_flat[candidate]
                source = candidate[matched]
                target = positions[source]
                rows.append(source)
                cols.append(target)
                data.append(np.ones(source.size, dtype=np.float64))
                degree[source] += 1
    index = np.arange(count, dtype=np.int64)
    rows.append(index)
    cols.append(index)
    data.append(diagonal)
    operator = csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(count, count),
    )
    return operator, np.flatnonzero(degree < 6)


def _sample_scalar(image: np.ndarray, coordinates: np.ndarray, order: int = 1) -> np.ndarray:
    return map_coordinates(
        np.asarray(image, dtype=np.float64),
        np.asarray(coordinates, dtype=np.float64).T,
        order=order,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )


class ElasticIntensityRefiner:
    def __init__(
        self,
        source: np.ndarray,
        target: np.ndarray,
        coordinates: np.ndarray,
        operator: csr_matrix,
        boundary: np.ndarray,
        alpha: float,
    ):
        self.source = np.asarray(source, dtype=np.float64)
        self.target = np.asarray(target, dtype=np.float64)
        self.coordinates = np.asarray(coordinates, dtype=np.float64)
        self.operator = operator * float(alpha)
        self.boundary = np.asarray(boundary, dtype=np.int64)
        self.free = np.ones(self.coordinates.shape[0], dtype=bool)
        self.free[self.boundary] = False
        self.target_samples = self.target[
            matlab_round(self.coordinates[:, 0]),
            matlab_round(self.coordinates[:, 1]),
            matlab_round(self.coordinates[:, 2]),
        ]
        self.source_gradient = np.gradient(self.source)
        self.origin = self.coordinates.copy()
        self.cost_history: list[float] = []

    def cost(self, coordinates: np.ndarray) -> float:
        residual = _sample_scalar(self.source, coordinates) - self.target_samples
        sim = float(np.dot(residual[self.free], residual[self.free]))
        regularization = 0.0
        displacement = coordinates - self.origin
        for axis in range(3):
            lap = self.operator @ displacement[:, axis]
            regularization += float(np.dot(lap[self.free], lap[self.free]))
        return sim + regularization

    def gradient(self, coordinates: np.ndarray) -> np.ndarray:
        residual = _sample_scalar(self.source, coordinates) - self.target_samples
        output = np.empty_like(coordinates)
        displacement = coordinates - self.origin
        for axis in range(3):
            lap = self.operator @ displacement[:, axis]
            reg_gradient = 2.0 * (self.operator.T @ lap)
            image_gradient = _sample_scalar(self.source_gradient[axis], coordinates)
            output[:, axis] = reg_gradient + 2.0 * residual * image_gradient
        output[~self.free] = 0.0
        return output

    def _backtrack(
        self,
        coordinates: np.ndarray,
        direction: np.ndarray,
        current_cost: float,
        slope: float,
        gamma: float,
    ) -> tuple[np.ndarray, float, int]:
        if slope >= 0.0:
            raise ValueError("nonlinear registration line search received a non-descent direction")
        step = np.asarray(direction, dtype=np.float64).copy()
        bound = 0.5 * slope
        for calls in range(1, 81):
            candidate = coordinates + step
            candidate_cost = self.cost(candidate)
            if candidate_cost <= current_cost + bound:
                return candidate, candidate_cost, calls
            step *= gamma
            bound *= gamma
            if np.linalg.norm(step) <= np.finfo(np.float64).eps:
                return coordinates.copy(), current_cost, calls
        return coordinates.copy(), current_cost, 80

    def optimize(self, iterations: int = 100) -> np.ndarray:
        gradient = self.gradient(self.coordinates)
        delta = -gradient
        self.coordinates, _, _ = self._backtrack(
            self.coordinates, delta, 1e100, float(np.sum(delta * gradient)), gamma=0.8
        )
        # The MATLAB code resets cost1(1) to the identity-map cost after its
        # initial full steepest-descent step; preserve that line-search behavior.
        current_cost = self.cost(self.origin)
        self.cost_history = [current_cost]
        for iteration in range(int(iterations)):
            displacement = self.coordinates - self.origin
            max_displacement = float(np.linalg.norm(displacement, axis=1).max(initial=0.0))
            if iteration > 5 and len(self.cost_history) >= 5:
                if abs(max_displacement - self._previous_displacement) < 1e-6:
                    break
            self._previous_displacement = max_displacement

            gradient = self.gradient(self.coordinates)
            previous_delta = delta
            delta = -gradient
            denom = float(np.sum(previous_delta * previous_delta))
            beta = float(np.sum(delta * (delta - previous_delta)) / denom) if denom > 0 else 0.0
            direction = delta + beta * delta
            direction[~self.free] = 0.0
            slope = float(np.sum(direction * gradient))
            self.coordinates, current_cost, _ = self._backtrack(
                self.coordinates, direction, current_cost, slope, gamma=0.2
            )
            self.cost_history.append(current_cost)
        return self.coordinates


def _surface_exclusion_indices(subbasename: str, resolution: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    pial = readdfs(f"{subbasename}.pial.cortex.dfs")
    inner = readdfs(f"{subbasename}.inner.cortex.dfs")
    vertices = np.vstack((np.asarray(pial.vertices), np.asarray(inner.vertices)))
    voxels = np.floor(vertices / resolution[None, :] + 0.5).astype(np.int64)
    valid = np.all((voxels >= 0) & (voxels < np.asarray(shape)[None, :]), axis=1)
    voxels = voxels[valid]
    return np.unique(np.ravel_multi_index(voxels.T, shape, order="F"))


def svreg_elastic_vol_reg_nonlin_cg(
    subbasename: str,
    tmp_base: str,
    iterations: int = 100,
    regul_alpha: float | None = None,
) -> np.ndarray:
    """Python implementation of SVReg's nonlinear intensity refinement."""
    subbasename = normalize_fileprefix(subbasename)
    vs_volume = load_nii_z(f"{tmp_base}.surfreg.nii.gz")
    target_volume = load_nii_z(f"{subbasename}.pvc.frac.nii.gz")
    map_volume = load_nii_z(f"{tmp_base}.surfreg.map.nii.gz")
    atlas_label = load_nii_z(f"{tmp_base}.target.label.nii.gz")
    atlas_pvc = load_nii_z(f"{tmp_base}.target.pvc.frac.nii.gz")
    source = gaussian_filter(np.asarray(vs_volume.img, dtype=np.float64), sigma=1.0, radius=4)
    target = gaussian_filter(np.asarray(target_volume.img, dtype=np.float64), sigma=1.0, radius=4)
    mask = (target > 0.0) | (source > 0.0)
    shape = tuple(int(v) for v in source.shape[:3])
    mask[[0, -1], :, :] = False
    mask[:, [0, -1], :] = False
    mask[:, :, [0, -1]] = False

    resolution = np.asarray(vs_volume.zooms[:3], dtype=np.float64)
    surface_flat = _surface_exclusion_indices(subbasename, resolution, shape)
    mask_flat = np.flatnonzero(mask.ravel(order="F"))
    if surface_flat.size:
        mask_flat = np.setdiff1d(mask_flat, surface_flat, assume_unique=True)
    if mask_flat.size == 0:
        raise ValueError("nonlinear intensity refinement mask is empty")
    coords = np.column_stack(np.unravel_index(mask_flat, shape, order="F"))
    operator, boundary = second_difference_laplacian(mask_flat, shape)
    if regul_alpha is None:
        regul_alpha = 0.35 / np.sqrt(float(np.prod(resolution)))
    refiner = ElasticIntensityRefiner(source, target, coords.astype(np.float64), operator, boundary, regul_alpha)
    optimized = refiner.optimize(iterations)

    refined_map = np.asarray(map_volume.img, dtype=np.float32).copy()
    refined_values = np.column_stack(
        [_sample_scalar(map_volume.img[..., axis], optimized) for axis in range(3)]
    ).astype(np.float32)
    refined_map[coords[:, 0], coords[:, 1], coords[:, 2], :] = refined_values
    dws_write_nii(f"{tmp_base}.svreg.map.nii.gz", refined_map, map_volume)
    shutil.copyfile(f"{tmp_base}.svreg.map.nii.gz", f"{subbasename}.svreg.map.nii.gz")

    labels = map_coordinates(
        np.asarray(atlas_label.img, dtype=np.float64),
        [refined_map[..., axis] - 1.0 for axis in range(3)],
        order=0,
        mode="constant",
        cval=0.0,
        prefilter=False,
    ).astype(np.int16)
    cerebrum = load_nii_z(f"{subbasename}.cerebrum.mask.nii.gz")
    label_header = cerebrum.header.copy()
    label_header.set_data_dtype(np.int16)
    label_volume = NiftiVolume(labels, cerebrum.affine, label_header, cerebrum.path)
    save_untouch_nii_gz(label_volume, f"{tmp_base}.svreg.label.nii.gz")
    save_untouch_nii_gz(label_volume, f"{subbasename}.svreg.init.label.nii.gz")

    pvc = map_coordinates(
        np.asarray(atlas_pvc.img, dtype=np.float64),
        [refined_map[..., axis] - 1.0 for axis in range(3)],
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    ).astype(np.float32)
    pvc = (np.floor(pvc.astype(np.float64) * 1e12) / 1e12).astype(np.float32)
    pvc *= np.asarray(target_volume.img > 0, dtype=np.float32)
    output = NiftiVolume(pvc, target_volume.affine, target_volume.header.copy(), target_volume.path)
    save_untouch_nii_gz(output, f"{tmp_base}.svreg.nii.gz")
    return refined_map
