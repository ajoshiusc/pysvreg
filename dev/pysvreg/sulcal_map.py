"""Map refined sulcal curves to cortical surfaces and their volume labels."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree

from . import native
from .refine_sulci_hemi import refine_sulci_hemi
from .runtime import normalize_fileprefix


_CURVE26_TO_36 = (1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 18, 19, 20, 21, 22, 23, 24, 25, 26, 29, 30, 31, 33, 35)


def _source_style_curve_distance(vertices: np.ndarray, adjacency, seeds: np.ndarray) -> np.ndarray:
    """Run the source implementation's two in-place relaxation sweeps."""
    return _source_style_curve_distances(vertices, adjacency, [seeds])[:, 0]


def _source_style_curve_distances(
    vertices: np.ndarray, adjacency, seed_groups: list[np.ndarray]
) -> np.ndarray:
    """Run the same ordered sweeps for multiple independent curves at once."""
    n_vertices = len(vertices)
    distances = np.full((n_vertices, len(seed_groups)), 100_000.0, dtype=np.float64)
    for column, seeds in enumerate(seed_groups):
        distances[np.asarray(seeds, dtype=np.int64), column] = 0.0

    indptr, indices = adjacency.indptr, adjacency.indices
    row_ids = np.repeat(np.arange(n_vertices, dtype=np.int64), np.diff(indptr))
    edge_lengths = np.linalg.norm(
        np.asarray(vertices, dtype=np.float64)[row_ids]
        - np.asarray(vertices, dtype=np.float64)[indices],
        axis=1,
    )
    for _ in range(2):
        for vertex in range(n_vertices):
            start, stop = indptr[vertex : vertex + 2]
            neighbors = indices[start:stop]
            if not len(neighbors):
                continue
            candidate = np.min(
                distances[neighbors] + edge_lengths[start:stop, None], axis=0
            )
            distances[vertex] = np.minimum(distances[vertex], candidate)
    return distances


def _curvature_positive(surface) -> np.ndarray:
    return native.patchcurvature_mean(surface) > 0


def _curve_labels(surface, curve_path: Path, hemi: str, dist_threshold: float):
    curves, _ = native.readdfc_sipi(str(curve_path))
    if len(curves) == 26:
        expanded = [np.zeros((0, 3), dtype=np.float32) for _ in range(36)]
        for index, target in enumerate(_CURVE26_TO_36):
            expanded[target - 1] = curves[index]
        curves = expanded

    vertices = np.asarray(surface.vertices, dtype=np.float64)
    point_tree = cKDTree(vertices)
    adjacency = native.vertices_connectivity_fast(surface)[1]
    min_dist = np.full(len(vertices), 100_000.0, dtype=np.float64)
    curve_id = np.zeros(len(vertices), dtype=np.int32)
    active_labels = [label for label, curve in enumerate(curves, start=1) if curve.size]
    active_curves = [curves[label - 1] for label in active_labels]
    seed_groups = []
    for curve in active_curves:
        _, seeds = point_tree.query(np.asarray(curve, dtype=np.float64), k=1)
        seed_groups.append(np.unique(np.asarray(seeds, dtype=np.int32)))
    if seed_groups:
        curve_distances = _source_style_curve_distances(vertices, adjacency, seed_groups)
    else:
        curve_distances = np.empty((len(vertices), 0), dtype=np.float64)
    for column, label in enumerate(active_labels):
        geodesic = curve_distances[:, column].copy()
        geodesic[geodesic > 20.0] = 1_000_000.0
        closer = geodesic < min_dist
        min_dist[closer] = geodesic[closer]
        curve_id[closer] = label

    curvature_positive = _curvature_positive(surface)
    attributes = curvature_positive.copy()
    attributes[min_dist < 2.0] = True
    source_labels = np.asarray(getattr(surface, "labels", np.zeros(len(vertices))), dtype=np.int64).reshape(-1)
    encoded = _source_topology_labels(surface, curve_id, min_dist, attributes, curvature_positive, source_labels, hemi, dist_threshold)
    return encoded, attributes, curves


def _source_topology_labels(
    surface,
    curve_ids: np.ndarray,
    distance: np.ndarray,
    attributes: np.ndarray,
    curvature_positive: np.ndarray,
    source_labels: np.ndarray,
    hemi: str,
    dist_threshold: float,
) -> np.ndarray:
    """Apply sulc_map_surf's face trimming, face median, and largest component rules."""
    faces = np.asarray(surface.faces, dtype=np.int64)
    n_vertices = int(surface.vertices.shape[0])
    labels = np.zeros(n_vertices, dtype=np.int32)
    if faces.size == 0:
        return labels.astype(np.uint16)

    # myclean_patch3 removes faces with two or more vertices outside the
    # positive-curvature/near-curve mask, then removes unused vertices.
    face_keep = np.sum(~attributes[faces], axis=1) <= 1
    retained_faces = faces[face_keep]
    if retained_faces.size == 0:
        return labels.astype(np.uint16)
    retained_vertices = np.unique(retained_faces.reshape(-1))
    provisional = np.zeros(n_vertices, dtype=np.int32)
    provisional[retained_vertices] = curve_ids[retained_vertices]
    provisional[distance > dist_threshold] = 0
    face_labels = np.median(provisional[faces], axis=1).astype(np.int32)

    incidence = native.face_v_conn(surface)
    face_adjacency = (incidence.T @ incidence).tocsr()
    if face_adjacency.nnz:
        face_adjacency.data[:] = 1
    face_adjacency.setdiag(1)
    hemisphere_bit = 1 if hemi == "left" else 0
    for curve_id in np.unique(curve_ids):
        if curve_id == 0:
            continue
        candidate_faces = np.flatnonzero(face_labels == int(curve_id))
        if candidate_faces.size == 0:
            continue
        subgraph = face_adjacency[candidate_faces][:, candidate_faces]
        n_components, component_ids = native.connected_components(subgraph, directed=False)
        if n_components > 1:
            component_sizes = np.bincount(component_ids, minlength=n_components)
            # myclean_patch_cc ignores a component when the input surface's
            # label at its first vertex is zero, then keeps the largest one.
            eligible = np.ones(n_components, dtype=bool)
            if source_labels.size == n_vertices:
                for component_id in range(n_components):
                    component_faces = candidate_faces[component_ids == component_id]
                    first_vertices = faces[component_faces, 0]
                    if np.median(source_labels[first_vertices]) == 0:
                        eligible[component_id] = False
            eligible_sizes = np.where(eligible, component_sizes, -1)
            if np.max(eligible_sizes) < 0:
                continue
            keep_component = int(np.argmax(eligible_sizes))
            candidate_faces = candidate_faces[component_ids == keep_component]
        elif source_labels.size == n_vertices and np.median(source_labels[faces[candidate_faces, 0]]) == 0:
            continue

        kept_vertices = np.unique(faces[candidate_faces].reshape(-1))
        labels[kept_vertices] = int(curve_id)

    labels[~curvature_positive] = 0
    labels[labels != 0] = 4000 + hemisphere_bit + 2 * (labels[labels != 0] - 1)
    return labels.astype(np.uint16)


def _map_surface(prefix: Path, hemi: str, xmlf: Path, xmlc: Path, dist_threshold: float) -> None:
    curve_path = Path(f"{prefix}.{hemi}.mapped.refined.dfc")
    if not curve_path.is_file():
        refine_sulci_hemi(str(prefix), hemi)
    if not curve_path.is_file():
        raise FileNotFoundError(curve_path)

    surface_path = Path(f"{prefix}.{hemi}.mid.cortex.svreg.dfs")
    base = native.readdfs(str(surface_path))
    smooth = native.smooth_cortex_fast(base, 0.1, 200)[0]
    labels, curvature_positive, curves = _curve_labels(smooth, curve_path, hemi, dist_threshold)

    smooth.labels = labels
    smooth.attributes = curvature_positive.astype(np.float32)
    smooth_path = Path(f"{prefix}.{hemi}.smooth.mid.cortex.sulci.dfs")
    # Keep the unsmoothed registered mesh in the primary output, matching the
    # MATLAB command; retain the smoothed mesh as a separate companion file.
    base.labels = labels
    native.writedfs(f"{prefix}.{hemi}.mid.cortex.sulci.dfs", base)
    native.recolor_by_label(f"{prefix}.{hemi}.mid.cortex.sulci.dfs", None, str(xmlf) if xmlf.is_file() else None)
    fine = native.smooth_cortex_fast(smooth, 0.1, 2200)[0]
    fine.labels = labels
    fine.attributes = curvature_positive.astype(np.float32)
    native.writedfs(str(smooth_path), fine)
    native.recolor_by_label(str(smooth_path), None, str(xmlf) if xmlf.is_file() else None)

    # Preserve/refresh the DFC header with the selected curve protocol.
    native.writedfc(str(curve_path), curves, str(xmlc))


def _map_volume(prefix: Path) -> None:
    label_path = Path(f"{prefix}.svreg.label.nii.gz")
    if not label_path.is_file():
        raise FileNotFoundError(label_path)
    label_img = nib.load(str(label_path))
    volume_labels = np.asanyarray(label_img.dataobj)
    zooms = np.asarray(label_img.header.get_zooms()[:3], dtype=np.float64)
    cortical = (np.mod(volume_labels, 1000) >= 100) & (np.mod(volume_labels, 1000) < 600)
    query_indices = np.argwhere(cortical)
    out = np.zeros(volume_labels.shape, dtype=volume_labels.dtype)
    if query_indices.size:
        points = []
        point_labels = []
        for hemi in ("left", "right"):
            sulcus = native.readdfs(f"{prefix}.{hemi}.mid.cortex.sulci.dfs")
            mid = np.asarray(sulcus.vertices, dtype=np.float64)
            inner = np.asarray(native.readdfs(f"{prefix}.{hemi}.inner.cortex.svreg.dfs").vertices, dtype=np.float64)
            pial = np.asarray(native.readdfs(f"{prefix}.{hemi}.pial.cortex.svreg.dfs").vertices, dtype=np.float64)
            half_in = 0.5 * (mid + inner)
            half_out = 0.5 * (mid + pial)
            points.extend((pial, inner, mid, half_in, half_out))
            point_labels.extend([sulcus.labels] * 5)
        surface_points = np.vstack(points)
        surface_labels = np.concatenate(point_labels).astype(np.uint16)
        tree = cKDTree(surface_points)
        source_labels = surface_labels
        coords = query_indices.astype(np.float64) * zooms[None, :]
        chunk = 250_000
        mapped = np.zeros(len(query_indices), dtype=volume_labels.dtype)
        for start in range(0, len(query_indices), chunk):
            stop = min(start + chunk, len(query_indices))
            _, nearest = tree.query(coords[start:stop], k=1, workers=-1)
            mapped[start:stop] = source_labels[nearest]
        out[tuple(query_indices.T)] = mapped

    result = nib.Nifti1Image(out, label_img.affine, label_img.header.copy())
    nib.save(result, str(prefix) + ".sulci.label.nii.gz")


def svreg_sulcal_map(args: list[str]) -> int:
    if len(args) < 2:
        print("usage: svreg_sulcal_map SUBJECT_PREFIX ATLAS_PREFIX [LABEL_XML] [CURVE_XML] [DISTANCE_MM]", file=sys.stderr)
        return 2
    subject = Path(normalize_fileprefix(args[0]))
    atlas = Path(normalize_fileprefix(args[1]))
    xmlf = Path(args[2]) if len(args) > 2 and Path(args[2]).is_file() else atlas.parent / "brainsuite_labeldescription.xml"
    protocol = Path(args[3]) if len(args) > 3 and Path(args[3]).is_file() else Path(f"{subject}.sulcal_protocol_HD.xml")
    if len(args) > 3 and Path(args[3]).is_file():
        shutil.copyfile(protocol, Path(f"{subject}.sulcal_protocol_HD.xml"))
    distance = float(args[4]) if len(args) > 4 else 10.0
    if not protocol.is_file():
        raise FileNotFoundError(protocol)
    print("Generating sulcal regions on cortical surfaces")
    for hemi in ("left", "right"):
        _map_surface(subject, hemi, xmlf, protocol, distance)
    print("Generating volumetric sulcal regions")
    _map_volume(subject)
    return 0


def main(argv: list[str] | None = None) -> int:
    return svreg_sulcal_map(list(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    raise SystemExit(main())

