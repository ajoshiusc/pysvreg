"""Curvature-weighted geodesic ROI boundary refinement from SVReg sources."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.csgraph import connected_components, dijkstra

from . import native


EXCLUDED_PAIRS = {
    (142, 150), (144, 146), (146, 172), (144, 172), (142, 144),
    (150, 182), (182, 222), (182, 242), (120, 182), (144, 150),
}


def _nondegenerate(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    eps = np.finfo(np.float64).eps
    d01 = np.sum(np.abs(tri[:, 0] - tri[:, 1]) > eps, axis=1)
    d02 = np.sum(np.abs(tri[:, 0] - tri[:, 2]) > eps, axis=1)
    d12 = np.sum(np.abs(tri[:, 1] - tri[:, 2]) > eps, axis=1)
    return (d01 * d02 * d12) > 0


def _compact_pair(surface, pair: tuple[int, int]):
    vertices = np.asarray(surface.vertices, dtype=np.float64)
    labels = np.asarray(surface.labels, dtype=np.int64).reshape(-1)
    faces = np.asarray(surface.faces, dtype=np.int64)
    keep = np.isin(labels[faces], pair).all(axis=1)
    faces = faces[keep]
    if not faces.size:
        return None, np.empty(0, dtype=np.int64)
    # MATLAB unique(...,'rows') emits faces in lexicographic sorted-row order.
    _, first = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
    faces = faces[first]
    faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 0] != faces[:, 2]) & (faces[:, 1] != faces[:, 2])]
    faces = faces[_nondegenerate(vertices, faces)]
    if not faces.size:
        return None, np.empty(0, dtype=np.int64)

    used = np.unique(faces)
    remap = np.full(vertices.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.size)
    faces = remap[faces]
    local_vertices, local_labels, original_ids = vertices[used], labels[used], used.copy()

    # myclean_patch3 rounds coordinates to 1e-15, uniques them, and carries
    # labels back from the first matching vertex in the input surface.
    rounded = np.rint(local_vertices * 1e15) * 1e-15
    local_vertices, first, inverse = np.unique(rounded, axis=0, return_index=True, return_inverse=True)
    local_labels, original_ids = local_labels[first], original_ids[first]
    faces = inverse[faces]
    _, first = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
    faces = faces[first]
    faces = faces[_nondegenerate(local_vertices, faces)]
    if not faces.size:
        return None, np.empty(0, dtype=np.int64)

    # faces2faces_connectivity marks faces that share an edge (at least two
    # vertices); two patches touching at only one vertex stay disconnected.
    tri_edges = np.vstack((faces[:, [0, 1]], faces[:, [0, 2]], faces[:, [1, 2]]))
    tri_edges.sort(axis=1)
    _, edge_ids = np.unique(tri_edges, axis=0, return_inverse=True)
    edge_incidence = csr_matrix(
        (np.ones(edge_ids.size, dtype=np.uint8),
         (edge_ids, np.tile(np.arange(faces.shape[0]), 3))),
        shape=(int(edge_ids.max()) + 1, faces.shape[0]),
    )
    face_graph = (edge_incidence.T @ edge_incidence).tocsr()
    face_graph.setdiag(0)
    face_graph.eliminate_zeros()
    ncomp, component = connected_components(face_graph, directed=False)
    scores = np.zeros(ncomp, dtype=np.int64)
    for ci in range(ncomp):
        ids = np.flatnonzero(component == ci)
        if ids.size and np.median(local_labels[faces[ids, 0]]) != 0:
            scores[ci] = ids.size
    if not np.any(scores):
        return None, np.empty(0, dtype=np.int64)
    faces = faces[component == int(np.argmax(scores))]
    used = np.unique(faces)
    remap = np.full(local_vertices.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.size)
    patch = SimpleNamespace(
        vertices=local_vertices[used].astype(np.float32),
        faces=remap[faces].astype(np.int32),
        labels=local_labels[used].astype(np.uint16),
    )
    return patch, original_ids[used]


def _mass_matrix(surface) -> csr_matrix:
    faces = np.asarray(surface.faces, dtype=np.int64)
    area = native.tri_area(faces, np.asarray(surface.vertices, dtype=np.float64))
    rows = np.concatenate([faces[:, i] for i in (0, 0, 0, 1, 1, 1, 2, 2, 2)])
    cols = np.concatenate([faces[:, i] for i in (0, 1, 2, 0, 1, 2, 0, 1, 2)])
    weights = np.concatenate([area / divisor for divisor in (6, 12, 12, 12, 6, 12, 12, 12, 6)])
    return csr_matrix((weights, (rows, cols)), shape=(len(surface.vertices), len(surface.vertices)))


def _matlab_pcg(matrix: csr_matrix, rhs: np.ndarray, diagonal: np.ndarray,
                initial: np.ndarray, maxiter: int = 300) -> tuple[np.ndarray, int]:
    """Port of mypcg.m, including its r'*M^-1*r stopping check."""
    x = np.asarray(initial, dtype=np.float64).copy()
    r = rhs - matrix @ x
    invdiag = 1.0 / np.asarray(diagonal, dtype=np.float64)
    z = invdiag * r
    p = z.copy()
    rtz = float(r @ z)
    iters = 0
    for i in range(maxiter):
        if rtz < 1e-200:
            break
        ap = matrix @ p
        alpha = rtz / float(p @ ap)
        x += alpha * p
        r -= alpha * ap
        z = invdiag * r
        new_rtz = float(r @ z)
        beta = new_rtz / rtz
        p = z + beta * p
        rtz = new_rtz
        iters = i + 1
    return x, iters


def _zero_level_count(surface, phi: np.ndarray) -> int:
    faces = np.asarray(surface.faces, dtype=np.int64)
    values = np.asarray(phi)[faces]
    crossing = (values.min(axis=1) < 0) & (values.max(axis=1) >= 0)
    if not np.any(crossing):
        return 0
    tri = faces[crossing]
    edges = np.unique(np.sort(np.vstack((tri[:, [0, 1]], tri[:, [0, 2]], tri[:, [1, 2]])), axis=1), axis=0)
    keep = phi[edges[:, 0]] * phi[edges[:, 1]] <= 0
    # get_zero_level_set returns one point for each sign-changing edge.
    return int(np.count_nonzero(keep))


def _flow_pair(surface, pair: tuple[int, int], *, maxiter: int = 300):
    labels = np.asarray(surface.labels, dtype=np.int64)
    faces = np.asarray(surface.faces, dtype=np.int64)
    vertices = np.asarray(surface.vertices, dtype=np.float64)
    face_label = np.median(labels[faces], axis=1)
    p1 = np.unique(faces[face_label == pair[0]])
    p2 = np.unique(faces[face_label == pair[1]])
    boundary = np.intersect1d(p1, p2, assume_unique=True)
    if not boundary.size:
        return None, {"reason": "no_shared_boundary"}

    _, adjacency = native.vertices_connectivity_fast(surface)
    rows, cols = adjacency.nonzero()
    lengths = np.linalg.norm(vertices[rows] - vertices[cols], axis=1)
    distance = dijkstra(csr_matrix((lengths, (rows, cols)), shape=adjacency.shape),
                        indices=boundary, min_only=True)
    if not np.all(np.isfinite(distance)):
        return None, {"reason": "disconnected_patch"}
    phi = np.asarray(distance, dtype=np.float64)
    phi[labels == pair[1]] *= -1.0
    scale = float(np.max(np.abs(phi), initial=0.0))
    if scale == 0:
        return None, {"reason": "zero_distance"}
    phi /= scale

    smooth1, smoothing = native.smooth_cortex_fast(surface, 0.1, 5)
    smooth2, _ = native.smooth_cortex_fast(smooth1, 0.2, 1000, smoothing)
    _, adjacency = native.vertices_connectivity_fast(smooth1)
    curvature, _ = native.curvature_cortex_fast(smooth2, 50.0, 0, adjacency)
    weight = (1.0 + np.asarray(curvature, dtype=np.float64)) ** 4
    stiffness, dx, dy = native.get_stiffness_matrix_tri_wt(smooth1, weight)
    mass = _mass_matrix(smooth1)
    tri_to_nodes = native.tri2nodes(smooth1)

    def update_force(phi_now):
        grad_norm = np.sqrt((dx @ phi_now) ** 2 + (dy @ phi_now) ** 2)
        grad_at_nodes = tri_to_nodes @ grad_norm
        gx, gy = dx @ phi_now, dy @ phi_now
        force = -((dx @ grad_at_nodes) * gx + (dy @ grad_at_nodes) * gy) / (grad_norm + 1e-6)
        return weight * (tri_to_nodes @ force)

    force = update_force(phi)
    initial = phi.copy()
    system = (mass + 0.5 * stiffness).tocsr()
    rhs_operator = mass - 0.5 * stiffness
    diagonal = np.asarray(system.diagonal())
    zero_points = _zero_level_count(smooth2, phi)
    nit = 4 * int(native.matlab_round(zero_points / 400.0))
    t1, t2 = phi.copy(), np.zeros_like(phi)
    total_pcg = 0
    for _ in range(nit):
        t1, n1 = _matlab_pcg(system, rhs_operator @ phi, diagonal, t1, maxiter)
        t2, n2 = _matlab_pcg(system, mass @ force, diagonal, t2, maxiter)
        total_pcg += n1 + n2
        phi = t1 + t2
        force = update_force(phi)

    delta = float(np.max(np.abs(initial - phi), initial=0.0))
    stats = {"zero_points": zero_points, "nit": nit, "pcg_iterations": total_pcg,
             "max_delta": delta, "accepted": bool(np.all(np.isfinite(initial + phi)) and delta <= 15.0)}
    return (phi if stats["accepted"] else None), stats


def refine_surface(
    surface,
    *,
    include_zero: bool = False,
    min_label: int | None = 100,
    max_label: int | None = 600,
    excluded_labels: tuple[int, ...] = (0, 324, 325),
    excluded_pairs: set[tuple[int, int]] | None = None,
    min_common_faces: int = 50,
):
    labels = np.asarray(surface.labels, dtype=np.int64)
    faces = np.asarray(surface.faces, dtype=np.int64)
    labs = np.unique(labels)
    if not include_zero:
        labs = labs[~np.isin(labs, np.asarray(excluded_labels, dtype=np.int64))]
    if min_label is not None:
        labs = labs[labs >= min_label]
    if max_label is not None:
        labs = labs[labs < max_label]
    face_labels = labels[faces]
    results = []
    skip_pairs = EXCLUDED_PAIRS if excluded_pairs is None else excluded_pairs
    for i, a in enumerate(labs):
        for b in labs[i + 1:]:
            pair = (int(a), int(b))
            if pair in skip_pairs:
                continue
            common = np.any(face_labels == a, axis=1) & np.any(face_labels == b, axis=1)
            if np.count_nonzero(common) < min_common_faces:
                continue
            patch, parent = _compact_pair(surface, pair)
            if patch is None or len(patch.faces) < 20:
                results.append((pair, {"reason": "small_patch"}))
                continue
            phi, stats = _flow_pair(patch, pair)
            if phi is None:
                results.append((pair, stats))
                continue
            # Both MATLAB refinement functions `continue` immediately when
            # this contour-derived iteration count is zero. In particular,
            # they do not threshold the initial signed distance field.
            if stats.get("nit") == 0:
                stats["reason"] = "zero_iterations"
                results.append((pair, stats))
                continue
            new = np.asarray(patch.labels, dtype=np.int64).copy()
            new[phi >= 0] = pair[0]
            new[phi < 0] = pair[1]
            before = labels[parent].copy()
            labels[parent] = new
            stats["changed"] = int(np.count_nonzero(before != new))
            results.append((pair, stats))
    surface.labels = labels.astype(np.uint16)
    return results
