"""Surface geodesic curve refinement used by ``refine_sulci_hemi``."""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

try:
    from . import native
except ImportError:
    import native

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree


def _edge_graph(surface, weighted: bool = False) -> csr_matrix:
    _, graph = native.vertices_connectivity_fast(surface)
    if not weighted:
        return graph.astype(np.float64)
    rows, cols = graph.nonzero()
    vertices = np.asarray(surface.vertices, dtype=np.float64)
    weights = np.linalg.norm(vertices[rows] - vertices[cols], axis=1)
    return csr_matrix((weights, (rows, cols)), shape=graph.shape)


def _shortest_path(graph: csr_matrix, start: int, end: int) -> np.ndarray:
    """Return an unweighted shortest path, retaining sorted-neighbor tie order."""
    start, end = int(start), int(end)
    if start == end:
        return np.asarray([start], dtype=np.int32)
    predecessor = np.full(graph.shape[0], -1, dtype=np.int64)
    predecessor[start] = start
    queue = deque([start])
    while queue:
        vertex = queue.popleft()
        for neighbor in graph.indices[graph.indptr[vertex] : graph.indptr[vertex + 1]]:
            if predecessor[neighbor] >= 0:
                continue
            predecessor[neighbor] = vertex
            if neighbor == end:
                queue.clear()
                break
            queue.append(int(neighbor))
    if predecessor[end] < 0:
        return np.zeros(0, dtype=np.int32)
    path = [end]
    while path[-1] != start:
        path.append(int(predecessor[path[-1]]))
    return np.asarray(path[::-1], dtype=np.int32)


def _compact_largest_face_component(vertices: np.ndarray, faces: np.ndarray):
    faces = np.asarray(faces, dtype=np.int64)
    if faces.size == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32), np.zeros(0, np.int32)
    _, first = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
    faces = faces[np.sort(first)]
    faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2])]
    if faces.size == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32), np.zeros(0, np.int32)

    n_vertices = int(np.max(faces)) + 1
    incidence = csr_matrix(
        (np.ones(faces.size, dtype=np.uint8), (faces.ravel(), np.repeat(np.arange(len(faces)), 3))),
        shape=(n_vertices, len(faces)),
    )
    face_graph = (incidence.T @ incidence).tocsr()
    face_graph.setdiag(0)
    face_graph.eliminate_zeros()
    _, components = connected_components(face_graph, directed=False)
    counts = np.bincount(components)
    keep = int(np.argmax(counts))
    faces = faces[components == keep]
    used = np.unique(faces)
    remap = np.full(n_vertices, -1, dtype=np.int32)
    remap[used] = np.arange(used.size, dtype=np.int32)
    return np.asarray(vertices[used], dtype=np.float32), remap[faces], used.astype(np.int32)


def _boundary_vertices(faces: np.ndarray) -> np.ndarray:
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges = np.sort(edges, axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    return np.unique(unique[counts == 1])


def _smooth_vertices(vertices: np.ndarray, graph: csr_matrix, alpha: float, iterations: int) -> np.ndarray:
    degree = np.asarray(graph.sum(axis=1)).ravel()
    degree[degree == 0] = 1.0
    smooth = diags(alpha / degree) @ graph + diags(np.full(graph.shape[0], 1.0 - alpha))
    out = np.asarray(vertices, dtype=np.float64)
    for _ in range(iterations):
        out = smooth @ out
    return out


def _curvature_sigmoid(vertices: np.ndarray, faces: np.ndarray, graph: csr_matrix, sigmoid: float) -> np.ndarray:
    tri = vertices[faces]
    face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area2 = np.linalg.norm(face_normals, axis=1)
    valid = area2 > np.finfo(float).eps
    face_normals[valid] /= area2[valid, None]
    face_normals[~valid] = 0
    rows = np.tile(faces.ravel(), 1)
    cols = np.repeat(np.arange(len(faces)), 3)
    incidence = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(vertices), len(faces)))
    area = 0.5 * area2
    mass = np.asarray(incidence @ area).ravel()
    mass[mass == 0] = 1.0
    normals = -(incidence @ (face_normals * area[:, None])) / mass[:, None]
    normal_len = np.linalg.norm(normals, axis=1)
    normal_len[normal_len == 0] = 1.0
    normals /= normal_len[:, None]

    edge_rows, edge_cols = graph.nonzero()
    edge = vertices[edge_cols] - vertices[edge_rows]
    edge_len = np.linalg.norm(edge, axis=1)
    edge_len[edge_len == 0] = 1.0
    edge /= edge_len[:, None]
    dots = np.einsum("ij,ij->i", normals[edge_rows], edge)
    angles = np.arccos(np.clip(dots, -1.0, 1.0))
    counts = np.asarray(graph.sum(axis=1)).ravel()
    counts[counts == 0] = 1.0
    curvature = np.bincount(edge_rows, weights=angles, minlength=len(vertices)) / counts - np.pi / 2
    return (1.0 / (1.0 + np.exp(np.clip(-curvature * sigmoid, -700, 700))) - 0.5)


def _fem_operators(vertices: np.ndarray, faces: np.ndarray, weight: np.ndarray):
    n_vertices = len(vertices)
    n_faces = len(faces)
    tri = vertices[faces].astype(np.float64)
    e21 = tri[:, 1] - tri[:, 0]
    e31 = tri[:, 2] - tri[:, 0]
    x2 = np.linalg.norm(e21, axis=1)
    x2[x2 == 0] = 1.0
    x3 = np.einsum("ij,ij->i", e31, e21 / x2[:, None])
    cross = np.cross(e21, e31)
    yunit = np.cross(cross, e21)
    ynorm = np.linalg.norm(yunit, axis=1)
    ynorm[ynorm == 0] = 1.0
    y3 = np.einsum("ij,ij->i", yunit, e31) / ynorm
    twice_area = np.abs(x2 * y3)
    twice_area[twice_area == 0] = np.finfo(float).tiny
    area = 0.5 * twice_area

    x1, y1, y2 = np.zeros(n_faces), np.zeros(n_faces), np.zeros(n_faces)
    tmp_a = np.column_stack((y2 - y3, y3 - y1, y1 - y2)) / twice_area[:, None]
    tmp_b = np.column_stack((x3 - x2, x1 - x3, x2 - x1)) / twice_area[:, None]
    rows = np.repeat(np.arange(n_faces), 3)
    cols = faces.ravel()
    dx = csr_matrix((tmp_a.ravel(), (rows, cols)), shape=(n_faces, n_vertices))
    dy = csr_matrix((tmp_b.ravel(), (rows, cols)), shape=(n_faces, n_vertices))

    incidence = csr_matrix((np.ones(3 * n_faces), (cols, rows)), shape=(n_vertices, n_faces))
    face_weight = (incidence.T @ weight) / 3.0 * area
    stiffness = dx.T @ diags(face_weight) @ dx + dy.T @ diags(face_weight) @ dy

    f = faces
    mass_rows = np.concatenate((f[:, 0], f[:, 0], f[:, 0], f[:, 1], f[:, 1], f[:, 1], f[:, 2], f[:, 2], f[:, 2]))
    mass_cols = np.concatenate((f[:, 0], f[:, 1], f[:, 2], f[:, 0], f[:, 1], f[:, 2], f[:, 0], f[:, 1], f[:, 2]))
    mass_vals = np.concatenate((area / 6, area / 12, area / 12, area / 12, area / 6, area / 12, area / 12, area / 12, area / 6))
    mass = csr_matrix((mass_vals, (mass_rows, mass_cols)), shape=(n_vertices, n_vertices))
    vertex_area = np.asarray(incidence @ area).ravel()
    vertex_area[vertex_area == 0] = 1.0
    vertex_face = diags(1.0 / vertex_area) @ incidence @ diags(area)
    return stiffness.tocsr(), mass, dx, dy, vertex_face.tocsr()


def _zero_level_set(vertices: np.ndarray, faces: np.ndarray, phi: np.ndarray):
    tri_phi = phi[faces]
    crossing = (tri_phi.min(axis=1) < 0) & (tri_phi.max(axis=1) >= 0)
    tri = faces[crossing]
    if tri.size == 0:
        return np.zeros((0, 3), np.float64), tri
    edges = np.unique(np.sort(np.vstack((tri[:, [0, 1]], tri[:, [0, 2]], tri[:, [1, 2]])), axis=1), axis=0)
    a, b = phi[edges[:, 0]].astype(np.float64), phi[edges[:, 1]].astype(np.float64)
    keep = a * b <= 0
    edges, a, b = edges[keep], a[keep], b[keep]
    a[a == 0] = np.finfo(float).eps
    b[b == 0] = np.finfo(float).eps
    w1 = -a / b
    w2 = -b / a
    wt1 = w1 / (w1 + w2)
    wt2 = w2 / (w1 + w2)
    points = wt2[:, None] * vertices[edges[:, 0]] + wt1[:, None] * vertices[edges[:, 1]]
    return points, tri


def _geodesic_refine_curve(surface, curve: np.ndarray, graph_weighted: csr_matrix) -> np.ndarray:
    curve = native.remove_curve_defects(curve)
    if curve.shape[0] == 0:
        return curve
    vertices = np.asarray(surface.vertices, dtype=np.float64)
    faces = np.asarray(surface.faces, dtype=np.int32)
    nearest = cKDTree(vertices).query(curve, k=1)[1].astype(np.int32)
    # The MATLAB code subsequently discards every face at 20 mm or farther,
    # so stop the full-hemisphere search at that radius.
    dist = dijkstra(graph_weighted, indices=np.unique(nearest), min_only=True, limit=20.0)
    in_roi = dist < 20.0
    roi_faces = faces[np.all(in_roi[faces], axis=1)]
    if roi_faces.shape[0] < 5:
        return curve

    roi_vertices, roi_faces, roi_ids = _compact_largest_face_component(vertices, roi_faces)
    if roi_faces.shape[0] < 5 or roi_vertices.shape[0] < 20:
        return curve
    to_roi = np.full(vertices.shape[0], -1, dtype=np.int32)
    to_roi[roi_ids] = np.arange(roi_ids.size, dtype=np.int32)
    roi_nearest = to_roi[nearest]
    if np.any(roi_nearest < 0):
        return curve

    patch = SimpleNamespace(vertices=roi_vertices, faces=roi_faces)
    graph = _edge_graph(patch)
    boundary = _boundary_vertices(roi_faces)
    if boundary.size == 0:
        return curve
    start_dist = np.linalg.norm(roi_vertices[boundary] - curve[0], axis=1)
    end_dist = np.linalg.norm(roi_vertices[boundary] - curve[-1], axis=1)
    start_boundary = int(boundary[np.argmin(start_dist - end_dist)])
    end_boundary = int(boundary[np.argmax(start_dist - end_dist)])

    path_start = _shortest_path(graph, start_boundary, int(roi_nearest[0]))
    path_end = _shortest_path(graph, int(roi_nearest[-1]), end_boundary)
    if path_start.size == 0 or path_end.size == 0:
        return curve
    sulind = [path_start]
    for a, b in zip(roi_nearest[:-1], roi_nearest[1:]):
        segment = _shortest_path(graph, int(a), int(b))
        if segment.size == 0:
            return curve
        sulind.append(segment[:-1])
    sulind.append(path_end)
    sulind = np.concatenate(sulind).astype(np.int32)

    cut_graph = graph.copy().tolil()
    cut_graph[sulind, :] = 0
    cut_graph[:, sulind] = 0
    cut_graph = cut_graph.tocsr()
    cut_graph.eliminate_zeros()
    _, components = connected_components(cut_graph, directed=False)
    components[sulind] = -1
    labels, counts = np.unique(components[components >= 0], return_counts=True)
    if labels.size < 2:
        return curve
    order = np.argsort(counts)
    largest = int(labels[order[-1]])
    sign = np.ones(roi_vertices.shape[0], dtype=np.float64)
    sign[components == largest] = -1.0
    smooth1 = _smooth_vertices(roi_vertices, graph, 0.1, 5)
    smooth2 = _smooth_vertices(smooth1, graph, 0.2, 1000)
    # This is the physical-edge geodesic distance field used by
    # signed_dist_func.m, seeded at the complete path.
    smoothed_patch = SimpleNamespace(vertices=smooth1, faces=roi_faces)
    phi = dijkstra(_edge_graph(smoothed_patch, weighted=True), indices=np.unique(sulind), min_only=True)
    phi *= sign
    max_abs = float(np.max(np.abs(phi)))
    if max_abs == 0:
        return curve
    phi /= max_abs

    weight = 1.0 + _curvature_sigmoid(smooth2, roi_faces, graph, 50.0)
    weight = weight ** 4
    stiffness, mass, dx, dy, vertex_face = _fem_operators(smooth1, roi_faces, weight)
    dt = 1.0
    matrix = (mass + 0.5 * dt * stiffness).tocsr()
    rhs_matrix = (mass - 0.5 * dt * stiffness).tocsr()
    solve = splu(matrix.tocsc()).solve

    def flow_term(values):
        gx, gy = dx @ values, dy @ values
        grad_norm = np.sqrt(gx * gx + gy * gy)
        grad_vertex = dx @ (vertex_face @ grad_norm) * gx + dy @ (vertex_face @ grad_norm) * gy
        grad_vertex = -grad_vertex / (grad_norm + 1e-6)
        return weight * (vertex_face @ grad_vertex)

    g_phi = flow_term(phi)
    t1, t2 = phi.copy(), np.zeros_like(phi)
    for _ in range(5):
        t1 = solve(rhs_matrix @ phi)
        t2 = solve(dt * (mass @ g_phi))
        phi = t1 + t2
        g_phi = flow_term(phi)

    _, zero_tri = _zero_level_set(smooth2, roi_faces, phi)
    if zero_tri.shape[0] < 5:
        return curve
    zero_vertices, zero_faces, zero_ids = _compact_largest_face_component(smooth2, zero_tri)
    if zero_faces.shape[0] == 0:
        return curve
    shared, boundary_idx, zero_idx = np.intersect1d(boundary, zero_ids, return_indices=True)
    if shared.size < 2:
        return curve
    first = zero_vertices[zero_idx[0]]
    farthest = int(zero_idx[1 + np.argmax(np.sum((zero_vertices[zero_idx[1:]] - first) ** 2, axis=1))])
    zero_surface = SimpleNamespace(vertices=zero_vertices, faces=zero_faces)
    zero_graph = _edge_graph(zero_surface)
    zero_path = _shortest_path(zero_graph, int(zero_idx[0]), farthest)
    if zero_path.size < 4:
        return np.zeros((0, 3), dtype=np.float32)

    total = path_start.size + path_end.size + len(roi_nearest)
    first_index = int(native.matlab_round(zero_path.size * path_start.size / total))
    last_index = int(native.matlab_round(zero_path.size * (total - path_end.size) / total))
    begin = min(first_index, last_index)
    end = min(last_index, zero_path.size)
    if begin < 1 or end <= begin:
        return curve
    selected = zero_path[begin - 1 : end]
    projected = cKDTree(smooth2).query(zero_vertices[selected], k=1)[1]
    return roi_vertices[projected].astype(np.float32)


def refine_curves_on_surface(surface, curves: list[np.ndarray]) -> list[np.ndarray]:
    weighted_graph = _edge_graph(surface, weighted=True)
    refined = []
    for curve in curves:
        refined.append(_geodesic_refine_curve(surface, curve, weighted_graph))
    return refined
