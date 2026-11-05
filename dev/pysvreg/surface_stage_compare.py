from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .dfsio import readdfs
except ImportError:
    from dfsio import readdfs


def _surface_metrics(lhs_path: str, rhs_path: str) -> dict[str, float | int | str]:
    lhs = readdfs(lhs_path)
    rhs = readdfs(rhs_path)
    out: dict[str, float | int | str] = {"lhs": lhs_path, "rhs": rhs_path}

    if lhs.vertices.shape == rhs.vertices.shape:
        vertex_diff = np.linalg.norm(lhs.vertices - rhs.vertices, axis=1)
        out["vertex_mean_abs"] = float(np.mean(vertex_diff))
        out["vertex_max_abs"] = float(np.max(vertex_diff))
    if hasattr(lhs, "u") and hasattr(rhs, "u") and np.asarray(lhs.u).shape == np.asarray(rhs.u).shape:
        uv_diff = np.linalg.norm(np.column_stack([lhs.u, lhs.v]) - np.column_stack([rhs.u, rhs.v]), axis=1)
        out["uv_mean_abs"] = float(np.mean(uv_diff))
        out["uv_max_abs"] = float(np.max(uv_diff))
    if hasattr(lhs, "labels") and hasattr(rhs, "labels") and np.asarray(lhs.labels).shape == np.asarray(rhs.labels).shape:
        out["label_diff_vertices"] = int(np.count_nonzero(np.asarray(lhs.labels) != np.asarray(rhs.labels)))
    return out


def compare_surface_stage(subject_prefix: str, reference_prefix: str, hemi: str) -> dict[str, object]:
    subject_prefix = str(Path(subject_prefix))
    reference_prefix = str(Path(reference_prefix))
    subject_name = Path(subject_prefix).name
    reference_name = Path(reference_prefix).name
    subject_tmp = f"{subject_prefix}.svreg.tmp/{subject_name}"
    reference_tmp = f"{reference_prefix}.svreg.tmp/{reference_name}"

    pairs = [
        (f"{subject_tmp}.{hemi}.mid.cortex.curvesmatch.dfs", f"{reference_tmp}.{hemi}.mid.cortex.curvesmatch.dfs"),
        (f"{subject_tmp}.target.{hemi}.mid.cortex.curvesmatch.dfs", f"{reference_tmp}.target.{hemi}.mid.cortex.curvesmatch.dfs"),
        (f"{subject_tmp}.{hemi}.mid.cortex.reg.dfs", f"{reference_tmp}.{hemi}.mid.cortex.reg.dfs"),
        (f"{subject_tmp}.target.{hemi}.mid.cortex.reg.dfs", f"{reference_tmp}.target.{hemi}.mid.cortex.reg.dfs"),
        (f"{subject_prefix}.{hemi}.mid.cortex.svreg.init.dfs", f"{reference_prefix}.{hemi}.mid.cortex.svreg.init.dfs"),
    ]

    report: dict[str, object] = {"subject_prefix": subject_prefix, "reference_prefix": reference_prefix, "hemi": hemi, "files": []}
    for lhs, rhs in pairs:
        if not Path(lhs).is_file() or not Path(rhs).is_file():
            report["files"].append({"lhs": lhs, "rhs": rhs, "missing": True})
            continue
        report["files"].append(_surface_metrics(lhs, rhs))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare Python and MATLAB SVReg surface-stage intermediates.")
    parser.add_argument("subject_prefix")
    parser.add_argument("reference_prefix")
    parser.add_argument("hemi", choices=["left", "right"])
    args = parser.parse_args(argv)
    print(json.dumps(compare_surface_stage(args.subject_prefix, args.reference_prefix, args.hemi), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
