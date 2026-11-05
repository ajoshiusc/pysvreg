from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from . import _deps  # noqa: F401
except ImportError:
    import _deps  # noqa: F401

import nibabel as nib
import numpy as np

try:
    from .dfsio import readdfs
except ImportError:
    from dfsio import readdfs


def _first_existing(paths: list[str]) -> str | None:
    for path in paths:
        if Path(path).is_file():
            return path
    return None


def _load_volume(path: str | None) -> np.ndarray | None:
    if path is None:
        return None
    return np.asarray(nib.load(path).dataobj)


def _load_surface(path: str | None):
    if path is None:
        return None
    return readdfs(path)


def _surface_metric(reference_prefix: str, candidate_prefix: str, hemi: str, field: str) -> dict[str, float | str | None]:
    ref_path = _first_existing(
        [
            f"{reference_prefix}.{hemi}.mid.cortex.svreg.dfs",
            f"{reference_prefix}.{hemi}.mid.cortex.reg.dfs",
            f"{reference_prefix}.{hemi}.mid.cortex.svreg.init.dfs",
        ]
    )
    cand_path = _first_existing(
        [
            f"{candidate_prefix}.{hemi}.mid.cortex.svreg.dfs",
            f"{candidate_prefix}.{hemi}.mid.cortex.reg.dfs",
            f"{candidate_prefix}.{hemi}.mid.cortex.svreg.init.dfs",
        ]
    )
    if ref_path is None or cand_path is None:
        return {"reference": ref_path, "candidate": cand_path, "mean_abs": None, "max_abs": None}

    ref = _load_surface(ref_path)
    cand = _load_surface(cand_path)
    if not hasattr(ref, field) or not hasattr(cand, field):
        return {"reference": ref_path, "candidate": cand_path, "mean_abs": None, "max_abs": None}

    ref_data = np.asarray(getattr(ref, field), dtype=np.float64)
    cand_data = np.asarray(getattr(cand, field), dtype=np.float64)
    n = min(ref_data.shape[0], cand_data.shape[0])
    if ref_data.ndim > 1:
        diff = np.linalg.norm(ref_data[:n] - cand_data[:n], axis=1)
    else:
        diff = np.abs(ref_data[:n] - cand_data[:n])
    return {
        "reference": ref_path,
        "candidate": cand_path,
        "mean_abs": float(np.mean(diff)) if diff.size else 0.0,
        "max_abs": float(np.max(diff)) if diff.size else 0.0,
    }


def compare_prefixes(reference_prefix: str, candidate_prefix: str) -> dict[str, object]:
    ref_label_path = _first_existing([f"{reference_prefix}.svreg.label.nii.gz", f"{reference_prefix}.svreg.init.label.nii.gz"])
    cand_label_path = _first_existing([f"{candidate_prefix}.svreg.label.nii.gz", f"{candidate_prefix}.svreg.init.label.nii.gz"])
    ref_map_path = _first_existing([f"{reference_prefix}.svreg.map.nii.gz", f"{reference_prefix}.surfreg.map.nii.gz"])
    cand_map_path = _first_existing([f"{candidate_prefix}.svreg.map.nii.gz", f"{candidate_prefix}.surfreg.map.nii.gz"])
    mask_path = _first_existing([f"{reference_prefix}.mask.nii.gz", f"{candidate_prefix}.mask.nii.gz"])

    ref_label = _load_volume(ref_label_path)
    cand_label = _load_volume(cand_label_path)
    ref_map = _load_volume(ref_map_path)
    cand_map = _load_volume(cand_map_path)
    mask = _load_volume(mask_path)

    result: dict[str, object] = {
        "reference_prefix": reference_prefix,
        "candidate_prefix": candidate_prefix,
        "volume_labels": {
            "reference": ref_label_path,
            "candidate": cand_label_path,
            "diff_voxels": None,
            "mean_abs": None,
        },
        "volume_map": {
            "reference": ref_map_path,
            "candidate": cand_map_path,
            "max_abs": None,
            "mean_abs": None,
            "max_abs_brain": None,
            "mean_abs_brain": None,
        },
        "surface_labels": {
            "left": _surface_metric(reference_prefix, candidate_prefix, "left", "labels"),
            "right": _surface_metric(reference_prefix, candidate_prefix, "right", "labels"),
        },
        "surface_attributes": {
            "left": _surface_metric(reference_prefix, candidate_prefix, "left", "attributes"),
            "right": _surface_metric(reference_prefix, candidate_prefix, "right", "attributes"),
        },
    }

    if ref_label is not None and cand_label is not None:
        shape = tuple(min(a, b) for a, b in zip(ref_label.shape, cand_label.shape))
        ref_crop = ref_label[tuple(slice(0, s) for s in shape)]
        cand_crop = cand_label[tuple(slice(0, s) for s in shape)]
        diff = np.abs(ref_crop.astype(np.float64) - cand_crop.astype(np.float64))
        result["volume_labels"] = {
            "reference": ref_label_path,
            "candidate": cand_label_path,
            "diff_voxels": int(np.count_nonzero(diff)),
            "mean_abs": float(np.mean(diff)),
        }

    if ref_map is not None and cand_map is not None:
        shape = tuple(min(a, b) for a, b in zip(ref_map.shape, cand_map.shape))
        ref_crop = ref_map[tuple(slice(0, s) for s in shape)]
        cand_crop = cand_map[tuple(slice(0, s) for s in shape)]
        diff = np.abs(ref_crop.astype(np.float64) - cand_crop.astype(np.float64))
        result["volume_map"]["max_abs"] = float(np.max(diff))
        result["volume_map"]["mean_abs"] = float(np.mean(diff))
        if mask is not None and mask.ndim >= 3 and diff.ndim == 4 and diff.shape[3] == 3:
            mask_shape = tuple(min(a, b) for a, b in zip(mask.shape[:3], diff.shape[:3]))
            mask_crop = mask[tuple(slice(0, s) for s in mask_shape)] > 0
            norm = np.linalg.norm(diff[: mask_shape[0], : mask_shape[1], : mask_shape[2], :], axis=3)
            masked = norm[mask_crop]
            if masked.size:
                result["volume_map"]["max_abs_brain"] = float(np.max(masked))
                result["volume_map"]["mean_abs_brain"] = float(np.mean(masked))

    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare MATLAB SVReg outputs against Python SVReg outputs.")
    parser.add_argument("reference_prefix", help="Reference output prefix, typically the MATLAB SVReg subject prefix")
    parser.add_argument("candidate_prefix", help="Candidate output prefix, typically the Python SVReg subject prefix")
    args = parser.parse_args(argv)
    print(json.dumps(compare_prefixes(args.reference_prefix, args.candidate_prefix), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
