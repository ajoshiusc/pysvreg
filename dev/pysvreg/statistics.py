"""Python equivalents of SVReg's ROI statistics command line programs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

import numpy as np

from . import native
from .runtime import remove_extn_basename


def _prefix(value: str) -> Path:
    path = Path(remove_extn_basename(value))
    if not path.parent or str(path.parent) == ".":
        path = Path.cwd() / path
    return path


def _nii_path(prefix: Path, suffix: str) -> Path:
    path = Path(str(prefix) + suffix)
    if not path.is_file() and str(path).endswith(".nii"):
        gz = Path(str(path) + ".gz")
        if gz.is_file():
            return gz
    return path


def _read_roi_list(path: Path) -> list[int]:
    result = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        token = line.split()[0].split(",")[0]
        try:
            result.append(int(float(token)))
        except ValueError:
            continue
    return result


def _labels(prefix: Path, flags: str, *, temporary: bool = False):
    candidates = [
        (".svreg.corr.label.nii", "corr"),
        (".svreg.label.nii", "label"),
        (".svreg.corr.manual.label.nii", "manual"),
    ]
    selected = None
    for suffix, kind in candidates:
        if _nii_path(prefix, suffix).is_file():
            selected = (suffix, kind)
    # MATLAB uses the corrected labels only when -r is present; a manual label
    # file, if present, takes precedence over either regular label file.
    manual = _nii_path(prefix, candidates[2][0])
    if manual.is_file():
        selected = candidates[2]
    elif "r" in flags:
        corrected = _nii_path(prefix, candidates[0][0])
        selected = candidates[0] if corrected.is_file() else None
    else:
        regular = _nii_path(prefix, candidates[1][0])
        selected = candidates[1] if regular.is_file() else None
    if selected is None:
        return None, None, None
    suffix, kind = selected
    volume = native.load_nii_z(str(_nii_path(prefix, suffix)))
    if kind == "label":
        volume.img = np.mod(volume.img, 1000)
    return volume, kind, _nii_path(prefix, suffix)


def _surface(prefix: Path, hemi: str, layer: str, manual: bool):
    suffix = f".{hemi}.{layer}.cortex" + (".manual" if manual else "") + ".svreg.dfs"
    return native.readdfs(str(prefix) + suffix)


def _manual_surfaces(prefix: Path) -> bool:
    return all(
        Path(str(prefix) + f".right.{layer}.cortex.manual.svreg.dfs").is_file()
        for layer in ("pial", "mid", "inner")
    )


def _volume_fractions(prefix: Path):
    vf = native.load_nii_z(str(_nii_path(prefix, ".pvc.frac.nii")))
    f = np.asarray(vf.img, dtype=np.float64)
    gm = (f > 1) * (1 - np.abs(f - 2))
    csf = (f < 2) * (1 - np.abs(f - 1))
    wm = (f > 2) * (1 - np.abs(f - 3))
    return gm, csf, wm


def _check_same_shape(a, b, what: str) -> None:
    if np.shape(a) != np.shape(b):
        raise ValueError(f"{what} dimensions differ: {np.shape(a)} vs {np.shape(b)}")


def _write_log(prefix: Path, name: str, args) -> None:
    with open(str(prefix) + ".svreg.log", "a", encoding="utf-8") as stream:
        stamp = datetime.now().strftime("%d-%b-%Y %H:%M:%S")
        stream.write(f"{stamp}:SVReg Python: {name} {prefix} {' '.join(args)}\n")


def generate_stats_xls(args: list[str]) -> int:
    if not args:
        print("usage: generate_stats_xls SUBJECT_PREFIX [flags]", file=sys.stderr)
        return 2
    prefix = _prefix(args[0])
    flags = "".join(args[1:])
    _write_log(prefix, "generate_stats_xls", args[1:])

    labels_vol, label_kind, _ = _labels(prefix, flags)
    target = _nii_path(prefix, ".target.label.nii")
    if (prefix.parent / "roi_list_svreg.txt").is_file():
        labs = _read_roi_list(prefix.parent / "roi_list_svreg.txt")
        labs = [lab for lab in labs if lab != 2000]
    elif target.is_file():
        labs = np.unique(native.load_nii_z(str(target)).img).astype(np.int64).tolist()
    elif labels_vol is not None:
        labs = np.unique(labels_vol.img).astype(np.int64).tolist()
    else:
        labs = []
    labs = sorted({1, 2, 3, *(int(x) for x in labs if x != 0)})

    manual = _manual_surfaces(prefix)
    layers = ("mid", "inner", "pial")
    surfaces = {
        (hemi, layer): _surface(prefix, hemi, layer, manual)
        for hemi in ("left", "right") for layer in layers
    }
    if len(labs) < 5:
        for hemi in ("left", "right"):
            surf = surfaces[(hemi, "mid")]
            labs = sorted(set(labs) | set(np.asarray(surf.labels).astype(int).tolist()))
        labs = sorted({1, 2, 3, *(x for x in labs if x != 0)})

    thickness_path = Path(str(prefix) + ".pvc-thickness_0-6mm.right.mid.cortex.dfs")
    if thickness_path.is_file():
        for hemi in ("left", "right"):
            thick = native.readdfs(str(prefix) + f".pvc-thickness_0-6mm.{hemi}.mid.cortex.dfs")
            surfaces[(hemi, "mid")].attributes = thick.attributes
    else:
        print("PVC Thickness is not available. Making roiwise thickness values NAN")
        for hemi in ("left", "right"):
            surfaces[(hemi, "mid")].attributes = np.full(
                len(surfaces[(hemi, "mid")].vertices), np.nan, dtype=np.float32
            )

    n = len(labs)
    thickness = np.zeros(n, dtype=float)
    areas = np.zeros((n, 3), dtype=float)
    for i, lab in enumerate(labs):
        left, right = surfaces[("left", "mid")], surfaces[("right", "mid")]
        vals = np.concatenate([
            np.asarray(left.attributes)[np.asarray(left.labels) == lab],
            np.asarray(right.attributes)[np.asarray(right.labels) == lab],
        ])
        thickness[i] = np.mean(vals) if vals.size else np.nan
        if lab > 600 or lab == 3:
            thickness[i] = 0
        if 100 <= lab <= 600:
            for side in ("left", "right"):
                mid = surfaces[(side, "mid")]
                faces = np.asarray(mid.faces)
                selected = (np.asarray(mid.labels)[faces] == lab).sum(axis=1) >= 2
                for j, layer in enumerate(layers):
                    surf = surfaces[(side, layer)]
                    areas[i, j] += native.tri_area(surf.faces, surf.vertices)[selected].sum()

    gmvol = csfvol = wmvol = total = np.zeros(n, dtype=float)
    volume_stats = labels_vol is not None and label_kind != "manual"
    labs_sub: list[int] = []
    labs_vol: list[float] = []
    if labels_vol is not None:
        gm, csf, wm = _volume_fractions(prefix)
        _check_same_shape(gm, labels_vol.img, "tissue fractions and labels")
        voxel_volume = float(np.prod(labels_vol.zooms[:3]))
        gmvol = np.asarray([gm[labels_vol.img == lab].sum() * voxel_volume for lab in labs])
        csfvol = np.asarray([csf[labels_vol.img == lab].sum() * voxel_volume for lab in labs])
        wmvol = np.asarray([wm[labels_vol.img == lab].sum() * voxel_volume for lab in labs])
        total = gmvol + wmvol
        for lab in (1, 2, 3):
            idx = labs.index(lab)
            gmvol[idx] = csfvol[idx] = wmvol[idx] = total[idx] = 0
        if 1 in labs:
            i = labs.index(1); csfvol[i] = csf.sum() * voxel_volume; total[i] = csfvol[i]
        if 2 in labs:
            i = labs.index(2); gmvol[i] = gm.sum() * voxel_volume; total[i] = gmvol[i]
        if 3 in labs:
            i = labs.index(3); wmvol[i] = wm.sum() * voxel_volume; total[i] = wmvol[i]
        # The MATLAB implementation currently constructs labs_sub by indexing
        # with an empty index array, so it never emits the optional extra rows.

    out = Path(str(prefix) + (".roiwise.manual.stats.txt" if manual else ".roiwise.stats.txt"))
    with out.open("w", encoding="utf-8") as fp:
        if volume_stats:
            fp.write("ROI_ID\tMean_Thickness(mm)\tGM_Volume(mm^3)\tCSF_Volume(mm^3)\tWM_Volume(mm^3)\tTotal_Volume(GM+WM)(mm^3)\tCortical_Area_mid(mm^2)\tCortical_Area_inner(mm^2)\tCortical_Area_pial(mm^2)\n")
            for i, lab in enumerate(labs):
                fp.write(f"{lab:d}\t{thickness[i]:.6f}\t{gmvol[i]:.6f}\t{csfvol[i]:.6f}\t{wmvol[i]:.6f}\t{total[i]:.6f}\t{areas[i,0]:.6f}\t{areas[i,1]:.6f}\t{areas[i,2]:.6f}\n")
        else:
            fp.write("ROI_ID\tMean_Thickness(mm)\tCortical_Area(mm^2)\n")
            for i, lab in enumerate(labs):
                fp.write(f"{lab:d}\t{thickness[i]:.6f}\t{areas[i,0]:.6f}\n")
        if volume_stats:
            for lab, vol in zip(labs_sub, labs_vol):
                fp.write(f"{lab:d}\t0.000000\t0.000000\t0.000000\t0.000000\t{vol:.6f}\n")
    return 0


def _weighted_mean_std(weights: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    weight_sum = float(weights.sum())
    if weight_sum == 0:
        return 0.0, 0.0
    mean = float(np.sum(weights * values) / weight_sum)
    count = int(np.count_nonzero(weights))
    if count < 2:
        return mean, 0.0
    variance = np.sum(weights * np.square(values - mean)) / (((count - 1) / count) * weight_sum)
    return mean, float(np.sqrt(max(variance, 0.0)))


def generate_vol_param_stats_xls(args: list[str]) -> int:
    if len(args) < 2:
        print("usage: generate_vol_param_stats_xls SUBJECT_PREFIX PARAM_NII [flags]", file=sys.stderr)
        return 2
    prefix = _prefix(args[0])
    param_path = Path(args[1])
    flags = "".join(args[2:])
    tmpdir = Path(str(prefix) + ".svreg.tmp")
    tmp_prefix = tmpdir / prefix.name if tmpdir.is_dir() else prefix
    _write_log(tmp_prefix, "generate_vol_param_stats_xls", args[1:])
    labels_vol, label_kind, _ = _labels(tmp_prefix, flags)
    if labels_vol is None:
        raise FileNotFoundError(f"SVREG label file missing for {tmp_prefix}")
    param = native.load_nii_z(str(param_path))
    _check_same_shape(param.img, labels_vol.img, "parameter map and labels")
    gm, csf, wm = _volume_fractions(prefix)
    _check_same_shape(gm, param.img, "parameter map and tissue fractions")
    labs = sorted({1, 2, 3, *(int(x) for x in np.unique(labels_vol.img) if x != 0)})
    values = np.asarray(param.img, dtype=np.float64)
    # The original MATLAB source references vl before loading it. Here labels
    # are taken from the selected label volume, matching the apparent intent.
    stat = np.zeros((len(labs), 8), dtype=float)
    for i, lab in enumerate(labs):
        roi = labels_vol.img == lab
        if np.any(roi):
            stat[i, 0:2] = _weighted_mean_std(gm[roi], values[roi])
            stat[i, 2:4] = _weighted_mean_std(wm[roi], values[roi])
            stat[i, 4:6] = _weighted_mean_std(csf[roi], values[roi])
            stat[i, 6] = float(np.mean(values[roi]))
            stat[i, 7] = float(np.std(values[roi], ddof=1)) if np.count_nonzero(roi) > 1 else 0.0
    # Labels 1, 2, and 3 denote pure CSF, GM, and WM summaries respectively.
    # Output columns are ordered GM, WM, CSF, so their tissue-specific weighted
    # statistics belong in columns 4:6, 0:2, and 2:4.
    for row_index, column_start, fraction in (
        (0, 4, csf),
        (1, 0, gm),
        (2, 2, wm),
    ):
        stat[row_index, :6] = 0
        stat[row_index, column_start : column_start + 2] = _weighted_mean_std(
            fraction.ravel(), values.ravel()
        )
        stat[row_index, 6] = float(np.mean(values))
        stat[row_index, 7] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0

    manual = _manual_surfaces(tmp_prefix)
    out = Path(str(tmp_prefix) + (".param.roiwise.manual.stats.txt" if manual else ".param.roiwise.stats.txt"))
    with out.open("w", encoding="utf-8") as fp:
        fp.write("ROI_ID\tGM voxels (mean)\tGM voxels (std dev)\tWM voxels (mean)\tWM voxels (std dev)\tCSF voxels (mean)\tCSF voxels (std dev)\tAll voxels (mean)\tAll voxels (std dev)\t\n")
        for lab, row in zip(labs, stat):
            fp.write(f"{lab:d}\t\t\t{row[0]:.6f}\t\t\t{row[1]:.6f}\t\t\t{row[2]:.6f}\t\t\t{row[3]:.6f}\t\t\t{row[4]:.6f}\t\t\t{row[5]:.6f}\t\t\t{row[6]:.6f}\t\t\t{row[7]:.6f}\n")
    if tmp_prefix != prefix:
        import shutil
        shutil.copyfile(out, Path(str(prefix) + ".param.roiwise" + (".manual" if manual else "") + ".stats.txt"))
        shutil.copyfile(str(tmp_prefix) + ".svreg.log", str(prefix) + ".svreg.log")
    return 0

