"""Python implementation of SVReg's subject-to-atlas packaging step."""

from __future__ import annotations

import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.io import loadmat, savemat
from scipy.ndimage import map_coordinates

from . import native
from .roi_refinement import refine_surface
from .runtime import normalize_fileprefix


def _copy(src: Path, dst: Path, *, required: bool = False) -> bool:
    if not src.is_file():
        if required:
            raise FileNotFoundError(src)
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return True


def _make_roi_list(prefix: Path) -> None:
    desc = {}
    xml = prefix.parent / "brainsuite_labeldescription.xml"
    if xml.is_file():
        root = ET.parse(xml).getroot()
        for label in root.iter("label"):
            attrs = label.attrib
            if "id" in attrs:
                desc[int(float(attrs["id"]))] = attrs.get("fullname", attrs.get("name", ""))
    label_path = Path(f"{prefix}.svreg.label.nii.gz")
    if not label_path.is_file():
        label_path = Path(f"{prefix}.label.nii.gz")
    if not label_path.is_file():
        return
    labels = np.unique(np.asanyarray(nib.load(str(label_path)).dataobj)).astype(np.int64)
    out = prefix.parent / "roi_list_svreg.txt"
    with out.open("w", encoding="utf-8") as stream:
        stream.write(f"# Atlas Name: {prefix.parent.name}\n")
        for label in labels:
            if int(label) in desc:
                stream.write(f"{int(label)}\t{desc[int(label)]}\n")
            else:
                stream.write(f"{int(label)}\n")


def _transfer_curvvar(prefix: Path) -> None:
    mat_path = prefix.parent / "curvvar.mat"
    if not mat_path.is_file():
        return
    data = loadmat(str(mat_path), simplify_cells=True)
    updated = {key: value for key, value in data.items() if not key.startswith("__")}
    for hemi in ("left", "right"):
        key = f"{hemi}var"
        if key not in data:
            continue
        atlas_path = prefix.parent / f"atlas.{hemi}.mid.cortex.svreg.dfs"
        subject_path = Path(f"{prefix}.{hemi}.mid.cortex.svreg.dfs")
        if not atlas_path.is_file() or not subject_path.is_file():
            continue
        atlas_surface = native.readdfs(str(atlas_path))
        subject_surface = native.readdfs(str(subject_path))
        values = np.asarray(data[key])
        if values.shape[0] != len(atlas_surface.vertices) and values.shape[-1] == len(atlas_surface.vertices):
            values = np.moveaxis(values, -1, 0)
        mapped = np.column_stack([
            native.interpolate_surface_values_by_uv(
                atlas_surface, values[:, i], subject_surface, method="linear"
            )
            for i in range(1 if values.ndim == 1 else values.shape[1])
        ]) if values.ndim > 1 else native.interpolate_surface_values_by_uv(
            atlas_surface, values, subject_surface, method="linear"
        )
        updated[key] = mapped
    if updated:
        savemat(str(mat_path), updated, do_compression=True)


def _label_surfaces_from_volume(prefix: Path) -> None:
    label_path = Path(f"{prefix}.label.nii.gz")
    bfc_path = Path(f"{prefix}.bfc.nii.gz")
    if not label_path.is_file() or not bfc_path.is_file():
        raise FileNotFoundError("-E requires .label.nii.gz and .bfc.nii.gz")
    image = nib.load(str(label_path))
    labels = np.asanyarray(image.dataobj)
    pixdim = np.asarray(image.header.get_zooms()[:3], dtype=np.float64)
    if np.any(pixdim <= 0):
        raise ValueError("label volume has invalid voxel dimensions")
    for hemi in ("left", "right"):
        surface_path = Path(f"{prefix}.{hemi}.mid.cortex.dfs")
        surf = native.readdfs(str(surface_path))
        vox = np.asarray(surf.vertices, dtype=np.float64) / pixdim[None, :]
        sampled = map_coordinates(
            labels.astype(np.float64, copy=False), vox.T, order=0, mode="nearest", prefilter=False
        )
        sampled = np.rint(sampled).astype(np.int64)
        if hemi == "left":
            selected = (sampled > 0) & ((sampled % 2) == 0)
        else:
            selected = (sampled % 2) == 1
        sampled[selected] += 1
        surf.labels = sampled.astype(np.uint16)
        native.writedfs(str(surface_path), surf)
        refined_path = f"{prefix}.{hemi}.mid.cortex.refined.dfs"
        refined_surface = native.corr_topology_labels(str(surface_path), refined_path)
        refine_surface(
            refined_surface,
            include_zero=True,
            min_label=None,
            max_label=None,
            excluded_labels=(),
            excluded_pairs=set(),
        )
        native.writedfs(refined_path, refined_surface)
        native.corr_topology_labels(refined_path)
        native.recolor_by_label(refined_path, str(prefix))
        shutil.copyfile(
            f"{prefix}.{hemi}.mid.cortex.refined.dfs", surface_path
        )


def svreg_make_atlas(args: list[str]) -> int:
    if len(args) < 2:
        print("usage: svreg_make_atlas SUBJECT_PREFIX ATLAS_PREFIX [flags]", file=sys.stderr)
        return 2
    subject = Path(normalize_fileprefix(args[0]))
    atlas = Path(normalize_fileprefix(args[1]))
    flags = "".join(args[2:])

    # Promote registered surfaces into ordinary BrainSuite surface names when
    # the subject has completed SVReg.
    registered = Path(f"{subject}.left.inner.cortex.svreg.dfs").is_file()
    if registered:
        for hemi in ("left", "right"):
            for layer in ("inner", "mid", "pial"):
                _copy(Path(f"{subject}.{hemi}.{layer}.cortex.svreg.dfs"),
                      Path(f"{subject}.{hemi}.{layer}.cortex.dfs"), required=True)

    for hemi in ("right", "left"):
        mid_path = Path(f"{subject}.{hemi}.mid.cortex.dfs")
        native.smooth_surf_hierarchy(str(mid_path), 10)
        for scale in range(1, 11):
            smooth_path = Path(f"{subject}.{hemi}.mid.cortex_smooth{scale}.dfs")
            if smooth_path.is_file():
                surface = native.readdfs(str(smooth_path))
                surface.faces = np.zeros((0, 3), dtype=np.int32)
                native.writedfs(str(smooth_path), surface)

    if "-E" not in flags:
        for xml in atlas.parent.glob("*.xml"):
            _copy(xml, subject.parent / xml.name)

    if not Path(f"{subject}.left.dfc").is_file():
        _copy(Path(f"{subject}.left.mapped.dfc"), Path(f"{subject}.left.dfc"), required=True)
        _copy(Path(f"{subject}.right.mapped.dfc"), Path(f"{subject}.right.dfc"), required=True)

    registered_label = Path(f"{subject}.svreg.label.nii.gz")
    if registered_label.is_file():
        target_label = Path(f"{subject}.label.nii.gz")
        _copy(registered_label, target_label, required=True)
        image = nib.load(str(target_label))
        data = np.asanyarray(image.dataobj).copy()
        mask = data != 2000
        data[mask] = np.mod(data[mask], 1000)
        nib.save(nib.Nifti1Image(data, image.affine, image.header.copy()), str(target_label))
        for mat in atlas.parent.glob("*.mat"):
            _copy(mat, subject.parent / mat.name)

    _transfer_curvvar(subject)
    hippo_target = Path(f"{subject}.hippo_carved.nii.gz")
    if not hippo_target.is_file():
        temp_hippo = Path(f"{subject}.svreg.tmp/{subject.name}.warped.hippo_carved.nii.gz")
        _copy(temp_hippo if temp_hippo.is_file() else Path(f"{subject}.cortex.dewisp.mask.nii.gz"), hippo_target, required=True)
    sulcal = subject.parent / "sulcal_protocol_HD.xml"
    if not sulcal.is_file():
        _copy(atlas.parent / "sulcal_protocol_HD.xml", sulcal)

    if "-E" in flags:
        _label_surfaces_from_volume(subject)
    _make_roi_list(subject)
    return 0


def main(argv: list[str] | None = None) -> int:
    return svreg_make_atlas(list(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    raise SystemExit(main())

