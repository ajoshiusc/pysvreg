from __future__ import annotations

import sys
from pathlib import Path
import shutil

try:
    from . import _deps  # noqa: F401
except ImportError:
    import _deps  # noqa: F401

import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import median_filter

try:
    from .native import (
        NiftiVolume,
        get_hippo_label,
        load_nii_z,
        nearest_fill,
        readdfs,
        save_untouch_nii_gz,
    )
    from .runtime import normalize_fileprefix
except ImportError:
    from native import NiftiVolume, get_hippo_label, load_nii_z, nearest_fill, readdfs, save_untouch_nii_gz
    from runtime import normalize_fileprefix


def _matlab_label_component_cleanup(image: np.ndarray) -> np.ndarray:
    """Port correct_vol_labels' imfill(...,6) and bwareaopen(...,26) loop."""
    output = np.asarray(image).copy()
    labels = np.unique(output)
    background_connectivity = ndi.generate_binary_structure(3, 1)
    component_connectivity = ndi.generate_binary_structure(3, 3)
    boxes = ndi.find_objects(output.astype(np.int32, copy=False))
    shape = np.asarray(output.shape)

    for value in labels:
        if value <= 0 or not np.isfinite(value) or float(value).is_integer() is False:
            continue
        label_value = int(value)
        if label_value - 1 >= len(boxes):
            continue
        bounds = boxes[label_value - 1]
        if bounds is None:
            continue
        padded = tuple(
            slice(max(0, axis_slice.start - 1), min(shape[axis], axis_slice.stop + 1))
            for axis, axis_slice in enumerate(bounds)
        )
        region = output[padded]
        filled = ndi.binary_fill_holes(region == value, structure=background_connectivity)
        filled_count = int(np.count_nonzero(filled))
        if filled_count == 0:
            continue
        components, _ = ndi.label(filled, structure=component_connectivity)
        counts = np.bincount(components.ravel())
        min_size = int(np.floor(filled_count / 2.0 + 0.5))
        keep = counts >= min_size
        if keep.size:
            keep[0] = False
        retained = keep[components]
        # MATLAB assigns the bwareaopen result only at filled-image voxels;
        # this also clears components removed by the half-area threshold.
        region[filled] = np.where(retained[filled], value, 0)
    return output


def _matlab_pial_surface_mask(surface, reference, *, mask_width: int = 1, face_chunk: int = 20000) -> np.ndarray:
    """Port surf2mask's four centroid subdivisions and morphology."""
    vertices = np.asarray(surface.vertices, dtype=np.float64)
    faces = np.asarray(surface.faces, dtype=np.int64)
    resolution = np.asarray(reference.zooms[:3], dtype=np.float64)
    shape = tuple(int(v) for v in reference.shape[:3])
    mask = np.zeros(shape, dtype=bool)

    def mark(points: np.ndarray) -> None:
        voxels = np.floor(points / resolution[None, :] + 0.5).astype(np.int64)
        valid = np.all((voxels >= 0) & (voxels < np.asarray(shape)[None, :]), axis=1)
        voxels = voxels[valid]
        if voxels.size:
            mask[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = True

    mark(vertices)
    for first in range(0, faces.shape[0], face_chunk):
        triangles = vertices[faces[first : first + face_chunk]]
        for _ in range(4):
            centroid = np.mean(triangles, axis=1)
            mark(centroid)
            child1 = np.stack((triangles[:, 1], centroid, triangles[:, 0]), axis=1)
            child2 = np.stack((triangles[:, 2], centroid, triangles[:, 1]), axis=1)
            child3 = np.stack((triangles[:, 0], centroid, triangles[:, 2]), axis=1)
            triangles = np.concatenate((child1, child2, child3), axis=0)

    dilation = np.ones((3 + int(mask_width),) * 3, dtype=bool)
    mask = ndi.binary_dilation(mask, structure=dilation)
    mask = ndi.binary_erosion(mask, structure=np.ones((3, 3, 3), dtype=bool), border_value=0)
    return ndi.binary_fill_holes(mask, structure=ndi.generate_binary_structure(3, 1))


def svreg_refinements(subbasename: str, atlas_name: str, postfix: str = "", *varargin: str) -> None:
    subbasename = normalize_fileprefix(subbasename)
    tmp_base = f"{subbasename}.svreg.tmp/{Path(subbasename).name}"
    _refine_vol_labels(tmp_base, postfix)
    if (not postfix) or ("USCBrain" in postfix) or ("BCI-DNI" in postfix) or ("USCLobes" in postfix):
        _correct_vol_labels(subbasename, postfix, "".join(varargin))
    else:
        shutil.copyfile(f"{tmp_base}.svreg.{postfix}ref.label.nii.gz", f"{subbasename}.svreg.{postfix}label.nii.gz")


def _refine_vol_labels(subbasename: str, postfix: str) -> None:
    vl = load_nii_z(f"{subbasename}.svreg.{postfix}label.nii.gz")
    sin = readdfs(f"{subbasename}.left.inner.cortex.svreg.{postfix}dfs")
    smid = readdfs(f"{subbasename}.left.mid.cortex.svreg.{postfix}dfs")
    spial = readdfs(f"{subbasename}.left.pial.cortex.svreg.{postfix}dfs")
    sinr = readdfs(f"{subbasename}.right.inner.cortex.svreg.{postfix}dfs")
    smidr = readdfs(f"{subbasename}.right.mid.cortex.svreg.{postfix}dfs")
    spialr = readdfs(f"{subbasename}.right.pial.cortex.svreg.{postfix}dfs")

    pth = Path(subbasename).parent
    xmlf = pth / "brainsuite_labeldescription.xml"
    ids = set(get_hippo_label(xmlf)) if xmlf.is_file() else set()

    if (not postfix) or ("USCBrain" in postfix) or ("BCI-DNI" in postfix) or ("USCLobes" in postfix):
        mask = (vl.img >= 100) & (vl.img < 600)
        for lab in ids:
            mask &= vl.img != lab
    else:
        cort_labs = set(np.unique(smid.labels)) | set(np.unique(smidr.labels))
        cort_labs.discard(0)
        mask = np.isin(vl.img, list(cort_labs))
        for lab in ids:
            mask &= vl.img != lab

    ind = np.argwhere(mask)
    if ind.size == 0:
        save_untouch_nii_gz(vl, f"{subbasename}.svreg.{postfix}ref.label.nii.gz")
        return

    dim = np.asarray(vl.zooms[:3], dtype=np.float32)
    query = ind * dim[None, :]

    def _stack(surface_inner, surface_mid, surface_pial):
        indkn = surface_mid.labels != 0
        spial2 = 0.5 * (surface_pial.vertices + surface_mid.vertices)
        points = np.vstack(
            [
                surface_inner.vertices[indkn],
                surface_mid.vertices[indkn],
                surface_pial.vertices[indkn],
                spial2[indkn],
            ]
        )
        values = np.concatenate([surface_mid.labels[indkn]] * 4)
        return points, values

    pts_l, vals_l = _stack(sin, smid, spial)
    pts_r, vals_r = _stack(sinr, smidr, spialr)
    points = np.vstack([pts_l, pts_r])
    values = np.concatenate([vals_l, vals_r])
    points, first = np.unique(points, axis=0, return_index=True)
    values = values[first]
    refined = nearest_fill(points, values, query).astype(np.int16)
    new_img = np.asarray(vl.img).copy()
    new_img[tuple(ind.T)] = refined
    save_untouch_nii_gz(
        NiftiVolume(new_img, vl.affine, vl.header.copy(), vl.path),
        f"{subbasename}.svreg.{postfix}ref.label.nii.gz",
    )


def _correct_vol_labels(subbasename: str, postfix: str, flags: str) -> None:
    tmp_base = f"{subbasename}.svreg.tmp/{Path(subbasename).name}"
    vl = load_nii_z(f"{tmp_base}.svreg.{postfix}ref.label.nii.gz")
    vlo = np.asarray(vl.img).copy()
    vmsk = load_nii_z(f"{subbasename}.cortex.dewisp.mask.nii.gz")
    vpvc = load_nii_z(f"{subbasename}.pvc.frac.nii.gz")

    wm = np.asarray(vl.img).copy()
    ind = (wm >= 100) & (wm < 600)
    wm = wm * (vpvc.img > 0)
    wm[ind & (vmsk.img > 0)] = 2000
    save_untouch_nii_gz(NiftiVolume(wm, vl.affine, vl.header.copy(), vl.path), f"{tmp_base}.svreg.{postfix}wm.label.nii.gz")

    v_img = median_filter(vlo, size=3)
    v_img[(vmsk.img == 0)] = vlo[(vmsk.img == 0)]
    v_img[(vlo < 100) | (vlo > 600)] = vlo[(vlo < 100) | (vlo > 600)]
    v_img = _matlab_label_component_cleanup(v_img)

    cortical_rois = set(np.unique(v_img[v_img < 670]).tolist()) | {850, 701, 720, 721, 614, 615, 616, 617, 612, 613, 630}
    non_cortical_rois = {900, 760, 780}
    cerebrum_mask = load_nii_z(f"{subbasename}.cerebrum.mask.nii.gz")
    cerebrum_region = np.asarray(cerebrum_mask.img) > 0
    pial_surface = readdfs(f"{subbasename}.pial.cortex.dfs")
    surface_mask = _matlab_pial_surface_mask(pial_surface, cerebrum_mask)
    if np.count_nonzero(surface_mask) < 0.5 * np.count_nonzero(cerebrum_region):
        pial_mask = cerebrum_region
    else:
        pial_mask = cerebrum_region & surface_mask
        pial_volume = NiftiVolume(
            (pial_mask.astype(np.uint8) * 255),
            cerebrum_mask.affine,
            cerebrum_mask.header.copy(),
            cerebrum_mask.path,
        )
        pial_volume.header.set_data_dtype(np.uint8)
        save_untouch_nii_gz(pial_volume, f"{tmp_base}.pial.mask.nii.gz")

    v_img[np.isin(v_img, list(non_cortical_rois)) & pial_mask & (v_img != 2000)] = 0
    v_img[np.isin(v_img, list(cortical_rois)) & ~pial_mask] = 0

    keep_mask = (v_img != 0) & (~np.isin(v_img, list(non_cortical_rois))) & pial_mask & (v_img != 2000)
    sample_pts = np.argwhere(keep_mask)
    matlab_nearest_jitter = np.asarray((0.753, 0.352, 0.551), dtype=np.float64)
    if sample_pts.size:
        fill_pts = np.argwhere(pial_mask & (v_img == 0))
        if fill_pts.size:
            v_img[tuple(fill_pts.T)] = nearest_fill(
                sample_pts,
                v_img[tuple(sample_pts.T)],
                fill_pts + matlab_nearest_jitter,
            )

    keep_mask = (v_img != 0) & (~np.isin(v_img, list(cortical_rois))) & ~pial_mask
    sample_pts = np.argwhere(keep_mask)
    fill_pts = np.argwhere((~pial_mask) & (vpvc.img > 0) & (v_img == 0))
    if sample_pts.size and fill_pts.size:
        v_img[tuple(fill_pts.T)] = nearest_fill(
            sample_pts,
            v_img[tuple(sample_pts.T)],
            fill_pts + matlab_nearest_jitter,
        )

    v_img = v_img * (vpvc.img > 0)
    save_untouch_nii_gz(NiftiVolume(v_img, vl.affine, vl.header.copy(), vl.path), f"{tmp_base}.svreg.{postfix}corr.label.nii.gz")

    dws_img = np.asarray(v_img).copy()
    ind = (dws_img >= 100) & (dws_img < 600)
    dws_img[dws_img == 3] = 2000
    dws_img[ind] = dws_img[ind] + 1000
    dws_img[ind & (vmsk.img > 0)] = dws_img[ind & (vmsk.img > 0)] + 1000
    dws_img[np.isin(dws_img, list(cortical_rois)) | (dws_img > 1000)] *= pial_mask[np.isin(dws_img, list(cortical_rois)) | (dws_img > 1000)]

    label_surfreg = load_nii_z(f"{tmp_base}.label.surfreg.nii.gz")
    msk_full = ndi.maximum_filter(
        np.asarray(label_surfreg.img), size=(3, 3, 3), mode="constant", cval=0
    )
    mask_cleanup = (dws_img != 740) & (
        np.isin(msk_full, list(cortical_rois)) | pial_mask
    )
    msk_full[mask_cleanup] = 0
    dws_img[
        (~np.isin(dws_img, list(cortical_rois)))
        & (dws_img <= 1000)
        & (msk_full == 0)
    ] = 0
    save_untouch_nii_gz(NiftiVolume(dws_img, vl.affine, vl.header.copy(), vl.path), f"{tmp_base}.svreg.{postfix}dws.label.nii.gz")
    shutil.copyfile(f"{tmp_base}.svreg.{postfix}dws.label.nii.gz", f"{subbasename}.svreg.{postfix}label.nii.gz")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2:
        print("usage: svreg_refinements.py SUBJECT_PREFIX ATLAS_PREFIX [postfix] [flags ...]")
        return 1
    postfix = argv[2] if len(argv) > 2 and not argv[2].startswith("-") else ""
    rest = argv[3:] if postfix else argv[2:]
    svreg_refinements(argv[0], argv[1], postfix, *rest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
