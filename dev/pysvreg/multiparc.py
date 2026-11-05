"""Transfer one or more atlas parcellations onto an SVReg subject."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates

from . import native
from .refine_ROIs2 import refine_ROIs2
from .runtime import normalize_fileprefix
from .svreg_refinements import svreg_refinements


def _atlas_jobs(multi_dir: Path, selector: str):
    if selector == "all":
        label_files = sorted(multi_dir.glob("*/*.label.nii.gz"))
    else:
        label_files = sorted((multi_dir / selector).glob("*.label.nii.gz"))
    for label_file in label_files:
        name = label_file.name[:-len(".label.nii.gz")]
        atlas_name = label_file.parent.name
        if atlas_name == "Schaefer2018":
            atlas_name = name[9:]
        yield label_file.with_name(name), atlas_name


def _surface_transfer(subject: Path, atlas_prefix: Path, atlas_name: str, output_dir: Path) -> None:
    tmp_dir = subject.parent / f"{subject.name}.svreg.tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_base = tmp_dir / subject.name
    xml_src = atlas_prefix.parent / "brainsuite_labeldescription.xml"
    xml_dst = subject.parent / f"brainsuite_labeldescription.{atlas_name}.xml"
    if xml_src.is_file():
        shutil.copyfile(xml_src, xml_dst)
        shutil.copyfile(xml_src, tmp_dir / "brainsuite_labeldescription.xml")

    special_atlas = any(token in atlas_name for token in ("USCBrain", "USCLobes", "BCI-DNI"))

    for hemi in ("left", "right"):
        subject_mid_path = Path(f"{subject}.{hemi}.mid.cortex.svreg.dfs")
        atlas_guidance_path = subject.parent / f"atlas.{hemi}.mid.cortex.svreg.dfs"
        atlas_parcel_path = Path(f"{atlas_prefix}.{hemi}.mid.cortex.dfs")
        for path in (subject_mid_path, atlas_guidance_path, atlas_parcel_path):
            if not path.is_file():
                raise FileNotFoundError(path)

        sub_mid = native.readdfs(str(subject_mid_path))
        atlas_guidance = native.readdfs(str(atlas_guidance_path))
        atlas_parcel = native.readdfs(str(atlas_parcel_path))
        if not hasattr(atlas_parcel, "labels"):
            raise ValueError(f"atlas parcel surface has no vertex labels: {atlas_parcel_path}")
        mapped_labels = native.interpolate_surface_values_by_uv(
            atlas_guidance,
            np.asarray(atlas_parcel.labels, dtype=np.float64),
            sub_mid,
            method="nearest",
        )
        sub_mid.labels = np.clip(np.rint(mapped_labels), 0, 65535).astype(np.uint16)

        for layer in ("mid", "inner", "pial"):
            src = subject_mid_path if layer == "mid" else Path(f"{subject}.{hemi}.{layer}.cortex.svreg.dfs")
            surf = native.readdfs(str(src))
            surf.labels = sub_mid.labels.copy()
            target = (
                Path(f"{tmp_base}.{hemi}.mid.cortex.reg.{atlas_name}dfs")
                if layer == "mid"
                else Path(f"{tmp_base}.{hemi}.{layer}.cortex.svreg.{atlas_name}dfs")
            )
            native.writedfs(str(target), surf)

        postfix = f"{atlas_name}"
        staged_mid = Path(f"{tmp_base}.{hemi}.mid.cortex.svreg.{postfix}dfs")
        if special_atlas:
            refine_ROIs2(str(subject), hemi, postfix)
            # The MATLAB multiparc command removes refined inner and pial
            # surfaces for these atlases; only the mid surface is published.
            for layer in ("inner", "pial"):
                Path(f"{subject}.{hemi}.{layer}.cortex.svreg.{postfix}dfs").unlink(missing_ok=True)
        else:
            shutil.copyfile(
                f"{tmp_base}.{hemi}.mid.cortex.reg.{postfix}dfs", staged_mid
            )
            mid = native.readdfs(str(staged_mid))
            for layer in ("inner", "pial"):
                side_path = Path(f"{tmp_base}.{hemi}.{layer}.cortex.svreg.{postfix}dfs")
                side = native.readdfs(str(side_path))
                for field in ("vColor", "attributes"):
                    if hasattr(mid, field):
                        setattr(side, field, np.asarray(getattr(mid, field)).copy())
                native.writedfs(str(side_path), side)
            shutil.copyfile(staged_mid, f"{subject}.{hemi}.mid.cortex.svreg.{postfix}dfs")

        subject_mid_out = Path(f"{subject}.{hemi}.mid.cortex.svreg.{postfix}dfs")
        native.recolor_by_label(
            str(subject_mid_out),
            str(atlas_prefix),
            str(xml_dst) if xml_dst.is_file() else None,
        )
        shutil.copyfile(
            subject_mid_out,
            output_dir / f"{subject.name}.{hemi}.mid.cortex.svreg.{postfix}.dfs",
        )


def _volume_transfer(subject: Path, atlas_prefix: Path, atlas_name: str) -> bool:
    map_path = Path(f"{subject}.svreg.map.nii.gz")
    label_path = Path(f"{atlas_prefix}.label.nii.gz")
    subject_label_path = Path(f"{subject}.svreg.label.nii.gz")
    if not map_path.is_file() or not label_path.is_file() or not subject_label_path.is_file():
        return False
    warp = nib.load(str(map_path))
    map_img = np.asanyarray(warp.dataobj)
    atlas = np.asanyarray(nib.load(str(label_path)).dataobj)
    if map_img.ndim != 4 or map_img.shape[3] < 3:
        raise ValueError(f"SVReg coordinate map must be 4D with at least 3 components: {map_path}")
    # MATLAB's interp3 uses column, row, slice coordinates. The stored SVReg
    # vector field is x,y,z in volume-axis order, hence the first two axes swap.
    coords = np.stack((map_img[..., 1] - 1.0, map_img[..., 0] - 1.0, map_img[..., 2] - 1.0), axis=0)
    values = map_coordinates(
        atlas.astype(np.float64, copy=False), coords, order=0, mode="constant", cval=0.0, prefilter=False
    )
    subject_image = nib.load(str(subject_label_path))
    out_image = nib.Nifti1Image(values.astype(np.asanyarray(subject_image.dataobj).dtype), subject_image.affine, subject_image.header.copy())
    tmp_base = subject.parent / f"{subject.name}.svreg.tmp" / subject.name
    nib.save(out_image, f"{tmp_base}.svreg.{atlas_name}label.nii.gz")
    return True


def svreg_multiparc(args: list[str]) -> int:
    if len(args) < 3:
        print("usage: svreg_multiparc SUBJECT_PREFIX MULTI_ATLAS_DIR BRAINSUITE_DIR [ATLAS_NAME|all]", file=sys.stderr)
        return 2
    subject = Path(normalize_fileprefix(args[0]))
    multi_dir = Path(args[1]).expanduser().resolve()
    # Retained for command-line compatibility; the current implementation uses
    # atlas files directly and does not invoke a BrainSuite executable.
    _brainsuite_path = Path(args[2]).expanduser()
    selector = args[3] if len(args) > 3 else "all"
    jobs = list(_atlas_jobs(multi_dir, selector))
    if not jobs:
        print(f"No labeled atlases found in {multi_dir} for {selector!r}", file=sys.stderr)
        return 1

    out_dir = subject.parent / "multiparc"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = subject.parent / f"{subject.name}.svreg.tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_base = tmp_dir / subject.name
    init_labels = Path(f"{subject}.svreg.init.label.nii.gz")
    if Path(f"{subject}.svreg.map.nii.gz").is_file() and init_labels.is_file():
        shutil.copyfile(init_labels, f"{tmp_base}.label.surfreg.nii.gz")

    for atlas_prefix, atlas_name in jobs:
        print(f"Processing Atlas:{atlas_name}\natlasbasename:{atlas_prefix}\n------")
        _surface_transfer(subject, atlas_prefix, atlas_name, out_dir)
        if _volume_transfer(subject, atlas_prefix, atlas_name):
            svreg_refinements(str(subject), str(atlas_prefix), atlas_name)
            refined_volume = Path(f"{subject}.svreg.{atlas_name}label.nii.gz")
            if refined_volume.is_file():
                shutil.move(
                    str(refined_volume),
                    out_dir / f"{subject.name}.svreg.{atlas_name}.label.nii.gz",
                )
        xml_src = subject.parent / f"brainsuite_labeldescription.{atlas_name}.xml"
        if xml_src.is_file():
            shutil.move(str(xml_src), out_dir / xml_src.name)
    return 0


def main(argv: list[str] | None = None) -> int:
    return svreg_multiparc(list(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    raise SystemExit(main())

