"""Prepare SVReg's per-subject staging directory from BrainSuite outputs."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from .native import calc_mid_surf
from .runtime import normalize_fileprefix


def prepare_files(args: list[str]) -> int:
    if len(args) < 2:
        print("usage: svreg_prepare_files SUBJECT_PREFIX ATLAS_PREFIX", file=sys.stderr)
        return 2

    subject = Path(normalize_fileprefix(args[0]))
    atlas = Path(normalize_fileprefix(args[1]))
    temp_dir = subject.parent / f"{subject.name}.svreg.tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    tmp = temp_dir / subject.name
    atlas_dir = atlas.parent
    missing: list[Path] = []

    def copy_required(src: Path, dst: Path) -> None:
        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
        else:
            missing.append(src)

    # The subject mid surface is a deterministic average of the inner and
    # pial meshes. BrainSuite commonly creates it earlier, so preserve it when
    # present and synthesize it only when missing.
    for hemi in ("left", "right"):
        for layer in ("inner", "pial"):
            copy_required(
                Path(f"{subject}.{hemi}.{layer}.cortex.dfs"),
                Path(f"{tmp}.{hemi}.{layer}.cortex.dfs"),
            )
        subject_mid = Path(f"{subject}.{hemi}.mid.cortex.dfs")
        if not subject_mid.is_file():
            inner = Path(f"{subject}.{hemi}.inner.cortex.dfs")
            pial = Path(f"{subject}.{hemi}.pial.cortex.dfs")
            if inner.is_file() and pial.is_file():
                calc_mid_surf(str(inner), str(pial), str(subject_mid))
        copy_required(subject_mid, Path(f"{tmp}.{hemi}.mid.cortex.dfs"))

        for layer in ("mid", "inner", "pial"):
            copy_required(
                Path(f"{atlas}.{hemi}.{layer}.cortex.dfs"),
                Path(f"{tmp}.target.{hemi}.{layer}.cortex.dfs"),
            )
        atlas_curve = Path(f"{atlas}.{hemi}.dfc")
        if atlas_curve.is_file():
            copy_required(atlas_curve, Path(f"{tmp}.target.{hemi}.dfc"))

    # Volumes and protocol files used by the hemisphere labeling and volume
    # registration stages are staged under the target basename.
    for suffix in (".cortex.dewisp.mask.nii.gz", ".bfc.nii.gz", ".pvc.frac.nii.gz", ".label.nii.gz"):
        copy_required(Path(f"{atlas}{suffix}"), Path(f"{tmp}.target{suffix}"))
    hippo = Path(f"{atlas}.hippo_carved.nii.gz")
    if not hippo.is_file():
        hippo = Path(f"{atlas}.cerebrum.mask.nii.gz")
    copy_required(hippo, Path(f"{tmp}.target.hippo_carved.nii.gz"))
    atlas_warp = Path(f"{atlas}.warp")
    if atlas_warp.is_file():
        copy_required(atlas_warp, Path(f"{tmp}_atlas.warp"))

    metadata = {
        atlas.parent / "brainsuite_labeldescription.xml": subject.parent / "brainsuite_labeldescription.xml",
        atlas.parent / "sulcal_protocol_HD.xml": Path(f"{tmp}.sulcal_protocol_HD.xml"),
        atlas.parent / "roi_list_svreg.txt": subject.parent / "roi_list_svreg.txt",
        atlas.parent.parent / "svregmanifest.xml": subject.parent / "svregmanifest.xml",
    }
    for src, dst in metadata.items():
        if src.is_file():
            copy_required(src, dst)
    if not (atlas.parent / "roi_list_svreg.txt").is_file():
        print("Atlas ROI list is missing; labels can still be read from the target volume.")

    bst_dir = atlas.parent / "bst_atlas"
    for name in ("brainsuite.icbm452.lpi.v08a.img", "brainsuite.icbm452.lpi.v08a.hdr"):
        src = bst_dir / name
        if src.is_file():
            copy_required(src, temp_dir / name)

    if missing:
        print("Missing required SVReg preparation inputs:", file=sys.stderr)
        for path in dict.fromkeys(missing):
            print(f"  {path}", file=sys.stderr)
        return 1
    print(f"SVReg inputs prepared in {temp_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(prepare_files(sys.argv[1:]))

