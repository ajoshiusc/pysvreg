from __future__ import annotations

import sys
from pathlib import Path
import shutil

try:
    from .native import copy_attrib_colors, corr_topology_labels, recolor_by_label, surface_jacobian_hemi, writedfs
    from .runtime import normalize_fileprefix
    from .roi_refinement import refine_surface
except ImportError:
    from native import copy_attrib_colors, corr_topology_labels, recolor_by_label, surface_jacobian_hemi, writedfs
    from runtime import normalize_fileprefix
    from roi_refinement import refine_surface


def refine_ROIs2(subbasename: str, hemi: str, *varargin: str) -> None:
    subbasename = normalize_fileprefix(subbasename)
    tmp_base = f"{subbasename}.svreg.tmp/{Path(subbasename).name}"
    flags = "".join(varargin)
    postfix = "" if "-" in flags and "BCI-DNI" not in flags else flags
    ext = f"svreg.{postfix}dfs" if postfix else "svreg.dfs"
    src = f"{tmp_base}.{hemi}.mid.cortex.reg.{postfix}dfs" if postfix else f"{tmp_base}.{hemi}.mid.cortex.reg.dfs"
    dst = f"{tmp_base}.{hemi}.mid.cortex.{ext}"
    surface = corr_topology_labels(src, dst)
    refine_surface(surface)
    writedfs(dst, surface)
    corr_topology_labels(dst)
    xml_name = f"brainsuite_labeldescription.{postfix}.xml" if postfix else "brainsuite_labeldescription.xml"
    xml_candidates = (Path(tmp_base).parent / xml_name, Path(subbasename).parent / xml_name)
    xmlf = next((path for path in xml_candidates if path.is_file()), None)
    if xmlf is not None:
        recolor_by_label(dst, None, str(xmlf))
    copy_attrib_colors(tmp_base, hemi, ext)
    shutil.copyfile(dst, f"{subbasename}.{hemi}.mid.cortex.{ext}")
    shutil.copyfile(f"{tmp_base}.{hemi}.inner.cortex.{ext}", f"{subbasename}.{hemi}.inner.cortex.{ext}")
    shutil.copyfile(f"{tmp_base}.{hemi}.pial.cortex.{ext}", f"{subbasename}.{hemi}.pial.cortex.{ext}")

    # MATLAB's refine_ROIs2 promotes the registered target mid-surface to the
    # subject directory. thickness2atlas, multiparc, and labelwith_atlas use
    # this per-subject atlas surface as their common UV reference.
    if not postfix:
        target_surface = f"{tmp_base}.target.{hemi}.mid.cortex.reg.dfs"
        atlas_surface = Path(subbasename).parent / f"atlas.{hemi}.mid.cortex.svreg.dfs"
        shutil.copyfile(target_surface, atlas_surface)
        surface_jacobian_hemi(subbasename, hemi)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2:
        print("usage: refine_ROIs2.py SUBJECT_PREFIX HEMI [flags ...]")
        return 1
    refine_ROIs2(argv[0], argv[1], *argv[2:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
