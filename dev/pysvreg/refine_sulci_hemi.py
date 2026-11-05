from __future__ import annotations

import sys
from pathlib import Path
import shutil

try:
    from .native import readdfc_sipi, readdfs, writedfc
    from .runtime import normalize_fileprefix
    from .sulcal_refinement import refine_curves_on_surface
except ImportError:
    from native import readdfc_sipi, readdfs, writedfc
    from runtime import normalize_fileprefix
    from sulcal_refinement import refine_curves_on_surface


def refine_sulci_hemi(subbasename: str, hemi: str, *varargin: str) -> None:
    subbasename = normalize_fileprefix(subbasename)
    tmp_base = f"{subbasename}.svreg.tmp/{Path(subbasename).name}"
    tmp_surface = Path(f"{tmp_base}.{hemi}.mid.cortex.reg.dfs")
    tmp_source = Path(f"{tmp_base}.{hemi}.mapped.dfc")
    use_tmp = tmp_surface.is_file() and tmp_source.is_file()

    source = tmp_source if use_tmp else Path(f"{subbasename}.{hemi}.mapped.dfc")
    surface = tmp_surface if use_tmp else Path(f"{subbasename}.{hemi}.mid.cortex.svreg.dfs")
    if not source.is_file() or not surface.is_file():
        raise FileNotFoundError(f"required surface or mapped curves are missing for {subbasename}.{hemi}")

    protocol = Path(f"{tmp_base}.sulcal_protocol_HD.xml") if use_tmp else Path(f"{subbasename}.sulcal_protocol_HD.xml")
    if not protocol.is_file():
        raise FileNotFoundError(f"sulcal protocol not found: {protocol}")

    # The MATLAB routine reads mapped.dfc, applies geodesic curvature flow,
    # and writes one mapped.refined.dfc output.
    curves, _ = readdfc_sipi(str(source))
    refined = refine_curves_on_surface(readdfs(str(surface)), curves)
    target = Path(f"{tmp_base}.{hemi}.mapped.refined.dfc") if use_tmp else Path(f"{subbasename}.{hemi}.mapped.refined.dfc")
    writedfc(str(target), refined, str(protocol))
    if use_tmp:
        shutil.copyfile(target, f"{subbasename}.{hemi}.mapped.refined.dfc")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2:
        print("usage: refine_sulci_hemi.py SUBJECT_PREFIX HEMI [flags ...]")
        return 1
    refine_sulci_hemi(argv[0], argv[1], *argv[2:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
