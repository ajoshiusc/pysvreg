from __future__ import annotations

import sys
from pathlib import Path

try:
    from .native import load_nii_z, make_unitball_map, readdfs
    from .runtime import normalize_fileprefix
except ImportError:
    from native import load_nii_z, make_unitball_map, readdfs
    from runtime import normalize_fileprefix


def volmap_ball(basename: str, resample_factor: str | int | float = 1, *varargin: str) -> None:
    basename = normalize_fileprefix(basename)
    path = Path(basename)
    is_target = path.name.endswith(".target")
    subject_name = path.name[:-7] if is_target else path.name
    tmp_base = str(path.parent / f"{subject_name}.svreg.tmp" / path.name)
    mask_base = tmp_base if is_target else basename
    surface_base = tmp_base
    mask = load_nii_z(f"{mask_base}.cortex.dewisp.mask.nii.gz")
    surfaces = [
        readdfs(f"{surface_base}.left.inner.cortex.reg.dfs"),
        readdfs(f"{surface_base}.right.inner.cortex.reg.dfs"),
        readdfs(f"{surface_base}.left.pial.cortex.reg.dfs"),
        readdfs(f"{surface_base}.right.pial.cortex.reg.dfs"),
    ]
    make_unitball_map(tmp_base, mask_base, tuple(mask.zooms[:3]), *surfaces)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: volmap_ball.py FILEPREFIX [resample_factor] [flags ...]")
        return 1
    basename = argv[0]
    resample_factor = argv[1] if len(argv) > 1 and not argv[1].startswith("-") else 1
    rest = argv[2:] if resample_factor != 1 else argv[1:]
    volmap_ball(basename, resample_factor, *rest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
