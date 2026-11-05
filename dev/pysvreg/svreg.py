from __future__ import annotations

import sys
from pathlib import Path

try:
    from .backend import StageExecutionError
    from .clean_intermediate_files import clean_intermediate_files
    from .disp1 import disp1
    from .refine_ROIs2 import refine_ROIs2
    from .refine_sulci_hemi import refine_sulci_hemi
    from .runtime import (
        append_log,
        default_atlasbasename,
        has_flag,
        join_flags,
        normalize_fileprefix,
        remove_flag_and_args,
        required_subject_inputs,
        run_parallel,
        stage_artifact_exists,
        top_level_log_path,
    )
    from .svreg_label_surf_hemi import svreg_label_surf_hemi
    from .svreg_refinements import svreg_refinements
    from .svreg_volreg import svreg_volreg
    from .volmap_ball import volmap_ball
except ImportError:
    from backend import StageExecutionError
    from clean_intermediate_files import clean_intermediate_files
    from disp1 import disp1
    from refine_ROIs2 import refine_ROIs2
    from refine_sulci_hemi import refine_sulci_hemi
    from runtime import (
        append_log,
        default_atlasbasename,
        has_flag,
        join_flags,
        normalize_fileprefix,
        remove_flag_and_args,
        required_subject_inputs,
        run_parallel,
        stage_artifact_exists,
        top_level_log_path,
    )
    from svreg_label_surf_hemi import svreg_label_surf_hemi
    from svreg_refinements import svreg_refinements
    from svreg_volreg import svreg_volreg
    from volmap_ball import volmap_ball


USAGE = """\
SVReg: Surface-Constrained Volumetric Registration

usage: svreg.py SUBJECT_PREFIX [ATLAS_PREFIX] [flags ...]

required input:
SUBJECT_PREFIX      path and filename prefix of the subject's BrainSuite outputs

optional input:
ATLAS_PREFIX        path and filename prefix of atlas files and labels

some optional flags:
-v#     Controls the verbosity of output messages (# is 0, 1, or 2)
-s      Checks if all files necessary for volume registration are present;
        if so, skip the surface registration and perform only volume registration
-S      surface registration only
-k      keep the intermediate files after the svreg sequence is complete
-t      display timestamps along with output messages
-U      single-threaded mode
"""


def parse_cli(argv: list[str]) -> tuple[str, str | None, list[str]]:
    if not argv:
        raise ValueError(USAGE)

    subbasename = argv[0]
    atlasbasename = None
    rest = list(argv[1:])
    if rest and not rest[0].startswith("-"):
        atlasbasename = rest.pop(0)
    return subbasename, atlasbasename, rest


def svreg(subbasename: str, atlasbasename: str | None = None, *varargin: str) -> int:
    subbasename = normalize_fileprefix(subbasename)
    subpath = Path(subbasename)
    pth = subpath.parent
    subname = subpath.name

    raw_varargin = list(varargin)
    raw_varargin.append("-r")

    tmpdir = pth / f"{subname}.svreg.tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)
    subbasename_tmp = str(tmpdir / subname)

    if atlasbasename and atlasbasename.startswith("-"):
        raw_varargin = [atlasbasename, *raw_varargin]
        atlasbasename = None

    atlasbasename = normalize_fileprefix(atlasbasename or default_atlasbasename())

    append_log(
        top_level_log_path(subbasename_tmp),
        "svreg",
        [subbasename, atlasbasename, *raw_varargin],
        include_timestamp=False,
        include_version=False,
    )

    volreg_varargin = list(raw_varargin)
    without_cbm = remove_flag_and_args(raw_varargin, "-cbm", 1)
    surreg_varargin = [token for token in without_cbm if token]
    without_cur = remove_flag_and_args(without_cbm, "-cur", 2)
    flags = join_flags(without_cur)

    if not has_flag(flags, "-gui"):
        disp1(
            "Running Python surface and volume registration for both hemispheres.",
            "main",
            flags,
        )
    else:
        disp1("StartSVREG:SVREG  (svreg)", "main", flags)

    for filename in required_subject_inputs(subbasename):
        if not Path(filename).is_file():
            disp1("The following file does not exist:", "main", flags)
            disp1(filename, "main", flags)
            disp1("Exiting the program", "main", flags)
            return 1

    hemi = ["left", "right"]
    skip_surface_stage = has_flag(flags, "-s") and stage_artifact_exists(
        f"{subbasename_tmp}.target.right.pial.cortex.reg.dfs"
    )

    if not skip_surface_stage:
        run_parallel(
            [
                lambda h=h: svreg_label_surf_hemi(subbasename, atlasbasename, h, *surreg_varargin)
                for h in hemi
            ],
            enabled=not has_flag(flags, "-U"),
            max_workers=2,
        )
        run_parallel(
            [lambda h=h: refine_ROIs2(subbasename, h, flags) for h in hemi],
            enabled=not has_flag(flags, "-U"),
            max_workers=2,
        )

    if not has_flag(flags, "-S"):
        atlasbasename_runtime = f"{subbasename}.target"

        if (not stage_artifact_exists(f"{subbasename_tmp}_unitball_map.mat")) or (not has_flag(flags, "-p")):
            run_parallel(
                [
                    lambda: volmap_ball(atlasbasename_runtime, 1, flags),
                    lambda: volmap_ball(subbasename, 1, flags),
                ],
                enabled=not has_flag(flags, "-U"),
                max_workers=2,
            )

        svreg_volreg(subbasename, atlasbasename_runtime, *volreg_varargin)
        svreg_refinements(subbasename, atlasbasename_runtime, "", flags)
        run_parallel(
            [lambda h=h: refine_sulci_hemi(subbasename, h, flags) for h in hemi],
            enabled=not has_flag(flags, "-U"),
            max_workers=2,
        )

    if not has_flag(flags, "-k"):
        clean_intermediate_files(subbasename)

    disp1("svreg sequence finished", "svreg", flags)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        subbasename, atlasbasename, rest = parse_cli(argv)
    except ValueError as exc:
        print(exc)
        return 1
    try:
        return svreg(subbasename, atlasbasename, *rest)
    except StageExecutionError as exc:
        print(exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
