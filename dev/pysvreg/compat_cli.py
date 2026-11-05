"""Command-name compatible entry point used by the former SVReg shell launchers."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

COMMANDS = {
    "svreg": ("dev.pysvreg.svreg", "main"),
    "svreg_label_surf_hemi": ("dev.pysvreg.svreg_label_surf_hemi", "main"),
    "refine_ROIs2": ("dev.pysvreg.refine_ROIs2", "main"),
    "refine_sulci_hemi": ("dev.pysvreg.refine_sulci_hemi", "main"),
    "volmap_ball": ("dev.pysvreg.volmap_ball", "main"),
    "svreg_volreg": ("dev.pysvreg.svreg_volreg", "main"),
    "svreg_refinements": ("dev.pysvreg.svreg_refinements", "main"),
    "clean_intermediate_files": ("dev.pysvreg.clean_intermediate_files", "main"),
    "svreg_smooth_vol_function": ("dev.pysvreg.standalone", "main"),
    "svreg_resample": ("dev.pysvreg.standalone", "main"),
    "svreg_get_mni_tal": ("dev.pysvreg.standalone", "main"),
    "svreg_mni2sub": ("dev.pysvreg.standalone", "main"),
    "svreg_apply_map": ("dev.pysvreg.standalone", "main"),
    "svreg_smooth_surf_function": ("dev.pysvreg.standalone", "main"),
    "svreg_thickness2atlas": ("dev.pysvreg.standalone", "main"),
    "svreg_labelwith_atlas": ("dev.pysvreg.standalone", "main"),
    "register_cc_curve": ("dev.pysvreg.standalone", "main"),
    "generate_stats_xls": ("dev.pysvreg.statistics", "generate_stats_xls"),
    "generate_vol_param_stats_xls": ("dev.pysvreg.statistics", "generate_vol_param_stats_xls"),
    "svreg_prepare_files": ("dev.pysvreg.prepare_files", "prepare_files"),
    "svreg_make_atlas": ("dev.pysvreg.make_atlas", "main"),
    "svreg_multiparc": ("dev.pysvreg.multiparc", "main"),
    "svreg_sulcal_map": ("dev.pysvreg.sulcal_map", "main"),
    "thicknessPVC": ("dev.pysvreg.thickness_pvc", "main"),
    "svreg_runall": ("dev.pysvreg.runall", "main"),
    "gui_bias_correct": ("dev.pysvreg.gui_bias_correct", "main"),
    "warp_points": ("dev.pysvreg.air_cli", "main"),
    "warp_coord_vol": ("dev.pysvreg.air_cli", "main"),
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: compat_cli.py COMMAND [arguments ...]", file=sys.stderr)
        return 2
    command, *args = argv
    command = Path(command).name
    if command.lower().endswith(".exe"):
        command = command[:-4]
    for suffix in ("_linux", "_mac"):
        if command.endswith(suffix):
            command = command[: -len(suffix)]
    if command not in COMMANDS:
        print(
            f"{command}: Python replacement is not implemented yet. "
            "See dev/pysvreg/README.md for the current coverage list.",
            file=sys.stderr,
        )
        return 78
    module_name, function_name = COMMANDS[command]
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    if module_name.endswith(("standalone", "air_cli")):
        return int(function(args, command=command))
    return int(function(args))


if __name__ == "__main__":
    raise SystemExit(main())
