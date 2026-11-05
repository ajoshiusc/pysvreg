"""Run the standard SVReg post-extraction processing sequence in Python."""

from __future__ import annotations

import sys

from .standalone import main as standalone_main
from .statistics import generate_stats_xls
from .svreg import svreg
from .thickness_pvc import thickness_pvc


USAGE = """\
SVReg run all: cortical thickness, surface and volume registration, thickness
mapping to the atlas, and ROI statistics.

usage: svreg_runall SUBJECT_PREFIX [ATLAS_PREFIX] [flags ...]
"""


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(USAGE)
        return 0
    subject = argv.pop(0)
    atlas = None
    if argv and not argv[0].startswith("-"):
        atlas = argv.pop(0)
    flags = argv

    print("[1/4] Estimating PVC cortical thickness")
    status = thickness_pvc([subject])
    if status:
        print("svreg_runall halted because thicknessPVC returned an error", file=sys.stderr)
        return status
    print("[2/4] Running surface-constrained volume registration")
    status = svreg(subject, atlas, *flags)
    if status:
        print("svreg_runall halted because svreg returned an error", file=sys.stderr)
        return status
    print("[3/4] Mapping thickness to atlas space")
    status = standalone_main([subject], command="svreg_thickness2atlas")
    if status:
        print("svreg_runall halted because svreg_thickness2atlas returned an error", file=sys.stderr)
        return status
    print("[4/4] Generating ROI statistics")
    status = generate_stats_xls([subject])
    if status:
        print("svreg_runall halted because generate_stats_xls returned an error", file=sys.stderr)
        return status
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

