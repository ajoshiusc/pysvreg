"""Deploy the Python SVReg commands into an existing POSIX BrainSuite tree.

The default mode prints the planned changes. Pass ``--apply`` to copy the
Python package into the BrainSuite SVReg directory and replace its MCR shell
launchers and extensionless compiled commands. Existing files are preserved
beside the replacements with a ``.pysvreg-original`` suffix.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dev.pysvreg.compat_cli import COMMANDS  # noqa: E402


def _check_python(python: str) -> str:
    python_path = Path(python).expanduser()
    executable = str(python_path.absolute()) if python_path.is_file() else shutil.which(python)
    if not executable:
        raise FileNotFoundError(f"Python interpreter not found: {python}")
    if os.name == "nt":
        raise RuntimeError("This deployer installs POSIX shell launchers; use pip console scripts on Windows.")
    subprocess.run(
        [executable, "-c", "import numpy, scipy, nibabel"],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return executable


def _launcher_text(command: str, python: str) -> str:
    # Python paths are inserted inside a shell parameter expansion, so use a
    # single-quoted shell literal and escape embedded single quotes correctly.
    python_literal = "'" + python.replace("'", "'\\''") + "'"
    return f'''#!/bin/sh
SVREG_BIN_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
SVREG_ROOT=$(CDPATH= cd -- "$SVREG_BIN_DIR/.." && pwd)
PYTHON=${{PYSVREG_PYTHON:-{python_literal}}}
export PYSVREG_ROOT="$SVREG_ROOT"
PYTHONPATH="$SVREG_ROOT${{PYTHONPATH:+:$PYTHONPATH}}"
export PYTHONPATH
exec "$PYTHON" -m dev.pysvreg.compat_cli {command} "$@"
'''


def _preserve(path: Path) -> None:
    if path.is_file() and not path.with_name(path.name + ".pysvreg-original").exists():
        shutil.copy2(path, path.with_name(path.name + ".pysvreg-original"))


def deploy(target: Path, python: str, *, apply: bool) -> list[Path]:
    target = target.expanduser().resolve()
    bin_dir = target / "bin"
    if not bin_dir.is_dir():
        raise FileNotFoundError(f"BrainSuite SVReg bin directory not found: {bin_dir}")

    package_source = REPO_ROOT / "dev" / "pysvreg"
    package_target = target / "dev" / "pysvreg"
    destinations = [package_target]
    for command in COMMANDS:
        destinations.extend((bin_dir / command, bin_dir / f"{command}.sh"))
    destinations.extend((bin_dir / "warp_points_linux", bin_dir / "warp_coord_vol_linux"))

    if not apply:
        return destinations

    python = _check_python(python)
    package_target.parent.mkdir(parents=True, exist_ok=True)
    if package_target.exists() and not package_target.with_name("pysvreg.pysvreg-original").exists():
        shutil.copytree(
            package_target,
            package_target.with_name("pysvreg.pysvreg-original"),
            dirs_exist_ok=True,
        )
    package_target.mkdir(parents=True, exist_ok=True)
    for source in package_source.iterdir():
        if source.is_file() and (source.suffix == ".py" or source.name in {"README.md", "COMPATIBILITY.md"}):
            shutil.copy2(source, package_target / source.name)

    launcher_commands = [(command, command) for command in COMMANDS]
    launcher_commands.extend(
        (("warp_points_linux", "warp_points"), ("warp_coord_vol_linux", "warp_coord_vol"))
    )
    for filename, command in launcher_commands:
        path = bin_dir / filename
        _preserve(path)
        path.write_text(_launcher_text(command, python), encoding="utf-8", newline="\n")
        path.chmod(path.stat().st_mode | 0o111)

        if filename in COMMANDS:
            shell_alias = bin_dir / f"{filename}.sh"
            _preserve(shell_alias)
            shell_alias.write_text(_launcher_text(command, python), encoding="utf-8", newline="\n")
            shell_alias.chmod(shell_alias.stat().st_mode | 0o111)

    return destinations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Install MATLAB/MCR-independent Python SVReg launchers into a POSIX BrainSuite tree."
    )
    parser.add_argument("brainsuite_svreg_dir", type=Path, help="BrainSuite SVReg directory containing bin/")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python 3.10+ interpreter with NumPy, SciPy, and NiBabel installed",
    )
    parser.add_argument("--apply", action="store_true", help="write the package and command launchers")
    args = parser.parse_args(argv)

    destinations = deploy(args.brainsuite_svreg_dir, args.python, apply=args.apply)
    if args.apply:
        print(f"Installed Python SVReg launchers in {args.brainsuite_svreg_dir.expanduser().resolve()}")
        print("Original commands were preserved with the .pysvreg-original suffix.")
        print("Set PYSVREG_PYTHON to override the interpreter selected for future launches.")
    else:
        print(f"Dry run: {len(destinations)} paths would be written under {args.brainsuite_svreg_dir}")
        print("Pass --apply to install. Existing command files will be preserved as .pysvreg-original.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
