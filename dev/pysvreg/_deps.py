from __future__ import annotations

from pathlib import Path
import sys


def _bootstrap_repo_venv() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    venv = repo_root / "dev" / "testing" / ".venv"
    candidates = [venv / "Lib" / "site-packages"]
    lib_dir = venv / "lib"
    if lib_dir.is_dir():
        candidates.extend(sorted(lib_dir.glob("python*/site-packages")))
        candidates.append(lib_dir / "site-packages")

    for site_packages in candidates:
        if not site_packages.is_dir():
            continue
        site_packages_str = str(site_packages)
        if site_packages_str not in sys.path:
            sys.path.insert(0, site_packages_str)


try:
    import nibabel as nib  # noqa: F401
    import scipy  # noqa: F401
except ImportError:
    _bootstrap_repo_venv()
