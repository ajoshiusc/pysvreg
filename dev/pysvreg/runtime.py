from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Sequence
import os
import shutil
import sys
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class RepoPaths:
    repo_root: Path
    src_dir: Path
    third_party_dir: Path
    gifti_dir: Path
    scripts_dir: Path


def get_repo_paths() -> RepoPaths:
    configured_root = os.getenv("PYSVREG_ROOT")
    repo_root = (
        Path(configured_root).expanduser().resolve()
        if configured_root
        else Path(__file__).resolve().parents[2]
    )
    if not configured_root and not (repo_root / "BrainSuiteAtlas1").is_dir():
        for candidate in (Path.cwd(), *Path.cwd().parents):
            if (candidate / "BrainSuiteAtlas1").is_dir() and (candidate / "dev" / "pysvreg").is_dir():
                repo_root = candidate.resolve()
                break
    return RepoPaths(
        repo_root=repo_root,
        src_dir=repo_root / "src",
        third_party_dir=repo_root / "3rdParty",
        gifti_dir=repo_root / "dev" / "matlab_gifti" / "gifti-main",
        scripts_dir=repo_root / "scripts",
    )


def remove_extn_basename(file_name: str) -> str:
    file_name = str(file_name)
    if len(file_name) >= 3 and file_name.lower().endswith(".gz"):
        file_name = file_name[:-3]

    while len(file_name) >= 4 and file_name.lower().endswith((".nii", ".img", ".hdr")):
        file_name = file_name[:-4]

    return file_name


def normalize_fileprefix(fileprefix: str) -> str:
    fileprefix = remove_extn_basename(fileprefix)
    path = Path(fileprefix)
    if not path.parent or str(path.parent) == ".":
        path = Path.cwd() / path.name
    return str(path.resolve())


def default_atlasbasename() -> str:
    return str(get_repo_paths().repo_root / "BrainSuiteAtlas1" / "mri")


def join_flags(tokens: Iterable[str | None]) -> str:
    return "".join(token for token in tokens if token)


def has_flag(flags: str, token: str) -> bool:
    return token in flags


def verbosity_from_flags(flags: str) -> int:
    if "-v" not in flags:
        return 2

    index = flags.find("-v")
    if index == -1 or index + 2 >= len(flags):
        return 2

    try:
        return int(flags[index + 2])
    except ValueError:
        return 2


def disp1(message: str, modn: str = "", flags: str = "") -> None:
    verbosity = verbosity_from_flags(flags)
    if verbosity == 0:
        return

    module_enabled = "m" in flags
    timestamp_enabled = "t" in flags
    now = datetime.now().strftime("%H:%M:%S")

    if not module_enabled:
        print(f"{now} {message}" if timestamp_enabled else message)
        return

    prefix = modn or ""
    if timestamp_enabled:
        print(f"SVREG:{prefix}: {now} {message}")
    else:
        print(f"{prefix}: {message}")


def get_svreg_version(subbasename: str, atlasbasename: str | None = None) -> tuple[str, str]:
    if atlasbasename:
        manifest = Path(remove_extn_basename(atlasbasename)).parent.parent / "svregmanifest.xml"
    else:
        manifest = Path(remove_extn_basename(subbasename)).parent / "svregmanifest.xml"

    if not manifest.is_file():
        print(
            f"SVReg version used to process subject {subbasename} could not be determined. Proceeding..."
        )
        return "", ""

    try:
        root = ET.parse(manifest).getroot()
        version = (root.findtext("version") or "").strip()
        build = (root.findtext("build") or "").strip()
        return version, build
    except ET.ParseError:
        print(
            f"SVReg version used to process subject {subbasename} could not be determined. Proceeding..."
        )
        return "", ""


def append_log(
    logfile: str | Path,
    command_name: str,
    args: Sequence[str],
    *,
    subbasename: str | None = None,
    atlasbasename: str | None = None,
    include_timestamp: bool = True,
    include_version: bool = True,
) -> None:
    logfile = Path(logfile)
    logfile.parent.mkdir(parents=True, exist_ok=True)

    prefix = ""
    if include_timestamp:
        prefix += f"{datetime.now().strftime('%d-%b-%Y %H:%M:%S')}:"

    if include_version and subbasename is not None:
        version, build = get_svreg_version(subbasename, atlasbasename)
        prefix += f"SVReg {version}({build}):"

    line = " ".join([command_name, *[str(arg) for arg in args]])
    with logfile.open("a", encoding="utf-8") as stream:
        stream.write(f"{prefix}{line}\n")


def top_level_log_path(subbasename_tmp: str) -> str:
    return f"{subbasename_tmp}.svreg.log"


def subject_log_path(subbasename: str) -> str:
    return f"{subbasename}.svreg.log"


def required_subject_inputs(subbasename: str) -> list[str]:
    return [
        f"{subbasename}.left.inner.cortex.dfs",
        f"{subbasename}.left.pial.cortex.dfs",
        f"{subbasename}.right.inner.cortex.dfs",
        f"{subbasename}.right.pial.cortex.dfs",
        f"{subbasename}.warp",
        f"{subbasename}.cortex.dewisp.mask.nii.gz",
        f"{subbasename}.cerebrum.mask.nii.gz",
        f"{subbasename}.bfc.nii.gz",
        f"{subbasename}.pvc.frac.nii.gz",
    ]


def remove_flag_and_args(tokens: Sequence[str], flag: str, nargs: int) -> list[str | None]:
    result: list[str | None] = list(tokens)
    index = 0
    while index < len(result):
        if result[index] == flag:
            result[index] = None
            for offset in range(1, nargs + 1):
                if index + offset < len(result):
                    result[index + offset] = None
            index += nargs
        index += 1
    return result


def clean_intermediate_files(subbasename: str, *varargin: str) -> None:
    subbasename = normalize_fileprefix(subbasename)
    subpath = Path(subbasename)
    tmpdir = subpath.parent / f"{subpath.name}.svreg.tmp"

    append_log(
        subject_log_path(subbasename),
        "clean_intermediate_files",
        [subbasename, *varargin],
        subbasename=subbasename,
    )

    try:
        shutil.rmtree(tmpdir)
    except FileNotFoundError:
        return
    except OSError:
        disp1("error deleting temp dir", "clean_intermediate_files", "")


def run_parallel(jobs: Sequence[Callable[[], None]], enabled: bool, max_workers: int | None = None) -> None:
    if not jobs:
        return

    if not enabled or len(jobs) == 1:
        for job in jobs:
            job()
        return

    with ThreadPoolExecutor(max_workers=max_workers or len(jobs)) as executor:
        futures = [executor.submit(job) for job in jobs]
        for future in futures:
            future.result()


def stage_artifact_exists(path: str) -> bool:
    return Path(path).is_file()


def current_platform_suffix() -> str:
    if os.name == "nt":
        return "cmd"
    if sys.platform == "darwin":
        return "mac"
    return "linux"
