from __future__ import annotations

import os
import shlex
import subprocess
from typing import Sequence

class StageExecutionError(RuntimeError):
    """Raised when a stage cannot be executed by the available backends."""


def resolve_external_stage_command(stage_name: str) -> list[str] | None:
    env_name = f"PYSVREG_{stage_name.upper()}_CMD".replace("-", "_")
    if os.getenv(env_name):
        return shlex.split(os.environ[env_name])
    return None


def run_external_stage(stage_name: str, args: Sequence[str]) -> None:
    """Run a Python stage or a user-supplied command without MATLAB fallback."""
    command = resolve_external_stage_command(stage_name)
    if command:
        subprocess.run([*command, *[str(arg) for arg in args]], check=True)
        return

    raise StageExecutionError(
        f"No native Python implementation is registered for stage `{stage_name}`. "
        f"Set `PYSVREG_{stage_name.upper()}_CMD` to an explicit compatible command."
    )
