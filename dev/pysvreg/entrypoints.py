"""Console-script adapters preserving the historical SVReg command names."""

from __future__ import annotations

import sys

from .compat_cli import COMMANDS, main as dispatch


def _make_entry(command: str):
    def run() -> int:
        return dispatch([command, *sys.argv[1:]])

    run.__name__ = f"run_{command}"
    run.__qualname__ = run.__name__
    run.__doc__ = f"Run the Python replacement for the {command} command."
    return run


for _command in COMMANDS:
    globals()[f"run_{_command}"] = _make_entry(_command)

del _command
