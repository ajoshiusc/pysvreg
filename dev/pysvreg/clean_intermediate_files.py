from __future__ import annotations

import sys

try:
    from .runtime import clean_intermediate_files
except ImportError:
    from runtime import clean_intermediate_files


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: clean_intermediate_files.py SUBJECT_PREFIX [flags ...]")
        return 1
    clean_intermediate_files(argv[0], *argv[1:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
