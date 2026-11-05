#!/bin/sh
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON=${PYTHON:-python3}
exec "$PYTHON" "$SCRIPT_DIR/../dev/pysvreg/compat_cli.py" "gui_bias_correct_linux" "$@"
