#!/usr/bin/env bash
# The Python runner also works directly on Windows.
set -euo pipefail
python "$(dirname "$0")/run_experiments.py" "$@"