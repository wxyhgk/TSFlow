#!/bin/sh
set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PYTHON="$PROJECT_DIR/.venv-uma/bin/python"

if test ! -x "$PYTHON"; then
  echo "Missing $PYTHON; create the environment first with: uv venv --python 3.13 .venv-uma" >&2
  exit 1
fi

exec uv pip install --python "$PYTHON" \
  -e "$PROJECT_DIR/packages/uma-workflows" \
  -e "$PROJECT_DIR/backend" \
  pymatgen
