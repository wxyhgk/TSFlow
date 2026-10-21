#!/bin/sh
set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
export PYTHONPATH="$PROJECT_DIR/backend:$PROJECT_DIR/packages/uma-workflows/src${PYTHONPATH:+:$PYTHONPATH}"
export UMA_WORKER_ENABLED=false

exec "$PROJECT_DIR/.venv-uma/bin/python" -m pytest -q -p no:cacheprovider \
  "$PROJECT_DIR/backend/tests/test_restored_backend.py" \
  "$PROJECT_DIR/packages/uma-workflows/tests/test_restored_ts_irc.py"
