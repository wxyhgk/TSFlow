#!/bin/sh
set -eu
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
export PYTHONPATH="$PROJECT_DIR/backend:$PROJECT_DIR/packages/uma-workflows/src${PYTHONPATH:+:$PYTHONPATH}"
export UMA_WORKER_ENABLED="${UMA_WORKER_ENABLED:-true}"
export UMA_OPTIMIZATION_DEVICE="${UMA_OPTIMIZATION_DEVICE:-cpu}"
cd "$PROJECT_DIR/backend"
exec "$PROJECT_DIR/.venv-uma/bin/python" -m uvicorn app.main:app --host 127.0.0.1 --port 8010
