import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest


# Unit/API tests must never recover or execute the workstation's durable queue.
os.environ["UMA_WORKER_ENABLED"] = "false"
# Configure paths before test modules import app.main and cache Settings.
_storage = TemporaryDirectory(prefix="uma-backend-tests-")
_root = Path(_storage.name)
os.environ["UMA_JOB_DATABASE_PATH"] = str(_root / "jobs.sqlite3")
os.environ["UMA_JOB_ARCHIVE_DIR"] = str(_root / "jobs")
os.environ["UMA_ARTIFACT_DIR"] = str(_root / "artifacts")


@pytest.fixture(scope="session", autouse=True)
def isolated_storage():
    yield
    _storage.cleanup()
