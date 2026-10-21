"""Behavioral checks for the restored local execution and persistence boundary."""

import asyncio
from pathlib import Path
import pytest
from app.contracts.jobs import JobCreate, JobStatus, CalculationResult
from app.contracts.actions import (
    ActionExecution,
    EvaluateEnergyForcesAction,
    ExecutionEnvelope,
)
from app.infrastructure.persistence.sqlite import SQLiteJobRepository
from app.infrastructure.persistence.protocol import IdempotencyConflict
from uuid import UUID


def payload():
    return JobCreate(
        calculation="single_point",
        structure={
            "atoms": [{"symbol": "H", "position": [0, 0, 0]}],
            "charge": -1,
            "spin": 1,
        },
    )


def test_idempotency_survives_restart_and_conflicts(tmp_path):
    async def run():
        path = tmp_path / "jobs.sqlite3"
        repo = SQLiteJobRepository(path)
        request = payload()
        action = EvaluateEnergyForcesAction(
            action_type="evaluate_energy_forces", structure=request.structure
        )
        envelope = ExecutionEnvelope(
            execution_id=UUID(int=0), payload=ActionExecution(action=action)
        )
        first, created = await repo.create_idempotent(
            request,
            idempotency_key="key",
            canonical_request_hash="one",
            execution_envelope=envelope,
        )
        assert created and first.execution_envelope.execution_id == first.id
        await repo.close()
        repo = SQLiteJobRepository(path)
        second, created = await repo.create_idempotent(
            request, idempotency_key="key", canonical_request_hash="one"
        )
        assert not created and second.id == first.id
        assert (
            second.execution_envelope.payload.action.action_type
            == "evaluate_energy_forces"
        )
        with pytest.raises(IdempotencyConflict):
            await repo.create_idempotent(
                request, idempotency_key="key", canonical_request_hash="two"
            )
        await repo.close()

    asyncio.run(run())


def test_cancelled_job_cannot_be_completed_by_late_worker(tmp_path):
    async def run():
        repo = SQLiteJobRepository(tmp_path / "jobs.sqlite3")
        job = await repo.create(payload())
        queue = await repo.subscribe(job.id)
        assert (await queue.get()).status == JobStatus.QUEUED
        await repo.start(job.id)
        await repo.cancel(job.id)
        result = CalculationResult(
            energy_ev=-1,
            forces_ev_per_a=[(0, 0, 0)],
            max_force_ev_per_a=0,
            final_structure=job.request.structure,
            elapsed_seconds=0,
        )
        await repo.succeed(job.id, result)
        assert (await repo.get(job.id)).status == JobStatus.CANCELLED
        assert (await repo.get(job.id)).result is None
        await repo.unsubscribe(job.id, queue)
        await repo.close()

    asyncio.run(run())


def test_summary_pagination_and_result_reload(tmp_path):
    async def run():
        path = tmp_path / "jobs.sqlite3"
        repo = SQLiteJobRepository(path)
        jobs = [await repo.create(payload()) for _ in range(3)]
        job = jobs[0]
        await repo.start(job.id)
        await repo.succeed(
            job.id,
            CalculationResult(
                energy_ev=-1,
                forces_ev_per_a=[(0, 0, 0)],
                max_force_ev_per_a=0,
                final_structure=job.request.structure,
                elapsed_seconds=0,
            ),
        )
        await repo.close()
        repo = SQLiteJobRepository(path)
        page, total = await repo.list_summaries(limit=1, offset=1)
        assert total == 3 and len(page) == 1
        succeeded, total = await repo.list_summaries(status=JobStatus.SUCCEEDED)
        assert total == 1 and succeeded[0].result_summary.energy_ev == -1
        assert (await repo.get(job.id)).result.energy_ev == -1
        await repo.close()

    asyncio.run(run())


def test_action_api_idempotency_and_availability(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from app.core.config import get_settings
    from app.main import create_app

    monkeypatch.setenv("UMA_JOB_DATABASE_PATH", str(tmp_path / "api.sqlite3"))
    monkeypatch.setenv("UMA_JOB_ARCHIVE_DIR", str(tmp_path / "archive"))
    monkeypatch.setenv("UMA_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("UMA_WORKER_ENABLED", "false")
    get_settings.cache_clear()

    class Worker:
        def __init__(self):
            self.submitted = []

        async def submit(self, job_id):
            self.submitted.append(job_id)

        async def stop(self):
            pass

    try:
        with TestClient(create_app()) as client:
            body = {
                "action_type": "evaluate_energy_forces",
                "structure": payload().structure.model_dump(mode="json"),
            }
            assert client.post("/api/v1/actions", json=body).status_code == 503
            worker = Worker()
            client.app.state.job_worker = worker
            headers = {"Idempotency-Key": "same"}
            first = client.post("/api/v1/actions", json=body, headers=headers)
            assert first.status_code == 202, first.text
            second = client.post("/api/v1/actions", json=body, headers=headers)
            assert (
                second.status_code == 202 and second.json()["id"] == first.json()["id"]
            )
            assert second.headers["Idempotent-Replayed"] == "true"
            assert len(worker.submitted) == 1
            assert client.get(first.json()["guidance_url"]).status_code == 200
            body["structure"]["name"] = "changed"
            assert (
                client.post("/api/v1/actions", json=body, headers=headers).status_code
                == 409
            )
    finally:
        get_settings.cache_clear()


def test_running_job_accepts_progress_and_preserves_it(tmp_path):
    from app.contracts.jobs import CalculationProgress

    async def run():
        path = tmp_path / "progress.sqlite3"
        repo = SQLiteJobRepository(path)
        job = await repo.create(payload())
        await repo.start(job.id)
        await repo.update_progress(
            job.id,
            CalculationProgress(
                step=2,
                max_steps=10,
                energy_ev=-1,
                max_force_ev_per_a=0.2,
                optimization_device="cpu",
                current_structure=job.request.structure,
                stage="irc_forward",
            ),
        )
        await repo.close()
        repo = SQLiteJobRepository(path)
        loaded = await repo.get(job.id)
        assert loaded.status == JobStatus.RUNNING and loaded.progress.step == 2
        assert loaded.progress.stage == "irc_forward"
        await repo.close()

    asyncio.run(run())
