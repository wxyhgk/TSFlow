import asyncio
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.artifacts import FileArtifactStore
from app.main import app
from app.repositories.jobs import InMemoryJobRepository
from app.schemas.jobs import (
    Atom,
    AtomicStructure,
    CalculationResult,
    JobCreate,
    OptimizationFrame,
)


def _structure() -> AtomicStructure:
    return AtomicStructure(
        atoms=[Atom(symbol="H", position=(0.0, 0.0, 0.0))],
        charge=-1,
        spin=1,
        name="hydride",
        source_format="xyz",
    )


def _result() -> CalculationResult:
    structure = _structure()
    return CalculationResult(
        energy_ev=-0.5,
        forces_ev_per_a=[(0.0, 0.0, 0.0)],
        max_force_ev_per_a=0.0,
        final_structure=structure,
        converged=True,
        elapsed_seconds=0.1,
        trajectory=[
            OptimizationFrame(
                step=0,
                energy_ev=-0.4,
                max_force_ev_per_a=0.1,
                structure=structure,
            )
        ],
    )


def test_artifact_store_verifies_content_hash(tmp_path: Path) -> None:
    store = FileArtifactStore(tmp_path)
    reference = store.write_json(
        uuid4(),
        "summary",
        "calculation_summary",
        {"energy_ev": -0.5},
    )

    assert store.read_verified(reference) == b'{"energy_ev":-0.5}'
    (tmp_path / reference.relative_path).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="size|checksum"):
        store.read_verified(reference)


def test_repository_dual_writes_result_artifacts(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = FileArtifactStore(tmp_path / "artifacts")
        repository = InMemoryJobRepository(
            tmp_path / "jobs",
            artifact_store=store,
        )
        payload = JobCreate(
            calculation="geometry_optimization",
            structure=_structure(),
        )
        job = await repository.create(payload, service_managed=True)
        await repository.start(job.id)
        completed = await repository.succeed(job.id, _result())

        assert completed is not None
        assert completed.artifact_error is None
        artifact_ids = {item.artifact_id for item in completed.artifacts}
        assert artifact_ids == {
            "calculation-summary",
            "optimization-trajectory-json",
            "optimization-trajectory-xyz",
        }
        for reference in completed.artifacts:
            assert store.read_verified(reference)

        restored = InMemoryJobRepository(
            tmp_path / "jobs",
            artifact_store=store,
        )
        restored_job = await restored.get(job.id)
        assert restored_job is not None
        assert restored_job.artifacts == completed.artifacts

    asyncio.run(scenario())


def test_artifact_download_checks_integrity(tmp_path: Path) -> None:
    with TestClient(app) as client:
        store = FileArtifactStore(tmp_path / "artifacts")
        repository = InMemoryJobRepository(
            tmp_path / "jobs",
            artifact_store=store,
        )
        client.app.state.artifact_store = store
        client.app.state.job_repository = repository
        payload = JobCreate(
            calculation="geometry_optimization",
            structure=_structure(),
        )
        job = client.portal.call(repository.create, payload)
        client.portal.call(repository.start, job.id)
        completed = client.portal.call(repository.succeed, job.id, _result())
        assert completed is not None
        reference = next(
            item
            for item in completed.artifacts
            if item.artifact_id == "optimization-trajectory-xyz"
        )

        response = client.get(
            f"/api/v1/jobs/{job.id}/artifacts/{reference.artifact_id}"
        )
        assert response.status_code == 200
        assert response.text.startswith("1\nstep=0")

        (store.root / reference.relative_path).write_bytes(b"tampered")
        rejected = client.get(
            f"/api/v1/jobs/{job.id}/artifacts/{reference.artifact_id}"
        )
        assert rejected.status_code == 409
