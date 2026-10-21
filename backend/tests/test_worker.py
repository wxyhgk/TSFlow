import asyncio
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import Callable

from app.repositories.jobs import InMemoryJobRepository
from app.schemas.jobs import (
    AtomicStructure,
    CalculationProgress,
    CalculationResult,
    JobCreate,
    JobStatus,
)
from app.services.job_worker import LocalJobWorker


class FakeRegistry:
    def get(self, _name: str):
        return SimpleNamespace(available=True, checkpoint="/tmp/fake-checkpoint.pt")


class FakeExecutor:
    def execute(
        self,
        request: JobCreate,
        _checkpoint: Path,
        _cancel_event: Event,
        progress_callback: Callable[[CalculationProgress], None] | None = None,
    ) -> CalculationResult:
        if progress_callback is not None:
            progress_callback(
                CalculationProgress(
                    step=1,
                    max_steps=1,
                    energy_ev=-76.0,
                    max_force_ev_per_a=0.0,
                    optimization_device="cpu",
                    current_structure=request.structure,
                )
            )
        return CalculationResult(
            energy_ev=-76.0,
            forces_ev_per_a=[(0.0, 0.0, 0.0)] * len(request.structure.atoms),
            max_force_ev_per_a=0.0,
            final_structure=AtomicStructure.model_validate(request.structure),
            elapsed_seconds=0.01,
        )


def test_local_worker_completes_a_job() -> None:
    async def scenario() -> None:
        repository = InMemoryJobRepository()
        worker = LocalJobWorker(repository, FakeRegistry(), FakeExecutor())  # type: ignore[arg-type]
        payload = JobCreate.model_validate(
            {
                "calculation": "single_point",
                "structure": {
                    "atoms": [
                        {"symbol": "O", "position": [0.0, 0.0, 0.0]},
                        {"symbol": "H", "position": [0.76, 0.58, 0.0]},
                        {"symbol": "H", "position": [-0.76, 0.58, 0.0]},
                    ],
                    "charge": 0,
                    "spin": 1,
                },
            }
        )
        job = await repository.create(payload)
        events = await repository.subscribe(job.id)
        assert events is not None
        await worker.start()
        await worker.submit(job.id)

        for _ in range(100):
            completed = await repository.get(job.id)
            if completed is not None and completed.status == JobStatus.SUCCEEDED:
                break
            await asyncio.sleep(0.01)
        await worker.stop()

        completed = await repository.get(job.id)
        assert completed is not None
        assert completed.status == JobStatus.SUCCEEDED
        assert completed.result is not None
        assert completed.result.energy_ev == -76.0
        assert completed.progress is not None
        assert completed.progress.step == 1
        latest_event = events.get_nowait()
        assert latest_event.status == JobStatus.SUCCEEDED
        await repository.unsubscribe(job.id, events)

    asyncio.run(scenario())


def test_repository_preserves_latest_path_energy_profile() -> None:
    async def scenario() -> None:
        repository = InMemoryJobRepository()
        payload = JobCreate.model_validate(
            {
                "calculation": "single_point",
                "structure": {
                    "atoms": [{"symbol": "H", "position": [0.0, 0.0, 0.0]}],
                    "charge": 0,
                    "spin": 1,
                },
            }
        )
        job = await repository.create(payload)
        await repository.start(job.id)
        await repository.update_progress(
            job.id,
            CalculationProgress(
                step=20,
                max_steps=100,
                energy_ev=-1.0,
                max_force_ev_per_a=0.2,
                optimization_device="mps",
                current_structure=payload.structure,
                stage="dmf",
                path_energies_ev=[-1.0, -0.7, -0.9],
                reactant_endpoint=payload.structure,
                product_endpoint=payload.structure,
            ),
        )
        await repository.update_progress(
            job.id,
            CalculationProgress(
                step=1,
                max_steps=100,
                energy_ev=-0.7,
                max_force_ev_per_a=0.1,
                optimization_device="mps",
                current_structure=payload.structure,
                stage="sella_mps",
            ),
        )

        updated = await repository.get(job.id)
        assert updated is not None
        assert updated.progress is not None
        assert updated.progress.stage == "sella_mps"
        assert updated.progress.path_energies_ev == [-1.0, -0.7, -0.9]
        assert updated.progress.reactant_endpoint == payload.structure
        assert updated.progress.product_endpoint == payload.structure

    asyncio.run(scenario())


def test_repository_restores_jobs_from_json_archive(tmp_path: Path) -> None:
    async def scenario() -> None:
        payload = JobCreate.model_validate(
            {
                "calculation": "single_point",
                "structure": {
                    "atoms": [{"symbol": "H", "position": [0.0, 0.0, 0.0]}],
                    "charge": -1,
                    "spin": 1,
                },
            }
        )
        repository = InMemoryJobRepository(tmp_path)
        created = await repository.create(payload)

        restored = InMemoryJobRepository(tmp_path)
        loaded = await restored.get(created.id)

        assert loaded is not None
        assert loaded.id == created.id
        assert loaded.request == payload

    asyncio.run(scenario())


def test_worker_recovers_only_service_managed_jobs(tmp_path: Path) -> None:
    async def scenario() -> None:
        payload = JobCreate.model_validate(
            {
                "calculation": "single_point",
                "structure": {
                    "atoms": [{"symbol": "H", "position": [0.0, 0.0, 0.0]}],
                    "charge": -1,
                    "spin": 1,
                },
            }
        )
        first = InMemoryJobRepository(tmp_path)
        managed = await first.create(payload, service_managed=True)
        unmanaged = await first.create(payload)
        await first.start(managed.id)

        restored = InMemoryJobRepository(tmp_path)
        worker = LocalJobWorker(restored, FakeRegistry(), FakeExecutor())  # type: ignore[arg-type]
        await worker.start()
        for _ in range(100):
            recovered = await restored.get(managed.id)
            if recovered is not None and recovered.status == JobStatus.SUCCEEDED:
                break
            await asyncio.sleep(0.01)
        await worker.stop()

        recovered = await restored.get(managed.id)
        untouched = await restored.get(unmanaged.id)
        assert recovered is not None
        assert recovered.status == JobStatus.SUCCEEDED
        assert recovered.recovery_count == 1
        assert untouched is not None
        assert untouched.status == JobStatus.QUEUED

    asyncio.run(scenario())
