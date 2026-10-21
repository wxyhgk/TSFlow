import asyncio
from pathlib import Path

from app.repositories.jobs import InMemoryJobRepository
from app.repositories.sqlite import SQLiteJobRepository
from app.schemas.jobs import (
    Atom,
    AtomicStructure,
    CalculationResult,
    JobCreate,
    JobStatus,
    OptimizationFrame,
)


def _payload(name: str = "hydride") -> JobCreate:
    return JobCreate(
        calculation="single_point",
        structure=AtomicStructure(
            atoms=[Atom(symbol="H", position=(0.0, 0.0, 0.0))],
            charge=-1,
            spin=1,
            name=name,
        ),
    )


def test_sqlite_repository_survives_restart(tmp_path: Path) -> None:
    async def scenario() -> None:
        database = tmp_path / "jobs.sqlite3"
        repository = SQLiteJobRepository(database)
        created = await repository.create(_payload(), service_managed=True)
        running = await repository.start(created.id)
        assert running is not None
        assert running.status == JobStatus.RUNNING
        await repository.close()

        restored = SQLiteJobRepository(database)
        loaded = await restored.get(created.id)
        assert loaded is not None
        assert loaded.status == JobStatus.RUNNING
        resumable = await restored.recover_service_jobs()
        assert resumable == [created.id]
        recovered = await restored.get(created.id)
        assert recovered is not None
        assert recovered.status == JobStatus.QUEUED
        assert recovered.recovery_count == 1
        await restored.close()

    asyncio.run(scenario())


def test_sqlite_imports_legacy_json_once(tmp_path: Path) -> None:
    async def scenario() -> None:
        archive = tmp_path / "legacy"
        legacy = InMemoryJobRepository(archive)
        created = await legacy.create(_payload("legacy"))

        database = tmp_path / "jobs.sqlite3"
        repository = SQLiteJobRepository(database, legacy_archive_dir=archive)
        imported = await repository.get(created.id)
        assert imported is not None
        assert imported.request.structure.name == "legacy"
        await repository.close()

        duplicate_safe = SQLiteJobRepository(database, legacy_archive_dir=archive)
        jobs = await duplicate_safe.list()
        assert [job.id for job in jobs].count(created.id) == 1
        await duplicate_safe.close()

    asyncio.run(scenario())


def test_sqlite_serializes_concurrent_creates(tmp_path: Path) -> None:
    async def scenario() -> None:
        repository = SQLiteJobRepository(tmp_path / "jobs.sqlite3")
        jobs = await asyncio.gather(
            *(repository.create(_payload(f"job-{index}")) for index in range(20))
        )
        assert len({job.id for job in jobs}) == 20
        await repository.close()

        restored = SQLiteJobRepository(tmp_path / "jobs.sqlite3")
        assert len(await restored.list()) == 20
        await restored.close()

    asyncio.run(scenario())


def test_sqlite_summary_query_does_not_return_trajectory(tmp_path: Path) -> None:
    async def scenario() -> None:
        repository = SQLiteJobRepository(tmp_path / "jobs.sqlite3")
        created = await repository.create(_payload())
        await repository.start(created.id)
        structure = created.request.structure
        completed = await repository.succeed(
            created.id,
            CalculationResult(
                energy_ev=-1.0,
                forces_ev_per_a=[(0.0, 0.0, 0.0)],
                max_force_ev_per_a=0.0,
                final_structure=structure,
                converged=True,
                elapsed_seconds=0.1,
                trajectory=[
                    OptimizationFrame(
                        step=0,
                        energy_ev=-0.9,
                        max_force_ev_per_a=0.1,
                        structure=structure,
                    )
                ],
            ),
        )
        assert completed is not None
        summaries = await repository.list_summaries()
        assert len(summaries) == 1
        assert summaries[0].result_summary is not None
        assert summaries[0].result_summary.has_trajectory is True
        assert summaries[0].result_summary.energy_ev == -1.0
        assert "trajectory" not in summaries[0].model_dump_json()
        await repository.close()

    asyncio.run(scenario())
