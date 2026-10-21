import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.application.jobs.process_worker import ProcessJobWorker
from app.application.jobs.local_worker import LocalJobWorker
from app.contracts.jobs import JobCreate, JobStatus
from app.core.config import Settings
from app.infrastructure.persistence.memory import InMemoryJobRepository


def test_default_retention():
    settings = Settings(_env_file=None)
    assert settings.worker_idle_timeout_seconds == 1800
    assert settings.worker_max_tasks_per_process == 0


def test_idle_retirement_waits_for_computation_and_restarts_deadline():
    async def run():
        worker = ProcessJobWorker(None, None, None, 'cpu', idle_timeout_seconds=.04)
        retired = asyncio.Event()
        started = asyncio.Event()
        finish = asyncio.Event()
        async def execute(_):
            started.set()
            await finish.wait()
        async def retire(**_):
            retired.set()
        worker._execute = execute
        worker._retire_process = retire
        await worker.submit('first')
        task = asyncio.create_task(worker._run())
        await started.wait()
        await asyncio.sleep(.08)
        assert not retired.is_set(), 'Never evict a running calculation'
        finish.set()
        await asyncio.wait_for(retired.wait(), 1)
        await worker._queue.put(None)
        await task
    asyncio.run(run())


def test_unlimited_worker_keeps_process_past_four_jobs():
    async def run():
        from uuid import uuid4
        request = JobCreate(calculation='single_point', structure={
            'atoms':[{'symbol':'H','position':[0,0,0]}], 'charge':0,'spin':2})
        job = SimpleNamespace(id=uuid4(), request=request, execution_envelope=None, status=JobStatus.RUNNING)
        repo = SimpleNamespace(start=AsyncMock(return_value=job), succeed=AsyncMock())
        registry = SimpleNamespace(get=lambda _: SimpleNamespace(available=True, checkpoint='model.pt'))
        worker = ProcessJobWorker(repo, registry, None, 'cpu')
        worker._ensure_process = AsyncMock()
        worker._receive_result = AsyncMock(return_value='result')
        worker._retire_process = AsyncMock()
        worker._process = SimpleNamespace(is_alive=lambda:True)
        worker._input_queue = SimpleNamespace(put=lambda _:None)
        worker._output_queue = object()
        worker._cancel_event = SimpleNamespace(clear=lambda:None,is_set=lambda:False)
        for _ in range(6):
            await worker._execute(job.id)
        worker._retire_process.assert_not_called()
        assert repo.succeed.await_count == 6
    asyncio.run(run())


def test_cancelled_queued_job_never_reaches_either_executor():
    async def run():
        request = JobCreate(calculation='single_point', structure={
            'atoms': [{'symbol': 'H', 'position': [0, 0, 0]}], 'charge': 0, 'spin': 2})
        for worker_type in (ProcessJobWorker, LocalJobWorker):
            repo = InMemoryJobRepository()
            job = await repo.create(request)
            await repo.cancel(job.id)
            registry = SimpleNamespace(get=lambda _: (_ for _ in ()).throw(
                AssertionError('cancelled job looked up a model')))
            if worker_type is ProcessJobWorker:
                worker = worker_type(repo, registry, None, 'cpu')
                worker._ensure_process = AsyncMock()
            else:
                executor = SimpleNamespace(execute=lambda *args: (_ for _ in ()).throw(
                    AssertionError('cancelled job ran calculation')))
                worker = worker_type(repo, registry, executor)
            await worker._execute(job.id)
            assert (await repo.get(job.id)).status == JobStatus.CANCELLED
            if worker_type is ProcessJobWorker:
                worker._ensure_process.assert_not_awaited()
    asyncio.run(run())
