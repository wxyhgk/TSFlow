import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from threading import Event
from typing import Callable, Protocol
from uuid import UUID

from app.infrastructure.persistence.protocol import JobRepository
from app.contracts.jobs import (
    CalculationProgress,
    CalculationResult,
    FrequencyAnalysisResult,
    JobCreate,
    JobStatus,
    TransitionStateResult,
)
from app.calculators.models import ModelRegistry


logger = logging.getLogger(__name__)


class JobExecutor(Protocol):
    def execute(
        self,
        request: JobCreate,
        checkpoint: Path | None,
        cancel_event: Event,
        progress_callback: Callable[[CalculationProgress], None] | None = None,
    ) -> CalculationResult | TransitionStateResult | FrequencyAnalysisResult: ...


class LocalJobWorker:
    """Single-consumer local queue for development and workstation execution."""

    def __init__(
        self,
        repository: JobRepository,
        model_registry: ModelRegistry,
        executor: JobExecutor,
    ) -> None:
        self._repository = repository
        self._model_registry = model_registry
        self._executor = executor
        self._queue: asyncio.Queue[UUID | None] = asyncio.Queue()
        self._cancel_events: dict[UUID, Event] = {}
        self._task: asyncio.Task[None] | None = None
        self._stopping = False
        self._thread_pool: ThreadPoolExecutor | None = None

    async def start(self) -> None:
        if self._task is None:
            self._stopping = False
            resumable = await self._repository.recover_service_jobs()
            self._thread_pool = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="uma-model"
            )
            self._task = asyncio.create_task(self._run(), name="uma-local-job-worker")
            for job_id in resumable:
                await self._queue.put(job_id)

    async def stop(self) -> None:
        if self._task is None:
            return
        self._stopping = True
        for event in self._cancel_events.values():
            event.set()
        await self._queue.put(None)
        await self._task
        self._task = None
        if self._thread_pool is not None:
            self._thread_pool.shutdown(wait=True)
            self._thread_pool = None

    async def submit(self, job_id: UUID) -> None:
        await self._queue.put(job_id)

    def cancel(self, job_id: UUID) -> None:
        event = self._cancel_events.get(job_id)
        if event is not None:
            event.set()

    async def _run(self) -> None:
        while True:
            job_id = await self._queue.get()
            try:
                if job_id is None:
                    return
                if self._stopping:
                    await self._repository.cancel(job_id)
                    continue
                await self._execute(job_id)
            finally:
                self._queue.task_done()

    async def _execute(self, job_id: UUID) -> None:
        job = await self._repository.start(job_id)
        if job is None or job.status != JobStatus.RUNNING:
            return

        model = self._model_registry.get(job.request.model_name)
        if model is None or not model.available:
            await self._repository.fail(job_id, "model checkpoint is not available")
            return

        cancel_event = Event()
        self._cancel_events[job_id] = cancel_event
        try:
            if self._thread_pool is None:
                raise RuntimeError("job worker execution thread is not running")
            loop = asyncio.get_running_loop()

            def report_progress(progress: CalculationProgress) -> None:
                future = asyncio.run_coroutine_threadsafe(
                    self._repository.update_progress(job_id, progress), loop
                )
                future.result(timeout=5)

            result = await loop.run_in_executor(
                self._thread_pool,
                partial(
                    self._executor.execute,
                    job.request,
                    Path(model.checkpoint) if model.checkpoint else None,
                    cancel_event,
                    report_progress,
                ),
            )
        except Exception as exc:
            if not cancel_event.is_set():
                logger.exception("calculation job %s failed", job_id)
                await self._repository.fail(job_id, str(exc))
        else:
            if not cancel_event.is_set():
                await self._repository.succeed(job_id, result)
        finally:
            self._cancel_events.pop(job_id, None)
