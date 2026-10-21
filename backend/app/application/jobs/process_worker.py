from __future__ import annotations

import asyncio
import logging
import multiprocessing
from pathlib import Path
from typing import Any
from uuid import UUID

from app.infrastructure.persistence.protocol import JobRepository
from app.application.jobs.child_process import calculation_process_main
from app.application.jobs.messages import receive_worker_result
from app.contracts.jobs import (
    CalculationProgress,
    CalculationResult,
    FrequencyAnalysisResult,
    JobStatus,
    TransitionStateResult,
)
from app.contracts.action_results import ActionResult
from app.calculators.models import ModelRegistry
from app.contracts.actions import ExecutionEnvelope, LegacyJobExecution
from app.infrastructure.artifacts import FileArtifactStore


logger = logging.getLogger(__name__)


class ProcessJobWorker:
    """Single-consumer queue backed by a bounded-lifetime calculator process."""

    def __init__(
        self,
        repository: JobRepository,
        model_registry: ModelRegistry,
        reference_dir: Path,
        optimization_device: str,
        *,
        max_tasks_per_process: int = 0,
        idle_timeout_seconds: float = 1_800.0,
        artifact_store: FileArtifactStore | None = None,
    ) -> None:
        if max_tasks_per_process < 0:
            raise ValueError("max_tasks_per_process must be non-negative")
        if idle_timeout_seconds <= 0:
            raise ValueError("idle_timeout_seconds must be positive")
        self._repository = repository
        self._model_registry = model_registry
        self._reference_dir = reference_dir
        self._optimization_device = optimization_device
        self._max_tasks_per_process = max_tasks_per_process
        self._idle_timeout_seconds = idle_timeout_seconds
        self._artifact_store = artifact_store
        self._queue: asyncio.Queue[UUID | None] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._stopping = False
        self._context = multiprocessing.get_context("spawn")
        self._process: Any | None = None
        self._input_queue: Any | None = None
        self._output_queue: Any | None = None
        self._cancel_event: Any | None = None
        self._tasks_in_process = 0
        self._current_job_id: UUID | None = None

    async def start(self) -> None:
        if self._task is not None:
            return
        self._stopping = False
        resumable = await self._repository.recover_service_jobs()
        self._task = asyncio.create_task(self._run(), name="uma-process-job-worker")
        for job_id in resumable:
            await self._queue.put(job_id)

    async def stop(self) -> None:
        if self._task is None:
            return
        self._stopping = True
        if self._cancel_event is not None:
            self._cancel_event.set()
        await self._queue.put(None)
        await self._task
        self._task = None
        await self._retire_process(force=True)

    async def submit(self, job_id: UUID) -> None:
        await self._queue.put(job_id)

    @property
    def process_pid(self) -> int | None:
        if self._process is not None and self._process.is_alive():
            return self._process.pid
        return None

    def cancel(self, job_id: UUID) -> None:
        if job_id == self._current_job_id and self._cancel_event is not None:
            self._cancel_event.set()

    async def _run(self) -> None:
        while True:
            try:
                job_id = await asyncio.wait_for(
                    self._queue.get(), timeout=self._idle_timeout_seconds
                )
            except asyncio.TimeoutError:
                await self._retire_process(force=False)
                continue
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

        await self._ensure_process()
        assert self._input_queue is not None
        assert self._output_queue is not None
        assert self._cancel_event is not None
        self._current_job_id = job_id
        self._cancel_event.clear()
        envelope = job.execution_envelope or ExecutionEnvelope(
            execution_id=job.id,
            payload=LegacyJobExecution(request=job.request),
        )
        self._input_queue.put(
            (
                envelope.model_dump(mode="json"),
                str(model.checkpoint) if model.checkpoint else None,
            )
        )
        process_failed = False
        try:
            result = await self._receive_result(job_id)
        except Exception as exc:
            process_failed = True
            if not self._cancel_event.is_set():
                logger.exception("calculation child process failed for job %s", job_id)
                await self._repository.fail(job_id, str(exc))
        else:
            if not self._cancel_event.is_set():
                await self._repository.succeed(job_id, result)
        finally:
            self._current_job_id = None
            self._tasks_in_process += 1
            if (
                process_failed
                or self._process is None
                or not self._process.is_alive()
                or (
                    self._max_tasks_per_process > 0
                    and self._tasks_in_process >= self._max_tasks_per_process
                )
            ):
                await self._retire_process(force=process_failed)

    async def _receive_result(
        self, job_id: UUID
    ) -> (
        CalculationResult
        | FrequencyAnalysisResult
        | TransitionStateResult
        | ActionResult
    ):
        assert self._output_queue is not None
        return await receive_worker_result(
            self._output_queue,
            lambda: self._process is not None and self._process.is_alive(),
            self._repository,
            job_id,
        )

    async def _ensure_process(self) -> None:
        if self._process is not None and self._process.is_alive():
            return
        await self._retire_process(force=True)
        self._input_queue = self._context.Queue(maxsize=1)
        self._output_queue = self._context.Queue(maxsize=16)
        self._cancel_event = self._context.Event()
        self._process = self._context.Process(
            target=calculation_process_main,
            args=(
                self._input_queue,
                self._output_queue,
                self._cancel_event,
                str(self._reference_dir),
                self._optimization_device,
                self._max_tasks_per_process,
                self._idle_timeout_seconds,
                str(self._artifact_store.root) if self._artifact_store else None,
            ),
            name="uma-calculation-worker",
            daemon=False,
        )
        self._process.start()
        self._tasks_in_process = 0

    async def _retire_process(self, *, force: bool) -> None:
        process = self._process
        if process is None:
            return
        if process.is_alive() and self._input_queue is not None:
            if force:
                if self._cancel_event is not None:
                    self._cancel_event.set()
                process.terminate()
            else:
                self._input_queue.put(None)
        await asyncio.to_thread(process.join, 10)
        if process.is_alive():
            process.kill()
            await asyncio.to_thread(process.join, 5)
        for channel in (self._input_queue, self._output_queue):
            if channel is not None:
                channel.close()
                channel.join_thread()
        self._process = None
        self._input_queue = None
        self._output_queue = None
        self._cancel_event = None
        self._tasks_in_process = 0
