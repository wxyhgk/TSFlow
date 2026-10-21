from __future__ import annotations

import asyncio
from typing import Protocol
from uuid import UUID

from app.contracts.jobs import (
    CalculationProgress,
    CalculationResult,
    FrequencyAnalysisResult,
    Job,
    JobCreate,
    JobSummary,
    JobStatus,
    TransitionStateResult,
)
from app.contracts.action_results import ActionResult
from app.contracts.actions import ExecutionEnvelope


JobResult = (
    CalculationResult | TransitionStateResult | FrequencyAnalysisResult | ActionResult
)


class IdempotencyConflict(ValueError):
    """An idempotency key was reused with a different canonical request."""


class JobRepository(Protocol):
    async def create(
        self,
        request: JobCreate,
        *,
        service_managed: bool = False,
        execution_envelope: ExecutionEnvelope | None = None,
    ) -> Job: ...

    async def create_idempotent(
        self,
        request: JobCreate,
        *,
        idempotency_key: str,
        canonical_request_hash: str,
        service_managed: bool = False,
        execution_envelope: ExecutionEnvelope | None = None,
    ) -> tuple[Job, bool]: ...

    async def recover_service_jobs(self) -> list[UUID]: ...
    async def get(self, job_id: UUID) -> Job | None: ...
    async def list(self) -> list[Job]: ...
    async def list_summaries(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        status: JobStatus | None = None,
    ) -> tuple[list[JobSummary], int]: ...
    async def cancel(self, job_id: UUID) -> Job | None: ...
    async def start(self, job_id: UUID) -> Job | None: ...
    async def succeed(self, job_id: UUID, result: JobResult) -> Job | None: ...
    async def fail(self, job_id: UUID, error: str) -> Job | None: ...

    async def update_progress(
        self,
        job_id: UUID,
        progress: CalculationProgress,
    ) -> Job | None: ...

    async def subscribe(self, job_id: UUID) -> asyncio.Queue[Job] | None: ...

    async def unsubscribe(
        self,
        job_id: UUID,
        queue: asyncio.Queue[Job],
    ) -> None: ...

    async def close(self) -> None: ...
