"""Serialized in-process job lifecycle, with optional JSON archives."""

import asyncio
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from app.contracts.jobs import Job, JobStatus, job_summary_from_job
from app.infrastructure.persistence.protocol import IdempotencyConflict

TERMINAL = {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}


class InMemoryJobRepository:
    def __init__(self, archive_dir=None, *, artifact_store=None):
        self._archive = Path(archive_dir) if archive_dir else None
        self._artifact_store = artifact_store
        self._jobs = {}
        self._lock = asyncio.Lock()
        self._subscribers = defaultdict(set)
        if self._archive:
            self._archive.mkdir(parents=True, exist_ok=True)
            for path in self._archive.glob("*.json"):
                job = Job.model_validate_json(path.read_text())
                self._jobs[job.id] = job

    def _save(self, job):
        if self._archive:
            path = self._archive / f"{job.id}.json"
            temp = path.with_suffix(".tmp")
            temp.write_text(job.model_dump_json())
            temp.replace(path)
        self._jobs[job.id] = job
        for queue in self._subscribers[job.id]:
            queue.put_nowait(job.model_copy(deep=True))

    def _create(
        self, request, service_managed, execution_envelope, key=None, digest=None
    ):
        now = datetime.now(timezone.utc)
        job_id = uuid4()
        if execution_envelope is not None:
            execution_envelope = execution_envelope.model_copy(
                update={"execution_id": job_id}
            )
        job = Job(
            id=job_id,
            status=JobStatus.QUEUED,
            request=request,
            created_at=now,
            updated_at=now,
            service_managed=service_managed,
            execution_envelope=execution_envelope,
            idempotency_key=key,
            canonical_request_hash=digest,
        )
        self._save(job)
        return job.model_copy(deep=True)

    async def create(self, request, *, service_managed=False, execution_envelope=None):
        async with self._lock:
            return self._create(request, service_managed, execution_envelope)

    async def create_idempotent(
        self,
        request,
        *,
        idempotency_key,
        canonical_request_hash,
        service_managed=False,
        execution_envelope=None,
    ):
        async with self._lock:
            for job in self._jobs.values():
                if job.idempotency_key == idempotency_key:
                    if job.canonical_request_hash != canonical_request_hash:
                        raise IdempotencyConflict(
                            "Idempotency key already used for another request"
                        )
                    return job.model_copy(deep=True), False
            return self._create(
                request,
                service_managed,
                execution_envelope,
                idempotency_key,
                canonical_request_hash,
            ), True

    async def get(self, job_id):
        job = self._jobs.get(job_id)
        return job.model_copy(deep=True) if job else None

    async def list(self):
        return [
            j.model_copy(deep=True)
            for j in sorted(
                self._jobs.values(), key=lambda j: j.created_at, reverse=True
            )
        ]

    async def list_summaries(self, *, limit=50, offset=0, status=None):
        jobs = [j for j in await self.list() if status is None or j.status == status]
        return [job_summary_from_job(j) for j in jobs[offset : offset + limit]], len(
            jobs
        )

    async def _transition(self, job_id, status, **updates):
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job.status in TERMINAL:
                return job.model_copy(deep=True)
            if (
                status == JobStatus.RUNNING
                and not updates
                and job.status != JobStatus.QUEUED
            ):
                return job.model_copy(deep=True)
            job = job.model_copy(
                deep=True,
                update=dict(
                    status=status, updated_at=datetime.now(timezone.utc), **updates
                ),
            )
            self._save(job)
            return job.model_copy(deep=True)

    async def start(self, job_id):
        return await self._transition(job_id, JobStatus.RUNNING)

    async def cancel(self, job_id):
        return await self._transition(job_id, JobStatus.CANCELLED)

    async def fail(self, job_id, error):
        return await self._transition(job_id, JobStatus.FAILED, error=error)

    async def succeed(self, job_id, result):
        artifacts = []
        if self._artifact_store:
            from app.infrastructure.artifacts.materialize import (
                materialize_result_artifacts,
            )

            job = await self.get(job_id)
            if job is None:
                return None
            if job.status in TERMINAL:
                return job
            artifacts = materialize_result_artifacts(
                self._artifact_store, job_id, result
            )
        return await self._transition(
            job_id, JobStatus.SUCCEEDED, result=result, artifacts=artifacts
        )

    async def update_progress(self, job_id, progress):
        return await self._transition(job_id, JobStatus.RUNNING, progress=progress)

    async def recover_service_jobs(self):
        async with self._lock:
            ids = []
            for job in list(self._jobs.values()):
                if job.service_managed and job.status not in TERMINAL:
                    if job.status == JobStatus.RUNNING:
                        job = job.model_copy(
                            update={
                                "status": JobStatus.QUEUED,
                                "recovery_count": job.recovery_count + 1,
                                "updated_at": datetime.now(timezone.utc),
                            }
                        )
                        self._save(job)
                    ids.append(job.id)
            return ids

    async def subscribe(self, job_id):
        if job_id not in self._jobs:
            return None
        queue = asyncio.Queue()
        self._subscribers[job_id].add(queue)
        queue.put_nowait(self._jobs[job_id].model_copy(deep=True))
        return queue

    async def unsubscribe(self, job_id, queue):
        self._subscribers[job_id].discard(queue)

    async def close(self):
        pass
