"""Legacy and compact server-sent job event streams."""

import asyncio
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.contracts.jobs import JobEvent, JobStatus, job_event_from_job


router = APIRouter()
_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


@router.get("/{job_id}/events")
async def stream_job_events(job_id: UUID, request: Request) -> StreamingResponse:
    repository = request.app.state.job_repository
    queue = await repository.subscribe(job_id)
    if queue is None:
        raise HTTPException(status_code=404, detail="job not found")

    async def event_stream():
        try:
            while True:
                try:
                    job = await asyncio.wait_for(queue.get(), timeout=15)
                except TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                yield f"event: job\ndata: {job.model_dump_json()}\n\n"
                if job.status in {
                    JobStatus.SUCCEEDED,
                    JobStatus.FAILED,
                    JobStatus.CANCELLED,
                }:
                    return
        finally:
            await repository.unsubscribe(job_id, queue)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


@router.get("/{job_id}/events-v2", response_model=None)
async def stream_job_events_v2(job_id: UUID, request: Request) -> StreamingResponse:
    repository = request.app.state.job_repository
    queue = await repository.subscribe(job_id)
    if queue is None:
        raise HTTPException(status_code=404, detail="job not found")

    async def event_stream():
        initial = True
        try:
            while True:
                try:
                    job = await asyncio.wait_for(queue.get(), timeout=15)
                except TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                event: JobEvent = job_event_from_job(job, initial=initial)
                initial = False
                yield f"event: job-summary\ndata: {event.model_dump_json()}\n\n"
                if event.event_type == "terminal":
                    return
        finally:
            await repository.unsubscribe(job_id, queue)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )
