"""Job collection and individual job query routes."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Request

from app.contracts.jobs import (
    Job,
    JobListResponse,
    JobStatus,
    JobSummaryListResponse,
)


router = APIRouter()


@router.get("", response_model=JobListResponse)
async def list_jobs(request: Request) -> JobListResponse:
    jobs = await request.app.state.job_repository.list()
    return JobListResponse(items=jobs, total=len(jobs))


@router.get("/summaries", response_model=JobSummaryListResponse)
async def list_job_summaries(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    job_status: Annotated[JobStatus | None, Query(alias="status")] = None,
) -> JobSummaryListResponse:
    jobs, total = await request.app.state.job_repository.list_summaries(
        limit=limit,
        offset=offset,
        status=job_status,
    )
    return JobSummaryListResponse(
        items=jobs,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{job_id}", response_model=Job)
async def get_job(job_id: UUID, request: Request) -> Job:
    job = await request.app.state.job_repository.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@router.delete("/{job_id}", response_model=Job)
async def cancel_job(job_id: UUID, request: Request) -> Job:
    job = await request.app.state.job_repository.cancel(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if request.app.state.job_worker is not None:
        request.app.state.job_worker.cancel(job_id)
    return job
