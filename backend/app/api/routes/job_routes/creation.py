"""Validated JSON job submission."""
from fastapi import APIRouter, Request, Response
from app.contracts.jobs import Job, JobCreate
from app.api.routes.job_routes.shared import store_and_enqueue_job
router=APIRouter()

@router.post('', response_model=Job, status_code=202)
async def create_job(payload: JobCreate, request: Request, response: Response):
    return await store_and_enqueue_job(payload, request, response)
