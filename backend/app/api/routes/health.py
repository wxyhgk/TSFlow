from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel


router = APIRouter()


class HealthResponse(BaseModel):
    status: Literal["ok"]
    environment: str
    checkpoints_available: int
    calculators_available: int
    worker_enabled: bool
    worker_pid: int | None
    worker_idle_timeout_seconds: float


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    models = request.app.state.model_registry.list_models()
    return HealthResponse(
        status="ok",
        environment=request.app.state.settings.environment,
        checkpoints_available=sum(
            model.available and model.engine == "uma" for model in models
        ),
        calculators_available=sum(model.available for model in models),
        worker_enabled=request.app.state.job_worker is not None,
        worker_pid=(
            request.app.state.job_worker.process_pid
            if request.app.state.job_worker is not None
            else None
        ),
        worker_idle_timeout_seconds=request.app.state.settings.worker_idle_timeout_seconds,
    )
