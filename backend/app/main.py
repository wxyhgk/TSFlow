from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.router import api_router
from app.infrastructure.artifacts import ArtifactHydrationError, FileArtifactStore
from app.core.config import get_settings
from app.infrastructure.persistence import (
    InMemoryJobRepository,
    SQLiteJobRepository,
)
from app.calculators.models import ModelRegistry


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.settings = settings
    app.state.model_registry = ModelRegistry(settings.checkpoint_dir)
    app.state.artifact_store = FileArtifactStore(settings.artifact_dir)
    if settings.job_repository_backend == "sqlite":
        app.state.job_repository = SQLiteJobRepository(
            settings.job_database_path,
            artifact_store=app.state.artifact_store,
            legacy_archive_dir=settings.job_archive_dir,
        )
    else:
        app.state.job_repository = InMemoryJobRepository(
            settings.job_archive_dir,
            artifact_store=app.state.artifact_store,
        )
    app.state.job_worker = None
    if settings.worker_enabled:
        from app.application.jobs import ProcessJobWorker

        app.state.job_worker = ProcessJobWorker(
            app.state.job_repository,
            app.state.model_registry,
            settings.reference_dir,
            settings.optimization_device,
            max_tasks_per_process=settings.worker_max_tasks_per_process,
            idle_timeout_seconds=settings.worker_idle_timeout_seconds,
            artifact_store=app.state.artifact_store,
        )
        await app.state.job_worker.start()
    try:
        yield
    finally:
        if app.state.job_worker is not None:
            await app.state.job_worker.stop()
        await app.state.job_repository.close()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="API for validated UMA atomistic simulation jobs.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(api_router, prefix=settings.api_prefix)

    @app.exception_handler(ArtifactHydrationError)
    async def artifact_hydration_error(
        _request: Request,
        exc: ArtifactHydrationError,
    ) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {"name": settings.app_name, "docs": "/docs"}

    return app


app = create_app()

