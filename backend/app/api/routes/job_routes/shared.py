"""Shared HTTP-layer validation and job submission helpers."""

from fastapi import HTTPException, Request, Response

from app.contracts.jobs import Job, JobCreate
from app.contracts.actions import ExecutionEnvelope
from app.infrastructure.persistence import IdempotencyConflict


async def store_and_enqueue_job(
    payload: JobCreate,
    request: Request,
    response: Response,
    *,
    idempotency_key: str | None = None,
    canonical_request_hash: str | None = None,
    execution_envelope: ExecutionEnvelope | None = None,
) -> Job:
    if payload.calculation == "transition_state_search":
        typed_action_type = None
        if execution_envelope is not None and execution_envelope.payload.execution_type == "action":
            typed_action_type = execution_envelope.payload.action.action_type
        path_action = typed_action_type in {
            "generate_path_candidates", "optimize_path_candidate",
            "probe_path_curvature", "run_collective_coordinate_scan",
        }
        if payload.transition_state_guess is None and not path_action:
            raise HTTPException(
                status_code=503,
                detail="Automatic path search is not yet restored; supply transition_state_guess",
            )
        if not path_action and (
            payload.parameters.saddle_final_refinement_engine != "geometric"
            or payload.parameters.irc_engine != "geometric"
        ):
            raise HTTPException(
                status_code=422,
                detail="Restored supplied-guess workflow currently supports geomeTRIC only",
            )
        if not path_action and (
            not payload.parameters.sella_refine
            or not payload.parameters.compute_frequencies
        ):
            raise HTTPException(
                status_code=422,
                detail="Supplied-guess validation requires saddle refinement and full frequencies",
            )
    validate_requested_model(request, payload)
    worker = request.app.state.job_worker
    created = True
    if idempotency_key is not None:
        if canonical_request_hash is None:
            raise ValueError("canonical_request_hash is required for idempotency")
        try:
            job, created = await request.app.state.job_repository.create_idempotent(
                payload,
                idempotency_key=idempotency_key,
                canonical_request_hash=canonical_request_hash,
                service_managed=worker is not None,
                execution_envelope=execution_envelope,
            )
        except IdempotencyConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        response.headers["Idempotent-Replayed"] = str(not created).lower()
    else:
        job = await request.app.state.job_repository.create(
            payload,
            service_managed=worker is not None,
            execution_envelope=execution_envelope,
        )
    if worker is not None and created:
        await worker.submit(job.id)
    prefix = request.app.state.settings.api_prefix
    response.headers["Location"] = f"{prefix}/jobs/{job.id}"
    return job


def validate_requested_model(request: Request, payload: JobCreate) -> None:
    model = request.app.state.model_registry.get(payload.model_name)
    if model is None:
        raise HTTPException(status_code=422, detail="unknown model_name")
    if not model.available:
        raise HTTPException(
            status_code=409,
            detail=model.availability_reason or "selected calculator is not available",
        )
    if model.archived:
        raise HTTPException(
            status_code=422, detail="archived models cannot run new jobs"
        )
    if payload.calculation not in model.capabilities:
        raise HTTPException(
            status_code=422,
            detail=f"{payload.model_name} does not support calculation {payload.calculation}",
        )
    if payload.task_name not in model.supported_tasks:
        raise HTTPException(
            status_code=422,
            detail=f"{payload.model_name} does not support task {payload.task_name}",
        )
