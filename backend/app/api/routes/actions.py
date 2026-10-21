import hashlib
import json
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Header, HTTPException, Request, Response, status

from app.api.routes.job_routes.shared import store_and_enqueue_job
from app.application.actions import (
    ACTION_CAPABILITIES,
    action_record_from_job,
    action_to_job,
    guidance_from_job,
)
from app.application.actions.algorithm_capabilities import validate_selected_algorithms
from app.application.actions.preflight import ActionPreflightError, preflight_action
from app.application.actions.algorithm_capabilities import transition_state_algorithm_capabilities
from app.contracts.algorithm_capabilities import TransitionStateAlgorithmCapabilitiesResponse
from app.contracts.actions import (
    ActionExecution,
    ActionCapabilitiesResponse,
    ActionGuidance,
    ActionRecord,
    ActionSpec,
    ExecutionEnvelope,
)


router = APIRouter()


@router.get(
    "/capabilities/transition-state-algorithms",
    response_model=TransitionStateAlgorithmCapabilitiesResponse,
)
async def list_transition_state_algorithms() -> TransitionStateAlgorithmCapabilitiesResponse:
    return TransitionStateAlgorithmCapabilitiesResponse(
        items=list(transition_state_algorithm_capabilities())
    )


@router.get("/capabilities/actions", response_model=ActionCapabilitiesResponse)
async def list_action_capabilities(request: Request) -> ActionCapabilitiesResponse:
    worker_available = request.app.state.job_worker is not None
    available_model = request.app.state.model_registry.get("uma-s-1p2p1")
    model_available = bool(available_model and available_model.available)
    available = worker_available and model_available
    if not worker_available:
        reason = "WORKER_DISABLED"
    elif not model_available:
        reason = "MODEL_CHECKPOINT_UNAVAILABLE"
    else:
        reason = None
    return ActionCapabilitiesResponse(
        items=[
            capability.model_copy(
                update={"available": available, "unavailable_reason": reason}
            )
            for capability in ACTION_CAPABILITIES
        ]
    )


@router.post(
    "/actions",
    response_model=ActionRecord,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_action(
    payload: ActionSpec,
    request: Request,
    response: Response,
    idempotency_key: Annotated[
        str | None,
        Header(alias="Idempotency-Key", min_length=1, max_length=200),
    ] = None,
) -> ActionRecord:
    if request.app.state.job_worker is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "WORKER_UNAVAILABLE",
                "message": "Enable UMA_WORKER_ENABLED before submitting actions.",
            },
        )
    try:
        try:
            validate_selected_algorithms(payload)
        except ValueError as exc:
            message = str(exc)
            if message.startswith("UNKNOWN_PATH_INITIALIZER:"):
                algorithm_id = message.split(":", 1)[1].strip()
                raise HTTPException(
                    status_code=422,
                    detail={"code": "UNKNOWN_PATH_INITIALIZER", "algorithm_id": algorithm_id},
                ) from exc
            if message.startswith("UNKNOWN_PATH_OPTIMIZER:"):
                algorithm_id = message.split(":", 1)[1].strip()
                raise HTTPException(
                    status_code=422,
                    detail={"code": "UNKNOWN_PATH_OPTIMIZER", "algorithm_id": algorithm_id},
                ) from exc
            raise
        preflight = await preflight_action(
            payload,
            repository=request.app.state.job_repository,
            artifact_store=request.app.state.artifact_store,
        )
    except ActionPreflightError as exc:
        raise HTTPException(status_code=exc.status_code, detail={"code": exc.code, "message": exc.message}) from exc
    if payload.action_type not in {
        "evaluate_energy_forces", "optimize_minimum", "analyze_frequencies",
        "transition_state_workflow", "generate_path_candidates",
        "optimize_path_candidate", "probe_path_curvature",
        "run_collective_coordinate_scan", "refine_saddle",
        "compute_full_hessian_and_modes", "run_bidirectional_irc",
    }:
        raise HTTPException(status_code=503, detail="This scientific workflow has not yet been restored")
    canonical_hash = hashlib.sha256(
        json.dumps(
            payload.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    job_request = action_to_job(payload)
    # The execution id is assigned by the repository, so a temporary UUID is
    # replaced below after creation only for display/provenance.  The worker
    # trusts the persisted job id when constructing its final envelope.
    envelope = ExecutionEnvelope(
        execution_id=UUID(int=0),
        payload=ActionExecution(
            action=payload,
            resolved_artifacts=preflight.resolved_artifacts,
            resolved_algorithms=preflight.resolved_algorithms,
        ),
    )
    job = await store_and_enqueue_job(
        job_request,
        request,
        response,
        idempotency_key=idempotency_key,
        canonical_request_hash=canonical_hash if idempotency_key else None,
        execution_envelope=envelope,
    )
    prefix = request.app.state.settings.api_prefix
    response.headers["Location"] = f"{prefix}/actions/{job.id}"
    return action_record_from_job(job, api_prefix=prefix)


@router.get("/actions/{action_id}", response_model=ActionRecord)
async def get_action(action_id: UUID, request: Request) -> ActionRecord:
    job = await request.app.state.job_repository.get(action_id)
    if job is None:
        raise HTTPException(status_code=404, detail="action not found")
    return action_record_from_job(
        job,
        api_prefix=request.app.state.settings.api_prefix,
    )


@router.get("/actions/{action_id}/guidance", response_model=ActionGuidance)
async def get_action_guidance(action_id: UUID, request: Request) -> ActionGuidance:
    job = await request.app.state.job_repository.get(action_id)
    if job is None:
        raise HTTPException(status_code=404, detail="action not found")
    return guidance_from_job(job)

