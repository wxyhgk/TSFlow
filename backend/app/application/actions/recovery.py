from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from app.contracts.action_results import (
    BidirectionalIrcActionResult,
    HessianModesActionResult,
    OptimizePathCandidateActionResult,
    ProbePathCurvatureActionResult,
    RunCollectiveCoordinateScanActionResult,
    SaddleRefinementActionResult,
)
from app.contracts.actions import (
    ActionSpec,
    ComputeFullHessianAndModesAction,
    OptimizePathCandidateAction,
    ProbePathCurvatureAction,
    RecoveryExecutionContext,
    RefineSaddleAction,
    RunBidirectionalIrcAction,
    RunCollectiveCoordinateScanAction,
)
from app.contracts.artifacts import ArtifactSelector
from app.contracts.job_records import Job, JobStatus
from app.infrastructure.persistence.protocol import JobRepository


RecoveryState = Literal["advanced", "complete", "blocked", "exhausted"]


@dataclass(frozen=True)
class RecoveryDecision:
    recovery_id: UUID
    state: RecoveryState
    code: str
    message: str
    next_action: ActionSpec | None = None
    next_context: RecoveryExecutionContext | None = None

    @property
    def terminal(self) -> bool:
        return self.next_action is None


async def decide_transition_state_recovery(
    job: Job,
    *,
    repository: JobRepository,
) -> RecoveryDecision:
    """Advance one inspectable, bounded TS-recovery step.

    The function never executes chemistry directly.  It only turns trusted,
    persisted action results into one whitelisted next Action.
    """
    envelope = job.execution_envelope
    if envelope is None or envelope.payload.execution_type != "action":
        return _stop(
            job.id,
            "blocked",
            "ACTION_LINEAGE_REQUIRED",
            "Recovery can only advance typed Actions with a persisted execution "
            "envelope.",
        )
    if job.status != JobStatus.SUCCEEDED or job.result is None:
        return _stop(
            job.id,
            "blocked",
            "ACTION_NOT_SUCCEEDED",
            "Wait for a successful terminal Action before advancing recovery.",
        )

    execution = envelope.payload
    action = execution.action
    result = job.result
    context = execution.recovery_context

    if context is None:
        if not (
            isinstance(action, OptimizePathCandidateAction)
            and isinstance(result, OptimizePathCandidateActionResult)
        ):
            return _stop(
                job.id,
                "blocked",
                "RECOVERY_ROOT_REQUIRED",
                "Start recovery from a completed optimize_path_candidate Action.",
            )
        context = RecoveryExecutionContext(
            recovery_id=job.id,
            parent_action_id=job.id,
            source_optimized_path=ArtifactSelector(
                source_action_id=job.id,
                artifact_id=result.optimized_path_artifact.artifact_id,
                expected_sha256=result.optimized_path_artifact.sha256,
            ),
            step_index=1,
        )

    root = await repository.get(context.source_optimized_path.source_action_id)
    root_action = _optimized_path_action(root)
    if root_action is None:
        return _stop(
            context.recovery_id,
            "blocked",
            "RECOVERY_SOURCE_INVALID",
            "The recovery chain no longer resolves to its optimized-path source "
            "Action.",
        )

    if isinstance(result, OptimizePathCandidateActionResult):
        if not result.converged:
            return _stop(
                context.recovery_id,
                "blocked",
                "PATH_NOT_CONVERGED",
                "Path optimization needs an explicit bounded retry before TS recovery.",
            )
        if result.has_interior_maximum:
            next_action = ProbePathCurvatureAction(
                action_type="probe_path_curvature",
                model_name=root_action.model_name,
                task_name=root_action.task_name,
                structure=root_action.structure,
                product_structure=root_action.product_structure,
                reaction_coordinates=root_action.reaction_coordinates,
                optimized_path=context.source_optimized_path,
            )
            return _advance(
                job,
                context,
                next_action,
                "PATH_CURVATURE_PROBE_QUEUED",
                "The optimized path has an interior maximum; nearby negative "
                "curvature will be screened before saddle refinement.",
            )
        return _collective_scan_or_stop(job, context, root_action)

    if isinstance(result, ProbePathCurvatureActionResult):
        if result.selected_structure is not None:
            return _refine(
                job,
                context,
                result.selected_structure,
                "A negative-directional-curvature path frame was selected for "
                "order-one saddle refinement.",
            )
        return _collective_scan_or_stop(job, context, root_action)

    if isinstance(result, RunCollectiveCoordinateScanActionResult):
        if result.selected_structure is not None:
            return _refine(
                job,
                context,
                result.selected_structure,
                "The converged interior maximum of the collective-coordinate scan "
                "was selected for order-one saddle refinement.",
            )
        return _stop(
            context.recovery_id,
            "exhausted",
            "COLLECTIVE_SCAN_NO_INTERIOR_MAXIMUM",
            "The bounded automatic collective scan found no converged interior "
            "maximum; AI input is required to change the coordinate or targets.",
        )

    if isinstance(result, SaddleRefinementActionResult):
        if result.stationary:
            next_action = ComputeFullHessianAndModesAction(
                action_type="compute_full_hessian_and_modes",
                model_name=action.model_name,
                task_name=action.task_name,
                structure=result.structure,
            )
            return _advance(
                job,
                context,
                next_action,
                "FULL_HESSIAN_QUEUED",
                "The force threshold is met; a full Hessian will classify the "
                "stationary point.",
            )
        if (
            context.collective_scan_attempts >= 1
            and context.saddle_refinement_attempts < 2
        ):
            return _refine(
                job,
                context,
                result.structure,
                "The post-scan saddle refinement is not stationary; one bounded "
                "continuation from its latest structure is allowed.",
            )
        return _collective_scan_or_stop(job, context, root_action)

    if isinstance(result, HessianModesActionResult):
        if result.pes_index1_saddle_valid:
            next_action = RunBidirectionalIrcAction(
                action_type="run_bidirectional_irc",
                model_name=action.model_name,
                task_name=action.task_name,
                structure=result.structure,
                hessian_artifact=result.hessian_artifact,
                mode_artifact=result.mode_artifact,
                expected_reactant=root_action.structure,
                expected_product=root_action.product_structure,
            )
            return _advance(
                job,
                context,
                next_action,
                "BIDIRECTIONAL_IRC_QUEUED",
                "The stationary full Hessian has exactly one imaginary mode; "
                "bidirectional IRC will determine connectivity.",
            )
        return _collective_scan_or_stop(job, context, root_action)

    if isinstance(result, BidirectionalIrcActionResult):
        endpoints_converged = (
            result.endpoint_a_relaxation.converged
            and result.endpoint_b_relaxation.converged
        )
        if result.path_complete and endpoints_converged:
            if (
                result.endpoint_match_evaluated
                and result.endpoint_matches_expected is True
            ):
                code = "VALIDATED_TRANSITION_STATE"
                message = (
                    "The index-1 saddle and both relaxed IRC endpoints match the "
                    "target reaction."
                )
            elif (
                result.endpoint_match_evaluated
                and result.endpoint_matches_expected is False
            ):
                code = "IRC_TOPOLOGY_MISMATCH"
                message = (
                    "The local saddle is valid, but the relaxed IRC endpoints do "
                    "not match the target reaction."
                )
            else:
                code = "IRC_CONNECTED_SADDLE"
                message = (
                    "The index-1 saddle has complete IRC connectivity without a "
                    "target endpoint comparison."
                )
            return _stop(context.recovery_id, "complete", code, message)
        if context.irc_attempts < 2 and isinstance(
            action,
            RunBidirectionalIrcAction,
        ):
            parameters = action.parameters.model_copy(
                update={
                    "steps": min(action.parameters.steps * 2, 1_000),
                    "endpoint_relaxation_max_steps": min(
                        action.parameters.endpoint_relaxation_max_steps * 2,
                        10_000,
                    ),
                }
            )
            return _advance(
                job,
                context,
                action.model_copy(update={"parameters": parameters}),
                "BOUNDED_IRC_RETRY_QUEUED",
                "IRC or endpoint relaxation was incomplete; one larger bounded "
                "retry is allowed.",
            )
        return _stop(
            context.recovery_id,
            "exhausted",
            "IRC_RECOVERY_EXHAUSTED",
            "IRC remained incomplete after the finite retry budget; its endpoints "
            "must not be labelled as final reactant or product.",
        )

    return _stop(
        context.recovery_id,
        "blocked",
        "UNSUPPORTED_RECOVERY_RESULT",
        f"Action result {type(result).__name__} is outside the recovery state machine.",
    )


def _optimized_path_action(job: Job | None) -> OptimizePathCandidateAction | None:
    if job is None or job.execution_envelope is None:
        return None
    payload = job.execution_envelope.payload
    if payload.execution_type != "action":
        return None
    if isinstance(payload.action, OptimizePathCandidateAction):
        return payload.action
    return None


def _collective_scan_or_stop(
    job: Job,
    context: RecoveryExecutionContext,
    root: OptimizePathCandidateAction,
) -> RecoveryDecision:
    if context.collective_scan_attempts >= 1:
        return _stop(
            context.recovery_id,
            "exhausted",
            "TS_RECOVERY_BUDGET_EXHAUSTED",
            "The finite automatic scan/refinement budget is exhausted; AI input "
            "is required.",
        )
    if not root.reaction_coordinates:
        return _stop(
            context.recovery_id,
            "blocked",
            "COLLECTIVE_COORDINATE_REQUIRED",
            "No reaction coordinate is available for an automatic collective "
            "scan; supply an explicit linear combination through a manual Action.",
        )
    next_action = RunCollectiveCoordinateScanAction(
        action_type="run_collective_coordinate_scan",
        model_name=root.model_name,
        task_name=root.task_name,
        structure=root.structure,
        product_structure=root.product_structure,
        reaction_coordinates=root.reaction_coordinates,
        optimized_path=context.source_optimized_path,
    )
    return _advance(
        job,
        context,
        next_action,
        "COLLECTIVE_SCAN_QUEUED",
        "Path curvature or saddle classification did not yield a usable index-1 "
        "seed; a bounded reaction-derived collective-coordinate scan will be run.",
    )


def _refine(
    job: Job,
    context: RecoveryExecutionContext,
    structure,
    message: str,
) -> RecoveryDecision:
    if context.saddle_refinement_attempts >= 2:
        return _stop(
            context.recovery_id,
            "exhausted",
            "SADDLE_REFINEMENT_BUDGET_EXHAUSTED",
            "The finite order-one saddle-refinement budget is exhausted.",
        )
    envelope = job.execution_envelope
    if envelope is None or envelope.payload.execution_type != "action":
        raise RuntimeError("recovery refinement requires an Action envelope")
    current_action = envelope.payload.action
    next_action = RefineSaddleAction(
        action_type="refine_saddle",
        model_name=current_action.model_name,
        task_name=current_action.task_name,
        structure=structure,
    )
    return _advance(job, context, next_action, "SADDLE_REFINEMENT_QUEUED", message)


def _advance(
    job: Job,
    context: RecoveryExecutionContext,
    action: ActionSpec,
    code: str,
    message: str,
) -> RecoveryDecision:
    if context.step_index >= 32:
        return _stop(
            context.recovery_id,
            "exhausted",
            "TS_RECOVERY_STEP_LIMIT_REACHED",
            "The recovery chain reached its server-controlled 32-step limit.",
        )
    next_context = context.model_copy(update={
        "parent_action_id": job.id,
        "step_index": context.step_index + 1,
        "saddle_refinement_attempts": context.saddle_refinement_attempts
        + int(isinstance(action, RefineSaddleAction)),
        "collective_scan_attempts": context.collective_scan_attempts
        + int(isinstance(action, RunCollectiveCoordinateScanAction)),
        "irc_attempts": context.irc_attempts
        + int(isinstance(action, RunBidirectionalIrcAction)),
    })
    return RecoveryDecision(
        recovery_id=context.recovery_id,
        state="advanced",
        code=code,
        message=message,
        next_action=action,
        next_context=next_context,
    )


def _stop(
    recovery_id: UUID,
    state: RecoveryState,
    code: str,
    message: str,
) -> RecoveryDecision:
    return RecoveryDecision(
        recovery_id=recovery_id,
        state=state,
        code=code,
        message=message,
    )
