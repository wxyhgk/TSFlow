from datetime import UTC, datetime
from uuid import uuid4

import pytest

from app.application.actions.mapping import action_to_job
from app.application.actions.recovery import decide_transition_state_recovery
from app.contracts.action_results import (
    HessianModesActionResult,
    OptimizePathCandidateActionResult,
    ProbePathCurvatureActionResult,
    SaddleRefinementActionResult,
)
from app.contracts.actions import (
    ActionExecution,
    ComputeFullHessianAndModesAction,
    ExecutionEnvelope,
    OptimizePathCandidateAction,
    ProbePathCurvatureAction,
    RecoveryExecutionContext,
    RefineSaddleAction,
    RunBidirectionalIrcAction,
    RunCollectiveCoordinateScanAction,
)
from app.contracts.artifacts import ArtifactReference, ArtifactSelector
from app.contracts.job_records import Job, JobStatus
from app.contracts.structures import AtomicStructure, ReactionCoordinate
from app.contracts.transition_states import EnergyProfilePoint


NOW = datetime.now(UTC)
STRUCTURE = AtomicStructure.model_validate({
    "atoms": [
        {"symbol": "H", "position": [0.0, 0.0, 0.0]},
        {"symbol": "F", "position": [1.0, 0.0, 0.0]},
    ],
    "charge": 0,
    "spin": 1,
})
PRODUCT = STRUCTURE.model_copy(deep=True)
PRODUCT.atoms[1].position = (1.2, 0.0, 0.0)


class FakeRepository:
    def __init__(self, jobs: list[Job]) -> None:
        self.jobs = {job.id: job for job in jobs}

    async def get(self, job_id):
        return self.jobs.get(job_id)


def artifact(artifact_id: str, kind: str) -> ArtifactReference:
    return ArtifactReference(
        artifact_id=artifact_id,
        kind=kind,
        media_type="application/json",
        relative_path=f"artifacts/{artifact_id}.json",
        sha256="a" * 64,
        size_bytes=10,
        created_at=NOW,
    )


def action_job(action, result, *, context=None, job_id=None) -> Job:
    job_id = job_id or uuid4()
    return Job(
        id=job_id,
        status=JobStatus.SUCCEEDED,
        request=action_to_job(action),
        created_at=NOW,
        updated_at=NOW,
        execution_envelope=ExecutionEnvelope(
            execution_id=job_id,
            payload=ActionExecution(
                action=action,
                recovery_context=context,
            ),
        ),
        result=result,
    )


def optimized_root(
    *,
    has_interior_maximum: bool = True,
) -> tuple[Job, ArtifactReference]:
    reference = artifact("optimized-path", "optimized_path_candidate")
    action = OptimizePathCandidateAction(
        action_type="optimize_path_candidate",
        structure=STRUCTURE,
        product_structure=PRODUCT,
        reaction_coordinates=[ReactionCoordinate(
            atom_i=0,
            atom_j=1,
            kind="breaking",
        )],
        path_candidate=ArtifactSelector(
            source_action_id=uuid4(),
            artifact_id="source-path",
            expected_sha256="b" * 64,
        ),
    )
    result = OptimizePathCandidateActionResult(
        source_path_artifact=artifact("source-path", "path_candidate"),
        optimized_path_artifact=reference,
        candidate_index=0,
        optimizer_algorithm_id="dmf.direct_maxflux",
        optimizer_algorithm_version="1",
        converged=True,
        iterations=5,
        optimizer_status="converged",
        energy_profile=[EnergyProfilePoint(
            path_coordinate=0.5,
            energy_ev=-1.0,
            relative_energy_ev=0.2,
        )],
        highest_path_coordinate=0.5,
        highest_structure=STRUCTURE,
        has_interior_maximum=has_interior_maximum,
        barrier_score_ev=0.2,
        coordination_penalty_ev=0.0,
        total_score_ev=0.2,
        same_kind_progress_imbalance=0.0,
        scientific_status="optimized_path_candidate",
        elapsed_seconds=1.0,
    )
    return action_job(action, result), reference


def recovery_context(root: Job, reference: ArtifactReference, **updates):
    values = {
        "recovery_id": root.id,
        "parent_action_id": root.id,
        "source_optimized_path": ArtifactSelector(
            source_action_id=root.id,
            artifact_id=reference.artifact_id,
            expected_sha256=reference.sha256,
        ),
        "step_index": 2,
    }
    values.update(updates)
    return RecoveryExecutionContext(**values)


@pytest.mark.asyncio
async def test_recovery_starts_with_curvature_probe() -> None:
    root, _ = optimized_root()
    decision = await decide_transition_state_recovery(
        root,
        repository=FakeRepository([root]),
    )

    assert decision.state == "advanced"
    assert decision.code == "PATH_CURVATURE_PROBE_QUEUED"
    assert isinstance(decision.next_action, ProbePathCurvatureAction)
    assert decision.next_context is not None
    assert decision.next_context.recovery_id == root.id


@pytest.mark.asyncio
async def test_missing_negative_curvature_falls_back_to_collective_scan() -> None:
    root, reference = optimized_root()
    context = recovery_context(root, reference)
    action = ProbePathCurvatureAction(
        action_type="probe_path_curvature",
        structure=STRUCTURE,
        product_structure=PRODUCT,
        reaction_coordinates=(
            root.execution_envelope.payload.action.reaction_coordinates
        ),
        optimized_path=context.source_optimized_path,
    )
    result = ProbePathCurvatureActionResult(
        source_optimized_path_artifact=reference,
        probes=[],
        peak_frame_index=1,
        selected_probe_index=None,
        selected_structure=None,
        force_evaluations=0,
        scientific_status="no_negative_directional_curvature",
        elapsed_seconds=1.0,
    )
    current = action_job(action, result, context=context)
    decision = await decide_transition_state_recovery(
        current,
        repository=FakeRepository([root, current]),
    )

    assert isinstance(decision.next_action, RunCollectiveCoordinateScanAction)
    assert decision.next_context.collective_scan_attempts == 1


@pytest.mark.asyncio
async def test_stationary_refinement_requires_full_hessian() -> None:
    root, reference = optimized_root()
    context = recovery_context(root, reference, saddle_refinement_attempts=1)
    action = RefineSaddleAction(
        action_type="refine_saddle",
        structure=STRUCTURE,
    )
    result = SaddleRefinementActionResult(
        structure=STRUCTURE,
        energy_ev=-1.0,
        forces_ev_per_a=[(0.0, 0.0, 0.0)] * 2,
        max_force_ev_per_a=0.01,
        converged=True,
        stationary=True,
        optimization_steps=2,
        elapsed_seconds=1.0,
    )
    current = action_job(action, result, context=context)
    decision = await decide_transition_state_recovery(
        current,
        repository=FakeRepository([root, current]),
    )

    assert isinstance(decision.next_action, ComputeFullHessianAndModesAction)
    assert decision.next_context.saddle_refinement_attempts == 1


@pytest.mark.asyncio
async def test_valid_hessian_advances_to_irc_with_expected_endpoints() -> None:
    root, reference = optimized_root()
    context = recovery_context(root, reference, saddle_refinement_attempts=1)
    action = ComputeFullHessianAndModesAction(
        action_type="compute_full_hessian_and_modes",
        structure=STRUCTURE,
    )
    result = HessianModesActionResult(
        structure=STRUCTURE,
        energy_ev=-1.0,
        max_force_ev_per_a=0.01,
        stationary=True,
        frequencies_cm1=[-300.0, 100.0],
        imaginary_frequencies_cm1=[-300.0],
        has_exactly_one_imaginary_mode=True,
        hessian_method="analytic",
        hessian_asymmetry_relative=0.0,
        hessian_quality_valid=True,
        pes_index1_saddle_valid=True,
        hessian_artifact=artifact("full-hessian", "full_hessian"),
        mode_artifact=artifact("normal-modes", "normal_modes"),
        elapsed_seconds=1.0,
    )
    current = action_job(action, result, context=context)
    decision = await decide_transition_state_recovery(
        current,
        repository=FakeRepository([root, current]),
    )

    assert isinstance(decision.next_action, RunBidirectionalIrcAction)
    assert decision.next_action.expected_reactant == STRUCTURE
    assert decision.next_action.expected_product == PRODUCT
    assert decision.next_context.irc_attempts == 1


@pytest.mark.asyncio
async def test_invalid_hessian_stops_after_scan_budget() -> None:
    root, reference = optimized_root()
    context = recovery_context(
        root,
        reference,
        saddle_refinement_attempts=2,
        collective_scan_attempts=1,
    )
    action = ComputeFullHessianAndModesAction(
        action_type="compute_full_hessian_and_modes",
        structure=STRUCTURE,
    )
    result = HessianModesActionResult(
        structure=STRUCTURE,
        energy_ev=-1.0,
        max_force_ev_per_a=0.01,
        stationary=True,
        frequencies_cm1=[100.0, 200.0],
        imaginary_frequencies_cm1=[],
        has_exactly_one_imaginary_mode=False,
        hessian_method="analytic",
        hessian_asymmetry_relative=0.0,
        hessian_quality_valid=True,
        pes_index1_saddle_valid=False,
        hessian_artifact=artifact("full-hessian", "full_hessian"),
        mode_artifact=artifact("normal-modes", "normal_modes"),
        elapsed_seconds=1.0,
    )
    current = action_job(action, result, context=context)
    decision = await decide_transition_state_recovery(
        current,
        repository=FakeRepository([root, current]),
    )

    assert decision.state == "exhausted"
    assert decision.code == "TS_RECOVERY_BUDGET_EXHAUSTED"
    assert decision.next_action is None
