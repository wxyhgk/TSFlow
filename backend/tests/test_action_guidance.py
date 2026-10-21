from datetime import UTC, datetime
from typing import get_args
from uuid import uuid4

from app.application.actions import ACTION_CAPABILITIES, guidance_from_job
from app.contracts.action_results import (
    HessianModesActionResult,
    OptimizePathCandidateActionResult,
    PathCurvatureProbeSummary,
    ProbePathCurvatureActionResult,
    RunCollectiveCoordinateScanActionResult,
    SaddleRefinementActionResult,
)
from app.contracts.actions import ActionType
from app.contracts.artifacts import ArtifactReference
from app.contracts.job_records import Job, JobStatus
from app.contracts.requests import JobCreate
from app.contracts.structures import Atom, AtomicStructure
from app.contracts.transition_states import EnergyProfilePoint
from app.contracts.transition_states import (
    CollectiveCoordinateDefinition,
    CollectiveScanPointResult,
)


def _structure() -> AtomicStructure:
    return AtomicStructure(
        atoms=[
            Atom(symbol="O", position=(0.0, 0.0, 0.0)),
            Atom(symbol="H", position=(0.7, 0.5, 0.0)),
            Atom(symbol="H", position=(-0.7, 0.5, 0.0)),
        ],
        charge=0,
        spin=1,
    )


def _artifact(kind: str) -> ArtifactReference:
    return ArtifactReference(
        artifact_id=f"test-{kind.replace('_', '-')}",
        kind=kind,
        media_type="application/octet-stream",
        relative_path=f"test/{kind}",
        sha256="0" * 64,
        size_bytes=1,
        created_at=datetime.now(UTC),
    )


def _job(result) -> Job:
    now = datetime.now(UTC)
    return Job(
        id=uuid4(),
        status=JobStatus.SUCCEEDED,
        request=JobCreate(
            calculation="single_point",
            structure=_structure(),
        ),
        result=result,
        created_at=now,
        updated_at=now,
    )


def test_capability_registry_covers_every_action_type() -> None:
    advertised = {item.action_type for item in ACTION_CAPABILITIES}
    assert advertised == set(get_args(ActionType))
    assert all(
        next_type in advertised
        for item in ACTION_CAPABILITIES
        for next_type in item.possible_next_actions
    )


def test_stationary_saddle_guidance_requests_full_hessian() -> None:
    structure = _structure()
    result = SaddleRefinementActionResult(
        structure=structure,
        energy_ev=-1.0,
        forces_ev_per_a=[(0.0, 0.0, 0.0)] * len(structure.atoms),
        max_force_ev_per_a=0.01,
        converged=True,
        stationary=True,
        optimization_steps=1,
        elapsed_seconds=0.1,
    )

    guidance = guidance_from_job(_job(result))

    assert guidance.scientific_state == "candidate"
    assert [item.action_type for item in guidance.next_actions] == [
        "compute_full_hessian_and_modes"
    ]
    assert guidance.next_actions[0].input_bindings == {
        "structure": "result.structure"
    }


def test_valid_hessian_guidance_exposes_executable_irc_action() -> None:
    structure = _structure()
    result = HessianModesActionResult(
        structure=structure,
        energy_ev=-1.0,
        max_force_ev_per_a=0.01,
        stationary=True,
        frequencies_cm1=[-500.0, 100.0],
        imaginary_frequencies_cm1=[-500.0],
        has_exactly_one_imaginary_mode=True,
        hessian_method="analytic",
        hessian_asymmetry_relative=0.0,
        hessian_quality_valid=True,
        pes_index1_saddle_valid=True,
        hessian_artifact=_artifact("full_hessian"),
        mode_artifact=_artifact("normal_modes"),
        elapsed_seconds=0.1,
    )

    guidance = guidance_from_job(_job(result))

    assert guidance.scientific_state == "index1_saddle"
    assert [item.action_type for item in guidance.next_actions] == [
        "run_bidirectional_irc"
    ]
    assert set(guidance.next_actions[0].input_bindings) == {
        "structure",
        "hessian_artifact",
        "mode_artifact",
    }


def test_optimized_path_guidance_only_refines_an_interior_maximum() -> None:
    structure = _structure()
    source = _artifact("path_candidate")
    optimized = _artifact("optimized_path_candidate")
    result = OptimizePathCandidateActionResult(
        source_path_artifact=source,
        optimized_path_artifact=optimized,
        candidate_index=0,
        optimizer_algorithm_id="dmf.direct_maxflux",
        optimizer_algorithm_version="1",
        converged=True,
        iterations=3,
        optimizer_status="ok",
        energy_profile=[EnergyProfilePoint(
            path_coordinate=0.5,
            energy_ev=-1.0,
            relative_energy_ev=0.2,
        )],
        highest_path_coordinate=0.5,
        highest_structure=structure,
        has_interior_maximum=True,
        barrier_score_ev=0.2,
        coordination_penalty_ev=0.0,
        total_score_ev=0.2,
        same_kind_progress_imbalance=0.0,
        scientific_status="optimized_path_candidate",
        elapsed_seconds=0.1,
    )

    candidate_guidance = guidance_from_job(_job(result))
    no_peak_guidance = guidance_from_job(
        _job(result.model_copy(update={"has_interior_maximum": False}))
    )

    assert candidate_guidance.scientific_state == "candidate"
    assert [item.action_type for item in candidate_guidance.next_actions] == [
        "probe_path_curvature"
    ]
    assert no_peak_guidance.scientific_state == "inconclusive"
    assert [item.action_type for item in no_peak_guidance.next_actions] == [
        "run_collective_coordinate_scan"
    ]


def test_negative_path_curvature_guidance_refines_only_selected_seed() -> None:
    structure = _structure()
    result = ProbePathCurvatureActionResult(
        source_optimized_path_artifact=_artifact("optimized_path_candidate"),
        probes=[PathCurvatureProbeSummary(
            frame_index=2,
            path_coordinate=0.5,
            structure=structure,
            directional_curvature_ev_per_a2=-0.4,
            gradient_projection_ev_per_a=0.2,
            max_force_ev_per_a=0.3,
            has_negative_directional_curvature=True,
        )],
        peak_frame_index=2,
        selected_probe_index=0,
        selected_structure=structure,
        force_evaluations=3,
        scientific_status="negative_directional_curvature_seed",
        elapsed_seconds=0.1,
    )

    candidate = guidance_from_job(_job(result))
    rejected = guidance_from_job(_job(result.model_copy(update={
        "selected_probe_index": None,
        "selected_structure": None,
        "scientific_status": "no_negative_directional_curvature",
    })))

    assert [item.action_type for item in candidate.next_actions] == [
        "refine_saddle"
    ]
    assert candidate.next_actions[0].input_bindings == {
        "structure": "result.selected_structure"
    }
    assert rejected.scientific_state == "inconclusive"
    assert [item.action_type for item in rejected.next_actions] == [
        "run_collective_coordinate_scan"
    ]


def test_collective_scan_guidance_refines_only_converged_interior_maximum() -> None:
    structure = _structure()
    result = RunCollectiveCoordinateScanActionResult(
        source_optimized_path_artifact=_artifact("optimized_path_candidate"),
        scan_artifact=_artifact("collective_coordinate_scan"),
        collective_coordinate=CollectiveCoordinateDefinition(terms=[{
            "atom_i": 0,
            "atom_j": 1,
            "coefficient": 1.0,
        }]),
        points=[CollectiveScanPointResult(
            target_value_a=value,
            coordinate_value_a=value,
            energy_ev=energy,
            relative_energy_ev=energy,
            derivative_ev_per_a=0.0,
            orthogonal_fmax_ev_per_a=0.01,
            converged=True,
            steps=2,
        ) for value, energy in ((0.8, 0.0), (1.0, 0.5), (1.2, 0.0))],
        interior_maxima=[1],
        selected_maximum_index=1,
        selected_structure=structure,
        scientific_status="collective_scan_seed",
        elapsed_seconds=0.1,
    )

    candidate = guidance_from_job(_job(result))
    missing = guidance_from_job(_job(result.model_copy(update={
        "interior_maxima": [],
        "selected_maximum_index": None,
        "selected_structure": None,
        "scientific_status": "collective_scan_no_interior_maximum",
    })))

    assert [item.action_type for item in candidate.next_actions] == [
        "refine_saddle"
    ]
    assert [item.action_type for item in missing.next_actions] == [
        "run_collective_coordinate_scan"
    ]
