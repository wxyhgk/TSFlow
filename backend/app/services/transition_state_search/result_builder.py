from __future__ import annotations

import time
from typing import Literal

from app.contracts.jobs import (
    BondChangeAssessment,
    EnergyProfilePoint,
    JobCreate,
    ReactionPathFrame,
    SaddleOptimizationFrame,
    TransitionStateResult,
)
from app.formats.gaussian import render_transition_state_input
from app.services.transition_state_search.irc_result_adapter import (
    irc_result_fields,
)
from app.services.transition_state_search.structures import (
    structure_from_atoms,
    structure_from_positions,
)
from uma_workflows.transition_states.workflow.metrics import max_force
from uma_workflows.transition_states.workflow.endpoints import EndpointStageResult
from uma_workflows.transition_states.workflow.path import PathStageResult
from uma_workflows.transition_states.workflow.saddle import SaddleStageResult
from uma_workflows.transition_states.workflow.irc import IrcStageResult
from uma_workflows.transition_states.outcomes import TransitionOutcome


def build_result(
    request: JobCreate,
    endpoints: EndpointStageResult,
    path_result: PathStageResult,
    saddle: SaddleStageResult,
    screening,
    irc: IrcStageResult,
    outcome: TransitionOutcome,
    *,
    device: Literal["cpu", "mps"],
    started: float,
    warnings: list[str],
) -> TransitionStateResult:
    params = request.parameters
    frequency = screening.frequency_analysis
    transition_structure = structure_from_atoms(
        request.structure,
        saddle.atoms,
        name="UMA/DMF transition-state candidate",
    )
    sampled_path = [
        ReactionPathFrame(
            path_coordinate=float(coordinate),
            structure=structure_from_positions(
                request.structure,
                positions,
                name=f"DMF path t={coordinate:.5f}",
            ),
        )
        for coordinate, positions in zip(
            path_result.sample_coordinates,
            path_result.sampled_positions,
            strict=True,
        )
    ]
    energy_profile = [
        EnergyProfilePoint(
            path_coordinate=float(coordinate),
            energy_ev=energy,
            relative_energy_ev=energy - path_result.cpu_energies_ev[0],
        )
        for coordinate, energy in zip(
            path_result.sample_coordinates,
            path_result.cpu_energies_ev,
            strict=True,
        )
    ]
    refinement = saddle.refinement
    optimization = path_result.optimization
    # The reusable workflow keeps algorithm metadata on its concrete
    # initializer/optimizer objects rather than duplicating it on every path
    # result. Older backend result code expected flattened fields, so derive
    # stable values here with explicit fallbacks for supplied guesses.
    initializer_algorithm_id = getattr(
        path_result,
        "initializer_algorithm_id",
        "fbenm.correlated",
    )
    initializer_algorithm_version = getattr(
        path_result,
        "initializer_algorithm_version",
        "1",
    )
    path_metadata = getattr(path_result.path, "metadata", {}) or {}
    optimizer_algorithm_id = getattr(
        path_result,
        "optimizer_algorithm_id",
        path_metadata.get("algorithm_id", "dmf.direct_maxflux"),
    )
    optimizer_algorithm_version = getattr(
        path_result,
        "optimizer_algorithm_version",
        path_metadata.get("algorithm_version", "1"),
    )
    irc_validated = irc.validation_passed is True
    if screening.passes_screening:
        forward_barrier = max(
            0.0,
            screening.energy_ev - path_result.cpu_energies_ev[0],
        )
        reverse_barrier = max(
            0.0,
            screening.energy_ev - path_result.cpu_energies_ev[-1],
        )
        barrier_source = (
            "validated_transition_state"
            if irc_validated
            else "screened_first_order_saddle"
        )
    else:
        forward_barrier = outcome.forward_path_rise_ev
        reverse_barrier = outcome.reverse_path_rise_ev
        barrier_source = (
            "converged_path_estimate"
            if optimization.converged
            else "unconverged_path_estimate"
        )
    return TransitionStateResult(
        transition_state=transition_structure,
        transition_state_energy_ev=screening.energy_ev,
        transition_state_forces_ev_per_a=[
            tuple(float(value) for value in row)
            for row in screening.forces_ev_per_a
        ],
        transition_state_max_force_ev_per_a=screening.max_force_ev_per_a,
        reactant_energy_ev=path_result.cpu_energies_ev[0],
        product_energy_ev=path_result.cpu_energies_ev[-1],
        forward_barrier_ev=forward_barrier,
        reverse_barrier_ev=reverse_barrier,
        barrier_source=barrier_source,
        barrier_is_validated=(screening.passes_screening and irc_validated),
        path=sampled_path,
        energy_profile=energy_profile,
        highest_path_coordinate=path_result.highest_coordinate,
        path_has_interior_maximum=path_result.has_interior_maximum,
        scientific_outcome=outcome.kind,
        scientific_outcome_reason=outcome.reason,
        forward_path_rise_ev=outcome.forward_path_rise_ev,
        reverse_path_rise_ev=outcome.reverse_path_rise_ev,
        near_barrierless_threshold_ev=(
            outcome.near_barrierless_threshold_ev
        ),
        near_barrierless_directions=list(
            outcome.near_barrierless_directions
        ),
        dmf_iterations=optimization.iterations,
        converged=optimization.converged,
        transition_state_stationary=screening.stationary,
        endpoint_preoptimization_performed=(
            endpoints.reactant.performed or endpoints.product.performed
        ),
        reactant_fragment_count=endpoints.reactant.fragment_count,
        product_fragment_count=endpoints.product.fragment_count,
        reactant_preoptimization_steps=(
            endpoints.reactant.mps_steps + endpoints.reactant.cpu_steps
        ),
        product_preoptimization_steps=(
            endpoints.product.mps_steps + endpoints.product.cpu_steps
        ),
        path_candidates_evaluated=path_result.candidates_evaluated,
        selected_path_candidate=path_result.selected_candidate,
        path_initializer_algorithm_id=initializer_algorithm_id,
        path_initializer_algorithm_version=initializer_algorithm_version,
        path_optimizer_algorithm_id=optimizer_algorithm_id,
        path_optimizer_algorithm_version=optimizer_algorithm_version,
        saddle_refinement_performed=(refinement is not None),
        saddle_refinement_converged=(
            refinement.converged if refinement else None
        ),
        sella_mps_steps=(refinement.mps_steps if refinement else 0),
        sella_cpu_steps=(refinement.cpu_steps if refinement else 0),
        saddle_trajectory=[
            SaddleOptimizationFrame(
                stage=frame.stage,
                step=frame.step,
                energy_ev=frame.energy_ev,
                max_force_ev_per_a=max_force(frame.forces_ev_per_a),
                structure=structure_from_positions(
                    request.structure,
                    frame.positions_a,
                    name=f"UMA saddle refinement: {frame.stage}",
                ),
            )
            for frame in saddle.trajectory
        ],
        frequency_analysis_performed=frequency is not None,
        frequency_scope=screening.frequency_scope,
        frequency_active_atom_indices=(
            list(screening.frequency_active_atom_indices) if frequency else []
        ),
        frequencies_cm1=(frequency.frequencies_cm1 if frequency else []),
        imaginary_frequencies_cm1=(
            frequency.imaginary_frequencies_cm1 if frequency else []
        ),
        has_exactly_one_imaginary_mode=(
            len(frequency.imaginary_frequencies_cm1) == 1
            if frequency
            else None
        ),
        primary_imaginary_mode=(
            [
                tuple(float(value) for value in row)
                for row in frequency.primary_imaginary_mode
            ]
            if frequency and frequency.primary_imaginary_mode is not None
            else None
        ),
        imaginary_mode_reaction_overlap=(
            screening.mode_reaction_overlap.score
            if screening.mode_reaction_overlap
            else None
        ),
        imaginary_mode_matches_reaction=(
            screening.mode_reaction_overlap.matches
            if screening.mode_reaction_overlap
            else None
        ),
        reaction_coordinate_consistent=(
            screening.reaction_coordinate_consistent
        ),
        reaction_definition_valid=screening.reaction_definition_valid,
        geometry_progress_consistent=(
            screening.geometry_progress_consistent
        ),
        mechanism_mode_consistent=screening.mechanism_mode_consistent,
        bond_changes=[
            BondChangeAssessment(
                atom_i=item.atom_i,
                atom_j=item.atom_j,
                kind=item.kind,
                reactant_distance_a=item.reactant_distance_a,
                transition_state_distance_a=item.transition_state_distance_a,
                product_distance_a=item.product_distance_a,
                path_fraction=item.path_fraction,
            )
            for item in screening.bond_changes
        ],
        hessian_quality_valid=screening.hessian_quality_valid,
        pes_index1_saddle_valid=screening.pes_index1_saddle_valid,
        local_saddle_valid=screening.local_saddle_valid,
        irc_eligible=screening.irc_eligible,
        passes_uma_ts_screening=screening.passes_screening,
        screening_diagnostic_codes=list(screening.diagnostic_codes),
        screening_failure_code=screening.failure_code,
        irc_failure_code=irc.failure_code,
        retry_actions=list(
            dict.fromkeys(screening.retry_actions + irc.retry_actions)
        ),
        **irc_result_fields(request.structure, irc),
        frequency_elapsed_seconds=(
            frequency.elapsed_seconds if frequency else None
        ),
        optimizer_status=optimization.optimizer_status,
        nmove=params.nmove,
        update_teval=bool(optimization.adaptive_sampling_enabled),
        convergence=params.dmf_convergence,
        optimization_device=device,
        gaussian_input=render_transition_state_input(
            transition_structure,
            energy_ev=screening.energy_ev,
            max_force_ev_per_a=screening.max_force_ev_per_a,
        ),
        elapsed_seconds=time.perf_counter() - started,
        warnings=warnings,
    )
