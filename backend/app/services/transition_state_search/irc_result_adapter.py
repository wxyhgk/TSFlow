from __future__ import annotations

from app.contracts.jobs import (
    AtomicStructure,
    IrcEndpointRelaxationResult,
    IrcPathFrameResult,
    OptimizationFrame,
)
from app.services.transition_state_search.structures import (
    structure_from_positions,
)
from uma_workflows.transition_states.workflow.irc import IrcStageResult


def irc_result_fields(
    template: AtomicStructure,
    irc: IrcStageResult,
) -> dict:
    result = irc.result
    topology = result.endpoint_topology if result else None
    raw_topology = result.raw_endpoint_topology if result else None
    endpoint_a = result.endpoint_a_relaxation if result else None
    endpoint_b = result.endpoint_b_relaxation if result else None
    roles_are_final = bool(
        topology
        and topology.matches_expected_endpoints
        and endpoint_a
        and endpoint_b
        and endpoint_a.converged
        and endpoint_b.converged
    )
    return {
        "irc_performed": irc.performed,
        "irc_validation_passed": irc.validation_passed,
        "irc_error": irc.error,
        "irc_frames": (
            [
                IrcPathFrameResult(
                    path_coordinate=frame.path_coordinate,
                    energy_ev=frame.energy_ev,
                    relative_energy_ev=frame.relative_energy_ev,
                    max_force_ev_per_a=frame.max_force_ev_per_a,
                    converged=frame.converged,
                    structure=structure_from_positions(
                        template,
                        frame.positions_a,
                        name=f"UMA IRC s={frame.path_coordinate:.5f}",
                    ),
                )
                for frame in result.frames
            ]
            if result
            else []
        ),
        "irc_forward_frame_count": (
            result.forward_frame_count if result else 0
        ),
        "irc_reverse_frame_count": (
            result.reverse_frame_count if result else 0
        ),
        "irc_forward_inner_loop_failed": (
            result.forward_inner_loop_failed if result else None
        ),
        "irc_reverse_inner_loop_failed": (
            result.reverse_inner_loop_failed if result else None
        ),
        "irc_path_complete": result.path_complete if result else None,
        "irc_endpoint_distance_delta_a": (
            result.endpoint_distance_delta_a if result else None
        ),
        "irc_relaxed_endpoint_distance_delta_a": (
            result.relaxed_endpoint_distance_delta_a if result else None
        ),
        "irc_raw_endpoint_matches_expected": (
            raw_topology.matches_expected_endpoints if raw_topology else None
        ),
        "irc_endpoint_matches_expected": (
            topology.matches_expected_endpoints if topology else None
        ),
        "irc_endpoint_assignment_final": roles_are_final,
        "irc_endpoint_a_role": (
            topology.endpoint_a_role if roles_are_final else None
        ),
        "irc_endpoint_b_role": (
            topology.endpoint_b_role if roles_are_final else None
        ),
        "irc_endpoint_a_relaxation": (
            _endpoint_relaxation(template, endpoint_a) if endpoint_a else None
        ),
        "irc_endpoint_b_relaxation": (
            _endpoint_relaxation(template, endpoint_b) if endpoint_b else None
        ),
    }


def _endpoint_relaxation(template, relaxation) -> IrcEndpointRelaxationResult:
    return IrcEndpointRelaxationResult(
        endpoint=relaxation.endpoint,
        performed=relaxation.performed,
        converged=relaxation.converged,
        initial_energy_ev=relaxation.initial_energy_ev,
        final_energy_ev=relaxation.final_energy_ev,
        final_max_force_ev_per_a=relaxation.final_max_force_ev_per_a,
        initial_structure=structure_from_positions(
            template,
            relaxation.initial_positions_a,
            name=f"UMA IRC {relaxation.endpoint}: finite endpoint",
        ),
        relaxed_endpoint_structure=structure_from_positions(
            template,
            relaxation.final_positions_a,
            name=(
                f"UMA IRC {relaxation.endpoint}: relaxed contact minimum"
                if relaxation.converged
                else f"UMA IRC {relaxation.endpoint}: unconverged relaxation endpoint"
            ),
        ),
        final_structure_label=(
            "relaxed_contact_minimum"
            if relaxation.converged
            else "unconverged_relaxation_endpoint"
        ),
        trajectory=[
            OptimizationFrame(
                step=frame.step,
                energy_ev=frame.energy_ev,
                max_force_ev_per_a=frame.max_force_ev_per_a,
                structure=structure_from_positions(
                    template,
                    frame.positions_a,
                    name=f"UMA IRC {relaxation.endpoint} relaxation step {frame.step}",
                ),
            )
            for frame in relaxation.trajectory
        ],
        error=relaxation.error,
    )
