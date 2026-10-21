from __future__ import annotations

from dataclasses import dataclass, replace
from threading import Event

from ase import Atoms

from uma_workflows.cancellation import raise_if_cancelled
from uma_workflows.geometry import set_nonperiodic_graph_cell
from uma_workflows.transition_states import TransitionStateScreeningResult
from uma_workflows.transition_states.outcomes import (
    TransitionOutcome,
    classify_transition_outcome,
)
from uma_workflows.transition_states.reaction_coordinates import (
    ReactionCoordinateLike,
)
from uma_workflows.transition_states.workflow.endpoints import (
    EndpointStageResult,
    prepare_endpoints,
)
from uma_workflows.transition_states.workflow.irc import (
    IrcStageResult,
    follow_and_validate_irc,
)
from uma_workflows.transition_states.workflow.models import (
    Device,
    TransitionSearchConfig,
    WorkflowProgressCallback,
)
from uma_workflows.transition_states.workflow.path import (
    PathStageResult,
    build_supplied_guess_reference_path,
    optimize_reaction_path,
)
from uma_workflows.transition_states.builtin_algorithms import build_builtin_algorithm_registry
from uma_workflows.transition_states.workflow.saddle import (
    SaddleStageResult,
    refine_saddle,
    refine_saddle_guess,
)
from uma_workflows.transition_states.workflow.screening import screen_saddle


@dataclass(frozen=True)
class TransitionSearchWorkflowResult:
    endpoints: EndpointStageResult
    path: PathStageResult
    saddle: SaddleStageResult
    screening: TransitionStateScreeningResult
    warnings: tuple[str, ...]
    irc: IrcStageResult | None = None
    outcome: TransitionOutcome | None = None


def run_transition_state_search(
    reactant: Atoms,
    product: Atoms,
    *,
    reaction_coordinates: tuple[ReactionCoordinateLike, ...],
    transition_state_guess: Atoms | None = None,
    config: TransitionSearchConfig,
    device: Device,
    calculator,
    cpu_calculator,
    analytic_hessian_calculator=None,
    cancel_event: Event,
    progress_callback: WorkflowProgressCallback | None = None,
) -> TransitionSearchWorkflowResult:
    images = (reactant, product) if transition_state_guess is None else (
        reactant,
        product,
        transition_state_guess,
    )
    set_nonperiodic_graph_cell(images)
    endpoints = prepare_endpoints(
        reactant,
        product,
        config=(
            config.endpoint
            if transition_state_guess is None
            else replace(config.endpoint, enabled=False)
        ),
        device=device,
        mps_calculator=calculator,
        cpu_calculator=cpu_calculator,
        cancel_event=cancel_event,
        progress_callback=progress_callback,
    )
    if transition_state_guess is None:
        path = optimize_reaction_path(
            reactant,
            product,
            endpoints,
            reaction_coordinates=reaction_coordinates,
            config=config.path,
            device=device,
            calculator=calculator,
            cpu_calculator=cpu_calculator,
            cancel_event=cancel_event,
            progress_callback=progress_callback,
            algorithm_registry=build_builtin_algorithm_registry(),
        )
    else:
        path = build_supplied_guess_reference_path(
            reactant,
            product,
            transition_state_guess,
            reaction_coordinates=reaction_coordinates,
            config=config.path,
            cpu_calculator=cpu_calculator,
            cancel_event=cancel_event,
        )
    raise_if_cancelled(cancel_event)
    transition = reactant.copy()
    transition.set_positions(
        getattr(path, "highest_positions", transition_state_guess.positions)
        if transition_state_guess is not None
        else path.highest_positions
    )
    if transition_state_guess is None:
        saddle = refine_saddle(
            path,
            transition,
            config=config.saddle,
            device=device,
            mps_calculator=calculator,
            cpu_calculator=cpu_calculator,
            cancel_event=cancel_event,
            progress_callback=progress_callback,
        )
    else:
        saddle = refine_saddle_guess(
            transition_state_guess,
            config=config.saddle,
            cpu_calculator=cpu_calculator,
            analytic_hessian_calculator=analytic_hessian_calculator,
            cancel_event=cancel_event,
            progress_callback=progress_callback,
            hessian_method=config.screening.hessian_method,
            finite_difference_max_atoms=config.screening.max_atoms,
        )
    screening = screen_saddle(
        path,
        saddle,
        config=config.screening,
        saddle_config=config.saddle,
        cpu_calculator=cpu_calculator,
        analytic_hessian_calculator=analytic_hessian_calculator,
        cancel_event=cancel_event,
        progress_callback=progress_callback,
    )
    irc = follow_and_validate_irc(
        path,
        saddle,
        screening,
        config=config.irc,
        cpu_calculator=cpu_calculator,
        cancel_event=cancel_event,
        progress_callback=progress_callback,
    )
    screening_complete = screening.frequency_scope == "full" and screening.frequency_analysis is not None
    candidate_is_minimum = bool(
        screening.frequency_analysis is not None
        and not screening.frequency_analysis.imaginary_frequencies_cm1
    )
    outcome = classify_transition_outcome(
        path.cpu_energies_ev,
        path_converged=path.optimization.converged,
        screening_complete=screening_complete,
        candidate_is_minimum=candidate_is_minimum,
        passes_ts_screening=screening.passes_screening,
        irc_required=config.irc.enabled,
        irc_validation_passed=irc.validation_passed,
        near_barrierless_threshold_ev=config.path.near_barrierless_threshold_ev,
    )
    warnings = (
        endpoints.warnings
        + path.warnings
        + saddle.warnings
        + screening.warnings
        + irc.warnings
    )
    return TransitionSearchWorkflowResult(
        endpoints=endpoints,
        path=path,
        saddle=saddle,
        screening=screening,
        warnings=warnings,
        irc=irc,
        outcome=outcome,
    )
