"""Screening adapter for a refined saddle candidate."""

from __future__ import annotations

from uma_workflows.transition_states.screening import (
    TransitionStateScreeningConfig,
    screen_transition_state,
)
from uma_workflows.transition_states.workflow.models import (
    SaddleConfig,
    ScreeningConfig,
    WorkflowProgress,
    WorkflowProgressCallback,
)
from uma_workflows.transition_states.workflow.path import PathStageResult
from uma_workflows.transition_states.workflow.saddle import SaddleStageResult


def screen_saddle(
    path_result: PathStageResult,
    saddle: SaddleStageResult,
    *,
    config: ScreeningConfig,
    saddle_config: SaddleConfig,
    cpu_calculator,
    analytic_hessian_calculator=None,
    cancel_event,
    progress_callback: WorkflowProgressCallback | None,
):
    def report_validation(energy: float, force: float) -> None:
        if progress_callback is None:
            return
        steps = saddle.refinement.cpu_steps if saddle.refinement else 0
        progress_callback(
            WorkflowProgress(
                stage="cpu_validation",
                step=steps,
                max_steps=saddle_config.max_steps,
                energy_ev=energy,
                max_force_ev_per_a=force,
                device="cpu",
                positions_a=saddle.atoms.positions.copy(),
                name="CPU UMA saddle validation",
            )
        )

    def report_frequency(coordinate: int, total: int, energy: float, force: float) -> None:
        if progress_callback is None:
            return
        progress_callback(
            WorkflowProgress(
                stage="frequencies",
                step=coordinate,
                max_steps=total,
                energy_ev=energy,
                max_force_ev_per_a=force,
                device="cpu",
                positions_a=saddle.atoms.positions.copy(),
                name="CPU UMA numerical Hessian",
            )
        )

    return screen_transition_state(
        path_result.problem,
        saddle.atoms,
        calculator=cpu_calculator,
        analytic_hessian_calculator=analytic_hessian_calculator,
        config=TransitionStateScreeningConfig(
            stationary_force_threshold_ev_per_a=saddle_config.cpu_fmax,
            compute_frequencies=config.compute_frequencies and path_result.has_interior_maximum,
            frequency_active_region_hops=config.active_region_hops,
            frequency_max_atoms=config.max_atoms,
            frequency_delta_a=config.delta_a,
            imaginary_frequency_threshold_cm1=config.imaginary_threshold_cm1,
            force_full_hessian=config.force_full_hessian,
            hessian_method=config.hessian_method,
            maximum_hessian_asymmetry=config.maximum_hessian_asymmetry,
        ),
        cancel_token=cancel_event,
        on_candidate_evaluated=report_validation,
        on_frequency_coordinate=report_frequency,
    )
