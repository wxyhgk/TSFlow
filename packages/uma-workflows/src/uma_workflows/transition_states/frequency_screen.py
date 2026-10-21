from __future__ import annotations

from dataclasses import dataclass

from ase import Atoms

from uma_workflows.cancellation import CancellationToken
from uma_workflows.frequencies import FrequencyAnalysis, numerical_frequency_analysis
from uma_workflows.transition_states.algorithm_api import TransitionStateProblem
from uma_workflows.transition_states.mode_overlap import (
    ModeReactionOverlap,
    analyze_mode_reaction_overlap,
)
from uma_workflows.transition_states.reaction_analysis import reaction_active_region


@dataclass(frozen=True)
class FrequencyScreen:
    analysis: FrequencyAnalysis | None
    scope: str
    overlap: ModeReactionOverlap | None
    active_indices: tuple[int, ...]
    warnings: tuple[str, ...]


def screen_frequencies(
    problem: TransitionStateProblem,
    atoms: Atoms,
    *,
    calculator,
    analytic_hessian_calculator,
    config,
    cancel_token: CancellationToken,
    on_coordinate,
    energy: float,
    max_force: float,
) -> FrequencyScreen:
    if not config.compute_frequencies:
        return FrequencyScreen(None, "none", None, (), ())
    active = _active_indices(problem, atoms, config)
    exceeds_finite_difference_limit = len(active) > config.frequency_max_atoms
    analytic_available = analytic_hessian_calculator is not None
    if exceeds_finite_difference_limit and (
        config.hessian_method == "finite_difference" or not analytic_available
    ):
        warning = (
            "Finite-difference Hessian was skipped because the selected active region has "
            f"{len(active)} atoms, exceeding "
            f"frequency_max_atoms={config.frequency_max_atoms}."
        )
        return FrequencyScreen(None, "none", None, active, (warning,))
    try:
        analysis = numerical_frequency_analysis(
            atoms,
            calculator=calculator,
            analytic_calculator=analytic_hessian_calculator,
            hessian_method=config.hessian_method,
            delta_a=config.frequency_delta_a,
            imaginary_threshold_cm1=config.imaginary_frequency_threshold_cm1,
            cancel_event=cancel_token,
            active_indices=list(active),
            allow_finite_difference_fallback=(
                not exceeds_finite_difference_limit
            ),
            on_coordinate=(
                (
                    lambda coordinate, total: on_coordinate(
                        coordinate, total, energy, max_force
                    )
                )
                if on_coordinate is not None
                else None
            ),
        )
    except Exception as exc:
        if config.hessian_method == "analytic" or exceeds_finite_difference_limit:
            warning = (
                "Hessian analysis was unavailable without an allowed numerical "
                f"fallback: {type(exc).__name__}: {exc}"
            )
            return FrequencyScreen(None, "none", None, active, (warning,))
        raise
    scope = "full" if len(active) == len(atoms) else "partial"
    warnings = _frequency_warnings(analysis, scope, config)
    overlap = None
    if (
        len(analysis.imaginary_frequencies_cm1) == 1
        and problem.reaction_coordinates
        and analysis.primary_imaginary_mode is not None
    ):
        overlap = analyze_mode_reaction_overlap(
            atoms,
            analysis.primary_imaginary_mode,
            problem.reaction_coordinates,
            minimum_score=config.minimum_mode_reaction_overlap,
        )
        if not overlap.matches:
            warnings.append(
                "The primary imaginary mode has insufficient overlap with the "
                "requested forming/breaking coordinates "
                f"(score={overlap.score:.3f})."
            )
    return FrequencyScreen(analysis, scope, overlap, active, tuple(warnings))


def _active_indices(problem, atoms, config) -> tuple[int, ...]:
    if problem.reaction_coordinates and not config.force_full_hessian:
        return tuple(
            sorted(
                reaction_active_region(
                    problem.reactant,
                    problem.product,
                    hops=config.frequency_active_region_hops,
                    expected_coordinates=problem.reaction_coordinates,
                )
            )
        )
    return tuple(range(len(atoms)))


def _frequency_warnings(analysis, scope: str, config) -> list[str]:
    warnings = []
    if scope == "partial":
        warnings.append(
            "A partial Hessian is a provisional screen only and cannot confirm "
            "the full system as a first-order saddle."
        )
    if len(analysis.imaginary_frequencies_cm1) != 1:
        warnings.append(
            "CPU UMA Hessian did not contain exactly one significant imaginary mode."
        )
    if analysis.hessian_asymmetry_relative > config.maximum_hessian_asymmetry:
        warnings.append(
            "The raw Hessian force Jacobian is insufficiently symmetric "
            f"(relative residual={analysis.hessian_asymmetry_relative:.3e}); "
            "its normal modes are not accepted as definitive."
        )
    if analysis.hessian_fallback_reason:
        warnings.append(
            "Analytic Hessian was unavailable and finite differences were used "
            f"instead: {analysis.hessian_fallback_reason}"
        )
    return warnings
