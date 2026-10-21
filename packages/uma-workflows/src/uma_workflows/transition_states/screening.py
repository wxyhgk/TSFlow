from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from ase import Atoms

from uma_workflows.cancellation import CancellationToken, raise_if_cancelled
from uma_workflows.frequencies import FrequencyAnalysis
from uma_workflows.transition_states.algorithm_api import TransitionStateProblem
from uma_workflows.transition_states.mode_overlap import ModeReactionOverlap
from uma_workflows.transition_states.reaction_analysis import (
    BondChangeAssessment,
    analyze_bond_changes,
)
from uma_workflows.transition_states.frequency_screen import screen_frequencies


@dataclass(frozen=True)
class TransitionStateScreeningConfig:
    stationary_force_threshold_ev_per_a: float
    compute_frequencies: bool
    frequency_active_region_hops: int
    frequency_max_atoms: int
    frequency_delta_a: float
    imaginary_frequency_threshold_cm1: float
    minimum_mode_reaction_overlap: float = 0.20
    force_full_hessian: bool = False
    hessian_method: str = "auto"
    maximum_hessian_asymmetry: float = 0.05


@dataclass(frozen=True)
class TransitionStateScreeningResult:
    energy_ev: float
    forces_ev_per_a: np.ndarray
    max_force_ev_per_a: float
    stationary: bool
    frequency_analysis: FrequencyAnalysis | None
    frequency_scope: str
    mode_reaction_overlap: ModeReactionOverlap | None
    frequency_active_atom_indices: tuple[int, ...]
    bond_changes: tuple[BondChangeAssessment, ...]
    reaction_definition_valid: bool | None
    geometry_progress_consistent: bool | None
    mechanism_mode_consistent: bool | None
    reaction_coordinate_consistent: bool | None
    hessian_quality_valid: bool
    pes_index1_saddle_valid: bool
    local_saddle_valid: bool
    irc_eligible: bool
    passes_screening: bool
    diagnostic_codes: tuple[str, ...]
    failure_code: str | None
    retry_actions: tuple[str, ...]
    warnings: tuple[str, ...]


def screen_transition_state(
    problem: TransitionStateProblem,
    candidate: Atoms,
    *,
    calculator,
    analytic_hessian_calculator=None,
    config: TransitionStateScreeningConfig,
    cancel_token: CancellationToken,
    on_candidate_evaluated: Callable[[float, float], None] | None = None,
    on_frequency_coordinate: Callable[[int, int, float, float], None] | None = None,
) -> TransitionStateScreeningResult:
    raise_if_cancelled(cancel_token)
    atoms = candidate.copy()
    atoms.calc = calculator
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=float)
    max_force = _max_force(forces)
    if on_candidate_evaluated is not None:
        on_candidate_evaluated(energy, max_force)
    raise_if_cancelled(cancel_token)
    stationary = max_force <= config.stationary_force_threshold_ev_per_a
    warnings: list[str] = []
    if not stationary:
        warnings.append(
            "The transition-state candidate is not a stationary point on the "
            f"CPU UMA surface (Fmax={max_force:.4f} eV/A); local saddle-point "
            "refinement is still required."
        )

    frequency = screen_frequencies(
        problem,
        atoms,
        calculator=calculator,
        analytic_hessian_calculator=analytic_hessian_calculator,
        config=config,
        cancel_token=cancel_token,
        on_coordinate=on_frequency_coordinate,
        energy=energy,
        max_force=max_force,
    )
    frequency_analysis = frequency.analysis
    frequency_scope = frequency.scope
    mode_overlap = frequency.overlap
    active_indices = frequency.active_indices
    warnings.extend(frequency.warnings)

    raise_if_cancelled(cancel_token)
    bond_analysis = analyze_bond_changes(
        problem.reactant,
        atoms,
        problem.product,
        expected_coordinates=problem.reaction_coordinates,
    )
    if bond_analysis.reaction_definition_valid is False:
        warnings.append(
            "The requested forming/breaking coordinates are inconsistent with "
            "the supplied reactant and product endpoints."
        )
    if bond_analysis.geometry_progress_consistent is False:
        warnings.append(
            "The saddle candidate is early, late, or endpoint-like along the "
            "expected bond changes. This geometry-progress assessment is "
            "advisory; bidirectional IRC connectivity is the final test."
        )
    hessian_quality_valid = (
        frequency_analysis is not None
        and frequency_analysis.hessian_asymmetry_relative
        <= config.maximum_hessian_asymmetry
    )
    mechanism_mode_consistent = (
        mode_overlap.matches if mode_overlap is not None else None
    )
    pes_index1_saddle_valid = (
        stationary
        and frequency_analysis is not None
        and frequency_scope == "full"
        and len(frequency_analysis.imaginary_frequencies_cm1) == 1
        and hessian_quality_valid
    )
    # Compatibility alias.  Local PES classification must not depend on the
    # user's proposed mechanism; an unexpected index-1 saddle is still a valid
    # candidate for bidirectional IRC discovery.
    local_saddle_valid = pes_index1_saddle_valid
    irc_eligible = pes_index1_saddle_valid
    diagnostics: list[str] = []
    if not stationary:
        diagnostics.append("TS_NOT_STATIONARY")
    if frequency_analysis is None or frequency_scope != "full":
        diagnostics.append("FULL_HESSIAN_REQUIRED")
    elif not frequency_analysis.imaginary_frequencies_cm1:
        diagnostics.append("SADDLE_COLLAPSED_TO_MINIMUM")
    elif len(frequency_analysis.imaginary_frequencies_cm1) > 1:
        diagnostics.append("MULTIPLE_IMAGINARY_MODES")
    if frequency_analysis is not None and not hessian_quality_valid:
        diagnostics.append("HESSIAN_QUALITY_INVALID")
    if problem.reaction_coordinates and (
        mode_overlap is None or not mode_overlap.matches
    ):
        diagnostics.append("MODE_REACTION_MISMATCH")
    if bond_analysis.reaction_definition_valid is False:
        diagnostics.append("REACTION_DEFINITION_MISMATCH")
    if bond_analysis.geometry_progress_consistent is False:
        diagnostics.append("GEOMETRY_PROGRESS_ADVISORY")
    hard_failure_codes = {
        "TS_NOT_STATIONARY",
        "FULL_HESSIAN_REQUIRED",
        "SADDLE_COLLAPSED_TO_MINIMUM",
        "MULTIPLE_IMAGINARY_MODES",
        "HESSIAN_QUALITY_INVALID",
    }
    failure_code = next(
        (code for code in diagnostics if code in hard_failure_codes),
        None,
    )
    return TransitionStateScreeningResult(
        energy_ev=energy,
        forces_ev_per_a=forces,
        max_force_ev_per_a=max_force,
        stationary=stationary,
        frequency_analysis=frequency_analysis,
        frequency_scope=frequency_scope,
        mode_reaction_overlap=mode_overlap,
        frequency_active_atom_indices=active_indices,
        bond_changes=bond_analysis.assessments,
        reaction_definition_valid=bond_analysis.reaction_definition_valid,
        geometry_progress_consistent=(
            bond_analysis.geometry_progress_consistent
        ),
        mechanism_mode_consistent=mechanism_mode_consistent,
        # Deprecated compatibility field.  It historically described
        # endpoint-relative geometry progress; new consumers should use the
        # two explicit fields above instead.
        reaction_coordinate_consistent=(
            bond_analysis.geometry_progress_consistent
        ),
        hessian_quality_valid=hessian_quality_valid,
        pes_index1_saddle_valid=pes_index1_saddle_valid,
        local_saddle_valid=local_saddle_valid,
        irc_eligible=irc_eligible,
        passes_screening=irc_eligible,
        diagnostic_codes=tuple(diagnostics),
        failure_code=failure_code,
        retry_actions=_retry_actions(failure_code),
        warnings=tuple(warnings),
    )


def _max_force(forces: np.ndarray) -> float:
    vectors = np.asarray(forces, dtype=float).reshape(-1, 3)
    return float(np.linalg.norm(vectors, axis=1).max(initial=0.0))


def _retry_actions(failure_code: str | None) -> tuple[str, ...]:
    if failure_code == "SADDLE_COLLAPSED_TO_MINIMUM":
        return (
            "screen_path_negative_curvature",
            "run_collective_coordinate_scan",
        )
    if failure_code in {
        "MULTIPLE_IMAGINARY_MODES",
        "MODE_REACTION_MISMATCH",
    }:
        return ("try_alternative_saddle_seed",)
    if failure_code == "TS_NOT_STATIONARY":
        return ("continue_saddle_refinement",)
    if failure_code == "REACTION_DEFINITION_MISMATCH":
        return ("review_reaction_definition",)
    return ()
