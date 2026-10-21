"""CPU supplied-saddle refinement, full-Hessian screening and bidirectional IRC.

This path intentionally does not initialize or optimize an endpoint-to-endpoint
DMF path. Failed scientific checks return evidence, never a validated TS label.
"""

from dataclasses import dataclass, field
import numpy as np
from ase import Atoms
from uma_workflows.cancellation import raise_if_cancelled
from uma_workflows.frequencies.workflow import numerical_frequency_analysis
from uma_workflows.optimization.geometric_saddle import refine_geometric_saddle
from uma_workflows.irc.workflow import BidirectionalIrcConfig, run_bidirectional_irc
from uma_workflows.irc.validation import IrcValidationError
from uma_workflows.transition_states.algorithm_api import TransitionStateProblem
from uma_workflows.transition_states.screening import (
    TransitionStateScreeningConfig,
    screen_transition_state,
)


@dataclass(frozen=True)
class SuppliedGuessConfig:
    fmax: float = 0.05
    max_steps: int = 100
    delta_a: float = 0.01
    imaginary_threshold_cm1: float = 100.0
    maximum_hessian_asymmetry: float = 0.05
    finite_difference_max_atoms: int = 30
    hessian_method: str = "auto"
    follow_irc: bool = True
    irc: BidirectionalIrcConfig = field(default_factory=BidirectionalIrcConfig)


@dataclass(frozen=True)
class SuppliedGuessResult:
    atoms: Atoms
    refinement: object
    screening: object
    irc: object | None
    irc_error: str | None
    irc_performed: bool
    validated: bool
    warnings: tuple[str, ...]


def run_supplied_guess(
    reactant,
    product,
    guess,
    *,
    calculator,
    config,
    cancel_event,
    reaction_coordinates=(),
    analytic_calculator=None,
    on_saddle_frame=None,
    on_irc_frame=None,
):
    symbols = guess.get_chemical_symbols()
    for image in (reactant, product, guess):
        if image.get_chemical_symbols() != symbols:
            raise ValueError("Endpoints and TS guess must have identical atom ordering")
        if image.pbc.any() or not np.isfinite(image.positions).all():
            raise ValueError(
                "Only finite nonperiodic molecular structures are supported"
            )
        if any(image.info.get(k) != guess.info.get(k) for k in ("charge", "spin")):
            raise ValueError(
                "Endpoints and TS guess must have matching charge and spin"
            )
    if (
        config.max_steps < 1
        or config.fmax <= 0
        or config.finite_difference_max_atoms < 1
    ):
        raise ValueError("Invalid refinement limits")
    raise_if_cancelled(cancel_event)
    atoms = guess.copy()
    atoms.set_constraint()
    can_finite_difference = len(atoms) <= config.finite_difference_max_atoms
    initial = numerical_frequency_analysis(
        atoms,
        calculator=calculator,
        analytic_calculator=analytic_calculator,
        hessian_method=config.hessian_method,
        delta_a=config.delta_a,
        imaginary_threshold_cm1=config.imaginary_threshold_cm1,
        cancel_event=cancel_event,
        allow_finite_difference_fallback=can_finite_difference,
    )
    refinement = refine_geometric_saddle(
        atoms,
        initial.hessian_ev_per_a2,
        calculator=calculator,
        fmax=config.fmax,
        max_steps=config.max_steps,
        cancel_event=cancel_event,
        on_frame=on_saddle_frame,
    )
    screening = screen_transition_state(
        TransitionStateProblem(reactant, product, tuple(reaction_coordinates)),
        atoms,
        calculator=calculator,
        analytic_hessian_calculator=analytic_calculator,
        config=TransitionStateScreeningConfig(
            stationary_force_threshold_ev_per_a=config.fmax,
            compute_frequencies=True,
            frequency_active_region_hops=2,
            frequency_max_atoms=config.finite_difference_max_atoms,
            frequency_delta_a=config.delta_a,
            imaginary_frequency_threshold_cm1=config.imaginary_threshold_cm1,
            force_full_hessian=True,
            hessian_method=config.hessian_method,
            maximum_hessian_asymmetry=config.maximum_hessian_asymmetry,
        ),
        cancel_token=cancel_event,
    )
    warnings = list(screening.warnings)
    if initial.hessian_fallback_reason:
        warnings.append(
            "Initial Hessian fell back to finite differences: "
            + initial.hessian_fallback_reason
        )
    irc = None
    error = None
    performed = False
    if config.follow_irc and screening.irc_eligible:
        performed = True
        frequency = screening.frequency_analysis
        try:
            irc = run_bidirectional_irc(
                atoms,
                frequency.hessian_ev_per_a2,
                validated_cartesian_mode=frequency.primary_imaginary_mode,
                calculator=calculator,
                expected_reactant=reactant,
                expected_product=product,
                config=config.irc,
                cancel_event=cancel_event,
                on_frame=on_irc_frame,
            )
        except IrcValidationError as exc:
            error = str(exc)
            warnings.append(error)
    elif config.follow_irc:
        error = screening.failure_code or "FULL_HESSIAN_SCREEN_REQUIRED"
        warnings.append("IRC skipped: " + error)
    validated = bool(
        irc is not None
        and irc.path_complete
        and irc.endpoint_a_relaxation.converged
        and irc.endpoint_b_relaxation.converged
        and irc.endpoint_topology.matches_expected_endpoints
        and irc.relaxed_endpoint_distance_delta_a
        >= config.irc.endpoint_distance_threshold_a
        and screening.pes_index1_saddle_valid
    )
    if irc is not None:
        warnings.extend(irc.warnings)
        if not validated:
            error = next(iter(irc.warnings), "IRC_VALIDATION_INCOMPLETE")
    return SuppliedGuessResult(
        atoms, refinement, screening, irc, error, performed, validated, tuple(warnings)
    )
