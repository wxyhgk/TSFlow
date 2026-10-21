from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


TransitionOutcomeKind = Literal[
    "validated_transition_state",
    "screened_first_order_saddle",
    "near_barrierless",
    "unvalidated_candidate",
    "path_search_inconclusive",
]
ReactionDirection = Literal["forward", "reverse"]


@dataclass(frozen=True)
class TransitionOutcome:
    kind: TransitionOutcomeKind
    reason: str
    forward_path_rise_ev: float
    reverse_path_rise_ev: float
    near_barrierless_threshold_ev: float
    near_barrierless_directions: tuple[ReactionDirection, ...]


def classify_transition_outcome(
    energies_ev: list[float] | tuple[float, ...] | np.ndarray,
    *,
    path_converged: bool,
    screening_complete: bool,
    candidate_is_minimum: bool,
    passes_ts_screening: bool,
    irc_required: bool = False,
    irc_validation_passed: bool | None = None,
    near_barrierless_threshold_ev: float,
) -> TransitionOutcome:
    """Classify a completed path without confusing scientific and job status."""
    energies = np.asarray(energies_ev, dtype=float).reshape(-1)
    if len(energies) < 2 or not np.all(np.isfinite(energies)):
        raise ValueError("at least two finite path energies are required")
    if near_barrierless_threshold_ev < 0.0:
        raise ValueError("near-barrierless threshold must be non-negative")

    peak = float(np.max(energies))
    forward_rise = max(0.0, peak - float(energies[0]))
    reverse_rise = max(0.0, peak - float(energies[-1]))
    directions: tuple[ReactionDirection, ...] = tuple(
        direction
        for direction, rise in (
            ("forward", forward_rise),
            ("reverse", reverse_rise),
        )
        if rise <= near_barrierless_threshold_ev
    )

    common = dict(
        forward_path_rise_ev=forward_rise,
        reverse_path_rise_ev=reverse_rise,
        near_barrierless_threshold_ev=near_barrierless_threshold_ev,
        near_barrierless_directions=directions,
    )
    if passes_ts_screening and (
        (irc_required and irc_validation_passed is True)
        or (not irc_required and screening_complete and not candidate_is_minimum)
    ):
        return TransitionOutcome(
            kind="validated_transition_state",
            reason=(
                "The candidate passed the stationary-point, full-Hessian, "
                "imaginary-mode, reaction-coordinate, and bidirectional IRC "
                "connectivity screens."
            ),
            **common,
        )
    if passes_ts_screening and not irc_required:
        return TransitionOutcome(
            kind="screened_first_order_saddle",
            reason=(
                "The candidate passed the stationary-point, full-Hessian, "
                "imaginary-mode, and reaction-coordinate screens, but IRC "
                "connectivity was not requested."
            ),
            **common,
        )
    if passes_ts_screening and irc_required:
        return TransitionOutcome(
            kind="unvalidated_candidate",
            reason=(
                "The candidate passed the first-order-saddle screen, but the "
                "requested bidirectional IRC did not validate connectivity to "
                "the expected endpoints."
            ),
            **common,
        )
    if not path_converged:
        return TransitionOutcome(
            kind="path_search_inconclusive",
            reason=(
                "The reaction path did not converge, so its apparent barrier "
                "cannot be classified."
            ),
            **common,
        )
    if screening_complete and candidate_is_minimum and directions:
        labels = " and ".join(directions)
        return TransitionOutcome(
            kind="near_barrierless",
            reason=(
                "The refined candidate is a full-Hessian minimum rather than a "
                "first-order saddle, and the converged path rise is below the "
                f"configured threshold in the {labels} direction."
            ),
            **common,
        )
    return TransitionOutcome(
        kind="unvalidated_candidate",
        reason=(
            "The candidate did not pass the complete first-order-saddle screen; "
            "the available evidence is insufficient for a near-barrierless label."
        ),
        **common,
    )
