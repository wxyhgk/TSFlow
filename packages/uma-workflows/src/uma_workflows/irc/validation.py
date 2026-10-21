from __future__ import annotations

import numpy as np


class IrcValidationError(RuntimeError):
    """Raised when an IRC trajectory fails a physical validity gate."""


def require_mode_gradient_bracket(
    negative_displacement_gradient: float,
    positive_displacement_gradient: float,
    *,
    tolerance: float = 0.0,
) -> tuple[float, float]:
    """Require a negative-curvature mode to bracket a projected-gradient root."""
    negative = float(negative_displacement_gradient)
    positive = float(positive_displacement_gradient)
    if not (negative > tolerance and positive < -tolerance):
        raise IrcValidationError(
            "TS_NOT_STATIONARY_ON_SELECTED_PES: imaginary-mode displacements "
            "do not bracket a projected-gradient root "
            f"(g(-dq)={negative:.6g}, g(+dq)={positive:.6g})"
        )
    return negative, positive


def first_step_projection(
    positions: np.ndarray,
    transition_state_positions: np.ndarray,
    masses: np.ndarray,
    cartesian_mode: np.ndarray,
) -> float:
    displacement = (
        np.asarray(positions, dtype=float)
        - np.asarray(transition_state_positions, dtype=float)
    ).reshape(-1)
    repeated_masses = np.repeat(np.asarray(masses, dtype=float), 3)
    return float(
        np.dot(displacement, repeated_masses * np.asarray(cartesian_mode).reshape(-1))
    )


def require_opposite_first_steps(
    forward_positions: np.ndarray,
    reverse_positions: np.ndarray,
    transition_state_positions: np.ndarray,
    masses: np.ndarray,
    forward_mode: np.ndarray,
    reverse_mode: np.ndarray,
) -> tuple[float, float]:
    forward = first_step_projection(
        forward_positions,
        transition_state_positions,
        masses,
        forward_mode,
    )
    reverse = first_step_projection(
        reverse_positions,
        transition_state_positions,
        masses,
        reverse_mode,
    )
    if forward * reverse >= 0:
        raise IrcValidationError(
            "TS_NOT_STATIONARY_ON_SELECTED_PES: corrected forward/reverse "
            "first steps are on "
            f"the same side (q_forward={forward:.6g}, q_reverse={reverse:.6g})"
        )
    return forward, reverse


def endpoint_internal_distance_delta(
    first_positions: np.ndarray,
    second_positions: np.ndarray,
) -> float:
    first = np.asarray(first_positions, dtype=float)
    second = np.asarray(second_positions, dtype=float)
    first_distances = np.linalg.norm(first[:, None] - first[None, :], axis=-1)
    second_distances = np.linalg.norm(second[:, None] - second[None, :], axis=-1)
    return float(np.max(np.abs(first_distances - second_distances)))


def mass_weighted_arc_coordinates(
    positions: list[np.ndarray],
    masses: np.ndarray,
) -> list[float]:
    if not positions:
        return []
    weights = np.sqrt(np.repeat(np.asarray(masses, dtype=float), 3))
    lengths = [0.0]
    for previous, current in zip(positions, positions[1:]):
        displacement = (
            np.asarray(current, dtype=float) - np.asarray(previous, dtype=float)
        ).reshape(-1)
        lengths.append(lengths[-1] + float(np.linalg.norm(displacement * weights)))
    total = lengths[-1]
    if total <= 1.0e-15:
        return [0.0] * len(lengths)
    return [length / total for length in lengths]
