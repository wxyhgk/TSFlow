from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from ase import Atoms

from uma_workflows.transition_states.reaction_coordinates import (
    ReactionCoordinateLike,
)


@dataclass(frozen=True)
class ModeCoordinateProjection:
    atom_i: int
    atom_j: int
    kind: str
    bond_length_derivative: float
    expected_direction: int
    direction_matches: bool


@dataclass(frozen=True)
class ModeReactionOverlap:
    score: float
    directional_consistency: float
    orientation: int
    matches: bool
    projections: tuple[ModeCoordinateProjection, ...]


def analyze_mode_reaction_overlap(
    atoms: Atoms,
    mode: np.ndarray,
    reaction_coordinates: Sequence[ReactionCoordinateLike],
    *,
    minimum_score: float = 0.20,
) -> ModeReactionOverlap:
    """Compare a Cartesian normal mode with requested bond-length changes."""
    if not reaction_coordinates:
        raise ValueError("mode overlap requires at least one reaction coordinate")
    displacements = np.asarray(mode, dtype=float)
    if displacements.shape != (len(atoms), 3):
        raise ValueError("normal mode shape must match the molecular structure")
    mode_norm = float(np.linalg.norm(displacements))
    if mode_norm <= 1.0e-12:
        raise ValueError("normal mode has zero norm")

    gradient = np.zeros_like(displacements)
    raw_derivatives: list[float] = []
    expected_directions: list[int] = []
    for coordinate in reaction_coordinates:
        first = coordinate.atom_i
        second = coordinate.atom_j
        delta = atoms.positions[second] - atoms.positions[first]
        distance = float(np.linalg.norm(delta))
        if distance <= 1.0e-12:
            raise ValueError("reaction-coordinate bond has zero length")
        unit = delta / distance
        derivative = float(
            np.dot(unit, displacements[second] - displacements[first])
        )
        expected = 1 if coordinate.kind == "breaking" else -1
        gradient[first] -= expected * unit
        gradient[second] += expected * unit
        raw_derivatives.append(derivative)
        expected_directions.append(expected)

    gradient_norm = float(np.linalg.norm(gradient))
    score = (
        abs(float(np.sum(displacements * gradient)))
        / (mode_norm * gradient_norm)
        if gradient_norm > 1.0e-12
        else 0.0
    )
    signed_sum = float(
        np.dot(
            np.asarray(raw_derivatives),
            np.asarray(expected_directions),
        )
    )
    orientation = 1 if signed_sum >= 0.0 else -1
    oriented = orientation * np.asarray(raw_derivatives)
    expected = np.asarray(expected_directions, dtype=float)
    directional_consistency = float(
        np.sum(np.abs(oriented) * ((oriented * expected) > 0.0))
        / max(float(np.sum(np.abs(oriented))), 1.0e-12)
    )
    projections = tuple(
        ModeCoordinateProjection(
            atom_i=coordinate.atom_i,
            atom_j=coordinate.atom_j,
            kind=coordinate.kind,
            bond_length_derivative=float(oriented[index]),
            expected_direction=expected_directions[index],
            direction_matches=bool(
                oriented[index] * expected_directions[index] > 0.0
            ),
        )
        for index, coordinate in enumerate(reaction_coordinates)
    )
    return ModeReactionOverlap(
        score=score,
        directional_consistency=directional_consistency,
        orientation=orientation,
        matches=score >= minimum_score and directional_consistency >= 0.75,
        projections=projections,
    )
