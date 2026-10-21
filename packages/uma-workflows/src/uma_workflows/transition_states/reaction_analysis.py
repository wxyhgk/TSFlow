from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii

from uma_workflows.transition_states.reaction_coordinates import (
    ReactionCoordinateLike,
)


@dataclass(frozen=True)
class BondChangeAssessment:
    atom_i: int
    atom_j: int
    kind: str
    reactant_distance_a: float
    transition_state_distance_a: float
    product_distance_a: float
    path_fraction: float


@dataclass(frozen=True)
class BondChangeAnalysis:
    assessments: tuple[BondChangeAssessment, ...]
    reaction_definition_valid: bool | None
    geometry_progress_consistent: bool | None


def analyze_bond_changes(
    reactant: Atoms,
    transition: Atoms,
    product: Atoms,
    expected_coordinates: Sequence[ReactionCoordinateLike] | None = None,
) -> BondChangeAnalysis:
    """Separate endpoint-definition validity from candidate geometry progress."""
    assessments = _collect_bond_change_assessments(
        reactant,
        transition,
        product,
        expected_coordinates,
    )
    if not assessments:
        return BondChangeAnalysis((), None, None)

    definition_valid: bool | None = True if expected_coordinates else None
    geometry_consistent = True
    for item in assessments:
        if item.kind == "coordination":
            continue
        total_change = item.product_distance_a - item.reactant_distance_a
        if abs(total_change) < 0.10:
            if definition_valid is not None:
                definition_valid = False
            geometry_consistent = False
            continue
        direction = 1.0 if total_change > 0.0 else -1.0
        if item.kind == "forming" and direction > 0.0:
            definition_valid = False
        elif item.kind == "breaking" and direction < 0.0:
            definition_valid = False

        progress_from_reactant = direction * (
            item.transition_state_distance_a - item.reactant_distance_a
        )
        progress_from_product = direction * (
            item.product_distance_a - item.transition_state_distance_a
        )
        minimum_progress = min(0.075, 0.15 * abs(total_change))
        if (
            progress_from_reactant < minimum_progress
            or progress_from_product < minimum_progress
        ):
            geometry_consistent = False

    return BondChangeAnalysis(
        assessments=tuple(assessments),
        reaction_definition_valid=definition_valid,
        geometry_progress_consistent=geometry_consistent,
    )


def assess_bond_changes(
    reactant: Atoms,
    transition: Atoms,
    product: Atoms,
    expected_coordinates: Sequence[ReactionCoordinateLike] | None = None,
) -> tuple[list[BondChangeAssessment], bool | None]:
    """Legacy adapter returning the former combined geometry assessment."""
    analysis = analyze_bond_changes(
        reactant,
        transition,
        product,
        expected_coordinates,
    )
    return list(analysis.assessments), analysis.geometry_progress_consistent


def _collect_bond_change_assessments(
    reactant: Atoms,
    transition: Atoms,
    product: Atoms,
    expected_coordinates: Sequence[ReactionCoordinateLike] | None = None,
) -> list[BondChangeAssessment]:
    symbols = reactant.get_chemical_symbols()
    reactant_distances = reactant.get_all_distances(mic=False)
    transition_distances = transition.get_all_distances(mic=False)
    product_distances = product.get_all_distances(mic=False)
    selected_pairs = (
        [(item.atom_i, item.atom_j, item.kind) for item in expected_coordinates]
        if expected_coordinates
        else [
            (first, second, "inferred")
            for first in range(len(reactant))
            for second in range(first + 1, len(reactant))
        ]
    )
    assessments: list[BondChangeAssessment] = []
    for first, second, kind in selected_pairs:
        start = float(reactant_distances[first, second])
        finish = float(product_distances[first, second])
        change = finish - start
        covalent_cutoff = 1.35 * float(
            covalent_radii[atomic_numbers[symbols[first]]]
            + covalent_radii[atomic_numbers[symbols[second]]]
        )
        if (
            kind == "inferred"
            and (abs(change) < 0.35 or min(start, finish) > covalent_cutoff)
        ):
            continue
        saddle = float(transition_distances[first, second])
        assessments.append(
            BondChangeAssessment(
                atom_i=first,
                atom_j=second,
                kind=kind,
                reactant_distance_a=start,
                transition_state_distance_a=saddle,
                product_distance_a=finish,
                path_fraction=(
                    (saddle - start) / change if abs(change) > 1.0e-8 else 0.0
                ),
            )
        )
    return assessments


def reaction_active_region(
    reactant: Atoms,
    product: Atoms,
    *,
    hops: int,
    expected_coordinates: Sequence[ReactionCoordinateLike] | None = None,
) -> set[int]:
    """Return atoms within covalent-graph hops of changing bonds."""
    symbols = reactant.get_chemical_symbols()
    radii = np.asarray(
        [covalent_radii[atomic_numbers[symbol]] for symbol in symbols],
        dtype=float,
    )
    start = reactant.get_all_distances(mic=False)
    finish = product.get_all_distances(mic=False)
    covalent = radii[:, None] + radii[None, :]
    adjacency = (start <= 1.25 * covalent) | (finish <= 1.25 * covalent)
    np.fill_diagonal(adjacency, False)
    changing = (np.abs(finish - start) >= 0.35) & (
        np.minimum(start, finish) <= 1.35 * covalent
    )
    if expected_coordinates:
        active = {
            index
            for coordinate in expected_coordinates
            for index in (coordinate.atom_i, coordinate.atom_j)
        }
    else:
        rows, columns = np.nonzero(np.triu(changing, k=1))
        active = set(int(index) for index in np.concatenate((rows, columns)))
    frontier = set(active)
    for _ in range(hops):
        neighbors: set[int] = set()
        for index in frontier:
            neighbors.update(int(value) for value in np.flatnonzero(adjacency[index]))
        frontier = neighbors - active
        active.update(neighbors)
    return active
