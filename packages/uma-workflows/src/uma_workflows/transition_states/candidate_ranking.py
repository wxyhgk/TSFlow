from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from ase import Atoms

from uma_workflows.transition_states.algorithm_api import PathCandidate
from uma_workflows.transition_states.reaction_coordinates import (
    ReactionCoordinateLike,
)


@dataclass(frozen=True)
class CoordinatedPathScore:
    total_score_ev: float
    barrier_score_ev: float
    coordination_penalty_ev: float
    same_kind_progress_imbalance: float
    peak_image_index: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "total_score_ev": self.total_score_ev,
            "barrier_score_ev": self.barrier_score_ev,
            "coordination_penalty_ev": self.coordination_penalty_ev,
            "same_kind_progress_imbalance": self.same_kind_progress_imbalance,
            "peak_image_index": self.peak_image_index,
        }


def coarse_path_barrier_score(
    positions,
    template: Atoms,
    calculator,
) -> float:
    energies: list[float] = []
    for coordinates in positions:
        image = template.copy()
        image.set_positions(coordinates)
        image.calc = calculator
        energies.append(float(image.get_potential_energy()))
    return max(energies) - min(energies[0], energies[-1])


def reaction_coordinate_progresses(
    positions,
    reactant: Atoms,
    product: Atoms,
    reaction_coordinates: Sequence[ReactionCoordinateLike],
) -> tuple[tuple[str, float], ...]:
    current = np.asarray(positions, dtype=float)
    progresses: list[tuple[str, float]] = []
    for coordinate in reaction_coordinates:
        atom_i = int(coordinate.atom_i)
        atom_j = int(coordinate.atom_j)
        start = float(reactant.get_distance(atom_i, atom_j))
        finish = float(product.get_distance(atom_i, atom_j))
        denominator = finish - start
        if abs(denominator) <= 1.0e-8:
            continue
        distance = float(np.linalg.norm(current[atom_i] - current[atom_j]))
        progress = float(np.clip((distance - start) / denominator, 0.0, 1.0))
        progresses.append((str(coordinate.kind), progress))
    return tuple(progresses)


def same_kind_progress_imbalance(
    positions,
    reactant: Atoms,
    product: Atoms,
    reaction_coordinates: Sequence[ReactionCoordinateLike],
) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for kind, progress in reaction_coordinate_progresses(
        positions,
        reactant,
        product,
        reaction_coordinates,
    ):
        grouped[kind].append(progress)
    return max(
        (
            max(values) - min(values)
            for values in grouped.values()
            if len(values) >= 2
        ),
        default=0.0,
    )


def coordinated_path_score(
    positions,
    template: Atoms,
    calculator,
    *,
    reactant: Atoms,
    product: Atoms,
    reaction_coordinates: Sequence[ReactionCoordinateLike],
    imbalance_tolerance: float = 0.25,
    imbalance_weight_ev: float = 2.0,
) -> CoordinatedPathScore:
    if not 0.0 <= imbalance_tolerance <= 1.0:
        raise ValueError("imbalance_tolerance must be between zero and one")
    if imbalance_weight_ev < 0.0:
        raise ValueError("imbalance_weight_ev must be non-negative")
    energies: list[float] = []
    sampled = np.asarray(positions, dtype=float)
    for coordinates in sampled:
        image = template.copy()
        image.set_positions(coordinates)
        image.calc = calculator
        energies.append(float(image.get_potential_energy()))
    return coordinated_path_geometry_score(
        sampled,
        energies,
        reactant=reactant,
        product=product,
        reaction_coordinates=reaction_coordinates,
        imbalance_tolerance=imbalance_tolerance,
        imbalance_weight_ev=imbalance_weight_ev,
    )


def coordinated_path_geometry_score(
    positions,
    energies_ev: Sequence[float],
    *,
    reactant: Atoms,
    product: Atoms,
    reaction_coordinates: Sequence[ReactionCoordinateLike],
    imbalance_tolerance: float = 0.25,
    imbalance_weight_ev: float = 2.0,
) -> CoordinatedPathScore:
    if not 0.0 <= imbalance_tolerance <= 1.0:
        raise ValueError("imbalance_tolerance must be between zero and one")
    if imbalance_weight_ev < 0.0:
        raise ValueError("imbalance_weight_ev must be non-negative")
    sampled = np.asarray(positions, dtype=float)
    energies = [float(value) for value in energies_ev]
    if not energies or len(energies) != len(sampled):
        raise ValueError("path energies must match the sampled positions")
    peak_index = int(np.argmax(energies))
    barrier = max(energies) - min(energies[0], energies[-1])
    imbalance = same_kind_progress_imbalance(
        sampled[peak_index],
        reactant,
        product,
        reaction_coordinates,
    )
    penalty = imbalance_weight_ev * max(imbalance - imbalance_tolerance, 0.0)
    return CoordinatedPathScore(
        total_score_ev=barrier + penalty,
        barrier_score_ev=barrier,
        coordination_penalty_ev=penalty,
        same_kind_progress_imbalance=imbalance,
        peak_image_index=peak_index,
    )


def select_best_candidate(
    candidates: Sequence[PathCandidate],
) -> tuple[int, PathCandidate]:
    if not candidates:
        raise ValueError("candidate selection requires at least one path")
    if any(candidate.score is None for candidate in candidates):
        raise ValueError("all path candidates must be scored before selection")
    index = min(
        range(len(candidates)),
        key=lambda item: (float(candidates[item].score), item),
    )
    return index, candidates[index]
