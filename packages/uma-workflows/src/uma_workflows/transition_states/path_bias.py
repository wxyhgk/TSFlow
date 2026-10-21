from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.data import atomic_numbers, covalent_radii

from uma_workflows.transition_states.reaction_coordinates import (
    ReactionCoordinateLike,
)


@dataclass(frozen=True)
class PathBias:
    symbols: tuple[str, ...]
    repulsion_cutoffs_a: tuple[tuple[int, int, float], ...]
    synchronized_coordinates: tuple[tuple[int, int, float, float], ...]
    scaffold_distances: tuple[tuple[int, int, float, float], ...]
    repulsion_strength_ev_per_a2: float
    repulsion_cutoff_scale: float
    synchronization_strength_ev: float
    coordinate_range_strength_ev: float
    coordinate_schedule_strength_ev: float
    scaffold_strength_ev_per_a2: float
    synchronization_groups: tuple[tuple[int, ...], ...] = ()

    def energy_and_forces(
        self,
        positions: np.ndarray,
        *,
        target_progress: float | None = None,
    ) -> tuple[float, np.ndarray]:
        coordinates = np.asarray(positions, dtype=float)
        forces = np.zeros_like(coordinates)
        energy = self._apply_repulsion(coordinates, forces)
        energy += self._apply_reaction_coordinates(
            coordinates,
            forces,
            target_progress=target_progress,
        )
        energy += self._apply_scaffold(
            coordinates,
            forces,
            target_progress=target_progress,
        )
        return energy, forces

    def _apply_repulsion(self, coordinates: np.ndarray, forces: np.ndarray) -> float:
        if self.repulsion_strength_ev_per_a2 <= 0.0:
            return 0.0
        energy = 0.0
        for first, second, cutoff in self.repulsion_cutoffs_a:
            delta = coordinates[first] - coordinates[second]
            distance = float(np.linalg.norm(delta))
            overlap = cutoff - distance
            if overlap <= 0.0:
                continue
            energy += 0.5 * self.repulsion_strength_ev_per_a2 * overlap**2
            pair_force = (
                self.repulsion_strength_ev_per_a2
                * overlap
                * delta
                / max(distance, 1.0e-8)
            )
            forces[first] += pair_force
            forces[second] -= pair_force
        return energy

    def _apply_reaction_coordinates(
        self,
        coordinates: np.ndarray,
        forces: np.ndarray,
        *,
        target_progress: float | None,
    ) -> float:
        if not self.synchronized_coordinates:
            return 0.0
        progresses: list[float] = []
        derivatives: list[tuple[int, int, float, np.ndarray]] = []
        for first, second, start, finish in self.synchronized_coordinates:
            delta = coordinates[first] - coordinates[second]
            distance = float(np.linalg.norm(delta))
            total_change = finish - start
            if abs(total_change) < 1.0e-8:
                continue
            progresses.append((distance - start) / total_change)
            derivatives.append(
                (first, second, total_change, delta / max(distance, 1.0e-8))
            )
        if not progresses:
            return 0.0
        values = np.asarray(progresses, dtype=float)
        energy = 0.0
        range_deviations = np.where(
            values < 0.0,
            values,
            np.where(values > 1.0, values - 1.0, 0.0),
        )
        energy += self._apply_progress_penalty(
            range_deviations,
            derivatives,
            self.coordinate_range_strength_ev,
            forces,
        )
        if target_progress is not None:
            energy += self._apply_progress_penalty(
                values - target_progress,
                derivatives,
                self.coordinate_schedule_strength_ev,
                forces,
            )
        groups = self.synchronization_groups or (tuple(range(len(values))),)
        for group in groups:
            if len(group) < 2:
                continue
            indices = np.asarray(group, dtype=int)
            group_values = values[indices]
            group_derivatives = [derivatives[index] for index in group]
            energy += self._apply_progress_penalty(
                group_values - float(np.mean(group_values)),
                group_derivatives,
                self.synchronization_strength_ev,
                forces,
            )
        return energy

    @staticmethod
    def _apply_progress_penalty(
        deviations: np.ndarray,
        derivatives: list[tuple[int, int, float, np.ndarray]],
        strength: float,
        forces: np.ndarray,
    ) -> float:
        if strength <= 0.0:
            return 0.0
        for deviation, (first, second, total_change, unit) in zip(
            deviations, derivatives, strict=True
        ):
            pair_force = -(strength * float(deviation) / total_change) * unit
            forces[first] += pair_force
            forces[second] -= pair_force
        return 0.5 * strength * float(np.dot(deviations, deviations))

    def _apply_scaffold(
        self,
        coordinates: np.ndarray,
        forces: np.ndarray,
        *,
        target_progress: float | None,
    ) -> float:
        if target_progress is None or self.scaffold_strength_ev_per_a2 <= 0.0:
            return 0.0
        energy = 0.0
        for first, second, start, finish in self.scaffold_distances:
            delta = coordinates[first] - coordinates[second]
            distance = float(np.linalg.norm(delta))
            target_distance = (1.0 - target_progress) * start + target_progress * finish
            deviation = distance - target_distance
            energy += 0.5 * self.scaffold_strength_ev_per_a2 * deviation**2
            pair_force = (
                -self.scaffold_strength_ev_per_a2
                * deviation
                * delta
                / max(distance, 1.0e-8)
            )
            forces[first] += pair_force
            forces[second] -= pair_force
        return energy


class PathBiasedCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, base_calculator: Calculator, bias: PathBias) -> None:
        super().__init__()
        self._base_calculator = base_calculator
        self._bias = bias

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        if atoms is None:
            raise ValueError("path-biased calculation requires atoms")
        image = atoms.copy()
        image.calc = self._base_calculator
        bias_energy, bias_forces = self._bias.energy_and_forces(atoms.positions)
        self.results = {
            "energy": float(image.get_potential_energy()) + bias_energy,
            "forces": np.asarray(image.get_forces(), dtype=float) + bias_forces,
        }


def dense_path_bias_value_and_gradient(
    bias: PathBias,
    positions: np.ndarray,
    basis_values: np.ndarray,
    path_coordinates: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    sample_forces = np.empty_like(positions)
    energies = np.empty(len(positions), dtype=float)
    for index, sample in enumerate(positions):
        energies[index], sample_forces[index] = bias.energy_and_forces(
            sample,
            target_progress=(
                float(path_coordinates[index])
                if path_coordinates is not None
                else None
            ),
        )
    return (
        float(np.mean(energies)),
        -np.einsum("bi,ias->bas", basis_values, sample_forces) / len(positions),
    )


def build_path_bias(
    reactant: Atoms,
    product: Atoms,
    reaction_coordinates: Sequence[ReactionCoordinateLike],
    *,
    repulsion_strength_ev_per_a2: float,
    repulsion_cutoff_scale: float,
    synchronization_strength_ev: float,
    coordinate_range_strength_ev: float,
    coordinate_schedule_strength_ev: float,
    scaffold_strength_ev_per_a2: float,
    synchronize_same_kind_only: bool = False,
) -> PathBias:
    symbols = tuple(reactant.get_chemical_symbols())
    radii = np.asarray(
        [covalent_radii[atomic_numbers[symbol]] for symbol in symbols],
        dtype=float,
    )
    start = reactant.get_all_distances(mic=False)
    finish = product.get_all_distances(mic=False)
    repulsion_cutoffs: list[tuple[int, int, float]] = []
    for first in range(len(symbols)):
        for second in range(first + 1, len(symbols)):
            bonded_cutoff = 1.25 * float(radii[first] + radii[second])
            bonded_distances = [
                float(distance)
                for distance in (start[first, second], finish[first, second])
                if distance <= bonded_cutoff
            ]
            base_cutoff = (
                min(bonded_distances)
                if bonded_distances
                else float(radii[first] + radii[second])
            )
            repulsion_cutoffs.append(
                (first, second, repulsion_cutoff_scale * base_cutoff)
            )

    included_coordinates = tuple(
        coordinate
        for coordinate in reaction_coordinates
        if coordinate.kind in {"forming", "breaking"}
        and abs(
            finish[coordinate.atom_i, coordinate.atom_j]
            - start[coordinate.atom_i, coordinate.atom_j]
        )
        >= 0.10
    )
    synchronized = tuple(
        (
            coordinate.atom_i,
            coordinate.atom_j,
            float(start[coordinate.atom_i, coordinate.atom_j]),
            float(finish[coordinate.atom_i, coordinate.atom_j]),
        )
        for coordinate in included_coordinates
    )
    synchronization_groups: tuple[tuple[int, ...], ...] = ()
    if synchronize_same_kind_only:
        synchronization_groups = tuple(
            tuple(
                index
                for index, coordinate in enumerate(included_coordinates)
                if coordinate.kind == kind
            )
            for kind in ("forming", "breaking")
            if sum(
                coordinate.kind == kind for coordinate in included_coordinates
            )
            >= 2
        )
    reaction_pairs = {
        (min(item.atom_i, item.atom_j), max(item.atom_i, item.atom_j))
        for item in reaction_coordinates
    }
    adjacency = np.zeros((len(symbols), len(symbols)), dtype=bool)
    for first in range(len(symbols)):
        for second in range(first + 1, len(symbols)):
            cutoff = 1.25 * float(radii[first] + radii[second])
            if start[first, second] <= cutoff or finish[first, second] <= cutoff:
                adjacency[first, second] = adjacency[second, first] = True
    scaffold = tuple(
        (
            first,
            second,
            float(start[first, second]),
            float(finish[first, second]),
        )
        for first in range(len(symbols))
        for second in range(first + 1, len(symbols))
        if (first, second) not in reaction_pairs
        and (
            bool(adjacency[first, second])
            or bool(np.any(adjacency[first] & adjacency[second]))
        )
    )
    return PathBias(
        symbols=symbols,
        repulsion_cutoffs_a=tuple(repulsion_cutoffs),
        synchronized_coordinates=synchronized,
        scaffold_distances=scaffold,
        repulsion_strength_ev_per_a2=repulsion_strength_ev_per_a2,
        repulsion_cutoff_scale=repulsion_cutoff_scale,
        synchronization_strength_ev=synchronization_strength_ev,
        coordinate_range_strength_ev=coordinate_range_strength_ev,
        coordinate_schedule_strength_ev=coordinate_schedule_strength_ev,
        scaffold_strength_ev_per_a2=scaffold_strength_ev_per_a2,
        synchronization_groups=synchronization_groups,
    )
