from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from uma_workflows.cancellation import CancellationToken
from uma_workflows.optimization import OptimizationStage, optimize_minimum_stages


@dataclass(frozen=True)
class CollectiveCoordinateTerm:
    atom_i: int
    atom_j: int
    coefficient: float


@dataclass(frozen=True)
class CollectiveScanPoint:
    target_value_a: float
    coordinate_value_a: float
    energy_ev: float
    derivative_ev_per_a: float
    orthogonal_fmax_ev_per_a: float
    converged: bool
    steps: int
    positions_a: np.ndarray


@dataclass(frozen=True)
class CollectiveScanResult:
    terms: tuple[CollectiveCoordinateTerm, ...]
    points: tuple[CollectiveScanPoint, ...]
    derivative_brackets: tuple[tuple[int, int], ...]
    interior_maxima: tuple[int, ...]

    @property
    def selected_maximum(self) -> CollectiveScanPoint | None:
        if not self.interior_maxima:
            return None
        return max(
            (self.points[index] for index in self.interior_maxima),
            key=lambda point: point.energy_ev,
        )


CollectiveScanCallback = Callable[[int, int, CollectiveScanPoint], None]


class HarmonicCollectiveCoordinateCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        calculator,
        terms: Sequence[CollectiveCoordinateTerm],
        target_value_a: float,
        strength_ev_per_a2: float,
    ) -> None:
        super().__init__()
        self.calculator = calculator
        self.terms = tuple(terms)
        self.target_value_a = float(target_value_a)
        self.strength_ev_per_a2 = float(strength_ev_per_a2)

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        if atoms is None:
            raise ValueError("collective-coordinate restraint requires atoms")
        image = atoms.copy()
        image.set_constraint()
        image.calc = self.calculator
        energy = float(image.get_potential_energy())
        forces = np.asarray(image.get_forces(), dtype=float)
        value, gradient = collective_coordinate_value_and_gradient(
            image.positions,
            self.terms,
        )
        delta = value - self.target_value_a
        energy += 0.5 * self.strength_ev_per_a2 * delta**2
        forces -= self.strength_ev_per_a2 * delta * gradient
        self.results = {"energy": energy, "forces": forces}


def relaxed_collective_coordinate_scan(
    seed: Atoms,
    *,
    terms: Sequence[CollectiveCoordinateTerm],
    target_values_a: Sequence[float],
    calculator,
    fmax: float,
    max_steps: int,
    cancel_event: CancellationToken,
    restraint_strength_ev_per_a2: float = 200.0,
    on_point: CollectiveScanCallback | None = None,
) -> CollectiveScanResult:
    """Relax all degrees of freedom except one restrained linear coordinate."""
    normalized_terms = _validate_terms(terms, len(seed))
    targets = tuple(sorted({float(value) for value in target_values_a}))
    if not targets or not all(np.isfinite(targets)):
        raise ValueError("collective scan requires finite target values")
    if fmax <= 0.0 or max_steps < 1:
        raise ValueError("collective scan optimization budget is invalid")
    if restraint_strength_ev_per_a2 <= 0.0:
        raise ValueError("collective scan restraint strength must be positive")

    seed_value, _ = collective_coordinate_value_and_gradient(
        seed.positions,
        normalized_terms,
    )
    split = min(
        range(len(targets)),
        key=lambda index: abs(targets[index] - seed_value),
    )
    branches = (targets[split:], tuple(reversed(targets[:split])))
    indexed: dict[float, CollectiveScanPoint] = {}
    completed = 0
    for branch in branches:
        atoms = seed.copy()
        for target in branch:
            point = _relax_collective_scan_point(
                atoms,
                terms=normalized_terms,
                target=target,
                calculator=calculator,
                fmax=fmax,
                max_steps=max_steps,
                restraint_strength=restraint_strength_ev_per_a2,
                cancel_event=cancel_event,
            )
            indexed[target] = point
            completed += 1
            if on_point is not None:
                on_point(completed, len(targets), point)

    points = tuple(indexed[target] for target in targets)
    brackets = tuple(
        (index, index + 1)
        for index in range(len(points) - 1)
        if _brackets_zero(
            points[index].derivative_ev_per_a,
            points[index + 1].derivative_ev_per_a,
        )
    )
    maxima = tuple(
        index
        for index in range(1, len(points) - 1)
        if points[index - 1].converged
        and points[index].converged
        and points[index + 1].converged
        and points[index].energy_ev > points[index - 1].energy_ev
        and points[index].energy_ev > points[index + 1].energy_ev
    )
    return CollectiveScanResult(normalized_terms, points, brackets, maxima)


def collective_coordinate_value_and_gradient(
    positions_a: np.ndarray,
    terms: Sequence[CollectiveCoordinateTerm],
) -> tuple[float, np.ndarray]:
    positions = np.asarray(positions_a, dtype=float)
    gradient = np.zeros_like(positions)
    value = 0.0
    for term in terms:
        vector = positions[term.atom_j] - positions[term.atom_i]
        distance = float(np.linalg.norm(vector))
        if distance <= 1.0e-12:
            raise ValueError("collective-coordinate term has zero distance")
        unit = vector / distance
        value += term.coefficient * distance
        gradient[term.atom_i] -= term.coefficient * unit
        gradient[term.atom_j] += term.coefficient * unit
    norm2 = float(np.vdot(gradient, gradient))
    if norm2 <= 1.0e-16:
        raise ValueError("collective-coordinate gradient is zero")
    return float(value), gradient


def collective_force_components(
    positions_a: np.ndarray,
    forces_ev_per_a: np.ndarray,
    terms: Sequence[CollectiveCoordinateTerm],
) -> tuple[float, float, float]:
    value, gradient = collective_coordinate_value_and_gradient(
        positions_a,
        terms,
    )
    forces = np.asarray(forces_ev_per_a, dtype=float)
    norm2 = float(np.vdot(gradient, gradient))
    force_projection = float(np.vdot(forces, gradient) / norm2)
    derivative = -force_projection
    orthogonal = forces - force_projection * gradient
    return (
        value,
        derivative,
        float(np.linalg.norm(orthogonal, axis=1).max(initial=0.0)),
    )


def _relax_collective_scan_point(
    atoms: Atoms,
    *,
    terms: tuple[CollectiveCoordinateTerm, ...],
    target: float,
    calculator,
    fmax: float,
    max_steps: int,
    restraint_strength: float,
    cancel_event: CancellationToken,
) -> CollectiveScanPoint:
    restrained = HarmonicCollectiveCoordinateCalculator(
        calculator,
        terms,
        target,
        restraint_strength,
    )
    result = optimize_minimum_stages(
        atoms,
        (OptimizationStage(
            name="relaxed_collective_coordinate_scan",
            calculator=restrained,
            fmax=fmax,
            max_steps=max_steps,
            synchronize_after=False,
        ),),
        cancel_event=cancel_event,
    )
    atoms.calc = calculator
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=float)
    value, derivative, orthogonal_fmax = collective_force_components(
        atoms.positions,
        forces,
        terms,
    )
    atoms.calc = None
    return CollectiveScanPoint(
        target_value_a=target,
        coordinate_value_a=value,
        energy_ev=energy,
        derivative_ev_per_a=derivative,
        orthogonal_fmax_ev_per_a=orthogonal_fmax,
        converged=result.converged,
        steps=sum(stage.steps for stage in result.stage_results),
        positions_a=atoms.positions.copy(),
    )


def _validate_terms(
    terms: Sequence[CollectiveCoordinateTerm],
    atom_count: int,
) -> tuple[CollectiveCoordinateTerm, ...]:
    normalized = tuple(terms)
    if not normalized:
        raise ValueError("collective coordinate requires at least one term")
    seen: set[tuple[int, int]] = set()
    for term in normalized:
        if term.atom_i == term.atom_j:
            raise ValueError("collective-coordinate atoms must be different")
        if min(term.atom_i, term.atom_j) < 0 or max(term.atom_i, term.atom_j) >= atom_count:
            raise IndexError("collective-coordinate atom index is out of range")
        if not np.isfinite(term.coefficient) or abs(term.coefficient) <= 1.0e-12:
            raise ValueError("collective-coordinate coefficient must be nonzero")
        pair = tuple(sorted((term.atom_i, term.atom_j)))
        if pair in seen:
            raise ValueError("collective coordinate contains a duplicate atom pair")
        seen.add(pair)
    return normalized


def _brackets_zero(first: float, second: float) -> bool:
    return first == 0.0 or second == 0.0 or np.signbit(first) != np.signbit(second)
