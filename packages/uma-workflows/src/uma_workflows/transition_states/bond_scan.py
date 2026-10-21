from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from ase import Atoms

from uma_workflows.cancellation import CancellationToken
from uma_workflows.endpoints.restraints import (
    HarmonicBondRestraintCalculator,
)
from uma_workflows.optimization import (
    OptimizationStage,
    optimize_minimum_stages,
)


@dataclass(frozen=True)
class BondScanPoint:
    target_distance_a: float
    distance_a: float
    energy_ev: float
    derivative_ev_per_a: float
    orthogonal_fmax_ev_per_a: float
    converged: bool
    steps: int
    positions_a: np.ndarray


@dataclass(frozen=True)
class BondScanResult:
    points: tuple[BondScanPoint, ...]
    derivative_brackets: tuple[tuple[int, int], ...]

    @property
    def best_point(self) -> BondScanPoint:
        return min(
            self.points,
            key=lambda point: (
                abs(point.derivative_ev_per_a),
                point.orthogonal_fmax_ev_per_a,
            ),
        )


BondScanCallback = Callable[[int, int, BondScanPoint], None]


def relaxed_bond_scan(
    seed: Atoms,
    *,
    atom_i: int,
    atom_j: int,
    distances_a: Sequence[float],
    calculator,
    fmax: float,
    max_steps: int,
    cancel_event: CancellationToken,
    restraint_strength_ev_per_a2: float = 200.0,
    on_point: BondScanCallback | None = None,
) -> BondScanResult:
    """Relax all coordinates except one restrained bond distance.

    Every reported energy and force is recomputed with the *unrestrained*
    calculator.  Distances are traversed outwards from the seed in both
    directions so each branch has an independent warm start.
    """
    targets = tuple(sorted({float(value) for value in distances_a}))
    if not targets:
        raise ValueError("bond scan requires at least one target distance")
    if atom_i == atom_j:
        raise ValueError("bond scan atoms must be different")
    if min(atom_i, atom_j) < 0 or max(atom_i, atom_j) >= len(seed):
        raise IndexError("bond scan atom index is out of range")

    seed_distance = float(seed.get_distance(atom_i, atom_j))
    split = min(range(len(targets)), key=lambda index: abs(targets[index] - seed_distance))
    branches = (targets[split:], tuple(reversed(targets[:split])))
    indexed: dict[float, BondScanPoint] = {}
    completed = 0
    for branch in branches:
        atoms = seed.copy()
        for target in branch:
            point = _relax_scan_point(
                atoms,
                atom_i=atom_i,
                atom_j=atom_j,
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
    return BondScanResult(points=points, derivative_brackets=brackets)


def _relax_scan_point(
    atoms: Atoms,
    *,
    atom_i: int,
    atom_j: int,
    target: float,
    calculator,
    fmax: float,
    max_steps: int,
    restraint_strength: float,
    cancel_event: CancellationToken,
) -> BondScanPoint:
    restrained = HarmonicBondRestraintCalculator(
        calculator,
        ((atom_i, atom_j, target),),
        strength_ev_per_a2=restraint_strength,
    )
    result = optimize_minimum_stages(
        atoms,
        (
            OptimizationStage(
                name="relaxed_bond_scan",
                calculator=restrained,
                fmax=fmax,
                max_steps=max_steps,
                synchronize_after=False,
            ),
        ),
        cancel_event=cancel_event,
    )
    atoms.calc = calculator
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=float)
    distance, derivative, orthogonal_fmax = bond_force_components(
        atoms.positions,
        forces,
        atom_i,
        atom_j,
    )
    atoms.calc = None
    return BondScanPoint(
        target_distance_a=target,
        distance_a=distance,
        energy_ev=energy,
        derivative_ev_per_a=derivative,
        orthogonal_fmax_ev_per_a=orthogonal_fmax,
        converged=result.converged,
        steps=sum(stage.steps for stage in result.stage_results),
        positions_a=atoms.positions.copy(),
    )


def bond_force_components(
    positions_a: np.ndarray,
    forces_ev_per_a: np.ndarray,
    atom_i: int,
    atom_j: int,
) -> tuple[float, float, float]:
    vector = positions_a[atom_j] - positions_a[atom_i]
    distance = float(np.linalg.norm(vector))
    if distance <= 1.0e-12:
        raise ValueError("bond scan coordinate has zero length")
    unit = vector / distance
    coordinate = np.zeros_like(forces_ev_per_a)
    coordinate[atom_i] = -unit
    coordinate[atom_j] = unit
    projection = float(np.sum(forces_ev_per_a * coordinate) / 2.0)
    orthogonal = forces_ev_per_a - projection * coordinate
    derivative = -projection
    return distance, derivative, float(np.linalg.norm(orthogonal, axis=1).max())


def _brackets_zero(first: float, second: float) -> bool:
    return first == 0.0 or second == 0.0 or np.signbit(first) != np.signbit(second)
