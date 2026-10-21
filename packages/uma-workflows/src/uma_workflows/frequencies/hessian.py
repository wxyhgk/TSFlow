from __future__ import annotations

import time
from typing import Callable

import numpy as np
from ase import Atoms

from uma_workflows.cancellation import CancellationToken, raise_if_cancelled
from uma_workflows.frequencies.models import NumericalHessian
from uma_workflows.frequencies.selection import validate_active_indices


def compute_numerical_hessian(
    atoms: Atoms,
    *,
    calculator,
    delta_a: float,
    cancel_event: CancellationToken,
    active_indices: list[int] | None = None,
    on_coordinate: Callable[[int, int], None] | None = None,
) -> NumericalHessian:
    """Compute a central finite-difference Cartesian Hessian from forces."""
    if delta_a <= 0:
        raise ValueError("finite-difference displacement must be positive")
    started = time.perf_counter()
    selected = validate_active_indices(atoms, active_indices)
    reference_positions = atoms.positions.copy()
    degrees_of_freedom = 3 * len(selected)
    hessian = np.empty((degrees_of_freedom, degrees_of_freedom), dtype=float)

    for coordinate in range(degrees_of_freedom):
        raise_if_cancelled(cancel_event)
        selected_atom_index, axis = divmod(coordinate, 3)
        atom_index = selected[selected_atom_index]
        displaced_forces: list[np.ndarray] = []
        for sign in (1.0, -1.0):
            raise_if_cancelled(cancel_event)
            displaced = atoms.copy()
            positions = reference_positions.copy()
            positions[atom_index, axis] += sign * delta_a
            displaced.set_positions(positions)
            displaced.calc = calculator
            displaced_forces.append(
                np.asarray(displaced.get_forces(), dtype=float)[selected].reshape(-1)
            )
        hessian[:, coordinate] = -(
            displaced_forces[0] - displaced_forces[1]
        ) / (2.0 * delta_a)
        if on_coordinate is not None:
            on_coordinate(coordinate + 1, degrees_of_freedom)
        raise_if_cancelled(cancel_event)

    asymmetry = _relative_asymmetry(hessian)
    return NumericalHessian(
        hessian_ev_per_a2=0.5 * (hessian + hessian.T),
        active_atom_indices=selected,
        elapsed_seconds=time.perf_counter() - started,
        method="finite_difference",
        asymmetry_relative=asymmetry,
    )


def compute_analytic_hessian(
    atoms: Atoms,
    *,
    calculator,
    active_indices: list[int] | None = None,
) -> NumericalHessian:
    """Read the exact coordinate derivative exposed by an ASE calculator."""
    started = time.perf_counter()
    selected = validate_active_indices(atoms, active_indices)
    image = atoms.copy()
    image.calc = calculator
    raw = np.asarray(calculator.get_property("hessian", image), dtype=float)
    full_shape = (3 * len(atoms), 3 * len(atoms))
    if raw.shape != full_shape:
        raise ValueError(f"analytic Hessian shape must be {full_shape}, got {raw.shape}")
    coordinates = [3 * atom + axis for atom in selected for axis in range(3)]
    active = raw[np.ix_(coordinates, coordinates)]
    return NumericalHessian(
        hessian_ev_per_a2=0.5 * (active + active.T),
        active_atom_indices=selected,
        elapsed_seconds=time.perf_counter() - started,
        method="analytic",
        asymmetry_relative=_relative_asymmetry(active),
    )


def _relative_asymmetry(hessian: np.ndarray) -> float:
    norm = float(np.linalg.norm(hessian))
    return float(np.linalg.norm(hessian - hessian.T) / max(norm, 1.0e-30))
