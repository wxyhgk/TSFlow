from threading import Event

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from uma_workflows.transition_states import (
    bond_force_components,
    relaxed_bond_scan,
)


class DoubleWellBondCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ):
        super().calculate(atoms, properties, system_changes)
        vector = atoms.positions[1] - atoms.positions[0]
        distance = float(np.linalg.norm(vector))
        derivative = 4.0 * distance * (distance**2 - 1.0)
        unit = vector / distance
        forces = np.zeros_like(atoms.positions)
        forces[0] = derivative * unit
        forces[1] = -forces[0]
        self.results = {
            "energy": (distance**2 - 1.0) ** 2,
            "forces": forces,
        }


def test_bond_force_components_reports_physical_derivative() -> None:
    positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    forces = np.array([[-1.5, 0.2, 0.0], [1.5, -0.2, 0.0]])

    distance, derivative, orthogonal_fmax = bond_force_components(
        positions,
        forces,
        0,
        1,
    )

    assert distance == pytest.approx(0.5)
    assert derivative == pytest.approx(-1.5)
    assert orthogonal_fmax == pytest.approx(0.2)


def test_relaxed_bond_scan_finds_projected_force_bracket() -> None:
    atoms = Atoms("HH", positions=[[0.0, 0, 0], [0.9, 0, 0]])

    result = relaxed_bond_scan(
        atoms,
        atom_i=0,
        atom_j=1,
        distances_a=(0.8, 1.0, 1.2),
        calculator=DoubleWellBondCalculator(),
        fmax=1.0e-5,
        max_steps=50,
        restraint_strength_ev_per_a2=1000.0,
        cancel_event=Event(),
    )

    assert len(result.points) == 3
    assert result.derivative_brackets == ((1, 2),)
    assert all(point.energy_ev >= 0.0 for point in result.points)
