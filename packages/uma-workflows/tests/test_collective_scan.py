from threading import Event

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from uma_workflows.transition_states.collective_scan import (
    CollectiveCoordinateTerm,
    collective_coordinate_value_and_gradient,
    relaxed_collective_coordinate_scan,
)


class InvertedBondCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        vector = atoms.positions[1] - atoms.positions[0]
        distance = float(np.linalg.norm(vector))
        derivative = -2.0 * (distance - 1.0)
        unit = vector / distance
        forces = np.zeros_like(atoms.positions)
        forces[0] = derivative * unit
        forces[1] = -forces[0]
        self.results = {
            "energy": 1.0 - (distance - 1.0) ** 2,
            "forces": forces,
        }


def test_linear_combination_value_and_gradient() -> None:
    positions = np.asarray([[0.0, 0, 0], [1.0, 0, 0], [3.0, 0, 0]])
    terms = (
        CollectiveCoordinateTerm(0, 1, 1.0),
        CollectiveCoordinateTerm(1, 2, -1.0),
    )

    value, gradient = collective_coordinate_value_and_gradient(positions, terms)

    assert value == pytest.approx(-1.0)
    assert gradient[:, 0].tolist() == pytest.approx([-1.0, 2.0, -1.0])


def test_relaxed_collective_scan_selects_interior_unbiased_maximum() -> None:
    atoms = Atoms("HH", positions=[[0.0, 0, 0], [1.0, 0, 0]])
    result = relaxed_collective_coordinate_scan(
        atoms,
        terms=(CollectiveCoordinateTerm(0, 1, 1.0),),
        target_values_a=(0.8, 1.0, 1.2),
        calculator=InvertedBondCalculator(),
        fmax=1.0e-5,
        max_steps=50,
        restraint_strength_ev_per_a2=1000.0,
        cancel_event=Event(),
    )

    assert result.interior_maxima == (1,)
    assert result.selected_maximum is not None
    assert result.selected_maximum.coordinate_value_a == pytest.approx(1.0)
    assert result.selected_maximum.energy_ev == pytest.approx(1.0)
