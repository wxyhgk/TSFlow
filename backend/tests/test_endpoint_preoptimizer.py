from threading import Event

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from app.services.endpoint_preoptimizer import covalent_fragments, preoptimize_endpoint


class FragmentBondCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        vector = atoms.positions[1] - atoms.positions[0]
        distance = np.linalg.norm(vector)
        derivative = distance - 0.75
        forces = np.zeros_like(atoms.positions)
        forces[0] = derivative * vector / distance
        forces[1] = -forces[0]
        self.results = {"energy": 0.5 * derivative**2, "forces": forces}


def test_preoptimization_relaxes_fragment_internal_bond_and_keeps_separation() -> None:
    atoms = Atoms("HHHe", positions=[[0, 0, 0], [0.7, 0, 0], [3.0, 0, 0]])
    calculator = FragmentBondCalculator()
    initial_anchor_distance = atoms.get_distance(0, 2)
    assert covalent_fragments(atoms) == [[0, 1], [2]]

    result = preoptimize_endpoint(
        atoms,
        optimization_device="cpu",
        mps_calculator=calculator,
        cpu_calculator=calculator,
        fmax=1.0e-4,
        max_steps=20,
        cancel_event=Event(),
    )

    assert result.performed
    assert result.fragment_count == 2
    assert result.converged
    assert atoms.get_distance(0, 1) == pytest.approx(0.75, abs=1.0e-3)
    assert atoms.get_distance(0, 2) == pytest.approx(initial_anchor_distance, abs=1.0e-6)
    assert not atoms.constraints


def test_preoptimization_does_not_skip_connected_product() -> None:
    atoms = Atoms("HH", positions=[[0, 0, 0], [0.70, 0, 0]])
    calculator = FragmentBondCalculator()
    assert covalent_fragments(atoms) == [[0, 1]]

    result = preoptimize_endpoint(
        atoms,
        optimization_device="cpu",
        mps_calculator=calculator,
        cpu_calculator=calculator,
        fmax=1.0e-4,
        max_steps=20,
        cancel_event=Event(),
    )

    assert result.performed
    assert result.fragment_count == 1
    assert result.anchor_indices == []
    assert result.converged
    assert atoms.get_distance(0, 1) == pytest.approx(0.75, abs=1.0e-3)
