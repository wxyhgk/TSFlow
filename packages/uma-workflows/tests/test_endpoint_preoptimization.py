from threading import Event

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from uma_workflows.endpoints import preoptimize_endpoint_stages
from uma_workflows.optimization import OptimizationStage


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


def test_endpoint_accepts_generic_optimization_stages() -> None:
    atoms = Atoms("HHHe", positions=[[0, 0, 0], [0.7, 0, 0], [3.0, 0, 0]])
    calculator = FragmentBondCalculator()

    result = preoptimize_endpoint_stages(
        atoms,
        stages=[
            OptimizationStage(
                name="custom_endpoint_stage",
                calculator=calculator,
                fmax=1.0e-4,
                max_steps=20,
            )
        ],
        cancel_event=Event(),
    )

    assert result.converged
    assert result.stage_results[0].name == "custom_endpoint_stage"
    assert atoms.get_distance(0, 1) == pytest.approx(0.75, abs=1.0e-3)
    assert not atoms.constraints


def test_endpoint_can_preserve_reaction_coordinate_distance() -> None:
    atoms = Atoms("HH", positions=[[0, 0, 0], [0.7, 0, 0]])

    result = preoptimize_endpoint_stages(
        atoms,
        stages=[
            OptimizationStage(
                name="topology_preserving_endpoint",
                calculator=FragmentBondCalculator(),
                fmax=1.0e-4,
                max_steps=20,
            )
        ],
        cancel_event=Event(),
        preserved_bonds=((0, 1),),
    )

    assert result.converged
    assert atoms.get_distance(0, 1) == pytest.approx(0.7, abs=1.0e-3)
    assert not atoms.constraints
