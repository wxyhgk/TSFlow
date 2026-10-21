from threading import Event

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from app.services.sella_refiner import (
    numerical_frequency_analysis,
    refine_first_order_saddle,
)


class AnalyticSaddleCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        positions = np.asarray(self.atoms.positions)
        coordinates = positions.reshape(-1)
        curvatures = np.ones_like(coordinates)
        curvatures[:3] = np.asarray([-1.0, 2.0, 3.0])
        self.results["energy"] = float(0.5 * np.sum(curvatures * coordinates**2))
        self.results["forces"] = (-curvatures * coordinates).reshape((-1, 3))


class BondSaddleCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        displacement = self.atoms.positions[1] - self.atoms.positions[0]
        distance = np.linalg.norm(displacement)
        reaction_coordinate = distance - 1.0
        derivative = -reaction_coordinate
        force_on_second = -derivative * displacement / distance
        self.results["energy"] = float(-0.5 * reaction_coordinate**2)
        self.results["forces"] = np.asarray([-force_on_second, force_on_second])


def test_numerical_frequency_analysis_finds_one_imaginary_mode() -> None:
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    calculator = AnalyticSaddleCalculator()
    atoms.calc = calculator
    progress = []

    result = numerical_frequency_analysis(
        atoms,
        calculator=calculator,
        delta_a=0.01,
        imaginary_threshold_cm1=50.0,
        cancel_event=Event(),
        on_coordinate=lambda step, total: progress.append((step, total)),
        project_rigid_body=False,
    )

    assert len(result.frequencies_cm1) == 3
    assert len(result.imaginary_frequencies_cm1) == 1
    assert result.primary_imaginary_mode is not None
    assert progress == [(1, 3), (2, 3), (3, 3)]


def test_partial_numerical_frequency_analysis_maps_mode_to_full_structure() -> None:
    atoms = Atoms("HH", positions=[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    calculator = AnalyticSaddleCalculator()
    result = numerical_frequency_analysis(
        atoms,
        calculator=calculator,
        delta_a=0.01,
        imaginary_threshold_cm1=50.0,
        cancel_event=Event(),
        active_indices=[0],
    )

    assert len(result.frequencies_cm1) == 3
    assert len(result.imaginary_frequencies_cm1) == 1
    assert result.primary_imaginary_mode is not None
    assert result.primary_imaginary_mode.shape == (2, 3)
    assert result.primary_imaginary_mode[1] == pytest.approx([0.0, 0.0, 0.0])


def test_sella_projects_cartesian_dmf_tangent_into_free_coordinate_space() -> None:
    atoms = Atoms("HH", positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    calculator = BondSaddleCalculator()
    refinement = refine_first_order_saddle(
        atoms,
        path_tangent=np.asarray([-1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        optimization_device="cpu",
        mps_calculator=calculator,
        cpu_calculator=calculator,
        mps_fmax=0.1,
        cpu_fmax=1.0e-4,
        max_steps=20,
        cancel_event=Event(),
    )

    assert refinement.converged is True
    assert atoms.get_distance(0, 1) == pytest.approx(1.0, abs=1.0e-5)
