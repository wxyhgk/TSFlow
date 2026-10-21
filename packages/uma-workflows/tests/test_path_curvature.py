from threading import Event

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from uma_workflows.transition_states.workflow.path_curvature import (
    probe_path_curvature,
)


class SaddleCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        coordinates = atoms.get_positions()
        curvatures = np.asarray([[-2.0, 3.0, 4.0]])
        self.results["energy"] = float(
            0.5 * np.sum(curvatures * coordinates**2)
        )
        self.results["forces"] = -curvatures * coordinates


def test_path_curvature_finds_nonstationary_negative_seed() -> None:
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    coordinates = np.linspace(0.0, 1.0, 5)
    positions = np.zeros((5, 1, 3))
    positions[:, 0, 0] = np.linspace(-0.4, 0.4, 5)
    positions[:, 0, 1] = 0.1
    energies = np.asarray([0.0, 0.3, 0.5, 0.3, 0.0])

    result = probe_path_curvature(
        atoms,
        positions,
        coordinates,
        energies,
        calculator=SaddleCalculator(),
        cancel_event=Event(),
        displacement_a=0.01,
        max_frames=3,
    )

    assert len(result.probes) == 3
    assert result.force_evaluations == 9
    assert result.selected_probe is not None
    assert result.selected_probe.frame_index == 2
    assert np.isclose(
        result.selected_probe.directional_curvature_ev_per_a2,
        -2.0,
    )
    assert result.selected_probe.max_force_ev_per_a > 0.0


def test_path_curvature_does_not_claim_positive_path_is_a_seed() -> None:
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    coordinates = np.linspace(0.0, 1.0, 3)
    positions = np.zeros((3, 1, 3))
    positions[:, 0, 1] = np.linspace(-0.2, 0.2, 3)

    result = probe_path_curvature(
        atoms,
        positions,
        coordinates,
        np.asarray([0.0, 0.5, 0.0]),
        calculator=SaddleCalculator(),
        cancel_event=Event(),
    )

    assert result.selected_probe is None
    assert result.probes[0].has_negative_directional_curvature is False
    assert "No probed frame" in result.warnings[-1]
