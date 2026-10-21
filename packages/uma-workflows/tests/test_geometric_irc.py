import logging
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from uma_workflows.irc.geometric_runner import (
    follow_geometric_irc_direction,
    regularize_irc_hessian,
)
from uma_workflows.irc.models import IrcDirectionResult, IrcFrame
from uma_workflows.irc.topology import (
    match_irc_endpoint_topologies,
    perceived_heavy_atom_bonds,
)
from uma_workflows.irc.workflow_internals import follow_direction
from uma_workflows.optimization import refine_geometric_saddle


class DoubleWellCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(
        self,
        atoms=None,
        properties=("energy",),
        system_changes=all_changes,
    ):
        super().calculate(atoms, properties, system_changes)
        assert atoms.info == {"charge": -1, "spin": 2}
        positions = atoms.positions
        vector = positions[1] - positions[0]
        distance = np.linalg.norm(vector)
        displacement = distance - 1.0
        energy = displacement**4 - 0.04 * displacement**2
        derivative = 4.0 * displacement**3 - 0.08 * displacement
        forces = np.zeros_like(positions)
        forces[0] += derivative * vector / distance
        forces[1] -= derivative * vector / distance
        target = np.sqrt(1.25)
        for index in (0, 1):
            vector = positions[2] - positions[index]
            distance = np.linalg.norm(vector)
            displacement = distance - target
            derivative = 2.0 * displacement
            energy += displacement**2
            forces[index] += derivative * vector / distance
            forces[2] -= derivative * vector / distance
        self.results = {"energy": energy, "forces": forces}


def numerical_hessian(atoms: Atoms, delta: float = 1.0e-4) -> np.ndarray:
    size = 3 * len(atoms)
    hessian = np.zeros((size, size))
    for coordinate in range(size):
        plus = atoms.copy()
        minus = atoms.copy()
        plus_positions = plus.positions.reshape(-1)
        minus_positions = minus.positions.reshape(-1)
        plus_positions[coordinate] += delta
        minus_positions[coordinate] -= delta
        plus.positions = plus_positions.reshape(-1, 3)
        minus.positions = minus_positions.reshape(-1, 3)
        plus.calc = DoubleWellCalculator()
        minus.calc = DoubleWellCalculator()
        hessian[:, coordinate] = -(
            plus.get_forces().reshape(-1) - minus.get_forces().reshape(-1)
        ) / (2.0 * delta)
    return 0.5 * (hessian + hessian.T)


def test_geometric_irc_follows_both_sides_of_first_order_saddle() -> None:
    atoms = Atoms(
        "HHH",
        positions=[[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )
    atoms.info.update({"charge": -1, "spin": 2})
    hessian = numerical_hessian(atoms)
    root_handlers = logging.getLogger().handlers[:]
    results = {}
    for direction in ("forward", "backward"):
        results[direction] = follow_geometric_irc_direction(
            atoms,
            hessian,
            direction,
            DoubleWellCalculator(),
            trust_a=0.10,
            fmax=0.05,
            steps=4,
            cancel_event=Event(),
        )

    forward = results["forward"].frames
    backward = results["backward"].frames
    assert len(forward) >= 2
    assert len(backward) >= 2
    assert forward[-1].energy_ev < forward[0].energy_ev
    assert backward[-1].energy_ev < backward[0].energy_ev
    forward_step = (forward[1].positions_a - atoms.positions).reshape(-1)
    backward_step = (backward[1].positions_a - atoms.positions).reshape(-1)
    assert np.dot(forward_step, backward_step) < 0.0
    assert logging.getLogger().handlers == root_handlers


def test_follow_direction_returns_geometric_irc_endpoint(monkeypatch) -> None:
    atoms = Atoms("HH", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    endpoint = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    result = IrcDirectionResult(
        frames=[
            IrcFrame(
                step=0,
                energy_ev=-1.0,
                max_force_ev_per_a=0.01,
                positions_a=endpoint,
                converged=True,
            )
        ],
        inner_loop_failed=False,
    )
    captured = {}

    def fake_follow(_atoms, _hessian, direction, *args, **kwargs):
        captured["direction"] = direction
        return result

    monkeypatch.setattr(
        "uma_workflows.irc.workflow_internals.follow_geometric_irc_direction",
        fake_follow,
    )
    calculator = DoubleWellCalculator()
    mode = np.ones(6)

    image, returned = follow_direction(
        atoms,
        np.eye(6),
        "reverse",
        mode,
        calculator,
        SimpleNamespace(engine="geometric", dx=0.1, fmax=0.05, steps=10),
        Event(),
        None,
    )

    assert returned is result
    assert captured["direction"] == "backward"
    assert image.calc is calculator
    assert np.allclose(image.positions, endpoint)
    assert np.array_equal(image.info["irc_initial_mode"], mode)


def test_irc_endpoint_topology_includes_transferred_hydrogen() -> None:
    reactant = Atoms(
        "HCO",
        positions=[[0.0, 0.0, 0.0], [1.05, 0.0, 0.0], [2.30, 0.0, 0.0]],
    )
    product = Atoms(
        "HCO",
        positions=[[2.20, 0.0, 0.0], [0.95, 0.0, 0.0], [1.25, 0.0, 0.0]],
    )

    match = match_irc_endpoint_topologies(
        reactant,
        reactant,
        reactant,
        product,
    )

    assert match.matches_expected_endpoints is False
    assert match.endpoint_b_mismatch_count > 0
    assert all(0 not in bond for bond in perceived_heavy_atom_bonds(reactant))


def test_geometric_rs_p_rfo_refines_first_order_saddle() -> None:
    atoms = Atoms(
        "HHH",
        positions=[[-0.52, 0.0, 0.0], [0.52, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )
    atoms.info.update({"charge": -1, "spin": 2})
    hessian = numerical_hessian(atoms)
    result = refine_geometric_saddle(
        atoms,
        hessian,
        calculator=DoubleWellCalculator(),
        fmax=0.0005,
        max_steps=20,
        cancel_event=Event(),
    )

    distance = atoms.get_distance(0, 1)
    final_forces = atoms.get_forces()
    final_hessian = numerical_hessian(atoms)
    assert result.converged
    assert distance == pytest.approx(1.0, abs=1.0e-3)
    assert np.linalg.norm(final_forces, axis=1).max() < 0.0005
    assert np.count_nonzero(np.linalg.eigvalsh(final_hessian) < -1.0e-4) == 1


def test_irc_hessian_regularization_keeps_only_primary_negative_mode() -> None:
    hessian = np.diag([-2.0, -0.2, -1.0e-12, 0.4, 0.8, 1.0])
    regularized = regularize_irc_hessian(hessian, np.ones(2))

    values = np.linalg.eigvalsh(regularized)
    assert np.count_nonzero(values < 0.0) == 1
    assert values[0] == pytest.approx(-2.0)
