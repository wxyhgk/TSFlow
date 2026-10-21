from ase import Atoms
from ase.calculators.morse import MorsePotential
import pytest

from app.services.geometric_optimizer import optimize_with_geometric


def test_tric_optimizer_uses_ase_energy_and_forces() -> None:
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]])
    live_steps: list[int] = []

    result = optimize_with_geometric(
        atoms,
        MorsePotential(r0=0.74),
        fmax_ev_per_a=0.05,
        max_steps=100,
        check_cancelled=lambda: None,
        on_evaluation=lambda frame: live_steps.append(frame.step),
    )

    final_fmax = max(
        float((force**2).sum() ** 0.5) for force in result.final.forces_ev_per_a
    )
    assert atoms.get_distance(0, 1) == pytest.approx(0.74, abs=0.01)
    assert final_fmax <= 0.05
    assert len(result.frames) >= 2
    assert live_steps == list(range(result.evaluations))
