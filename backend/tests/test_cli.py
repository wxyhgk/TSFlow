from pathlib import Path

import pytest

from app.cli import build_parser
from app.schemas.jobs import Atom, AtomicStructure, CalculationResult
from app.services.gaussian_input import render_gaussian_input
from app.services.sdf_distance_geometry import generate_structure_from_2d_sdf


FIXTURES = Path(__file__).parent / "fixtures"


def test_documented_cli_syntax_is_accepted() -> None:
    args = build_parser().parse_args(["-sdf", "-2d", "example.sdf"])
    assert args.sdf is True
    assert args.distance_geometry is True
    assert args.input == Path("example.sdf")
    assert args.model == "uma-s-1p2p1"
    assert args.output is None


def test_openchemlib_distance_geometry_adds_hydrogens_and_3d_coordinates() -> None:
    try:
        structure = generate_structure_from_2d_sdf(
            FIXTURES / "ethane-2d.sdf",
            charge=0,
            spin=1,
        )
    except ValueError as exc:
        if "OpenChemLib is unavailable" in str(exc) or "Node.js is required" in str(exc):
            pytest.skip(str(exc))
        raise
    assert structure.name == "ethane-2d"
    assert len(structure.atoms) == 8
    assert sum(atom.symbol == "H" for atom in structure.atoms) == 6
    assert max(abs(atom.position[2]) for atom in structure.atoms) > 0.1
    assert structure.comment is not None and "seed 0x5eed" in structure.comment


def test_gaussian_input_contains_link0_charge_spin_and_coordinates() -> None:
    structure = AtomicStructure(
        atoms=[
            Atom(symbol="O", position=(0.0, 0.0, 0.0)),
            Atom(symbol="H", position=(0.8, 0.5, 0.0)),
            Atom(symbol="H", position=(-0.8, 0.5, 0.0)),
        ],
        charge=0,
        spin=1,
        name="water",
        source_format="sdf",
    )
    result = CalculationResult(
        energy_ev=-10.0,
        forces_ev_per_a=[(0.0, 0.0, 0.0)] * 3,
        max_force_ev_per_a=0.01,
        final_structure=structure,
        converged=True,
        optimization_steps=2,
        optimization_device="cpu",
        optimizer="geomeTRIC-TRIC",
        elapsed_seconds=1.0,
    )
    text = render_gaussian_input(structure, result, stem="water-optimized")
    assert text.startswith("%chk=water-optimized.chk\n%nprocshared=8\n%mem=8GB\n")
    assert "#p B3LYP/6-31G(d) Opt Freq\n\n" in text
    assert "\n0 1\n" in text
    assert sum(line.startswith(("O ", "H ")) for line in text.splitlines()) == 3
    assert text.endswith("\n\n")
