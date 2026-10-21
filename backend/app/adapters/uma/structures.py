"""Conversions between backend structures and ASE atoms."""

import hashlib
import json
from threading import Event

from ase import Atoms
from uma_workflows.runtime import CalculationCancelled

from app.contracts.jobs import Atom, AtomicStructure, JobCreate


def atomic_structure_to_atoms(structure: AtomicStructure) -> Atoms:
    atoms = Atoms(
        symbols=[atom.symbol for atom in structure.atoms],
        positions=[atom.position for atom in structure.atoms],
        pbc=False,
    )
    atoms.info.update({"charge": structure.charge, "spin": structure.spin})
    return atoms


def structure_fingerprint(structure: AtomicStructure) -> str:
    payload = {
        "atoms": [
            {"symbol": atom.symbol, "position": list(atom.position)}
            for atom in structure.atoms
        ],
        "charge": structure.charge,
        "spin": structure.spin,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def to_atoms(request: JobCreate) -> Atoms:
    return atomic_structure_to_atoms(request.structure)


def structure_from_template(
    template: AtomicStructure,
    atoms: Atoms,
) -> AtomicStructure:
    return AtomicStructure(
        atoms=[
            Atom(symbol=symbol, position=tuple(float(value) for value in position))
            for symbol, position in zip(
                atoms.get_chemical_symbols(), atoms.positions, strict=True
            )
        ],
        charge=template.charge,
        spin=template.spin,
        name=template.name,
        comment=template.comment,
        source_format=template.source_format,
    )


def prepare_nonperiodic_graph_box(atoms: Atoms):
    spans = atoms.positions.max(axis=0) - atoms.positions.min(axis=0)
    atoms.set_cell(tuple(float(span + 20.0) for span in spans))
    before = atoms.positions.copy()
    atoms.center()
    return atoms.positions[0] - before[0]


def max_force(forces) -> float:
    return float((forces**2).sum(axis=1).max() ** 0.5)


def structure_from_atoms(
    request: JobCreate,
    atoms: Atoms,
    translation=None,
) -> AtomicStructure:
    positions = atoms.positions if translation is None else atoms.positions - translation
    return AtomicStructure(
        atoms=[
            Atom(symbol=symbol, position=tuple(float(value) for value in position))
            for symbol, position in zip(
                atoms.get_chemical_symbols(), positions, strict=True
            )
        ],
        charge=request.structure.charge,
        spin=request.structure.spin,
        name=request.structure.name,
        comment=request.structure.comment,
        source_format=request.structure.source_format,
    )


def raise_if_cancelled(cancel_event: Event) -> None:
    if cancel_event.is_set():
        raise CalculationCancelled("calculation cancelled")
