from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes


BondTarget = tuple[int, int, float]


class HarmonicBondRestraintCalculator(Calculator):
    """Add endpoint-only harmonic bond restraints to an ASE calculator."""

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        calculator,
        targets: Sequence[BondTarget],
        strength_ev_per_a2: float = 100.0,
    ) -> None:
        super().__init__()
        self.calculator = calculator
        self.targets = tuple(targets)
        self.strength = strength_ev_per_a2

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        if atoms is None:
            raise ValueError("endpoint restraint requires atoms")
        image = atoms.copy()
        image.set_constraint()
        image.calc = self.calculator
        energy = float(image.get_potential_energy())
        forces = np.asarray(image.get_forces(), dtype=float)
        for atom_i, atom_j, target in self.targets:
            vector = image.positions[atom_j] - image.positions[atom_i]
            distance = float(np.linalg.norm(vector))
            if distance <= 1.0e-12:
                continue
            delta = distance - target
            energy += 0.5 * self.strength * delta**2
            force = self.strength * delta * vector / distance
            forces[atom_i] += force
            forces[atom_j] -= force
        self.results = {"energy": energy, "forces": forces}
