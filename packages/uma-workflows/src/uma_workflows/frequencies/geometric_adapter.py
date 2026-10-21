from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.units import Bohr, Hartree
from geometric.normal_modes import frequency_analysis


def geometric_vibrational_modes(
    atoms: Atoms,
    hessian_ev_per_a2: np.ndarray,
) -> tuple[list[float], np.ndarray, int]:
    """Use geomeTRIC's Eckart-frame vibrational analysis for a full molecule."""
    if len(atoms) == 1:
        return [], np.empty((0, 1, 3)), 3
    coordinates_bohr = np.asarray(atoms.positions, dtype=float).reshape(-1) / Bohr
    hessian_au = np.asarray(hessian_ev_per_a2, dtype=float) * Bohr**2 / Hartree
    frequencies, flat_modes, _ = frequency_analysis(
        coordinates_bohr,
        hessian_au,
        elem=atoms.get_chemical_symbols(),
        mass=atoms.get_masses(),
        verbose=0,
        normalized=True,
    )
    modes = np.asarray(flat_modes, dtype=float).reshape(-1, len(atoms), 3)
    return (
        [float(value) for value in frequencies],
        modes,
        3 * len(atoms) - len(frequencies),
    )
