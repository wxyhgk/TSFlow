from __future__ import annotations

import logging
import logging.config
from contextlib import contextmanager
from pathlib import Path
from threading import Lock

import numpy as np
from ase import Atoms, units
from geometric.ase_engine import EngineASE
from geometric.molecule import Molecule
from geometric.nifty import logger as geometric_logger

from uma_workflows.cancellation import CancellationToken, raise_if_cancelled


_GEOMETRIC_LOCK = Lock()


class RecordingASEEngine(EngineASE):
    """geomeTRIC ASE engine that preserves molecular metadata and evaluations."""

    def __init__(
        self,
        molecule: Molecule,
        calculator,
        cancel_event: CancellationToken,
        template: Atoms,
    ):
        super().__init__(molecule, calculator)
        self.ase_atoms = template.copy()
        self.ase_atoms.calc = calculator
        self.cancel_event = cancel_event
        self.evaluations: list[tuple[np.ndarray, float, np.ndarray]] = []

    def calc_new(self, coords, dirname):
        raise_if_cancelled(self.cancel_event)
        result = super().calc_new(coords, dirname)
        self.evaluations.append(
            (
                np.asarray(coords).reshape(-1, 3) * units.Bohr,
                float(result["energy"]) * units.Hartree,
                -np.asarray(result["gradient"]).reshape(-1, 3)
                * units.Hartree
                / units.Bohr,
            )
        )
        return result


def atoms_to_molecule(atoms: Atoms) -> Molecule:
    molecule = Molecule()
    molecule.elem = atoms.get_chemical_symbols()
    molecule.xyzs = [atoms.positions.copy()]
    molecule.comms = ["UMA structure"]
    molecule.build_topology()
    return molecule


def nearest_evaluation(
    positions: np.ndarray,
    evaluations: list[tuple[np.ndarray, float, np.ndarray]],
) -> tuple[float, np.ndarray]:
    distances = [
        float(np.max(np.abs(candidate - positions))) for candidate, _, _ in evaluations
    ]
    _, energy, forces = evaluations[int(np.argmin(distances))]
    return energy, forces


@contextmanager
def isolated_geometric_logging(log_path: Path):
    """Keep geomeTRIC from replacing the host application's root logger."""
    with _GEOMETRIC_LOCK:
        original_config = logging.config.fileConfig
        original_handlers = geometric_logger.handlers[:]
        original_propagate = geometric_logger.propagate
        handler = logging.FileHandler(log_path)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logging.config.fileConfig = lambda *args, **kwargs: None
        geometric_logger.handlers = [handler]
        geometric_logger.propagate = False
        try:
            yield
        finally:
            handler.close()
            geometric_logger.handlers = original_handlers
            geometric_logger.propagate = original_propagate
            logging.config.fileConfig = original_config
