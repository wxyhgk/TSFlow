from __future__ import annotations

import re
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
from ase import Atoms, units
from geometric.errors import GeomOptNotConvergedError
from geometric.optimize import run_optimizer

from uma_workflows.cancellation import CancellationToken, raise_if_cancelled
from uma_workflows.geometric_runtime import (
    RecordingASEEngine,
    atoms_to_molecule,
    isolated_geometric_logging,
    nearest_evaluation,
)
from uma_workflows.irc.models import IrcDirectionResult, IrcFrame


_ENERGY = re.compile(r"Energy\s+([-+0-9.eE]+)")


def regularize_irc_hessian(
    hessian_ev_per_a2: np.ndarray,
    masses: np.ndarray,
) -> np.ndarray:
    """Regularize an already screened Hessian only for IRC initialization."""
    root_masses = np.sqrt(np.repeat(np.asarray(masses, dtype=float), 3))
    mass_weighted = (
        np.asarray(hessian_ev_per_a2, dtype=float)
        / root_masses[:, None]
        / root_masses[None, :]
    )
    values, vectors = np.linalg.eigh(0.5 * (mass_weighted + mass_weighted.T))
    negative = np.flatnonzero(values < 0.0)
    for index in negative[1:]:
        values[index] = max(abs(values[index]), 1.0e-10)
    regularized = (vectors * values) @ vectors.T
    return regularized * root_masses[:, None] * root_masses[None, :]


def _read_xyz(path: Path) -> list[tuple[np.ndarray, float | None]]:
    lines = path.read_text().splitlines()
    frames: list[tuple[np.ndarray, float | None]] = []
    cursor = 0
    while cursor < len(lines):
        count = int(lines[cursor].strip())
        comment = lines[cursor + 1]
        coordinates = [
            [float(value) for value in lines[cursor + 2 + i].split()[1:4]]
            for i in range(count)
        ]
        match = _ENERGY.search(comment)
        energy = float(match.group(1)) * units.Hartree if match else None
        frames.append((np.asarray(coordinates), energy))
        cursor += count + 2
    return frames


def _nearest_evaluation(
    positions: np.ndarray,
    evaluations: list[tuple[np.ndarray, float, np.ndarray]],
) -> tuple[float, float]:
    distances = [
        float(np.max(np.abs(candidate - positions))) for candidate, _, _ in evaluations
    ]
    index = int(np.argmin(distances))
    _, energy, forces = evaluations[index]
    return energy, float(np.linalg.norm(forces, axis=1).max(initial=0.0))


def _trajectory(
    path: Path,
    transition_state: Atoms,
    engine: RecordingASEEngine,
    fmax: float,
) -> list[IrcFrame]:
    raw = _read_xyz(path)
    if np.linalg.norm(raw[-1][0] - transition_state.positions) < np.linalg.norm(
        raw[0][0] - transition_state.positions
    ):
        raw.reverse()
    frames: list[IrcFrame] = []
    for positions, file_energy in raw:
        if frames and np.max(np.abs(positions - frames[-1].positions_a)) < 1.0e-8:
            continue
        energy, forces = nearest_evaluation(positions, engine.evaluations)
        max_force = float(np.linalg.norm(forces, axis=1).max(initial=0.0))
        frames.append(
            IrcFrame(
                step=len(frames),
                energy_ev=file_energy if file_energy is not None else energy,
                max_force_ev_per_a=max_force,
                positions_a=positions,
                converged=max_force <= fmax,
            )
        )
    return frames


def follow_geometric_irc_direction(
    atoms: Atoms,
    hessian_ev_per_a2: np.ndarray,
    direction: str,
    calculator,
    *,
    trust_a: float,
    fmax: float,
    steps: int,
    cancel_event: CancellationToken,
    regularize_secondary_negative_modes: bool = False,
    on_frame: Callable[[IrcFrame], None] | None = None,
) -> IrcDirectionResult:
    """Follow one Gonzalez-Schlegel IRC direction with geomeTRIC."""
    molecule = atoms_to_molecule(atoms)
    engine = RecordingASEEngine(molecule, calculator, cancel_event, atoms)
    hessian = np.asarray(hessian_ev_per_a2)
    if regularize_secondary_negative_modes:
        hessian = regularize_irc_hessian(hessian, atoms.get_masses())
    hessian_au = (hessian * units.Bohr**2 / units.Hartree).tolist()
    with tempfile.TemporaryDirectory(prefix="uma-geometric-irc-") as directory:
        prefix = Path(directory) / direction
        with isolated_geometric_logging(Path(directory) / "geometric.log"):
            try:
                run_optimizer(
                    customengine=engine,
                    input=str(Path(directory) / "transition-state.xyz"),
                    prefix=str(prefix),
                    irc=True,
                    irc_direction=direction,
                    hessian="first",
                    hess_data=hessian_au,
                    frequency=True,
                    maxiter=steps,
                    trust=trust_a,
                    tmax=trust_a,
                    verbose=False,
                )
                incomplete = False
            except GeomOptNotConvergedError:
                incomplete = True
        raise_if_cancelled(cancel_event)
        paths = list(Path(directory).glob("*_irc.xyz"))
        if not paths:
            raise RuntimeError("geomeTRIC IRC did not produce a trajectory")
        frames = _trajectory(paths[0], atoms, engine, fmax)
    if on_frame is not None:
        for frame in frames:
            on_frame(frame)
    return IrcDirectionResult(frames=frames, inner_loop_failed=incomplete)
