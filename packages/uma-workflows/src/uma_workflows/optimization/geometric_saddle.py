from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
from ase import Atoms, units
from geometric.errors import GeomOptNotConvergedError
from geometric.molecule import Molecule
from geometric.optimize import run_optimizer

from uma_workflows.cancellation import CancellationToken, raise_if_cancelled
from uma_workflows.geometric_runtime import (
    RecordingASEEngine,
    atoms_to_molecule,
    isolated_geometric_logging,
    nearest_evaluation,
)
from uma_workflows.optimization.models import (
    OptimizationStageResult,
    SaddleRefinement,
    SellaFrame,
)


def _trajectory(
    path: Path,
    initial_positions: np.ndarray,
    engine: RecordingASEEngine,
) -> list[SellaFrame]:
    molecule = Molecule(str(path), build_topology=False)
    positions = [np.asarray(value, dtype=float) for value in molecule.xyzs]
    if np.linalg.norm(positions[-1] - initial_positions) < np.linalg.norm(
        positions[0] - initial_positions
    ):
        positions.reverse()
    frames: list[SellaFrame] = []
    for current in positions:
        if frames and np.max(np.abs(current - frames[-1].positions_a)) < 1.0e-8:
            continue
        energy, forces = nearest_evaluation(current, engine.evaluations)
        frames.append(
            SellaFrame(
                stage="geometric_cpu",
                step=len(frames),
                energy_ev=energy,
                forces_ev_per_a=forces.copy(),
                positions_a=current.copy(),
            )
        )
    return frames


def refine_geometric_saddle(
    atoms: Atoms,
    hessian_ev_per_a2: np.ndarray,
    *,
    calculator,
    fmax: float,
    max_steps: int,
    cancel_event: CancellationToken,
    trust_a: float = 0.03,
    max_trust_a: float = 0.10,
    on_frame: Callable[[SellaFrame], None] | None = None,
) -> SaddleRefinement:
    """Refine a first-order saddle with geomeTRIC RS-P-RFO."""
    size = 3 * len(atoms)
    hessian = np.asarray(hessian_ev_per_a2, dtype=float)
    if hessian.shape != (size, size):
        raise ValueError(
            f"geomeTRIC TS refinement requires Hessian shape {(size, size)}"
        )
    raise_if_cancelled(cancel_event)
    molecule = atoms_to_molecule(atoms)
    engine = RecordingASEEngine(molecule, calculator, cancel_event, atoms)
    hessian_au = (hessian * units.Bohr**2 / units.Hartree).tolist()
    gradient_threshold = fmax * units.Bohr / units.Hartree
    with tempfile.TemporaryDirectory(prefix="uma-geometric-ts-") as directory:
        prefix = Path(directory) / "transition-state"
        with isolated_geometric_logging(Path(directory) / "geometric.log"):
            try:
                run_optimizer(
                    customengine=engine,
                    input=str(Path(directory) / "initial.xyz"),
                    prefix=str(prefix),
                    transition=True,
                    hessian="first",
                    hess_data=hessian_au,
                    frequency=False,
                    maxiter=max_steps,
                    trust=trust_a,
                    tmax=max_trust_a,
                    convergence_energy=1.0e6,
                    convergence_grms=gradient_threshold,
                    convergence_gmax=gradient_threshold,
                    convergence_drms=1.0e6,
                    convergence_dmax=1.0e6,
                    verbose=False,
                )
                converged = True
            except GeomOptNotConvergedError:
                converged = False
        raise_if_cancelled(cancel_event)
        trajectory_path = prefix.with_name(prefix.name + "_optim.xyz")
        if not trajectory_path.exists():
            raise RuntimeError("geomeTRIC TS refinement produced no trajectory")
        frames = _trajectory(trajectory_path, atoms.positions, engine)
    if not frames:
        raise RuntimeError("geomeTRIC TS refinement produced no accepted frames")
    atoms.positions = frames[-1].positions_a
    atoms.calc = calculator
    if on_frame is not None:
        for frame in frames:
            on_frame(frame)
    stage = OptimizationStageResult(
        name="geometric_cpu",
        device="cpu",
        steps=max(0, len(frames) - 1),
        converged=converged,
    )
    return SaddleRefinement(
        converged=converged,
        mps_steps=0,
        cpu_steps=stage.steps,
        frames=frames,
        stage_results=(stage,),
    )
