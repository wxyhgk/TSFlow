from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.constraints import FixInternals

from uma_workflows.cancellation import CancellationToken
from uma_workflows.optimization import relax_constrained_minimum
from uma_workflows.transition_states.path_bias import PathBias


def prepare_relaxed_scan_seed(
    atoms: Atoms,
    *,
    path,
    path_bias: PathBias,
    highest_coordinate: float,
    requested_progress: float,
    optimization_device: str,
    mps_calculator,
    cpu_calculator,
    fmax: float,
    max_steps: int,
    cancel_token: CancellationToken,
) -> tuple[np.ndarray, tuple[str, ...]]:
    progress = max(requested_progress, highest_coordinate)
    atoms.set_positions(path.evaluate([progress])[0])
    scan_bonds = [
        [
            (1.0 - progress) * start + progress * finish,
            [first, second],
        ]
        for first, second, start, finish in path_bias.synchronized_coordinates
    ]
    atoms.set_constraint(FixInternals(bonds=scan_bonds))
    try:
        result = relax_constrained_minimum(
            atoms,
            optimization_device=optimization_device,
            mps_calculator=mps_calculator,
            cpu_calculator=cpu_calculator,
            fmax=fmax,
            max_steps=max_steps,
            cancel_event=cancel_token,
        )
    finally:
        atoms.set_constraint()
    warnings = [
        "A constrained UMA relaxed-scan point was optimized at reaction "
        f"progress {progress:.2f} before unrestrained Sella refinement."
    ]
    if not result.converged:
        warnings.append(
            "The constrained relaxed-scan point did not reach its requested "
            "force threshold before Sella."
        )
    tangent = path.evaluate([progress], derivative_order=1)[0].reshape(-1)
    return tangent, tuple(warnings)
