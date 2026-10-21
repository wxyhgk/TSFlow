from __future__ import annotations

from collections.abc import Callable

import numpy as np
from ase import Atoms

from uma_workflows.cancellation import CancellationToken
from uma_workflows.optimization import (
    OptimizationStage,
    SaddleRefinement,
    SellaFrame,
    refine_saddle_stages,
)
from uma_workflows.transition_states.path_bias import (
    PathBias,
    PathBiasedCalculator,
)


def refine_corridor_seed_then_unbiased(
    atoms: Atoms,
    *,
    tangent: np.ndarray,
    path_bias: PathBias,
    optimization_device: str,
    mps_calculator,
    cpu_calculator,
    mps_fmax: float,
    cpu_fmax: float,
    max_steps: int,
    use_path_mode_hessian: bool,
    cancel_token: CancellationToken,
    on_frame: Callable[[SellaFrame], None] | None,
) -> SaddleRefinement:
    seed_calculator = (
        mps_calculator
        if optimization_device == "mps"
        else cpu_calculator
    )
    return refine_saddle_stages(
        atoms,
        path_tangent=tangent,
        stages=[
            OptimizationStage(
                name=(
                    "sella_mps"
                    if optimization_device == "mps"
                    else "sella_cpu"
                ),
                device=optimization_device,
                calculator=PathBiasedCalculator(seed_calculator, path_bias),
                fmax=max(mps_fmax, 0.03),
                max_steps=max_steps,
            ),
            OptimizationStage(
                name="sella_cpu",
                device="cpu",
                calculator=cpu_calculator,
                fmax=cpu_fmax,
                max_steps=max_steps,
            ),
        ],
        use_path_mode_hessian=use_path_mode_hessian,
        cancel_event=cancel_token,
        on_frame=on_frame,
    )
