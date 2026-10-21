from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
from ase import Atoms
from sella import Sella

from uma_workflows.accelerators import synchronize
from uma_workflows.cancellation import CancellationToken, raise_if_cancelled
from uma_workflows.optimization.models import (
    OptimizationStage,
    OptimizationStageResult,
    SaddleRefinement,
    SellaFrame,
)


def refine_saddle_stages(
    atoms: Atoms,
    *,
    path_tangent: np.ndarray,
    stages: Sequence[OptimizationStage],
    use_path_mode_hessian: bool = False,
    cancel_event: CancellationToken,
    on_frame: Callable[[SellaFrame], None] | None = None,
) -> SaddleRefinement:
    """Run an arbitrary ordered sequence of first-order Sella stages."""
    raise_if_cancelled(cancel_event)
    if not stages:
        raise ValueError("saddle refinement requires at least one stage")
    frames: list[SellaFrame] = []
    stage_results: list[OptimizationStageResult] = []
    for stage in stages:
        raise_if_cancelled(cancel_event)
        atoms.calc = stage.calculator
        steps, converged = _run_sella_stage(
            atoms,
            stage=stage.name,
            fmax=stage.fmax,
            max_steps=stage.max_steps,
            use_path_mode_hessian=use_path_mode_hessian,
            path_tangent=path_tangent,
            cancel_event=cancel_event,
            frames=frames,
            on_frame=on_frame,
        )
        stage_results.append(
            OptimizationStageResult(
                name=stage.name,
                device=stage.device,
                steps=steps,
                converged=converged,
            )
        )
        if stage.synchronize_after:
            synchronize(stage.device)

    return SaddleRefinement(
        converged=stage_results[-1].converged,
        mps_steps=sum(
            result.steps for result in stage_results if result.device == "mps"
        ),
        cpu_steps=sum(
            result.steps for result in stage_results if result.device == "cpu"
        ),
        frames=frames,
        stage_results=tuple(stage_results),
    )


def refine_first_order_saddle(
    atoms: Atoms,
    *,
    path_tangent: np.ndarray,
    optimization_device: Literal["cpu", "mps"],
    mps_calculator,
    cpu_calculator,
    mps_fmax: float,
    cpu_fmax: float,
    max_steps: int,
    cpu_refine: bool = True,
    use_path_mode_hessian: bool = False,
    cancel_event: CancellationToken,
    on_frame: Callable[[SellaFrame], None] | None = None,
) -> SaddleRefinement:
    """Compatibility adapter for the former fixed MPS-to-CPU workflow."""
    stages: list[OptimizationStage] = []
    if optimization_device == "mps":
        stages.append(
            OptimizationStage(
                name="sella_mps",
                device="mps",
                calculator=mps_calculator,
                fmax=mps_fmax,
                max_steps=max_steps,
            )
        )
    if optimization_device == "cpu" or cpu_refine:
        stages.append(
            OptimizationStage(
                name="sella_cpu",
                device="cpu",
                calculator=cpu_calculator,
                fmax=cpu_fmax,
                max_steps=max_steps,
            )
        )
    return refine_saddle_stages(
        atoms,
        path_tangent=path_tangent,
        stages=stages,
        use_path_mode_hessian=use_path_mode_hessian,
        cancel_event=cancel_event,
        on_frame=on_frame,
    )


def _run_sella_stage(
    atoms: Atoms,
    *,
    stage: str,
    fmax: float,
    max_steps: int,
    use_path_mode_hessian: bool,
    path_tangent: np.ndarray,
    cancel_event: CancellationToken,
    frames: list[SellaFrame],
    on_frame: Callable[[SellaFrame], None] | None,
) -> tuple[int, bool]:
    cartesian_tangent = np.asarray(path_tangent, dtype=float).reshape(-1)
    tangent_norm = float(np.linalg.norm(cartesian_tangent))
    if tangent_norm <= 1.0e-12:
        raise ValueError("DMF path tangent is zero; cannot initialize saddle mode")
    sella_options: dict[str, object] = {}
    if use_path_mode_hessian:
        unit_tangent = cartesian_tangent / tangent_norm
        initial_hessian = 0.50 * np.eye(cartesian_tangent.size)
        initial_hessian -= 1.00 * np.outer(unit_tangent, unit_tangent)
        sella_options.update({"eig": False, "H0": initial_hessian})
    optimizer = Sella(
        atoms,
        order=1,
        internal=False,
        v0=None,
        logfile=None,
        trajectory=None,
        **sella_options,
    )
    optimizer.pes.v0 = cartesian_tangent @ optimizer.pes.get_Ufree()
    last_step = -1
    converged = False
    for converged in optimizer.irun(fmax=fmax, steps=max_steps):
        raise_if_cancelled(cancel_event)
        step = int(optimizer.nsteps)
        if step == last_step:
            continue
        forces = np.asarray(atoms.get_forces(), dtype=float)
        frame = SellaFrame(
            stage=stage,
            step=step,
            energy_ev=float(atoms.get_potential_energy()),
            forces_ev_per_a=forces.copy(),
            positions_a=atoms.positions.copy(),
        )
        frames.append(frame)
        if on_frame is not None:
            on_frame(frame)
        raise_if_cancelled(cancel_event)
        last_step = step
    return max(0, int(optimizer.nsteps)), bool(converged)
