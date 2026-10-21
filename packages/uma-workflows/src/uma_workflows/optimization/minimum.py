from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

from ase import Atoms
from sella import Sella

from uma_workflows.accelerators import synchronize
from uma_workflows.cancellation import CancellationToken, raise_if_cancelled
from uma_workflows.optimization.models import (
    MinimumRefinement,
    OptimizationStage,
    OptimizationStageResult,
)


MinimumStepCallback = Callable[[OptimizationStage, int, int, Atoms], None]


def optimize_minimum_stages(
    atoms: Atoms,
    stages: Sequence[OptimizationStage],
    *,
    cancel_event: CancellationToken,
    on_step: MinimumStepCallback | None = None,
) -> MinimumRefinement:
    """Run an arbitrary ordered sequence of Sella order-0 stages."""
    if not stages:
        raise ValueError("minimum optimization requires at least one stage")
    stage_results: list[OptimizationStageResult] = []
    completed_steps = 0
    total_planned_steps = sum(stage.max_steps for stage in stages)
    for stage in stages:
        atoms.calc = stage.calculator

        def report(stage_step: int, current_atoms: Atoms) -> None:
            if on_step is not None:
                on_step(
                    stage,
                    completed_steps + stage_step,
                    total_planned_steps,
                    current_atoms,
                )

        steps, converged = _run_minimum_stage(
            atoms,
            fmax=stage.fmax,
            max_steps=stage.max_steps,
            cancel_event=cancel_event,
            on_step=report,
        )
        stage_results.append(
            OptimizationStageResult(
                name=stage.name,
                device=stage.device,
                steps=steps,
                converged=converged,
            )
        )
        completed_steps += steps
        if stage.synchronize_after:
            synchronize(stage.device)

    return MinimumRefinement(
        converged=stage_results[-1].converged,
        mps_steps=sum(
            result.steps for result in stage_results if result.device == "mps"
        ),
        cpu_steps=sum(
            result.steps for result in stage_results if result.device == "cpu"
        ),
        stage_results=tuple(stage_results),
    )


def relax_constrained_minimum(
    atoms: Atoms,
    *,
    optimization_device: Literal["cpu", "mps"],
    mps_calculator,
    cpu_calculator,
    fmax: float,
    max_steps: int,
    cancel_event: CancellationToken,
) -> MinimumRefinement:
    """Compatibility adapter for the former fixed MPS-to-CPU workflow."""
    stages: list[OptimizationStage] = []
    if optimization_device == "mps":
        stages.append(
            OptimizationStage(
                name="minimum_mps",
                device="mps",
                calculator=mps_calculator,
                fmax=fmax,
                max_steps=max_steps,
            )
        )
    stages.append(
        OptimizationStage(
            name="minimum_cpu",
            device="cpu",
            calculator=cpu_calculator,
            fmax=fmax,
            max_steps=max_steps,
        )
    )
    return optimize_minimum_stages(
        atoms,
        stages,
        cancel_event=cancel_event,
    )


def _run_minimum_stage(
    atoms: Atoms,
    *,
    fmax: float,
    max_steps: int,
    cancel_event: CancellationToken,
    on_step: Callable[[int, Atoms], None] | None = None,
) -> tuple[int, bool]:
    raise_if_cancelled(cancel_event)
    optimizer = Sella(
        atoms,
        order=0,
        internal=False,
        logfile=None,
        trajectory=None,
    )
    last_step = -1
    converged = False
    for converged in optimizer.irun(fmax=fmax, steps=max_steps):
        raise_if_cancelled(cancel_event)
        step = int(optimizer.nsteps)
        if step == last_step:
            continue
        if on_step is not None:
            on_step(step, atoms)
        raise_if_cancelled(cancel_event)
        last_step = step
    return max(0, int(optimizer.nsteps)), bool(converged)
