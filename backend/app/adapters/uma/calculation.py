"""Single-point and geometry-optimization execution."""

import time
from math import ceil
from pathlib import Path
from threading import Event
from typing import Any, Callable, Literal

from uma_workflows.runtime import synchronize

from app.adapters.uma.structures import (
    max_force,
    prepare_nonperiodic_graph_box,
    raise_if_cancelled,
    structure_from_atoms,
    to_atoms,
)
from app.contracts.jobs import (
    CalculationProgress,
    CalculationResult,
    JobCreate,
    OptimizationFrame,
)
from app.adapters.uma.geometric import GeometricFrame, optimize_with_geometric


MAX_TRAJECTORY_FRAMES = 200


def run_energy_or_optimization(
    executor: Any,
    request: JobCreate,
    checkpoint: Path,
    cancel_event: Event,
    progress_callback: Callable[[CalculationProgress], None] | None,
) -> CalculationResult:
    started = time.perf_counter()
    raise_if_cancelled(cancel_event)
    atoms = to_atoms(request)
    warnings: list[str] = []
    initial_energy: float | None = None
    converged: bool | None = None
    steps = 0
    device: Literal["cpu", "mps"] | None = None
    trajectory: list[OptimizationFrame] = []

    if request.calculation == "geometry_optimization":
        requested_device = request.parameters.optimization_device
        device = (executor._resolve_optimization_device(requested_device)
                  if requested_device else executor._resolve_optimization_device())
        if (requested_device or executor._optimization_device) != "cpu" and device == "cpu":
            warnings.append("MPS unavailable; geometry optimization fell back to CPU.")
        translation = prepare_nonperiodic_graph_box(atoms) if device == "mps" else None
        if device == "mps":
            warnings.append(
                "Experimental MPS acceleration was used for optimization; "
                "final energy and forces were recomputed on CPU."
            )
        atoms.calc = executor._calculator(checkpoint, device, request.task_name)
        initial_energy = float(atoms.get_potential_energy())
        synchronize(device)
        sample_interval = max(
            1,
            ceil(request.parameters.max_steps / MAX_TRAJECTORY_FRAMES),
        )
        last_reported = -1

        def report(frame: GeometricFrame) -> None:
            nonlocal last_reported
            if frame.step == last_reported:
                return
            atoms.set_positions(frame.positions_a)
            if progress_callback is not None:
                progress_callback(
                    CalculationProgress(
                        step=min(frame.step, request.parameters.max_steps),
                        max_steps=request.parameters.max_steps,
                        energy_ev=frame.energy_ev,
                        max_force_ev_per_a=max_force(frame.forces_ev_per_a),
                        optimization_device=device,
                        current_structure=structure_from_atoms(
                            request,
                            atoms,
                            translation,
                        ),
                    )
                )
            last_reported = frame.step

        result = optimize_with_geometric(
            atoms,
            atoms.calc,
            fmax_ev_per_a=request.parameters.fmax,
            max_steps=request.parameters.max_steps,
            check_cancelled=lambda: raise_if_cancelled(cancel_event),
            on_evaluation=report,
        )
        steps = max(0, len(result.frames) - 1)
        final_force = max_force(result.final.forces_ev_per_a)
        for frame in result.frames:
            if (
                frame.step != 0
                and frame.step % sample_interval != 0
                and frame is not result.final
            ):
                continue
            atoms.set_positions(frame.positions_a)
            trajectory.append(
                OptimizationFrame(
                    step=frame.step,
                    energy_ev=frame.energy_ev,
                    max_force_ev_per_a=max_force(frame.forces_ev_per_a),
                    structure=structure_from_atoms(request, atoms, translation),
                )
            )
        atoms.set_positions(result.final.positions_a)
        if progress_callback is not None and last_reported != result.final.step:
            progress_callback(
                CalculationProgress(
                    step=steps,
                    max_steps=request.parameters.max_steps,
                    energy_ev=result.final.energy_ev,
                    max_force_ev_per_a=final_force,
                    optimization_device=device,
                    current_structure=structure_from_atoms(request, atoms, translation),
                )
            )
        synchronize(device)
        if translation is not None:
            atoms.positions -= translation

    raise_if_cancelled(cancel_event)
    atoms.calc = executor._calculator(checkpoint, "cpu", request.task_name)
    energy = float(atoms.get_potential_energy())
    forces = atoms.get_forces()
    force = max_force(forces)
    if request.calculation == "geometry_optimization":
        converged = force <= request.parameters.fmax
    return CalculationResult(
        energy_ev=energy,
        initial_energy_ev=initial_energy,
        forces_ev_per_a=[tuple(float(value) for value in row) for row in forces],
        max_force_ev_per_a=force,
        final_structure=structure_from_atoms(request, atoms),
        converged=converged,
        optimization_steps=steps,
        optimization_device=device,
        optimizer=(
            "geomeTRIC-TRIC"
            if request.calculation == "geometry_optimization"
            else None
        ),
        elapsed_seconds=time.perf_counter() - started,
        warnings=warnings,
        trajectory=trajectory,
    )
