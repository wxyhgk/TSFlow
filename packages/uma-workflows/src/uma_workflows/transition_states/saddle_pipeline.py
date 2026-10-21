from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms

from uma_workflows.cancellation import CancellationToken, raise_if_cancelled
from uma_workflows.optimization import (
    SaddleRefinement,
    SellaFrame,
    refine_first_order_saddle,
)
from uma_workflows.transition_states.algorithm_api import TransitionStateProblem
from uma_workflows.transition_states.corridor_refinement import (
    refine_corridor_seed_then_unbiased,
)
from uma_workflows.transition_states.path_bias import PathBias
from uma_workflows.transition_states.relaxed_scan import (
    prepare_relaxed_scan_seed,
)
from uma_workflows.transition_states.reaction_analysis import reaction_active_region


class ParametricPathLike(Protocol):
    def evaluate(
        self,
        coordinates,
        *,
        derivative_order: int = 0,
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class SaddlePipelineConfig:
    optimization_device: Literal["cpu", "mps"]
    mps_fmax: float
    cpu_fmax: float
    max_steps: int
    cpu_refine: bool = True
    use_path_mode_hessian: bool = False
    active_region_hops: int | None = None
    relaxed_scan: bool = False
    relaxed_scan_progress: float = 0.5
    relaxed_scan_fmax: float = 0.05
    relaxed_scan_max_steps: int = 100


@dataclass(frozen=True)
class SaddlePipelineResult:
    atoms: Atoms
    refinement: SaddleRefinement
    active_atom_indices: tuple[int, ...]
    relaxed_scan_performed: bool
    warnings: tuple[str, ...]


def refine_path_saddle(
    problem: TransitionStateProblem,
    candidate: Atoms,
    *,
    path: ParametricPathLike,
    highest_coordinate: float,
    path_bias: PathBias,
    mps_calculator,
    cpu_calculator,
    config: SaddlePipelineConfig,
    cancel_token: CancellationToken,
    on_frame: Callable[[SellaFrame], None] | None = None,
) -> SaddlePipelineResult:
    """Prepare and refine a first-order saddle from a parametric path maximum."""
    raise_if_cancelled(cancel_token)
    atoms = candidate.copy()
    tangent = path.evaluate(
        [highest_coordinate],
        derivative_order=1,
    )[0].reshape(-1)
    warnings: list[str] = []
    relaxed_scan_performed = False
    if config.relaxed_scan and path_bias.synchronized_coordinates:
        tangent, scan_warnings = prepare_relaxed_scan_seed(
            atoms,
            path=path,
            path_bias=path_bias,
            highest_coordinate=highest_coordinate,
            requested_progress=config.relaxed_scan_progress,
            optimization_device=config.optimization_device,
            mps_calculator=mps_calculator,
            cpu_calculator=cpu_calculator,
            fmax=config.relaxed_scan_fmax,
            max_steps=config.relaxed_scan_max_steps,
            cancel_token=cancel_token,
        )
        relaxed_scan_performed = True
        warnings.extend(scan_warnings)

    corridor_seed = (
        bool(path_bias.synchronized_coordinates)
        and not relaxed_scan_performed
    )
    if corridor_seed:
        warnings.append(
            "Sella first used a bond-aware reaction-corridor seed, then removed "
            "the bias for an explicit CPU saddle refinement."
        )

    active_indices = tuple(range(len(atoms)))
    if config.active_region_hops is not None:
        active = reaction_active_region(
            problem.reactant,
            problem.product,
            hops=config.active_region_hops,
            expected_coordinates=problem.reaction_coordinates,
        )
        if not active:
            active = set(range(len(atoms)))
            warnings.append(
                "The requested Sella active region was empty; refinement fell "
                "back to the full molecular system."
            )
        active_indices = tuple(sorted(active))
        fixed = sorted(set(range(len(atoms))) - active)
        atoms.set_constraint(FixAtoms(indices=fixed))
        warnings.append(
            f"Sella refinement used a {len(active)}-atom active region; "
            f"{len(fixed)} spectator atoms were fixed."
        )
    try:
        if corridor_seed:
            refinement = refine_corridor_seed_then_unbiased(
                atoms,
                tangent=tangent,
                path_bias=path_bias,
                optimization_device=config.optimization_device,
                mps_calculator=mps_calculator,
                cpu_calculator=cpu_calculator,
                mps_fmax=config.mps_fmax,
                cpu_fmax=config.cpu_fmax,
                max_steps=config.max_steps,
                use_path_mode_hessian=config.use_path_mode_hessian,
                cancel_token=cancel_token,
                on_frame=on_frame,
            )
        else:
            refinement = refine_first_order_saddle(
                atoms,
                path_tangent=tangent,
                optimization_device=config.optimization_device,
                mps_calculator=mps_calculator,
                cpu_calculator=cpu_calculator,
                mps_fmax=config.mps_fmax,
                cpu_fmax=config.cpu_fmax,
                max_steps=config.max_steps,
                cpu_refine=config.cpu_refine,
                use_path_mode_hessian=config.use_path_mode_hessian,
                cancel_event=cancel_token,
                on_frame=on_frame,
            )
    finally:
        atoms.set_constraint()
    if (
        config.optimization_device == "mps"
        and not config.cpu_refine
        and not corridor_seed
    ):
        warnings.append(
            "Iterative CPU Sella refinement was skipped for fast screening; "
            "the caller must perform an unbiased CPU energy/force validation."
        )
    elif not refinement.converged:
        warnings.append(
            "Sella did not converge the CPU first-order saddle within the "
            "requested step limit."
        )
    atoms.calc = None
    return SaddlePipelineResult(
        atoms=atoms,
        refinement=refinement,
        active_atom_indices=active_indices,
        relaxed_scan_performed=relaxed_scan_performed,
        warnings=tuple(warnings),
    )
