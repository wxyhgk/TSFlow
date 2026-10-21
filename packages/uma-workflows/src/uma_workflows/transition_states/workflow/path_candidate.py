from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uma_workflows.accelerators import synchronize
from uma_workflows.cancellation import CancellationToken
from uma_workflows.transition_states.algorithm_api import (
    PathCandidate,
    PathOptimizationResult,
    PathProgressCallback,
    TransitionStateProblem,
)
from uma_workflows.transition_states.algorithm_registry import AlgorithmRegistry
from uma_workflows.transition_states.builtin_algorithms import (
    build_builtin_algorithm_registry,
)
from uma_workflows.transition_states.candidate_ranking import (
    CoordinatedPathScore,
    coordinated_path_geometry_score,
)
from uma_workflows.transition_states.direct_maxflux import DirectMaxFluxConfig
from uma_workflows.transition_states.path_analysis import (
    evaluate_path_energies,
    select_path_maximum,
)
from uma_workflows.transition_states.path_bias import PathBias
from uma_workflows.transition_states.workflow.models import Device, PathConfig
from uma_workflows.transition_states.workflow.path_bias import prepare_path_bias


@dataclass(frozen=True)
class OptimizedPathCandidateResult:
    problem: TransitionStateProblem
    bias: PathBias
    candidate_index: int
    optimization: PathOptimizationResult
    sample_coordinates: np.ndarray
    sampled_positions: np.ndarray
    cpu_energies_ev: tuple[float, ...]
    highest_coordinate: float
    highest_positions: np.ndarray
    has_interior_maximum: bool
    score: CoordinatedPathScore
    optimizer_algorithm_id: str
    optimizer_algorithm_version: str
    warnings: tuple[str, ...] = ()


def optimize_path_candidate(
    problem: TransitionStateProblem,
    candidate: PathCandidate,
    *,
    config: PathConfig,
    device: Device,
    calculator,
    cpu_calculator,
    cancel_event: CancellationToken,
    progress: PathProgressCallback | None = None,
    algorithm_registry: AlgorithmRegistry | None = None,
) -> OptimizedPathCandidateResult:
    bias, bias_warnings = prepare_path_bias(
        problem.reactant,
        problem.product,
        problem.reaction_coordinates,
        config,
    )
    aligned_reactant = problem.reactant.copy()
    aligned_product = problem.product.copy()
    aligned_reactant.set_positions(candidate.path.positions_a[0])
    aligned_product.set_positions(candidate.path.positions_a[-1])
    aligned_problem = TransitionStateProblem(
        reactant=aligned_reactant,
        product=aligned_product,
        reaction_coordinates=problem.reaction_coordinates,
        metadata=problem.metadata,
    )
    registry = algorithm_registry or build_builtin_algorithm_registry()
    optimizer = registry.create_optimizer(
        config.optimizer_id,
        DirectMaxFluxConfig(
            nmove=config.nmove,
            max_iterations=config.max_iterations,
            convergence=config.convergence,
            path_bias=bias,
            update_teval=config.update_teval,
            accept_every_trial_step=config.accept_every_trial_step,
            after_force_evaluation=lambda: synchronize(device),
        ),
        candidate=candidate,
    )
    optimization = optimizer.optimize(
        aligned_problem,
        candidate,
        calculator=calculator,
        cancel_token=cancel_event,
        progress=progress,
    )
    sample_coordinates = np.linspace(0.0, 1.0, config.path_points)
    sampled_positions = optimization.path.evaluate(sample_coordinates)
    energies = evaluate_path_energies(
        sampled_positions,
        aligned_reactant,
        cpu_calculator,
        cancel_event=cancel_event,
    )
    highest_index, has_maximum = select_path_maximum(energies)
    score = coordinated_path_geometry_score(
        sampled_positions,
        energies,
        reactant=aligned_reactant,
        product=aligned_product,
        reaction_coordinates=problem.reaction_coordinates,
    )
    candidate_index = int(
        candidate.candidate_index if candidate.candidate_index is not None else 0
    )
    warnings = [*bias_warnings, *optimization.warnings]
    if not optimization.converged:
        warnings.append(
            "Path optimization did not converge: "
            f"{optimization.optimizer_status}"
        )
    if not has_maximum:
        warnings.append(
            "The unbiased CPU path has no interior energy maximum; no endpoint "
            "may be promoted to a transition-state seed."
        )
    return OptimizedPathCandidateResult(
        problem=aligned_problem,
        bias=bias,
        candidate_index=candidate_index,
        optimization=optimization,
        sample_coordinates=sample_coordinates,
        sampled_positions=sampled_positions,
        cpu_energies_ev=tuple(float(value) for value in energies),
        highest_coordinate=float(sample_coordinates[highest_index]),
        highest_positions=sampled_positions[highest_index].copy(),
        has_interior_maximum=has_maximum,
        score=score,
        optimizer_algorithm_id=optimizer.metadata.algorithm_id,
        optimizer_algorithm_version=optimizer.metadata.version,
        warnings=tuple(warnings),
    )
