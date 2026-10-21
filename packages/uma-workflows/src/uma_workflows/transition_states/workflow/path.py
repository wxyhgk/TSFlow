"""Path-stage adapters for the reusable transition-state workflow."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from ase import Atoms

from uma_workflows.accelerators import synchronize
from uma_workflows.cancellation import CancellationToken, raise_if_cancelled
from uma_workflows.transition_states.algorithm_api import (
    PathOptimizationResult,
    ReactionPath,
    TransitionStateProblem,
)
from uma_workflows.transition_states.algorithm_registry import AlgorithmRegistry
from uma_workflows.transition_states.builtin_algorithms import build_builtin_algorithm_registry
from uma_workflows.transition_states.candidate_ranking import select_best_candidate
from uma_workflows.transition_states.direct_maxflux import (
    DirectMaxFluxConfig,
    DirectMaxFluxPathOptimizer,
)
from uma_workflows.transition_states.fbenm import FbenmConfig, FbenmPathInitializer
from uma_workflows.transition_states.path_analysis import (
    evaluate_path_energies,
    select_path_maximum,
)
from uma_workflows.transition_states.path_bias import PathBias, PathBiasedCalculator
from uma_workflows.transition_states.workflow.endpoints import EndpointStageResult
from uma_workflows.transition_states.workflow.models import (
    Device,
    PathConfig,
    WorkflowProgress,
    WorkflowProgressCallback,
)
from uma_workflows.transition_states.workflow.path_bias import prepare_path_bias


@dataclass(frozen=True)
class PathStageResult:
    problem: TransitionStateProblem
    bias: PathBias
    path: ReactionPath
    optimization: PathOptimizationResult
    sample_coordinates: np.ndarray
    sampled_positions: np.ndarray
    cpu_energies_ev: list[float]
    highest_coordinate: float
    highest_positions: np.ndarray
    has_interior_maximum: bool
    candidates_evaluated: int
    selected_candidate: int
    warnings: tuple[str, ...]
    initializer_algorithm_id: str = "fbenm.correlated"
    initializer_algorithm_version: str = "1"
    optimizer_algorithm_id: str = "dmf.direct_maxflux"
    optimizer_algorithm_version: str = "1"


def optimize_reaction_path(
    reactant: Atoms,
    product: Atoms,
    endpoints: EndpointStageResult,
    *,
    reaction_coordinates,
    config: PathConfig,
    device: Device,
    calculator,
    cpu_calculator,
    cancel_event: CancellationToken,
    progress_callback: WorkflowProgressCallback | None,
    algorithm_registry: AlgorithmRegistry | None = None,
) -> PathStageResult:
    warnings: list[str] = []
    bias, bias_warnings = prepare_path_bias(
        reactant, product, tuple(reaction_coordinates), config
    )
    warnings.extend(bias_warnings)
    problem = TransitionStateProblem(
        reactant=reactant,
        product=product,
        reaction_coordinates=tuple(reaction_coordinates),
    )
    if endpoints is None:
        candidate_count = config.ensemble_size
    else:
        candidate_count = config.ensemble_size if max(
            endpoints.reactant.fragment_count,
            endpoints.product.fragment_count,
        ) > 1 else 1
    registry = algorithm_registry or build_builtin_algorithm_registry()
    initializer = registry.create_initializer(
        config.initializer_id,
        FbenmConfig(
            nmove=max(3, config.nmove),
            ensemble_size=candidate_count,
            validation_points=max(3, config.path_points),
            score_points=min(max(3, config.path_points), 7),
            set_shared_graph_cell=device == "mps",
        )
    )

    def report_fbenm(_stage, step, total, _snapshot) -> None:
        if progress_callback is not None:
            progress_callback(
                WorkflowProgress(
                    stage="fbenm",
                    step=step,
                    max_steps=total,
                    energy_ev=0.0,
                    max_force_ev_per_a=0.0,
                    device=device,
                    positions_a=reactant.positions.copy(),
                    name="FB-ENM path initialization",
                )
            )

    initialization = initializer.generate(
        problem,
        calculator=PathBiasedCalculator(calculator, bias),
        cancel_token=cancel_event,
        progress=report_fbenm,
    )
    warnings.extend(initialization.warnings)
    if any(candidate.score is None for candidate in initialization.candidates):
        initial_candidate = initialization.candidates[0]
    else:
        _, initial_candidate = select_best_candidate(initialization.candidates)
    candidate_total = len(initialization.candidates)
    selected_candidate = int(initial_candidate.candidate_index or 0)
    reactant.set_positions(initial_candidate.path.positions_a[0])
    product.set_positions(initial_candidate.path.positions_a[-1])

    def report_dmf(_stage, step, total, snapshot) -> None:
        if progress_callback is None or snapshot is None:
            return
        progress_callback(
            WorkflowProgress(
                stage="dmf",
                step=step,
                max_steps=total,
                energy_ev=float(snapshot.energy_ev or 0.0),
                max_force_ev_per_a=float(snapshot.max_force_ev_per_a or 0.0),
                device=device,
                positions_a=snapshot.path.positions_a[0].copy(),
                name="Direct MaxFlux path optimization",
                path_energies_ev=tuple(snapshot.path_energies_ev),
            )
        )

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
        candidate=initial_candidate,
    )
    optimization = optimizer.optimize(
        problem,
        initial_candidate,
        calculator=calculator,
        cancel_token=cancel_event,
        progress=report_dmf,
    )
    warnings.extend(optimization.warnings)
    if not optimization.converged:
        warnings.append(f"DMF/IPOPT did not converge: {optimization.optimizer_status}")

    coordinates = np.linspace(0.0, 1.0, config.path_points)
    positions = optimization.path.evaluate(coordinates)
    energies = evaluate_path_energies(
        positions, reactant, cpu_calculator, cancel_event=cancel_event
    )
    highest_index, has_maximum = select_path_maximum(energies)
    if not has_maximum:
        warnings.append(
            "The final CPU UMA path has no interior energy maximum; Sella and "
            "frequency analysis must not use an endpoint as a TS guess."
        )
    return PathStageResult(
        problem=problem,
        bias=bias,
        path=optimization.path,
        optimization=optimization,
        sample_coordinates=coordinates,
        sampled_positions=positions,
        cpu_energies_ev=energies,
        highest_coordinate=float(coordinates[highest_index]),
        highest_positions=positions[highest_index].copy(),
        has_interior_maximum=has_maximum,
        candidates_evaluated=candidate_total,
        selected_candidate=selected_candidate,
        warnings=tuple(warnings),
        initializer_algorithm_id=initializer.metadata.algorithm_id,
        initializer_algorithm_version=initializer.metadata.version,
        optimizer_algorithm_id=optimizer.metadata.algorithm_id,
        optimizer_algorithm_version=optimizer.metadata.version,
    )


def build_supplied_guess_reference_path(
    reactant: Atoms,
    product: Atoms,
    guess: Atoms,
    *,
    reaction_coordinates,
    config: PathConfig,
    cpu_calculator,
    cancel_event: CancellationToken,
) -> PathStageResult:
    """Wrap a supplied TS geometry without launching FB-ENM or DMF."""
    raise_if_cancelled(cancel_event)
    symbols = reactant.get_chemical_symbols()
    if product.get_chemical_symbols() != symbols or guess.get_chemical_symbols() != symbols:
        raise ValueError("supplied path endpoints and guess must share atom ordering")
    sample_coordinates = np.asarray([0.0, 0.5, 1.0], dtype=float)
    sampled_positions = np.stack(
        [reactant.positions.copy(), guess.positions.copy(), product.positions.copy()]
    )
    energies = evaluate_path_energies(
        sampled_positions, reactant, cpu_calculator, cancel_event=cancel_event
    )
    peak_index, has_maximum = select_path_maximum(energies)
    path = ReactionPath(
        positions_a=sampled_positions,
        coordinates=sample_coordinates,
        metadata={"source": "supplied_transition_state_guess"},
    )
    optimization = PathOptimizationResult(
        path=path,
        converged=False,
        iterations=0,
        optimizer_status="supplied_transition_state_guess",
        warnings=("No FB-ENM or DMF optimization was run for the supplied guess.",),
    )
    bias, _ = prepare_path_bias(reactant, product, tuple(reaction_coordinates), config)
    return PathStageResult(
        problem=TransitionStateProblem(reactant, product, tuple(reaction_coordinates)),
        bias=bias,
        path=path,
        optimization=optimization,
        sample_coordinates=sample_coordinates,
        sampled_positions=sampled_positions,
        cpu_energies_ev=energies,
        highest_coordinate=float(sample_coordinates[peak_index]),
        highest_positions=sampled_positions[peak_index].copy(),
        has_interior_maximum=has_maximum,
        candidates_evaluated=0,
        selected_candidate=0,
        warnings=optimization.warnings,
    )
