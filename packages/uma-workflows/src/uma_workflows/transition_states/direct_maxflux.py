from __future__ import annotations

import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from uma_workflows.cancellation import CancellationToken, raise_if_cancelled
from uma_workflows.transition_states.algorithm_api import (
    AlgorithmMetadata,
    PathCandidate,
    PathOptimizationResult,
    PathProgressCallback,
    PathProgressSnapshot,
    ReactionPath,
    TransitionStateProblem,
)
from uma_workflows.transition_states.direct_maxflux_runtime import (
    ReportingDirectMaxFlux,
    validate_finite_dmf_state,
)
from uma_workflows.transition_states.path_bias import PathBias
from uma_workflows.transition_states.path_geometry import validate_path_geometry


@dataclass(frozen=True)
class DirectMaxFluxConfig:
    nmove: int
    max_iterations: int
    convergence: str | float
    path_bias: PathBias
    update_teval: bool = True
    accept_every_trial_step: bool = False
    result_points: int = 101
    path_bias_samples: int = 41
    after_force_evaluation: Callable[[], None] | None = None

    def __post_init__(self) -> None:
        if self.nmove < 3:
            raise ValueError("Direct MaxFlux nmove must be at least 3")
        if self.max_iterations < 1:
            raise ValueError("Direct MaxFlux max_iterations must be positive")
        if self.result_points < 3 or self.path_bias_samples < 3:
            raise ValueError("Direct MaxFlux path sampling requires at least three points")


class DirectMaxFluxPathOptimizer:
    metadata = AlgorithmMetadata(
        algorithm_id="dmf.direct_maxflux",
        version="1",
        display_name="Direct MaxFlux",
    )

    def __init__(self, config: DirectMaxFluxConfig) -> None:
        self.config = config

    def optimize(
        self,
        problem: TransitionStateProblem,
        initial_path: PathCandidate,
        *,
        calculator,
        cancel_token: CancellationToken,
        progress: PathProgressCallback | None = None,
    ) -> PathOptimizationResult:
        reference_images = self._reference_images(problem, initial_path)
        coefficients = self._initial_coefficients(initial_path)

        def report_iteration(
            iteration: int,
            path: ReportingDirectMaxFlux,
        ) -> None:
            raise_if_cancelled(cancel_token)
            validate_finite_dmf_state(path)
            if progress is None or not path.history.images_tmax:
                return
            energies = tuple(float(value + path.e0) for value in path.energies)
            forces = np.asarray(path.forces, dtype=float)
            progress(
                "dmf",
                min(iteration, self.config.max_iterations),
                self.config.max_iterations,
                PathProgressSnapshot(
                    path=ReactionPath(
                        positions_a=np.asarray(
                            [path.history.images_tmax[-1].positions],
                            dtype=float,
                        ),
                        coordinates=np.asarray([0.5]),
                    ),
                    energy_ev=max(energies),
                    path_energies_ev=energies,
                    max_force_ev_per_a=self._max_force(forces),
                ),
            )
            raise_if_cancelled(cancel_token)

        with tempfile.TemporaryDirectory(prefix="uma-dmf-") as temp_dir:
            native_path = ReportingDirectMaxFlux(
                reference_images,
                coefs=coefficients,
                nmove=self.config.nmove,
                update_teval=self.config.update_teval,
                remove_rotation_and_translation=True,
                calc_factory=lambda _index: calculator,
                cancel_token=cancel_token,
                iteration_callback=report_iteration,
                path_bias=self.config.path_bias,
                path_bias_samples=self.config.path_bias_samples,
            )
            native_path.add_ipopt_options(
                {
                    "output_file": str(Path(temp_dir) / "dmf-ipopt.out"),
                    "max_iter": self.config.max_iterations,
                    "print_level": 0,
                    "accept_every_trial_step": (
                        "yes" if self.config.accept_every_trial_step else "no"
                    ),
                }
            )
            _, solve_info = native_path.solve(tol=self.config.convergence)
            raise_if_cancelled(cancel_token)
            native_path.get_forces()
            if self.config.after_force_evaluation is not None:
                self.config.after_force_evaluation()

        validate_finite_dmf_state(native_path)
        result_coordinates = np.linspace(0.0, 1.0, self.config.result_points)
        result_positions = native_path.get_positions(t=result_coordinates)
        result_tangents = native_path.get_positions(t=result_coordinates, nu=1)
        validate_path_geometry(
            result_positions,
            problem.reactant.get_chemical_symbols(),
            stage="DMF",
        )
        status_code = int(solve_info.get("status", -999))
        optimizer_status = solve_info.get("status_msg", "unknown IPOPT status")
        if isinstance(optimizer_status, bytes):
            optimizer_status = optimizer_status.decode(
                "utf-8",
                errors="replace",
            )
        warnings: list[str] = []
        if native_path.teval_fallback_reason is not None:
            warnings.append(
                "PyDMF adaptive energy-point updates became ill-conditioned; "
                "the calculation continued with the last valid fixed evaluation grid."
            )
        iterations = max(0, len(native_path.history.images_tmax) - 1)
        return PathOptimizationResult(
            path=ReactionPath(
                positions_a=result_positions,
                coordinates=result_coordinates,
                tangents_a=result_tangents,
                metadata={"algorithm_id": self.metadata.algorithm_id},
            ),
            converged=status_code in {0, 1},
            iterations=iterations,
            optimizer_status=str(optimizer_status),
            status_code=status_code,
            adaptive_sampling_enabled=bool(native_path.update_teval),
            warnings=tuple(warnings),
            metadata={
                "teval_fallback_reason": native_path.teval_fallback_reason,
            },
        )

    @staticmethod
    def _initial_coefficients(candidate: PathCandidate) -> np.ndarray | None:
        parameterization = candidate.parameterization
        if parameterization is None:
            return None
        if parameterization.kind != "pydmf.bspline":
            raise ValueError(
                "Direct MaxFlux does not support initial path parameterization "
                f"{parameterization.kind!r}"
            )
        return parameterization.as_array().copy()

    @staticmethod
    def _reference_images(
        problem: TransitionStateProblem,
        candidate: PathCandidate,
    ) -> list:
        positions = np.asarray(candidate.path.positions_a, dtype=float)
        if positions.shape[1] != len(problem.reactant):
            raise ValueError("initial path atom count does not match TS problem")
        images = []
        for coordinates in positions:
            image = problem.reactant.copy()
            image.set_positions(coordinates)
            images.append(image)
        return images

    @staticmethod
    def _max_force(forces) -> float:
        values = np.asarray(forces, dtype=float).reshape(-1, 3)
        return float(np.linalg.norm(values, axis=1).max(initial=0.0))
