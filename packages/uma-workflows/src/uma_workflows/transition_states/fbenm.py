from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dmf import interpolate_fbenm

from uma_workflows.cancellation import CancellationToken, raise_if_cancelled
from uma_workflows.geometry import set_nonperiodic_graph_cell
from uma_workflows.transition_states.algorithm_api import (
    AlgorithmMetadata,
    PathCandidate,
    PathParameterization,
    PathInitializationResult,
    PathProgressCallback,
    ReactionPath,
    TransitionStateProblem,
)
from uma_workflows.transition_states.candidate_ranking import (
    coarse_path_barrier_score,
)
from uma_workflows.transition_states.path_geometry import validate_path_geometry
from uma_workflows.transition_states.reference_paths import (
    generate_directional_reference_paths,
)


@dataclass(frozen=True)
class FbenmConfig:
    nmove: int
    ensemble_size: int = 1
    validation_points: int = 101
    score_points: int = 7
    set_shared_graph_cell: bool = False

    def __post_init__(self) -> None:
        if self.nmove < 3:
            raise ValueError("FB-ENM nmove must be at least 3")
        if self.ensemble_size < 1:
            raise ValueError("FB-ENM ensemble size must be positive")
        if self.validation_points < 3 or self.score_points < 3:
            raise ValueError("FB-ENM sampling requires at least three points")


class FbenmPathInitializer:
    metadata = AlgorithmMetadata(
        algorithm_id="fbenm.correlated",
        version="1",
        display_name="Correlated FB-ENM",
    )

    def __init__(self, config: FbenmConfig) -> None:
        self.config = config

    def generate(
        self,
        problem: TransitionStateProblem,
        *,
        calculator,
        cancel_token: CancellationToken,
        progress: PathProgressCallback | None = None,
    ) -> PathInitializationResult:
        ensemble = generate_directional_reference_paths(
            problem.reactant,
            problem.product,
            ensemble_size=self.config.ensemble_size,
        )
        valid: list[PathCandidate] = []
        warnings = list(ensemble.warnings)
        symbols = problem.reactant.get_chemical_symbols()
        validation_coordinates = np.linspace(
            0.0,
            1.0,
            self.config.validation_points,
        )
        score_coordinates = np.linspace(0.0, 1.0, self.config.score_points)
        with tempfile.TemporaryDirectory(prefix="uma-fbenm-") as temp_dir:
            for candidate_index, reference_images in enumerate(ensemble.candidates):
                raise_if_cancelled(cancel_token)
                if self.config.set_shared_graph_cell:
                    set_nonperiodic_graph_cell(reference_images)
                if progress is not None:
                    progress(
                        "fbenm",
                        candidate_index,
                        len(ensemble.candidates),
                        None,
                    )
                raise_if_cancelled(cancel_token)
                try:
                    raw_path = interpolate_fbenm(
                        reference_images,
                        nmove=self.config.nmove,
                        correlated=True,
                        fbenm_only_endpoints=True,
                        output_file=str(
                            Path(temp_dir) / f"fbenm-ipopt-{candidate_index}.out"
                        ),
                        dmf_options={
                            "remove_rotation_and_translation": True,
                        },
                        ipopt_options={"print_level": 0},
                    )
                    sampled = raw_path.get_positions(t=validation_coordinates)
                    validate_path_geometry(
                        sampled,
                        symbols,
                        stage=f"FB-ENM candidate {candidate_index}",
                    )
                    score = coarse_path_barrier_score(
                        raw_path.get_positions(t=score_coordinates),
                        reference_images[0],
                        calculator,
                    )
                    valid.append(
                        PathCandidate(
                            path=ReactionPath(
                                positions_a=sampled,
                                coordinates=validation_coordinates,
                                metadata={
                                    "algorithm_id": self.metadata.algorithm_id,
                                    "candidate_index": candidate_index,
                                },
                            ),
                            score=score,
                            source=self.metadata.algorithm_id,
                            parameterization=PathParameterization.from_array(
                                "pydmf.bspline",
                                np.asarray(raw_path.coefs, dtype=float),
                            ),
                            metadata={
                                "candidate_index": candidate_index,
                                "reference_images": reference_images,
                            },
                        )
                    )
                except (RuntimeError, ValueError, FloatingPointError) as exc:
                    warnings.append(
                        f"FB-ENM candidate {candidate_index} was rejected: {exc}"
                    )
        if not valid:
            details = " | ".join(warnings)
            raise RuntimeError(
                "all directional FB-ENM path candidates failed"
                + (f": {details}" if details else "")
            )
        return PathInitializationResult(
            candidates=tuple(valid),
            warnings=tuple(warnings),
        )
