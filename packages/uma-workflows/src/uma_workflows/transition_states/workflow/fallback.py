from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from ase import Atoms

from uma_workflows.cancellation import CancellationToken, raise_if_cancelled
from uma_workflows.transition_states.screening import TransitionStateScreeningResult
from uma_workflows.transition_states.workflow.models import (
    Device,
    SaddleConfig,
    ScreeningConfig,
    WorkflowProgressCallback,
)
from uma_workflows.transition_states.workflow.path import PathStageResult
from uma_workflows.transition_states.workflow.saddle import (
    SaddleStageResult,
    refine_saddle,
)
from uma_workflows.transition_states.workflow.screening import screen_saddle


FallbackStage = Literal[
    "initial_refinement",
    "path_curvature_screen",
    "alternative_seed_refinement",
    "collective_coordinate_scan",
]


@dataclass(frozen=True)
class SaddleFallbackAttempt:
    stage: FallbackStage
    attempt: int
    path_coordinate: float | None
    failure_code: str | None
    local_saddle_valid: bool
    irc_eligible: bool


@dataclass(frozen=True)
class SaddleFallbackResult:
    saddle: SaddleStageResult
    screening: TransitionStateScreeningResult
    attempts: tuple[SaddleFallbackAttempt, ...]
    performed: bool
    recovered: bool
    warnings: tuple[str, ...]


def recover_collapsed_saddle(
    path: PathStageResult,
    initial_saddle: SaddleStageResult,
    initial_screening: TransitionStateScreeningResult,
    *,
    config: SaddleConfig,
    screening_config: ScreeningConfig,
    device: Device,
    mps_calculator,
    cpu_calculator,
    analytic_hessian_calculator,
    cancel_event: CancellationToken,
    progress_callback: WorkflowProgressCallback | None,
) -> SaddleFallbackResult:
    """Run a bounded recovery sequence after refinement collapses to a minimum."""
    attempts = [
        _attempt("initial_refinement", 0, path.highest_coordinate, initial_screening)
    ]
    if (
        not config.automatic_fallback
        or initial_screening.failure_code != "SADDLE_COLLAPSED_TO_MINIMUM"
    ):
        return SaddleFallbackResult(
            initial_saddle,
            initial_screening,
            tuple(attempts),
            False,
            False,
            (),
        )

    warnings = [
        "The default saddle refinement collapsed to a minimum; bounded automatic "
        "fallback was started."
    ]
    probe_config = replace(screening_config, force_full_hessian=True)
    candidate_probes: list[
        tuple[float, float, int, TransitionStateScreeningResult]
    ] = []
    for probe_number, index in enumerate(
        _nearby_interior_indices(path, config.fallback_nearby_frames),
        start=1,
    ):
        raise_if_cancelled(cancel_event)
        coordinate = float(path.sample_coordinates[index])
        atoms = path.problem.reactant.copy()
        atoms.set_positions(path.sampled_positions[index])
        probe_saddle = SaddleStageResult(atoms, None, (), ())
        probe = screen_saddle(
            path,
            probe_saddle,
            config=probe_config,
            saddle_config=config,
            cpu_calculator=cpu_calculator,
            analytic_hessian_calculator=analytic_hessian_calculator,
            cancel_event=cancel_event,
            progress_callback=progress_callback,
        )
        attempts.append(
            _attempt("path_curvature_screen", probe_number, coordinate, probe)
        )
        frequency = probe.frequency_analysis
        has_one_negative_mode = (
            frequency is not None
            and probe.frequency_scope == "full"
            and len(frequency.imaginary_frequencies_cm1) == 1
            and probe.hessian_quality_valid
        )
        if has_one_negative_mode:
            # The user's reaction hypothesis ranks candidates but must not
            # discard a genuine negative-curvature seed for an unexpected
            # mechanism.
            overlap = (
                probe.mode_reaction_overlap.score
                if probe.mode_reaction_overlap is not None
                else 0.0
            )
            distance = abs(coordinate - path.highest_coordinate)
            candidate_probes.append((-overlap, distance, index, probe))

    candidate_probes.sort(key=lambda item: item[:2])
    for candidate_number, (_, _, index, _) in enumerate(
        candidate_probes[: config.fallback_max_candidates],
        start=1,
    ):
        coordinate = float(path.sample_coordinates[index])
        candidate_path = replace(
            path,
            highest_coordinate=coordinate,
            highest_positions=path.sampled_positions[index].copy(),
        )
        atoms = path.problem.reactant.copy()
        atoms.set_positions(path.sampled_positions[index])
        saddle = refine_saddle(
            candidate_path,
            atoms,
            config=replace(config, relaxed_scan=False),
            device=device,
            mps_calculator=mps_calculator,
            cpu_calculator=cpu_calculator,
            analytic_hessian_calculator=analytic_hessian_calculator,
            cancel_event=cancel_event,
            progress_callback=progress_callback,
        )
        screening = screen_saddle(
            candidate_path,
            saddle,
            config=screening_config,
            saddle_config=config,
            cpu_calculator=cpu_calculator,
            analytic_hessian_calculator=analytic_hessian_calculator,
            cancel_event=cancel_event,
            progress_callback=progress_callback,
        )
        attempts.append(
            _attempt(
                "alternative_seed_refinement",
                candidate_number,
                coordinate,
                screening,
            )
        )
        if screening.irc_eligible:
            warnings.append(
                f"Automatic fallback recovered an IRC-eligible saddle from path "
                f"coordinate {coordinate:.4f}."
            )
            return SaddleFallbackResult(
                saddle,
                screening,
                tuple(attempts),
                True,
                True,
                tuple(warnings),
            )

    if config.relaxed_scan and path.bias.synchronized_coordinates:
        atoms = path.problem.reactant.copy()
        atoms.set_positions(path.highest_positions)
        saddle = refine_saddle(
            path,
            atoms,
            config=replace(config, relaxed_scan=True),
            device=device,
            mps_calculator=mps_calculator,
            cpu_calculator=cpu_calculator,
            analytic_hessian_calculator=analytic_hessian_calculator,
            cancel_event=cancel_event,
            progress_callback=progress_callback,
        )
        screening = screen_saddle(
            path,
            saddle,
            config=screening_config,
            saddle_config=config,
            cpu_calculator=cpu_calculator,
            analytic_hessian_calculator=analytic_hessian_calculator,
            cancel_event=cancel_event,
            progress_callback=progress_callback,
        )
        attempts.append(
            _attempt(
                "collective_coordinate_scan",
                1,
                path.highest_coordinate,
                screening,
            )
        )
        if screening.irc_eligible:
            warnings.append(
                "Automatic fallback recovered an IRC-eligible saddle through the "
                "collective-coordinate relaxed scan."
            )
            return SaddleFallbackResult(
                saddle,
                screening,
                tuple(attempts),
                True,
                True,
                tuple(warnings),
            )

    warnings.append(
        "Bounded automatic saddle fallback was exhausted without recovering an "
        "IRC-eligible first-order saddle."
    )
    return SaddleFallbackResult(
        initial_saddle,
        initial_screening,
        tuple(attempts),
        True,
        False,
        tuple(warnings),
    )


def _nearby_interior_indices(
    path: PathStageResult,
    limit: int,
) -> tuple[int, ...]:
    if limit <= 0 or len(path.sample_coordinates) <= 2:
        return ()
    peak_index = int(
        np.argmin(np.abs(path.sample_coordinates - path.highest_coordinate))
    )
    indices = [
        index
        for index in range(1, len(path.sample_coordinates) - 1)
        if index != peak_index
    ]
    indices.sort(
        key=lambda index: (
            abs(index - peak_index),
            -float(path.cpu_energies_ev[index]),
            index,
        )
    )
    return tuple(indices[:limit])


def _attempt(
    stage: FallbackStage,
    attempt: int,
    coordinate: float | None,
    screening: TransitionStateScreeningResult,
) -> SaddleFallbackAttempt:
    return SaddleFallbackAttempt(
        stage=stage,
        attempt=attempt,
        path_coordinate=coordinate,
        failure_code=screening.failure_code,
        local_saddle_valid=screening.local_saddle_valid,
        irc_eligible=screening.irc_eligible,
    )
