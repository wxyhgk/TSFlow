from types import SimpleNamespace
from threading import Event

import numpy as np
from ase import Atoms

from uma_workflows.transition_states.workflow import fallback
from uma_workflows.transition_states.workflow.fallback import (
    recover_collapsed_saddle,
)
from uma_workflows.transition_states.workflow.models import (
    SaddleConfig,
    ScreeningConfig,
)
from uma_workflows.transition_states.workflow.path import PathStageResult
from uma_workflows.transition_states.workflow.saddle import SaddleStageResult


def _screening(
    *,
    failure_code: str | None,
    local: bool = False,
    eligible: bool = False,
    one_negative_mode: bool = False,
):
    frequency = SimpleNamespace(
        imaginary_frequencies_cm1=[-300.0] if one_negative_mode else [],
    )
    return SimpleNamespace(
        failure_code=failure_code,
        local_saddle_valid=local,
        irc_eligible=eligible,
        frequency_analysis=frequency,
        frequency_scope="full",
        hessian_quality_valid=True,
        mode_reaction_overlap=SimpleNamespace(matches=True, score=0.8),
        reaction_definition_valid=True,
    )


def _path() -> PathStageResult:
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    coordinates = np.linspace(0.0, 1.0, 5)
    positions = np.zeros((5, 1, 3))
    positions[:, 0, 0] = coordinates
    return PathStageResult(
        problem=SimpleNamespace(
            reactant=atoms,
            product=atoms.copy(),
            reaction_coordinates=(SimpleNamespace(),),
        ),
        bias=SimpleNamespace(synchronized_coordinates=((0, 0, 0.0, 1.0),)),
        path=SimpleNamespace(),
        optimization=SimpleNamespace(),
        sample_coordinates=coordinates,
        sampled_positions=positions,
        cpu_energies_ev=[0.0, 0.4, 0.8, 0.5, 0.1],
        highest_coordinate=0.5,
        highest_positions=positions[2],
        has_interior_maximum=True,
        candidates_evaluated=1,
        selected_candidate=0,
        warnings=(),
    )


def test_collapsed_saddle_recovers_from_nearby_negative_curvature_frame(
    monkeypatch,
) -> None:
    path = _path()
    initial_saddle = SaddleStageResult(path.problem.reactant.copy(), None, (), ())
    initial_screening = _screening(failure_code="SADDLE_COLLAPSED_TO_MINIMUM")
    probe = _screening(
        failure_code="TS_NOT_STATIONARY",
        one_negative_mode=True,
    )
    recovered = _screening(failure_code=None, local=True, eligible=True)
    screens = iter((probe, probe, recovered))
    refined_coordinates: list[float] = []

    monkeypatch.setattr(fallback, "screen_saddle", lambda *args, **kwargs: next(screens))

    def fake_refine(candidate_path, atoms, **kwargs):
        refined_coordinates.append(candidate_path.highest_coordinate)
        return SaddleStageResult(atoms, SimpleNamespace(), (), ())

    monkeypatch.setattr(fallback, "refine_saddle", fake_refine)
    result = recover_collapsed_saddle(
        path,
        initial_saddle,
        initial_screening,
        config=SaddleConfig(
            fallback_nearby_frames=2,
            fallback_max_candidates=1,
        ),
        screening_config=ScreeningConfig(),
        device="cpu",
        mps_calculator=None,
        cpu_calculator=None,
        analytic_hessian_calculator=None,
        cancel_event=Event(),
        progress_callback=None,
    )

    assert result.performed is True
    assert result.recovered is True
    assert refined_coordinates == [0.75]
    assert [attempt.stage for attempt in result.attempts] == [
        "initial_refinement",
        "path_curvature_screen",
        "path_curvature_screen",
        "alternative_seed_refinement",
    ]


def test_fallback_nearby_frame_selection_is_bounded_and_excludes_peak() -> None:
    path = _path()

    assert fallback._nearby_interior_indices(path, 2) == (3, 1)
    assert fallback._nearby_interior_indices(path, 0) == ()
