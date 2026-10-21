from __future__ import annotations

import pytest

from uma_workflows.transition_states import classify_transition_outcome


def test_validated_saddle_takes_precedence_over_a_shallow_path() -> None:
    result = classify_transition_outcome(
        [0.0, 0.01, -0.2],
        path_converged=True,
        screening_complete=True,
        candidate_is_minimum=False,
        passes_ts_screening=True,
        near_barrierless_threshold_ev=0.05,
    )

    assert result.kind == "validated_transition_state"
    assert result.near_barrierless_directions == ("forward",)


def test_complete_screen_classifies_only_the_shallow_direction() -> None:
    result = classify_transition_outcome(
        [0.0, 0.002047, -0.50836],
        path_converged=True,
        screening_complete=True,
        candidate_is_minimum=True,
        passes_ts_screening=False,
        near_barrierless_threshold_ev=0.05,
    )

    assert result.kind == "near_barrierless"
    assert result.forward_path_rise_ev == pytest.approx(0.002047)
    assert result.reverse_path_rise_ev == pytest.approx(0.510407)
    assert result.near_barrierless_directions == ("forward",)


def test_missing_full_hessian_cannot_claim_near_barrierless() -> None:
    result = classify_transition_outcome(
        [0.0, 0.001, -0.3],
        path_converged=True,
        screening_complete=False,
        candidate_is_minimum=False,
        passes_ts_screening=False,
        near_barrierless_threshold_ev=0.05,
    )

    assert result.kind == "unvalidated_candidate"


def test_unconverged_path_is_inconclusive_even_when_apparently_flat() -> None:
    result = classify_transition_outcome(
        [0.0, 0.001, -0.3],
        path_converged=False,
        screening_complete=True,
        candidate_is_minimum=True,
        passes_ts_screening=False,
        near_barrierless_threshold_ev=0.05,
    )

    assert result.kind == "path_search_inconclusive"


def test_classifier_rejects_invalid_energy_input() -> None:
    with pytest.raises(ValueError, match="two finite"):
        classify_transition_outcome(
            [0.0],
            path_converged=True,
            screening_complete=True,
            candidate_is_minimum=True,
            passes_ts_screening=False,
            near_barrierless_threshold_ev=0.05,
        )


def test_nonstationary_shallow_candidate_is_not_called_barrierless() -> None:
    result = classify_transition_outcome(
        [0.0, 0.001, -0.3],
        path_converged=True,
        screening_complete=True,
        candidate_is_minimum=False,
        passes_ts_screening=False,
        near_barrierless_threshold_ev=0.05,
    )

    assert result.kind == "unvalidated_candidate"
