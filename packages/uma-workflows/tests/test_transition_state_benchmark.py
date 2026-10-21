from uma_workflows.benchmarks import (
    RegressionAssertion,
    evaluate_transition_state_result,
)


def test_transition_state_regression_accepts_job_envelope() -> None:
    report = evaluate_transition_state_result(
        "example",
        [
            RegressionAssertion("scientific_outcome", "eq", "validated_transition_state"),
            RegressionAssertion("transition_state_max_force_ev_per_a", "le", 0.001),
            RegressionAssertion("imaginary_frequencies_cm1", "length_eq", 1),
            RegressionAssertion("imaginary_frequencies_cm1.0", "abs_between", [150, 250]),
        ],
        {
            "status": "succeeded",
            "result": {
                "scientific_outcome": "validated_transition_state",
                "transition_state_max_force_ev_per_a": 0.0002,
                "imaginary_frequencies_cm1": [-190.0],
            },
        },
    )

    assert report.passed
    assert all(check.passed for check in report.checks)


def test_transition_state_regression_reports_missing_and_failed_values() -> None:
    report = evaluate_transition_state_result(
        "example",
        [
            RegressionAssertion("irc_validation_passed", "eq", True),
            RegressionAssertion("imaginary_mode_reaction_overlap", "ge", 0.5),
        ],
        {"irc_validation_passed": False},
    )

    assert not report.passed
    assert report.checks[0].actual is False
    assert report.checks[0].error is None
    assert report.checks[1].actual is None
    assert report.checks[1].error is not None
