import numpy as np
import pytest

from app.services.irc_validation import (
    IrcValidationError,
    endpoint_internal_distance_delta,
    mass_weighted_arc_coordinates,
    require_mode_gradient_bracket,
    require_opposite_first_steps,
)


def test_rejects_imaginary_mode_without_projected_gradient_root() -> None:
    with pytest.raises(
        IrcValidationError, match="TS_NOT_STATIONARY_ON_SELECTED_PES"
    ):
        require_mode_gradient_bracket(-0.0040, -0.0050)


def test_accepts_imaginary_mode_that_brackets_projected_gradient_root() -> None:
    assert require_mode_gradient_bracket(0.0030, -0.0040) == pytest.approx(
        (0.0030, -0.0040)
    )


def test_rejects_corrected_irc_steps_on_same_side() -> None:
    transition_state = np.zeros((2, 3))
    forward = transition_state.copy()
    reverse = transition_state.copy()
    forward[0, 0] = 0.1
    reverse[0, 0] = 0.05
    mode = np.zeros((2, 3))
    mode[0, 0] = 1.0

    with pytest.raises(
        IrcValidationError, match="TS_NOT_STATIONARY_ON_SELECTED_PES"
    ):
        require_opposite_first_steps(
            forward,
            reverse,
            transition_state,
            np.array([1.0, 1.0]),
            mode,
            mode,
        )


def test_accepts_opposite_corrected_irc_steps() -> None:
    transition_state = np.zeros((1, 3))
    mode = np.array([[1.0, 0.0, 0.0]])
    projections = require_opposite_first_steps(
        np.array([[0.1, 0.0, 0.0]]),
        np.array([[-0.2, 0.0, 0.0]]),
        transition_state,
        np.array([12.0]),
        mode,
        mode,
    )
    assert projections[0] > 0
    assert projections[1] < 0


def test_endpoint_delta_ignores_rigid_translation() -> None:
    first = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    translated = first + np.array([4.0, -2.0, 3.0])
    assert endpoint_internal_distance_delta(first, translated) == pytest.approx(0.0)


def test_mass_weighted_arc_coordinates_are_normalized() -> None:
    positions = [
        np.array([[0.0, 0.0, 0.0]]),
        np.array([[1.0, 0.0, 0.0]]),
        np.array([[3.0, 0.0, 0.0]]),
    ]
    assert mass_weighted_arc_coordinates(positions, np.array([4.0])) == pytest.approx(
        [0.0, 1.0 / 3.0, 1.0]
    )
