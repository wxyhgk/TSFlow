"""Backward-compatible imports for the extracted reusable IRC package."""

from uma_workflows.irc.validation import (
    IrcValidationError,
    endpoint_internal_distance_delta,
    first_step_projection,
    mass_weighted_arc_coordinates,
    require_mode_gradient_bracket,
    require_opposite_first_steps,
)

__all__ = [
    "IrcValidationError",
    "endpoint_internal_distance_delta",
    "first_step_projection",
    "mass_weighted_arc_coordinates",
    "require_mode_gradient_bracket",
    "require_opposite_first_steps",
]
