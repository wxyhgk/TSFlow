from __future__ import annotations

from ase import Atoms

from uma_workflows.transition_states.workflow.models import PathConfig
from uma_workflows.transition_states import PathBias, build_path_bias
from uma_workflows.transition_states.reaction_coordinates import (
    ReactionCoordinateLike,
)


def prepare_path_bias(
    reactant: Atoms,
    product: Atoms,
    reaction_coordinates: tuple[ReactionCoordinateLike, ...],
    config: PathConfig,
) -> tuple[PathBias, tuple[str, ...]]:
    strengths = {
        "repulsion": config.repulsion_strength_ev_per_a2,
        "synchronization": config.synchronization_strength_ev,
        "range": config.coordinate_range_strength_ev,
        "schedule": config.coordinate_schedule_strength_ev,
        "scaffold": config.scaffold_strength_ev_per_a2,
    }
    if config.bias_mode == "native":
        strengths = {name: 0.0 for name in strengths}
    elif config.bias_mode == "asynchronous":
        strengths["schedule"] = 0.0
        strengths["scaffold"] = 0.0
    elif config.bias_mode != "legacy_synchronous":
        raise ValueError(f"unsupported path bias mode: {config.bias_mode}")

    bias = build_path_bias(
        reactant,
        product,
        reaction_coordinates,
        repulsion_strength_ev_per_a2=strengths["repulsion"],
        repulsion_cutoff_scale=config.repulsion_cutoff_scale,
        synchronization_strength_ev=strengths["synchronization"],
        coordinate_range_strength_ev=strengths["range"],
        coordinate_schedule_strength_ev=strengths["schedule"],
        scaffold_strength_ev_per_a2=strengths["scaffold"],
        synchronize_same_kind_only=config.bias_mode == "asynchronous",
    )
    warnings = [f"DMF path bias mode: {config.bias_mode}."]
    if bias.repulsion_strength_ev_per_a2 > 0.0:
        warnings.append(
            "DMF used a continuous short-range repulsion bias; final path "
            "energies were recomputed without the bias on CPU UMA."
        )
    if len(bias.synchronized_coordinates) >= 2:
        if config.bias_mode == "asynchronous" and bias.synchronization_groups:
            warnings.append(
                "DMF synchronized only reaction coordinates of the same kind; "
                "forming and breaking groups remained mutually asynchronous, "
                "with no path-coordinate schedule or interpolated scaffold."
            )
        elif bias.synchronization_strength_ev > 0.0:
            warnings.append(
                "DMF synchronized the normalized progress of multiple specified "
                "bond changes; final path energies were recomputed without the bias."
            )
    return bias, tuple(warnings)
