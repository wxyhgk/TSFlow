"""Intrinsic reaction coordinate workflows and validation utilities."""

from uma_workflows._lazy import resolve_export


_EXPORTS = {
    "BidirectionalIrcConfig": "uma_workflows.irc.workflow",
    "BidirectionalIrcResult": "uma_workflows.irc.workflow",
    "IrcDirectionResult": "uma_workflows.irc.models",
    "IrcEndpointTopologyMatch": "uma_workflows.irc.topology",
    "IrcEndpointRelaxation": "uma_workflows.irc.workflow",
    "IrcFrame": "uma_workflows.irc.models",
    "follow_geometric_irc_direction": "uma_workflows.irc.geometric_runner",
    "IrcPathFrame": "uma_workflows.irc.workflow",
    "IrcValidationError": "uma_workflows.irc.validation",
    "endpoint_internal_distance_delta": "uma_workflows.irc.validation",
    "first_step_projection": "uma_workflows.irc.validation",
    "follow_sella_irc_direction": "uma_workflows.irc.runner",
    "mass_weighted_arc_coordinates": "uma_workflows.irc.validation",
    "match_irc_endpoint_topologies": "uma_workflows.irc.topology",
    "perceived_covalent_bonds": "uma_workflows.irc.topology",
    "perceived_heavy_atom_bonds": "uma_workflows.irc.topology",
    "prepare_sella_irc": "uma_workflows.irc.runner",
    "regularize_irc_hessian": "uma_workflows.irc.geometric_runner",
    "require_mode_gradient_bracket": "uma_workflows.irc.validation",
    "require_opposite_first_steps": "uma_workflows.irc.validation",
    "run_bidirectional_irc": "uma_workflows.irc.workflow",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    return resolve_export(name, _EXPORTS, globals())


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
