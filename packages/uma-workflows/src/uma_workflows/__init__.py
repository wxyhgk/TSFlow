"""Reusable UMA computational chemistry workflows.

Public symbols are resolved lazily so importing the base package does not load
Sella, FairChem, SciPy, or PyTorch.
"""

from uma_workflows._lazy import resolve_export


_EXPORTS = {
    "IrcEndpointTopologyMatch": "uma_workflows.irc.topology",
    "IrcValidationError": "uma_workflows.irc.validation",
    "endpoint_internal_distance_delta": "uma_workflows.irc.validation",
    "first_step_projection": "uma_workflows.irc.validation",
    "mass_weighted_arc_coordinates": "uma_workflows.irc.validation",
    "match_irc_endpoint_topologies": "uma_workflows.irc.topology",
    "perceived_heavy_atom_bonds": "uma_workflows.irc.topology",
    "require_mode_gradient_bracket": "uma_workflows.irc.validation",
    "require_opposite_first_steps": "uma_workflows.irc.validation",
    "set_nonperiodic_graph_cell": "uma_workflows.geometry",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    return resolve_export(name, _EXPORTS, globals())


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
