"""Endpoint topology and preoptimization workflows."""

from uma_workflows._lazy import resolve_export


_EXPORTS = {
    "EndpointPreoptimization": "uma_workflows.endpoints.preoptimization",
    "covalent_fragments": "uma_workflows.endpoints.preoptimization",
    "preoptimize_endpoint": "uma_workflows.endpoints.preoptimization",
    "preoptimize_endpoint_stages": "uma_workflows.endpoints.preoptimization",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    return resolve_export(name, _EXPORTS, globals())


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
