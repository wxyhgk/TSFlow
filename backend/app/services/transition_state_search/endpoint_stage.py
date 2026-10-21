"""Compatibility exports for the endpoint stage moved to the workflow package."""

from uma_workflows.transition_states.workflow.endpoints import (
    EndpointStageResult,
    prepare_endpoints,
)

__all__ = ["EndpointStageResult", "prepare_endpoints"]
