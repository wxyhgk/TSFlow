from __future__ import annotations

from uma_workflows.transition_states import build_builtin_algorithm_registry

from app.contracts.actions import (
    ActionSpec,
    GeneratePathCandidatesAction,
    OptimizePathCandidateAction,
    TransitionStateWorkflowAction,
)
from app.contracts.algorithm_capabilities import (
    TransitionStateAlgorithmCapability,
)


def transition_state_algorithm_capabilities() -> tuple[
    TransitionStateAlgorithmCapability,
    ...,
]:
    registry = build_builtin_algorithm_registry()
    return tuple(
        TransitionStateAlgorithmCapability(
            algorithm_id=item.metadata.algorithm_id,
            version=item.metadata.version,
            display_name=item.metadata.display_name,
            role=item.role,
            selector_parameter=(
                "path_initializer_id"
                if item.role == "path_initializer"
                else "path_optimizer_id"
            ),
            config_type=item.config_type.__name__,
            capabilities=list(item.capabilities),
            accepts_unparameterized=(
                item.compatibility.accepts_unparameterized
            ),
            accepted_parameterizations=(
                sorted(item.compatibility.accepted_parameterizations)
                if item.compatibility.accepted_parameterizations is not None
                else None
            ),
            produced_parameterization=(
                item.compatibility.produced_parameterization
            ),
        )
        for item in registry.describe()
    )


def validate_selected_algorithms(action: ActionSpec) -> None:
    if not isinstance(
        action,
        (
            GeneratePathCandidatesAction,
            OptimizePathCandidateAction,
            TransitionStateWorkflowAction,
        ),
    ):
        return
    registry = build_builtin_algorithm_registry()
    available = {
        "path_initializer": {
            registration.metadata.algorithm_id
            for registration in registry.describe("path_initializer")
        },
        "path_optimizer": {
            registration.metadata.algorithm_id
            for registration in registry.describe("path_optimizer")
        },
    }
    if isinstance(action, GeneratePathCandidatesAction):
        selected = {"path_initializer": action.parameters.initializer_id}
    elif isinstance(action, OptimizePathCandidateAction):
        selected = {"path_optimizer": action.parameters.optimizer_id}
    else:
        selected = {
            "path_initializer": action.parameters.path_initializer_id,
            "path_optimizer": action.parameters.path_optimizer_id,
        }
    for role, algorithm_id in selected.items():
        if algorithm_id not in available.get(role, set()):
            raise ValueError(f"UNKNOWN_{role.upper()}: {algorithm_id}")
