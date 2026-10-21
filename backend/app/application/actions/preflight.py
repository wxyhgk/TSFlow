from __future__ import annotations

from dataclasses import dataclass, field

from app.adapters.uma.path_artifacts import (
    PathArtifactValidationError,
    load_optimized_path_artifact,
    load_path_candidate,
)
from app.application.actions.artifact_resolver import resolve_action_artifact
from app.contracts.actions import (
    ActionSpec,
    OptimizePathCandidateAction,
    ProbePathCurvatureAction,
    RunCollectiveCoordinateScanAction,
)
from app.contracts.artifacts import ArtifactReference
from app.infrastructure.artifacts import FileArtifactStore
from app.infrastructure.persistence.protocol import JobRepository
from uma_workflows.transition_states import build_builtin_algorithm_registry


@dataclass(frozen=True)
class ActionPreflightResolution:
    resolved_artifacts: dict[str, ArtifactReference] = field(default_factory=dict)
    resolved_algorithms: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionPreflightError(ValueError):
    status_code: int
    code: str
    message: str

    def __str__(self) -> str:
        return self.message


async def preflight_action(
    action: ActionSpec,
    *,
    repository: JobRepository,
    artifact_store: FileArtifactStore | None,
) -> ActionPreflightResolution:
    if not isinstance(
        action,
        (
            OptimizePathCandidateAction,
            ProbePathCurvatureAction,
            RunCollectiveCoordinateScanAction,
        ),
    ):
        return ActionPreflightResolution()
    if artifact_store is None:
        raise ActionPreflightError(
            503,
            "ARTIFACT_STORE_UNAVAILABLE",
            "This action requires the configured artifact store.",
        )
    if isinstance(
        action,
        (ProbePathCurvatureAction, RunCollectiveCoordinateScanAction),
    ):
        reference = await resolve_action_artifact(
            action.optimized_path,
            repository=repository,
            store=artifact_store,
            allowed_source_action_types=frozenset({"optimize_path_candidate"}),
            expected_kind="optimized_path_candidate",
        )
        try:
            artifact = load_optimized_path_artifact(
                artifact_store,
                reference,
                reactant=action.structure,
                product=action.product_structure,
                model_name=action.model_name,
                task_name=action.task_name,
                reaction_coordinates=action.reaction_coordinates,
            )
        except PathArtifactValidationError as exc:
            raise ActionPreflightError(422, exc.code, exc.message) from exc
        if not artifact.converged:
            raise ActionPreflightError(
                409,
                "PATH_NOT_CONVERGED",
                "This operation requires a converged optimized path.",
            )
        if (
            isinstance(action, ProbePathCurvatureAction)
            and not artifact.has_interior_maximum
        ):
            raise ActionPreflightError(
                422,
                "PATH_NO_INTERIOR_MAXIMUM",
                "Curvature probing requires an interior path-energy maximum.",
            )
        return ActionPreflightResolution(
            resolved_artifacts={"optimized_path": reference},
        )
    reference = await resolve_action_artifact(
        action.path_candidate,
        repository=repository,
        store=artifact_store,
        allowed_source_action_types=frozenset({"generate_path_candidates"}),
        expected_kind="path_candidate",
    )
    try:
        _, candidate = load_path_candidate(
            artifact_store,
            reference,
            reactant=action.structure,
            product=action.product_structure,
            model_name=action.model_name,
            task_name=action.task_name,
            reaction_coordinates=action.reaction_coordinates,
        )
    except PathArtifactValidationError as exc:
        raise ActionPreflightError(422, exc.code, exc.message) from exc
    registry = build_builtin_algorithm_registry()
    try:
        registration = registry.validate_optimizer_candidate(
            action.parameters.optimizer_id,
            candidate,
        )
    except ValueError as exc:
        raise ActionPreflightError(
            422,
            "PATH_PARAMETERIZATION_INCOMPATIBLE",
            str(exc),
        ) from exc
    version = registration.metadata.version
    if action.parameters.expected_optimizer_version != version:
        raise ActionPreflightError(
            409,
            "ALGORITHM_VERSION_MISMATCH",
            "The selected path optimizer version differs from the pinned version.",
        )
    return ActionPreflightResolution(
        resolved_artifacts={"path_candidate": reference},
        resolved_algorithms={"path_optimizer": version},
    )
