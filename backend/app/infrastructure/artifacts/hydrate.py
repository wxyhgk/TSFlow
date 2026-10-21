from __future__ import annotations

import json
from typing import Any

from app.contracts.artifacts import ArtifactReference
from app.contracts.jobs import (
    CalculationResult,
    EnergyProfilePoint,
    IrcEndpointRelaxationResult,
    IrcPathFrameResult,
    Job,
    OptimizationFrame,
    ReactionPathFrame,
    SaddleOptimizationFrame,
    TransitionStateResult,
)
from app.infrastructure.artifacts.store import FileArtifactStore


class ArtifactHydrationError(ValueError):
    """A database result references a missing, corrupt, or invalid artifact."""


def compact_result_for_storage(
    result: Any,
    artifacts: list[ArtifactReference],
    store: FileArtifactStore | None,
) -> tuple[dict[str, Any], list[str]]:
    payload = result.model_dump(mode="json")
    references = {artifact.artifact_id: artifact for artifact in artifacts}
    external_fields: list[str] = []

    def externalize(field: str, artifact_id: str, empty_value: Any) -> None:
        reference = references.get(artifact_id)
        if reference is None or store is None:
            return
        try:
            store.read_verified(reference)
        except (FileNotFoundError, ValueError):
            return
        payload[field] = empty_value
        external_fields.append(field)

    if isinstance(result, CalculationResult):
        externalize("trajectory", "optimization-trajectory-json", [])
    elif isinstance(result, TransitionStateResult):
        externalize("path", "reaction-path-json", [])
        externalize("energy_profile", "energy-profile", [])
        externalize("saddle_trajectory", "saddle-trajectory-json", [])
        externalize("irc_frames", "irc-trajectory-json", [])
        externalize(
            "irc_endpoint_a_relaxation",
            "endpoint-a-relaxation-json",
            None,
        )
        externalize(
            "irc_endpoint_b_relaxation",
            "endpoint-b-relaxation-json",
            None,
        )
    return payload, external_fields


def hydrate_job_result(
    job: Job,
    external_fields: list[str],
    store: FileArtifactStore | None,
) -> Job:
    if not external_fields or job.result is None:
        return job
    if store is None:
        raise ArtifactHydrationError("artifact store is unavailable")
    references = {artifact.artifact_id: artifact for artifact in job.artifacts}
    result = job.result
    updates: dict[str, Any] = {}
    if isinstance(result, CalculationResult):
        mappings: dict[str, tuple[str, Any]] = {
            "trajectory": ("optimization-trajectory-json", OptimizationFrame),
        }
    elif isinstance(result, TransitionStateResult):
        mappings = {
            "path": ("reaction-path-json", ReactionPathFrame),
            "energy_profile": ("energy-profile", EnergyProfilePoint),
            "saddle_trajectory": (
                "saddle-trajectory-json",
                SaddleOptimizationFrame,
            ),
            "irc_frames": ("irc-trajectory-json", IrcPathFrameResult),
            "irc_endpoint_a_relaxation": (
                "endpoint-a-relaxation-json",
                IrcEndpointRelaxationResult,
            ),
            "irc_endpoint_b_relaxation": (
                "endpoint-b-relaxation-json",
                IrcEndpointRelaxationResult,
            ),
        }
    else:
        mappings = {}
    for field in external_fields:
        mapping = mappings.get(field)
        if mapping is None:
            raise ArtifactHydrationError(f"unsupported external result field: {field}")
        artifact_id, model = mapping
        reference = references.get(artifact_id)
        if reference is None:
            raise ArtifactHydrationError(
                f"result field {field} is missing artifact {artifact_id}"
            )
        try:
            payload = json.loads(store.read_verified(reference))
            if field.startswith("irc_endpoint_"):
                updates[field] = model.model_validate(payload)
            else:
                if not isinstance(payload, list):
                    raise ValueError("trajectory artifact must contain a JSON array")
                updates[field] = [model.model_validate(item) for item in payload]
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
            raise ArtifactHydrationError(
                f"cannot restore result field {field}: {exc}"
            ) from exc
    return job.model_copy(update={"result": result.model_copy(update=updates)})
