from __future__ import annotations

from dataclasses import dataclass

from app.contracts.artifacts import ArtifactReference, ArtifactSelector
from app.contracts.job_records import JobStatus
from app.infrastructure.artifacts import FileArtifactStore
from app.infrastructure.persistence.protocol import JobRepository


@dataclass(frozen=True)
class ArtifactSelectionError(ValueError):
    status_code: int
    code: str
    message: str

    def __str__(self) -> str:
        return self.message


async def resolve_action_artifact(
    selector: ArtifactSelector,
    *,
    repository: JobRepository,
    store: FileArtifactStore | None,
    allowed_source_action_types: frozenset[str],
    expected_kind: str,
    expected_media_type: str = "application/json",
    expected_schema_version: int = 1,
) -> ArtifactReference:
    if store is None:
        raise ArtifactSelectionError(
            503,
            "ARTIFACT_STORE_UNAVAILABLE",
            "This action requires the configured artifact store.",
        )
    source = await repository.get(selector.source_action_id)
    if source is None:
        raise ArtifactSelectionError(
            404,
            "SOURCE_ACTION_NOT_FOUND",
            "The source action does not exist.",
        )
    if source.status != JobStatus.SUCCEEDED:
        raise ArtifactSelectionError(
            409,
            "SOURCE_ACTION_NOT_SUCCEEDED",
            "The source action has not succeeded.",
        )
    if source.artifact_error is not None:
        raise ArtifactSelectionError(
            409,
            "SOURCE_ACTION_ARTIFACTS_INCOMPLETE",
            "The source action did not persist a complete artifact set.",
        )
    envelope = source.execution_envelope
    if envelope is None or envelope.payload.execution_type != "action":
        raise ArtifactSelectionError(
            422,
            "SOURCE_ACTION_TYPE_MISMATCH",
            "The source job is not an eligible typed action.",
        )
    source_action_type = envelope.payload.action.action_type
    if source_action_type not in allowed_source_action_types:
        raise ArtifactSelectionError(
            422,
            "SOURCE_ACTION_TYPE_MISMATCH",
            f"Action {source_action_type} cannot produce this input.",
        )
    matches = [
        reference
        for reference in source.artifacts
        if reference.artifact_id == selector.artifact_id
    ]
    if not matches:
        raise ArtifactSelectionError(
            404,
            "ARTIFACT_NOT_FOUND",
            "The source action does not contain the selected artifact.",
        )
    if len(matches) != 1:
        raise ArtifactSelectionError(
            409,
            "SOURCE_ACTION_ARTIFACTS_INCOMPLETE",
            "The source action contains duplicate artifact identifiers.",
        )
    reference = matches[0]
    if reference.sha256 != selector.expected_sha256:
        raise ArtifactSelectionError(
            409,
            "ARTIFACT_DIGEST_MISMATCH",
            "The expected artifact digest does not match the source manifest.",
        )
    if reference.kind != expected_kind:
        raise ArtifactSelectionError(
            422,
            "ARTIFACT_KIND_MISMATCH",
            f"Expected artifact kind {expected_kind}.",
        )
    if reference.media_type != expected_media_type:
        raise ArtifactSelectionError(
            422,
            "ARTIFACT_MEDIA_TYPE_UNSUPPORTED",
            f"Expected artifact media type {expected_media_type}.",
        )
    if reference.schema_version != expected_schema_version:
        raise ArtifactSelectionError(
            422,
            "PATH_ARTIFACT_SCHEMA_UNSUPPORTED",
            f"Expected artifact schema version {expected_schema_version}.",
        )
    try:
        store.read_verified(reference)
    except (FileNotFoundError, ValueError) as exc:
        raise ArtifactSelectionError(
            409,
            "ARTIFACT_INTEGRITY_FAILED",
            "The selected artifact failed integrity verification.",
        ) from exc
    return reference
