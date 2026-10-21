from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from pydantic import ValidationError

from app.adapters.uma.structures import structure_fingerprint
from app.contracts.artifacts import ArtifactReference
from app.contracts.path_artifacts import OptimizedPathArtifactV1, PathArtifactV1
from app.contracts.structures import AtomicStructure, ReactionCoordinate
from app.infrastructure.artifacts import FileArtifactStore
from uma_workflows.transition_states.algorithm_api import (
    PathCandidate,
    PathParameterization,
    ReactionPath,
)


MAX_PATH_ARTIFACT_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True)
class PathArtifactValidationError(ValueError):
    code: str
    message: str

    def __str__(self) -> str:
        return self.message


def load_path_candidate(
    store: FileArtifactStore,
    reference: ArtifactReference,
    *,
    reactant: AtomicStructure,
    product: AtomicStructure,
    model_name: str,
    task_name: str,
    reaction_coordinates: list[ReactionCoordinate],
) -> tuple[PathArtifactV1, PathCandidate]:
    artifact = load_path_artifact(store, reference)
    _validate_bindings(
        artifact,
        reference,
        reactant=reactant,
        product=product,
        model_name=model_name,
        task_name=task_name,
        reaction_coordinates=reaction_coordinates,
    )
    path = ReactionPath(
        positions_a=np.asarray(artifact.path.positions_a, dtype=float),
        coordinates=np.asarray(artifact.path.coordinates, dtype=float),
        tangents_a=(
            np.asarray(artifact.path.tangents_a, dtype=float)
            if artifact.path.tangents_a is not None
            else None
        ),
        metadata=artifact.path.metadata,
    )
    parameterization = (
        PathParameterization.from_array(
            artifact.parameterization.kind,
            np.asarray(artifact.parameterization.coefficients, dtype=float),
        )
        if artifact.parameterization is not None
        else None
    )
    return artifact, PathCandidate(
        path=path,
        score=artifact.score_ev,
        source=artifact.source,
        candidate_index=artifact.candidate_index,
        parameterization=parameterization,
        warnings=tuple(artifact.warnings),
        metadata=artifact.metadata,
    )


def load_path_artifact(
    store: FileArtifactStore,
    reference: ArtifactReference,
) -> PathArtifactV1:
    if reference.kind != "path_candidate":
        raise PathArtifactValidationError(
            "ARTIFACT_KIND_MISMATCH",
            "The selected artifact is not a path_candidate.",
        )
    if reference.media_type != "application/json":
        raise PathArtifactValidationError(
            "ARTIFACT_MEDIA_TYPE_UNSUPPORTED",
            "Path candidates must use application/json.",
        )
    if reference.schema_version != 1:
        raise PathArtifactValidationError(
            "PATH_ARTIFACT_SCHEMA_UNSUPPORTED",
            f"Unsupported path artifact schema version: {reference.schema_version}.",
        )
    if reference.size_bytes > MAX_PATH_ARTIFACT_BYTES:
        raise PathArtifactValidationError(
            "PATH_ARTIFACT_TOO_LARGE",
            "The selected path artifact exceeds the server size limit.",
        )
    try:
        data = store.read_verified(reference)
    except (FileNotFoundError, ValueError) as exc:
        raise PathArtifactValidationError(
            "ARTIFACT_INTEGRITY_FAILED",
            "The selected path artifact failed integrity verification.",
        ) from exc
    try:
        payload = json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise PathArtifactValidationError(
            "PATH_ARTIFACT_SCHEMA_INVALID",
            "The selected path artifact is not valid JSON.",
        ) from exc
    try:
        return PathArtifactV1.model_validate(payload)
    except ValidationError as exc:
        raise PathArtifactValidationError(
            "PATH_ARTIFACT_SCHEMA_INVALID",
            "The selected path artifact does not satisfy schema version 1.",
        ) from exc


def load_optimized_path_artifact(
    store: FileArtifactStore,
    reference: ArtifactReference,
    *,
    reactant: AtomicStructure,
    product: AtomicStructure,
    model_name: str,
    task_name: str,
    reaction_coordinates: list[ReactionCoordinate],
) -> OptimizedPathArtifactV1:
    """Load and validate an optimized path against its source problem."""
    if reference.kind != "optimized_path_candidate":
        raise PathArtifactValidationError(
            "ARTIFACT_KIND_MISMATCH",
            "The selected artifact is not an optimized_path_candidate.",
        )
    if reference.media_type != "application/json":
        raise PathArtifactValidationError(
            "ARTIFACT_MEDIA_TYPE_UNSUPPORTED",
            "Optimized path candidates must use application/json.",
        )
    if reference.schema_version != 1:
        raise PathArtifactValidationError(
            "OPTIMIZED_PATH_ARTIFACT_SCHEMA_UNSUPPORTED",
            f"Unsupported optimized path artifact schema version: {reference.schema_version}.",
        )
    if reference.size_bytes > MAX_PATH_ARTIFACT_BYTES:
        raise PathArtifactValidationError(
            "OPTIMIZED_PATH_ARTIFACT_TOO_LARGE",
            "The selected optimized path artifact exceeds the server size limit.",
        )
    try:
        data = store.read_verified(reference)
    except (FileNotFoundError, ValueError) as exc:
        raise PathArtifactValidationError(
            "ARTIFACT_INTEGRITY_FAILED",
            "The selected optimized path artifact failed integrity verification.",
        ) from exc
    try:
        payload = json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise PathArtifactValidationError(
            "OPTIMIZED_PATH_ARTIFACT_SCHEMA_INVALID",
            "The selected optimized path artifact is not valid JSON.",
        ) from exc
    try:
        artifact = OptimizedPathArtifactV1.model_validate(payload)
    except ValidationError as exc:
        raise PathArtifactValidationError(
            "OPTIMIZED_PATH_ARTIFACT_SCHEMA_INVALID",
            "The selected optimized path artifact does not satisfy schema version 1.",
        ) from exc
    _validate_optimized_bindings(
        artifact,
        reference,
        reactant=reactant,
        product=product,
        model_name=model_name,
        task_name=task_name,
        reaction_coordinates=reaction_coordinates,
    )
    return artifact


def _validate_optimized_bindings(
    artifact: OptimizedPathArtifactV1,
    reference: ArtifactReference,
    *,
    reactant: AtomicStructure,
    product: AtomicStructure,
    model_name: str,
    task_name: str,
    reaction_coordinates: list[ReactionCoordinate],
) -> None:
    if artifact.model_name != model_name or artifact.task_name != task_name:
        raise PathArtifactValidationError(
            "PATH_PES_IDENTITY_MISMATCH",
            "The optimized path artifact was generated for a different model or task.",
        )
    if (
        artifact.reactant_sha256 != structure_fingerprint(reactant)
        or artifact.product_sha256 != structure_fingerprint(product)
    ):
        raise PathArtifactValidationError(
            "PATH_ENDPOINT_BINDING_MISMATCH",
            "The optimized path artifact is bound to different endpoint structures.",
        )
    if artifact.atom_symbols != [atom.symbol for atom in reactant.atoms]:
        raise PathArtifactValidationError(
            "PATH_ENDPOINT_BINDING_MISMATCH",
            "The optimized path artifact atom ordering does not match.",
        )
    if artifact.charge != reactant.charge or artifact.spin != reactant.spin:
        raise PathArtifactValidationError(
            "PATH_ENDPOINT_BINDING_MISMATCH",
            "The optimized path artifact charge or spin does not match.",
        )
    expected = [coordinate.model_dump(mode="json") for coordinate in reaction_coordinates]
    actual = [coordinate.model_dump(mode="json") for coordinate in artifact.reaction_coordinates]
    if actual != expected:
        raise PathArtifactValidationError(
            "PATH_REACTION_COORDINATES_MISMATCH",
            "The optimized path artifact uses a different reaction-coordinate definition.",
        )
    if reference.frame_count != len(artifact.path.positions_a):
        raise PathArtifactValidationError(
            "OPTIMIZED_PATH_ARTIFACT_SCHEMA_INVALID",
            "The optimized path artifact frame count does not match its manifest.",
        )
def _validate_bindings(
    artifact: PathArtifactV1,
    reference: ArtifactReference,
    *,
    reactant: AtomicStructure,
    product: AtomicStructure,
    model_name: str,
    task_name: str,
    reaction_coordinates: list[ReactionCoordinate],
) -> None:
    if artifact.model_name != model_name or artifact.task_name != task_name:
        raise PathArtifactValidationError(
            "PATH_PES_IDENTITY_MISMATCH",
            "The path artifact was generated for a different model or task.",
        )
    if (
        artifact.reactant_sha256 != structure_fingerprint(reactant)
        or artifact.product_sha256 != structure_fingerprint(product)
    ):
        raise PathArtifactValidationError(
            "PATH_ENDPOINT_BINDING_MISMATCH",
            "The path artifact is bound to different endpoint structures.",
        )
    symbols = [atom.symbol for atom in reactant.atoms]
    if (
        artifact.atom_symbols != symbols
        or artifact.charge != reactant.charge
        or artifact.spin != reactant.spin
    ):
        raise PathArtifactValidationError(
            "PATH_ENDPOINT_BINDING_MISMATCH",
            "The path artifact atom ordering, charge, or spin does not match.",
        )
    expected_coordinates = [
        coordinate.model_dump(mode="json") for coordinate in reaction_coordinates
    ]
    actual_coordinates = [
        coordinate.model_dump(mode="json")
        for coordinate in artifact.reaction_coordinates
    ]
    if actual_coordinates != expected_coordinates:
        raise PathArtifactValidationError(
            "PATH_REACTION_COORDINATES_MISMATCH",
            "The path artifact uses a different reaction-coordinate definition.",
        )
    if reference.frame_count != len(artifact.path.positions_a):
        raise PathArtifactValidationError(
            "PATH_ARTIFACT_SCHEMA_INVALID",
            "The path artifact frame count does not match its manifest.",
        )
