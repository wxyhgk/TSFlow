import asyncio
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from app.adapters.uma.path_artifacts import (
    PathArtifactValidationError,
    load_path_candidate,
)
from app.application.actions.artifact_resolver import (
    ArtifactSelectionError,
    resolve_action_artifact,
)
from app.contracts.actions import (
    ActionExecution,
    ExecutionEnvelope,
    GeneratePathCandidatesAction,
)
from app.contracts.artifacts import ArtifactSelector
from app.contracts.job_records import Job, JobStatus
from app.contracts.requests import JobCreate
from app.contracts.structures import Atom, AtomicStructure
from app.infrastructure.artifacts import FileArtifactStore


def _structure(x: float) -> AtomicStructure:
    return AtomicStructure(
        atoms=[Atom(symbol="H", position=(x, 0.0, 0.0))],
        charge=-1,
        spin=1,
    )


def _payload(reactant: AtomicStructure, product: AtomicStructure) -> dict:
    from app.adapters.uma.structures import structure_fingerprint

    return {
        "schema_version": 1,
        "model_name": "uma-s-1p2p1",
        "task_name": "omol",
        "reactant_sha256": structure_fingerprint(reactant),
        "product_sha256": structure_fingerprint(product),
        "atom_symbols": ["H"],
        "charge": -1,
        "spin": 1,
        "initializer_algorithm_id": "fbenm.correlated",
        "initializer_algorithm_version": "1",
        "candidate_index": 0,
        "score_ev": 0.2,
        "source": "fbenm.correlated",
        "path": {
            "positions_a": [
                [[0.0, 0.0, 0.0]],
                [[0.5, 0.0, 0.0]],
                [[1.0, 0.0, 0.0]],
            ],
            "coordinates": [0.0, 0.5, 1.0],
            "tangents_a": None,
            "metadata": {},
        },
        "parameterization": {
            "kind": "pydmf.bspline",
            "coefficients": [
                [[0.0, 0.0, 0.0]],
                [[0.5, 0.0, 0.0]],
                [[1.0, 0.0, 0.0]],
            ],
        },
        "reaction_coordinates": [],
        "warnings": [],
        "metadata": {},
    }


def test_load_path_candidate_validates_and_rebuilds_domain_object(
    tmp_path: Path,
) -> None:
    reactant = _structure(0.0)
    product = _structure(1.0)
    store = FileArtifactStore(tmp_path)
    reference = store.write_json(
        uuid4(),
        "path-candidate-0",
        "path_candidate",
        _payload(reactant, product),
        frame_count=3,
    )

    artifact, candidate = load_path_candidate(
        store,
        reference,
        reactant=reactant,
        product=product,
        model_name="uma-s-1p2p1",
        task_name="omol",
        reaction_coordinates=[],
    )

    assert artifact.candidate_index == 0
    assert candidate.path.positions_a.shape == (3, 1, 3)
    assert candidate.parameterization is not None
    assert candidate.parameterization.kind == "pydmf.bspline"


def test_path_candidate_rejects_nonfinite_and_binding_mismatch(
    tmp_path: Path,
) -> None:
    reactant = _structure(0.0)
    product = _structure(1.0)
    store = FileArtifactStore(tmp_path)
    payload = _payload(reactant, product)
    payload["path"]["positions_a"][1][0][0] = float("nan")
    reference = store.write_json(
        uuid4(),
        "path-candidate-0",
        "path_candidate",
        payload,
        frame_count=3,
    )
    with pytest.raises(
        PathArtifactValidationError,
        match="schema version 1",
    ):
        load_path_candidate(
            store,
            reference,
            reactant=reactant,
            product=product,
            model_name="uma-s-1p2p1",
            task_name="omol",
            reaction_coordinates=[],
        )

    valid_reference = store.write_json(
        uuid4(),
        "path-candidate-0",
        "path_candidate",
        _payload(reactant, product),
        frame_count=3,
    )
    with pytest.raises(
        PathArtifactValidationError,
        match="different endpoint",
    ):
        load_path_candidate(
            store,
            valid_reference,
            reactant=reactant,
            product=_structure(2.0),
            model_name="uma-s-1p2p1",
            task_name="omol",
            reaction_coordinates=[],
        )


class _Repository:
    def __init__(self, job: Job) -> None:
        self.job = job

    async def get(self, _job_id):
        return self.job


def test_selector_resolves_only_manifest_owned_artifact(tmp_path: Path) -> None:
    reactant = _structure(0.0)
    product = _structure(1.0)
    store = FileArtifactStore(tmp_path)
    source_id = uuid4()
    reference = store.write_json(
        source_id,
        "path-candidate-0",
        "path_candidate",
        _payload(reactant, product),
        frame_count=3,
    )
    action = GeneratePathCandidatesAction(
        action_type="generate_path_candidates",
        structure=reactant,
        product_structure=product,
    )
    now = datetime.now(UTC)
    job = Job(
        id=source_id,
        status=JobStatus.SUCCEEDED,
        request=JobCreate(
            calculation="transition_state_search",
            structure=reactant,
            product_structure=product,
        ),
        execution_envelope=ExecutionEnvelope(
            execution_id=source_id,
            payload=ActionExecution(action=action),
        ),
        artifacts=[reference],
        created_at=now,
        updated_at=now,
    )
    selector = ArtifactSelector(
        source_action_id=source_id,
        artifact_id=reference.artifact_id,
        expected_sha256=reference.sha256,
    )

    resolved = asyncio.run(resolve_action_artifact(
        selector,
        repository=_Repository(job),  # type: ignore[arg-type]
        store=store,
        allowed_source_action_types=frozenset({"generate_path_candidates"}),
        expected_kind="path_candidate",
    ))
    assert resolved == reference

    wrong_digest = selector.model_copy(update={"expected_sha256": "0" * 64})
    with pytest.raises(ArtifactSelectionError) as captured:
        asyncio.run(resolve_action_artifact(
            wrong_digest,
            repository=_Repository(job),  # type: ignore[arg-type]
            store=store,
            allowed_source_action_types=frozenset({"generate_path_candidates"}),
            expected_kind="path_candidate",
        ))
    assert captured.value.code == "ARTIFACT_DIGEST_MISMATCH"
