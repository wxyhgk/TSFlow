from fastapi.testclient import TestClient

from app.main import app
from app.adapters.uma.structures import structure_fingerprint
from app.contracts.action_results import (
    GeneratePathCandidatesActionResult,
    PathCandidateSummary,
)
from app.contracts.structures import AtomicStructure


STRUCTURE = {
    "atoms": [
        {"symbol": "O", "position": [0.0, 0.0, 0.0]},
        {"symbol": "H", "position": [0.76, 0.58, 0.0]},
        {"symbol": "H", "position": [-0.76, 0.58, 0.0]},
    ],
    "charge": 0,
    "spin": 1,
}


def test_action_capabilities_only_advertise_executable_actions() -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/capabilities/actions")

    assert response.status_code == 200
    items = response.json()["items"]
    assert [item["action_type"] for item in items] == [
        "evaluate_energy_forces",
        "optimize_minimum",
        "analyze_frequencies",
        "generate_path_candidates",
        "optimize_path_candidate",
        "probe_path_curvature",
        "run_collective_coordinate_scan",
        "transition_state_workflow",
        "refine_saddle",
        "compute_full_hessian_and_modes",
        "run_bidirectional_irc",
    ]
    assert all(item["available"] is True for item in items)
    assert all(item["execution_backend"] == "job_bridge" for item in items)


def test_typed_action_is_bridged_to_existing_job() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/actions",
            json={
                "action_type": "optimize_minimum",
                "structure": STRUCTURE,
                "parameters": {"fmax": 0.03, "max_steps": 250},
            },
        )
        assert response.status_code == 202
        action = response.json()
        assert action["action_type"] == "optimize_minimum"
        assert action["id"] == action["job_id"]
        assert response.headers["location"].endswith(action["id"])

        job = client.get(action["job_url"])
        assert job.status_code == 200
        job_request = job.json()["request"]
        assert job_request["calculation"] == "geometry_optimization"
        assert job_request["parameters"]["fmax"] == 0.03
        assert job_request["parameters"]["max_steps"] == 250
        assert job.json()["execution_envelope"]["schema_version"] == 1
        assert (
            job.json()["execution_envelope"]["payload"]["action"]["action_type"]
            == "optimize_minimum"
        )

        fetched = client.get(f"/api/v1/actions/{action['id']}")
        assert fetched.status_code == 200
        assert fetched.json()["action_type"] == "optimize_minimum"


def test_action_union_rejects_unknown_or_irrelevant_parameters() -> None:
    with TestClient(app) as client:
        unknown = client.post(
            "/api/v1/actions",
            json={"action_type": "run_python", "structure": STRUCTURE},
        )
        irrelevant = client.post(
            "/api/v1/actions",
            json={
                "action_type": "optimize_minimum",
                "structure": STRUCTURE,
                "parameters": {"irc_steps": 20},
            },
        )

    assert unknown.status_code == 422
    assert irrelevant.status_code == 422


def test_transition_state_macro_keeps_server_validation_profile() -> None:
    product = STRUCTURE | {
        "atoms": [
            {"symbol": "O", "position": [0.0, 0.0, 0.0]},
            {"symbol": "H", "position": [0.90, 0.40, 0.0]},
            {"symbol": "H", "position": [-0.90, 0.40, 0.0]},
        ]
    }
    with TestClient(app) as client:
        accepted = client.post(
            "/api/v1/actions",
            json={
                "action_type": "transition_state_workflow",
                "structure": STRUCTURE,
                "product_structure": product,
                "parameters": {
                    "sella_cpu_fmax": 1.5,
                    "imaginary_frequency_threshold_cm1": 1.0,
                    "maximum_hessian_asymmetry": 0.9,
                    "irc_fmax": 1.5,
                    "irc_inner_fmax": 1.5,
                    "irc_endpoint_relaxation_fmax": 1.5,
                },
            },
        )
        rejected = client.post(
            "/api/v1/actions",
            json={
                "action_type": "transition_state_workflow",
                "structure": STRUCTURE,
                "product_structure": product,
                "validation_profile_id": "agent-relaxed",
            },
        )
        normalized = client.get(accepted.json()["job_url"])

    assert accepted.status_code == 202
    assert rejected.status_code == 422
    parameters = normalized.json()["request"]["parameters"]
    assert parameters["sella_cpu_fmax"] == 0.05
    assert parameters["imaginary_frequency_threshold_cm1"] == 100.0
    assert parameters["maximum_hessian_asymmetry"] == 0.05
    assert parameters["irc_fmax"] == 0.05
    assert parameters["irc_inner_fmax"] == 0.01
    assert parameters["irc_endpoint_relaxation_fmax"] == 0.05


def test_generate_path_candidates_is_persisted_as_a_typed_primitive() -> None:
    product = STRUCTURE | {
        "atoms": [
            {"symbol": "O", "position": [0.0, 0.0, 0.0]},
            {"symbol": "H", "position": [0.90, 0.40, 0.0]},
            {"symbol": "H", "position": [-0.90, 0.40, 0.0]},
        ]
    }
    with TestClient(app) as client:
        enable_stub_worker(client)
        response = client.post(
            "/api/v1/actions",
            json={
                "action_type": "generate_path_candidates",
                "structure": STRUCTURE,
                "product_structure": product,
                "parameters": {
                    "initializer_id": "fbenm.correlated",
                    "nmove": 8,
                    "ensemble_size": 2,
                    "evaluation_device": "cpu",
                    "path_bias_mode": "native",
                },
            },
        )
        assert response.status_code == 202
        job = client.get(response.json()["job_url"])

    payload = job.json()
    assert payload["request"]["calculation"] == "transition_state_search"
    assert payload["request"]["parameters"]["path_initializer_id"] == (
        "fbenm.correlated"
    )
    assert payload["request"]["parameters"]["nmove"] == 8
    assert payload["execution_envelope"]["payload"]["action"]["action_type"] == (
        "generate_path_candidates"
    )


def test_optimize_path_candidate_resolves_server_owned_source_artifact() -> None:
    product = STRUCTURE | {
        "atoms": [
            {"symbol": "O", "position": [0.0, 0.0, 0.0]},
            {"symbol": "H", "position": [0.90, 0.40, 0.0]},
            {"symbol": "H", "position": [-0.90, 0.40, 0.0]},
        ]
    }
    reactant_model = AtomicStructure.model_validate(STRUCTURE)
    product_model = AtomicStructure.model_validate(product)
    with TestClient(app) as client:
        enable_stub_worker(client)
        generated = client.post(
            "/api/v1/actions",
            json={
                "action_type": "generate_path_candidates",
                "structure": STRUCTURE,
                "product_structure": product,
                "parameters": {"ensemble_size": 1},
            },
        )
        source_id = generated.json()["id"]
        store = client.app.state.artifact_store
        reference = store.write_json(
            __import__("uuid").UUID(source_id),
            "path-candidate-0",
            "path_candidate",
            {
                "schema_version": 1,
                "model_name": "uma-s-1p2p1",
                "task_name": "omol",
                "reactant_sha256": structure_fingerprint(reactant_model),
                "product_sha256": structure_fingerprint(product_model),
                "atom_symbols": ["O", "H", "H"],
                "charge": 0,
                "spin": 1,
                "initializer_algorithm_id": "fbenm.correlated",
                "initializer_algorithm_version": "1",
                "candidate_index": 0,
                "score_ev": 0.1,
                "source": "fbenm.correlated",
                "path": {
                    "positions_a": [
                        [atom["position"] for atom in STRUCTURE["atoms"]],
                        [
                            [0.0, 0.0, 0.0],
                            [0.83, 0.49, 0.0],
                            [-0.83, 0.49, 0.0],
                        ],
                        [atom["position"] for atom in product["atoms"]],
                    ],
                    "coordinates": [0.0, 0.5, 1.0],
                    "tangents_a": None,
                    "metadata": {},
                },
                "parameterization": {
                    "kind": "pydmf.bspline",
                    "coefficients": [
                        [atom["position"] for atom in STRUCTURE["atoms"]],
                        [
                            [0.0, 0.0, 0.0],
                            [0.83, 0.49, 0.0],
                            [-0.83, 0.49, 0.0],
                        ],
                        [atom["position"] for atom in product["atoms"]],
                    ],
                },
                "reaction_coordinates": [],
                "warnings": [],
                "metadata": {},
            },
            frame_count=3,
        )
        repository = client.app.state.job_repository
        client.portal.call(repository.start, __import__("uuid").UUID(source_id))
        completed = client.portal.call(
            repository.succeed,
            __import__("uuid").UUID(source_id),
            GeneratePathCandidatesActionResult(
                initializer_algorithm_id="fbenm.correlated",
                initializer_algorithm_version="1",
                candidates=[PathCandidateSummary(
                    candidate_index=0,
                    score_ev=0.1,
                    source="fbenm.correlated",
                    frame_count=3,
                    parameterization_kind="pydmf.bspline",
                    preview_structure=reactant_model,
                    path_artifact=reference,
                )],
                elapsed_seconds=0.1,
            ),
        )
        assert completed is not None
        assert {item.artifact_id for item in completed.artifacts} >= {
            "path-candidate-0"
        }
        response = client.post(
            "/api/v1/actions",
            json={
                "action_type": "optimize_path_candidate",
                "structure": STRUCTURE,
                "product_structure": product,
                "path_candidate": {
                    "source_action_id": source_id,
                    "artifact_id": reference.artifact_id,
                    "expected_sha256": reference.sha256,
                },
            },
        )
        assert response.status_code == 202, response.json()
        job = client.get(response.json()["job_url"]).json()

    resolved = job["execution_envelope"]["payload"]["resolved_artifacts"]
    assert resolved["path_candidate"]["sha256"] == reference.sha256
    assert job["execution_envelope"]["payload"]["resolved_algorithms"] == {
        "path_optimizer": "1"
    }


def test_action_idempotency_replays_same_job_and_rejects_key_reuse() -> None:
    first_payload = {
        "action_type": "optimize_minimum",
        "structure": STRUCTURE,
        "parameters": {"fmax": 0.03, "max_steps": 100},
    }
    with TestClient(app) as client:
        first = client.post(
            "/api/v1/actions",
            json=first_payload,
            headers={"Idempotency-Key": "agent-step-1"},
        )
        replay = client.post(
            "/api/v1/actions",
            json=first_payload,
            headers={"Idempotency-Key": "agent-step-1"},
        )
        conflict = client.post(
            "/api/v1/actions",
            json=first_payload
            | {"parameters": {"fmax": 0.02, "max_steps": 100}},
            headers={"Idempotency-Key": "agent-step-1"},
        )

    assert first.status_code == 202
    assert first.headers["idempotent-replayed"] == "false"
    assert replay.status_code == 202
    assert replay.headers["idempotent-replayed"] == "true"
    assert replay.json()["id"] == first.json()["id"]
    assert conflict.status_code == 409

