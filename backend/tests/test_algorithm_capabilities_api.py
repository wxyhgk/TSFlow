from fastapi.testclient import TestClient

from app.main import app


STRUCTURE = {
    "atoms": [
        {"symbol": "H", "position": [-1.0, 0.0, 0.0]},
        {"symbol": "H", "position": [1.0, 0.0, 0.0]},
    ],
    "charge": 0,
    "spin": 1,
}
PRODUCT = STRUCTURE | {
    "atoms": [
        {"symbol": "H", "position": [-0.8, 0.0, 0.0]},
        {"symbol": "H", "position": [0.8, 0.0, 0.0]},
    ]
}


class StubWorker:
    async def submit(self, _job_id) -> None:
        return None

    async def stop(self) -> None:
        return None

    def cancel(self, _job_id) -> None:
        return None


def test_algorithm_capabilities_expose_registered_selectors() -> None:
    with TestClient(app) as client:
        response = client.get(
            "/api/v1/capabilities/transition-state-algorithms"
        )

    assert response.status_code == 200
    items = response.json()["items"]
    assert [item["algorithm_id"] for item in items] == [
        "fbenm.correlated",
        "dmf.direct_maxflux",
    ]
    assert [item["selector_parameter"] for item in items] == [
        "path_initializer_id",
        "path_optimizer_id",
    ]
    assert items[0]["produced_parameterization"] == "pydmf.bspline"
    assert items[1]["accepted_parameterizations"] == ["pydmf.bspline"]


def test_unknown_algorithm_is_rejected_before_job_creation() -> None:
    with TestClient(app) as client:
        client.app.state.job_worker = StubWorker()
        response = client.post(
            "/api/v1/actions",
            json={
                "action_type": "transition_state_workflow",
                "structure": STRUCTURE,
                "product_structure": PRODUCT,
                "parameters": {"path_initializer_id": "unknown.initializer"},
            },
        )

    assert response.status_code == 422
    assert response.json()["detail"] == {
        "code": "UNKNOWN_PATH_INITIALIZER",
        "algorithm_id": "unknown.initializer",
    }


def test_selected_algorithms_are_preserved_in_execution_envelope() -> None:
    with TestClient(app) as client:
        client.app.state.job_worker = StubWorker()
        response = client.post(
            "/api/v1/actions",
            json={
                "action_type": "transition_state_workflow",
                "structure": STRUCTURE,
                "product_structure": PRODUCT,
                "parameters": {
                    "path_initializer_id": "fbenm.correlated",
                    "path_optimizer_id": "dmf.direct_maxflux",
                },
            },
        )
        job = client.get(response.json()["job_url"])

    assert response.status_code == 202
    action = job.json()["execution_envelope"]["payload"]["action"]
    assert action["parameters"]["path_initializer_id"] == "fbenm.correlated"
    assert action["parameters"]["path_optimizer_id"] == "dmf.direct_maxflux"
    request_parameters = job.json()["request"]["parameters"]
    assert request_parameters["path_initializer_id"] == "fbenm.correlated"
    assert request_parameters["path_optimizer_id"] == "dmf.direct_maxflux"
