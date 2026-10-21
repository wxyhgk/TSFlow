from __future__ import annotations

import ast
from pathlib import Path

from app.schemas.jobs import JobCreate
from app.services.transition_state_search.request_adapter import (
    config_from_request,
)


PURE_STAGE_MODULES = (
    "endpoints.py",
    "metrics.py",
    "models.py",
    "path.py",
    "path_bias.py",
    "runner.py",
    "saddle.py",
    "screening.py",
)


def test_request_adapter_maps_transition_search_parameters() -> None:
    request = JobCreate.model_validate(
        {
            "calculation": "transition_state_search",
            "structure": {
                "atoms": [{"symbol": "H", "position": [0, 0, 0]}],
                "charge": 0,
                "spin": 2,
            },
            "product_structure": {
                "atoms": [{"symbol": "H", "position": [0.5, 0, 0]}],
                "charge": 0,
                "spin": 2,
            },
            "parameters": {
                "path_initializer_id": "test.initializer",
                "path_optimizer_id": "test.optimizer",
                "endpoint_preopt": False,
                "nmove": 7,
                "path_bias_mode": "native",
                "path_points": 19,
                "near_barrierless_threshold_ev": 0.025,
                "sella_cpu_refine": True,
                "saddle_final_refinement_engine": "geometric",
                "frequency_max_atoms": 42,
                "irc_follow": True,
                "irc_engine": "geometric",
            },
        }
    )

    config = config_from_request(request)

    assert config.endpoint.enabled is False
    assert config.path.initializer_id == "test.initializer"
    assert config.path.optimizer_id == "test.optimizer"
    assert config.path.nmove == 7
    assert config.path.bias_mode == "native"
    assert config.path.path_points == 19
    assert config.path.near_barrierless_threshold_ev == pytest.approx(0.025)
    assert config.saddle.cpu_refine is True
    assert config.saddle.final_refinement_engine == "geometric"
    assert config.screening.max_atoms == 42
    assert config.screening.force_full_hessian is True
    assert config.screening.hessian_method == "auto"
    assert config.screening.maximum_hessian_asymmetry == pytest.approx(0.05)
    assert config.irc.enabled is True
    assert config.irc.engine == "geometric"


def test_computational_stages_do_not_import_api_schemas() -> None:
    service_dir = (
        Path(__file__).parents[2]
        / "packages"
        / "uma-workflows"
        / "src"
        / "uma_workflows"
        / "transition_states"
        / "workflow"
    )

    for filename in PURE_STAGE_MODULES:
        tree = ast.parse((service_dir / filename).read_text())
        imports = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        assert not any(
            module == "app" or module.startswith("app.")
            for module in imports
        ), filename
