from fastapi import APIRouter

from app.api.routes import create_router
from app.api.routes.registry import RouteDefinition, build_router


def test_default_route_package_preserves_public_route_groups() -> None:
    paths = {route.path for route in create_router().routes}

    assert "/health" in paths
    assert "/models" in paths
    assert "/actions" in paths
    assert "/jobs" in paths


def test_route_registry_allows_replacing_a_module() -> None:
    replacement = APIRouter()

    @replacement.get("/health")
    async def replacement_health() -> dict[str, str]:
        return {"status": "replacement"}

    router = build_router(
        [RouteDefinition("health", lambda: replacement, tags=("system",))]
    )

    assert {route.path for route in router.routes} == {"/health"}


def test_route_definition_rejects_invalid_prefix() -> None:
    try:
        RouteDefinition("health", lambda: APIRouter(), prefix="health")
    except ValueError as exc:
        assert "prefix" in str(exc)
    else:
        raise AssertionError("invalid route prefix was accepted")
