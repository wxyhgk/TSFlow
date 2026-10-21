"""API route package and default route composition."""

from __future__ import annotations

from collections.abc import Iterable

from fastapi import APIRouter

from app.api.routes import actions, health, jobs, models
from app.api.routes.registry import RouteDefinition, build_router


DEFAULT_ROUTE_DEFINITIONS: tuple[RouteDefinition, ...] = (
    RouteDefinition("health", lambda: health.router, tags=("system",)),
    RouteDefinition(
        "models",
        lambda: models.router,
        prefix="/models",
        tags=("models",),
    ),
    RouteDefinition("actions", lambda: actions.router, tags=("actions",)),
    RouteDefinition("jobs", lambda: jobs.router, tags=("jobs",)),
)


def create_router(
    definitions: Iterable[RouteDefinition] = DEFAULT_ROUTE_DEFINITIONS,
) -> APIRouter:
    """Create the public API router from the selected route modules."""
    return build_router(definitions)


__all__ = [
    "DEFAULT_ROUTE_DEFINITIONS",
    "RouteDefinition",
    "build_router",
    "create_router",
]

