"""Composable route registration for the HTTP API."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from fastapi import APIRouter


RouterFactory = Callable[[], APIRouter]


@dataclass(frozen=True)
class RouteDefinition:
    """One independently replaceable router module."""

    name: str
    factory: RouterFactory
    prefix: str = ""
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("route definition name cannot be empty")
        if self.prefix and not self.prefix.startswith("/"):
            raise ValueError("route definition prefix must start with '/'")


def build_router(definitions: Iterable[RouteDefinition]) -> APIRouter:
    """Build an aggregate router from independently replaceable modules."""
    aggregate = APIRouter()
    for definition in definitions:
        aggregate.include_router(
            definition.factory(),
            prefix=definition.prefix,
            tags=list(definition.tags),
        )
    return aggregate
