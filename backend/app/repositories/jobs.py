"""Compatibility import for the relocated in-memory repository."""

from app.infrastructure.persistence.memory import InMemoryJobRepository

__all__ = ["InMemoryJobRepository"]
