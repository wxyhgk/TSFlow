"""Compatibility import for the relocated artifact store."""

from app.infrastructure.artifacts.store import FileArtifactStore

__all__ = ["FileArtifactStore"]
