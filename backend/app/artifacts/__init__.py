"""Deprecated artifact imports; use ``app.infrastructure.artifacts``."""

from app.infrastructure.artifacts import ArtifactHydrationError, FileArtifactStore

__all__ = ["ArtifactHydrationError", "FileArtifactStore"]
