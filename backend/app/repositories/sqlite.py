"""Compatibility import for the relocated SQLite repository."""

from app.infrastructure.persistence.sqlite import SQLiteJobRepository

__all__ = ["SQLiteJobRepository"]
