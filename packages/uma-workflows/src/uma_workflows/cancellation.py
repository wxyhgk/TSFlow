from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class CancellationToken(Protocol):
    """Minimal cancellation interface accepted by reusable workflows."""

    def is_set(self) -> bool: ...


class CalculationCancelled(RuntimeError):
    """Raised when a reusable workflow is cancelled by its caller."""


def raise_if_cancelled(cancel_token: CancellationToken) -> None:
    if cancel_token.is_set():
        raise CalculationCancelled("calculation cancelled")
