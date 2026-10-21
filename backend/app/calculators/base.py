from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class CalculatorBundle:
    """Calculators and capability metadata for one model selection."""

    engine: str
    method: str
    calculator: Any
    hessian_calculator: Any | None = None
    capabilities: frozenset[str] = field(default_factory=frozenset)
    warnings: tuple[str, ...] = ()


class CalculatorProvider(Protocol):
    engine: str

    def create(self, *, device: str, task_name: str) -> CalculatorBundle:
        """Build calculators for an ASE-compatible workflow."""
