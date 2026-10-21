"""Compatibility imports; use :mod:`app.calculators.models` for new code."""

from app.calculators.models import (
    BASE_TASKS,
    MODEL_DEFINITIONS,
    V12_TASKS,
    ModelDefinition,
    ModelRegistry,
)

__all__ = [
    "BASE_TASKS",
    "MODEL_DEFINITIONS",
    "V12_TASKS",
    "ModelDefinition",
    "ModelRegistry",
]
