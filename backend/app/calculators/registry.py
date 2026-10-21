from __future__ import annotations

from pathlib import Path
from typing import Any

from app.calculators.models import ModelDefinition, ModelRegistry


class CalculatorRegistry:
    """Resolve a registered model to its engine-specific provider."""

    def __init__(
        self,
        model_registry: ModelRegistry,
        reference_dir: Path,
        optimization_device: str,
    ):
        self.model_registry = model_registry
        self.reference_dir = reference_dir
        self.optimization_device = optimization_device

    def definition(self, model_name: str) -> ModelDefinition:
        definition = self.model_registry.get_definition(model_name)
        if definition is None:
            raise ValueError(f"unknown model_name: {model_name}")
        return definition

    def provider(self, model_name: str, checkpoint: Path | None = None) -> Any:
        definition = self.definition(model_name)
        if definition.engine == "uma":
            from app.adapters.uma import UmaExecutor

            return UmaExecutor(self.reference_dir, self.optimization_device)
        if definition.engine == "xtb":
            from app.adapters.xtb import XtbExecutor

            return XtbExecutor()
        if definition.engine == "gaussian":
            from app.adapters.gaussian import GaussianExecutor

            return GaussianExecutor()
        raise ValueError(f"unsupported calculator engine: {definition.engine}")
