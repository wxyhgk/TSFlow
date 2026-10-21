from __future__ import annotations

import importlib
import sys
from collections.abc import Callable

import numpy as np

# PyTorch and Homebrew IPOPT can link different OpenMP runtimes on macOS.
# Loading an installed PyTorch first lets dyld reuse the active runtime. The
# standalone DMF extra still works when PyTorch is not installed.
if sys.platform == "darwin":
    try:
        importlib.import_module("torch")
    except ImportError:
        pass

from dmf import DirectMaxFlux, interpolate_fbenm

from uma_workflows.cancellation import CancellationToken
from uma_workflows.transition_states.path_bias import (
    PathBias,
    dense_path_bias_value_and_gradient,
)

class ReportingDirectMaxFlux(DirectMaxFlux):
    def __init__(
        self,
        *args,
        cancel_token: CancellationToken,
        iteration_callback: Callable[[int, "ReportingDirectMaxFlux"], None],
        path_bias: PathBias,
        path_bias_samples: int,
        **kwargs,
    ) -> None:
        self._cancel_token = cancel_token
        self._iteration_callback = iteration_callback
        self._path_bias = path_bias
        self.teval_fallback_reason: str | None = None
        super().__init__(*args, **kwargs)
        coordinates = np.linspace(0.0, 1.0, path_bias_samples)
        self._path_bias_coordinates = coordinates
        self._path_bias_basis = self._get_basis_values(coordinates)[0]

    def _get_objective(self):
        physical_objective = super()._get_objective()
        positions = self.get_positions(P=(self._path_bias_basis,))
        bias_value, _ = dense_path_bias_value_and_gradient(
            self._path_bias,
            positions,
            self._path_bias_basis,
            self._path_bias_coordinates,
        )
        return physical_objective + bias_value

    def _get_grad_objective(self):
        physical_gradient = super()._get_grad_objective()
        positions = self.get_positions(P=(self._path_bias_basis,))
        _, bias_gradient = dense_path_bias_value_and_gradient(
            self._path_bias,
            positions,
            self._path_bias_basis,
            self._path_bias_coordinates,
        )
        return physical_gradient + bias_gradient

    def intermediate(self, *args):
        try:
            super().intermediate(*args)
        except (FloatingPointError, ValueError) as exc:
            if not self.update_teval:
                raise
            self.update_teval = False
            self.teval_fallback_reason = str(exc)
        self._iteration_callback(int(args[1]), self)
        return not self._cancel_token.is_set()


def validate_finite_dmf_state(path: DirectMaxFlux) -> None:
    for label, values in (
        ("coordinates", path.coefs),
        ("energies", path.energies),
        ("forces", path.forces),
    ):
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"DMF produced non-finite {label}")
