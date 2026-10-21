from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np


Device = Literal["cpu", "mps"]


@dataclass(frozen=True)
class EndpointConfig:
    enabled: bool = True
    fmax: float = 0.10
    max_steps: int = 50


@dataclass(frozen=True)
class PathConfig:
    initializer_id: str = "fbenm.correlated"
    optimizer_id: str = "dmf.direct_maxflux"
    nmove: int = 10
    ensemble_size: int = 3
    max_iterations: int = 200
    convergence: Literal["tight", "middle", "loose"] = "middle"
    path_points: int = 31
    update_teval: bool = True
    accept_every_trial_step: bool = False
    bias_mode: Literal["native", "asynchronous", "legacy_synchronous"] = (
        "asynchronous"
    )
    repulsion_strength_ev_per_a2: float = 75.0
    repulsion_cutoff_scale: float = 0.80
    synchronization_strength_ev: float = 5.0
    coordinate_range_strength_ev: float = 100.0
    coordinate_schedule_strength_ev: float = 20.0
    scaffold_strength_ev_per_a2: float = 5.0
    near_barrierless_threshold_ev: float = 0.05


@dataclass(frozen=True)
class SaddleConfig:
    enabled: bool = True
    final_refinement_engine: Literal["geometric", "sella"] = "geometric"
    mps_fmax: float = 0.10
    cpu_refine: bool = False
    cpu_fmax: float = 0.05
    max_steps: int = 100
    active_region_hops: int | None = None
    path_mode_only: bool = False
    relaxed_scan: bool = True
    relaxed_scan_progress: float = 0.50
    relaxed_scan_fmax: float = 0.05
    relaxed_scan_max_steps: int = 100
    automatic_fallback: bool = True
    fallback_nearby_frames: int = 7
    fallback_max_candidates: int = 3


@dataclass(frozen=True)
class ScreeningConfig:
    compute_frequencies: bool = True
    active_region_hops: int = 2
    max_atoms: int = 30
    delta_a: float = 0.01
    imaginary_threshold_cm1: float = 100.0
    force_full_hessian: bool = False
    hessian_method: Literal["auto", "analytic", "finite_difference"] = "auto"
    maximum_hessian_asymmetry: float = 0.05


@dataclass(frozen=True)
class IrcConfig:
    enabled: bool = False
    engine: Literal["geometric", "sella"] = "geometric"
    steps: int = 30
    dx: float = 0.12
    inner_steps: int = 10
    inner_fmax: float = 0.01
    fmax: float = 0.05
    endpoint_distance_threshold_a: float = 0.02
    require_endpoint_match: bool = True
    relax_endpoints: bool = True
    endpoint_relaxation_fmax: float = 0.05
    endpoint_relaxation_max_steps: int = 100


@dataclass(frozen=True)
class TransitionSearchConfig:
    endpoint: EndpointConfig = field(default_factory=EndpointConfig)
    path: PathConfig = field(default_factory=PathConfig)
    saddle: SaddleConfig = field(default_factory=SaddleConfig)
    screening: ScreeningConfig = field(default_factory=ScreeningConfig)
    irc: IrcConfig = field(default_factory=IrcConfig)


@dataclass(frozen=True)
class WorkflowProgress:
    stage: str
    step: int
    max_steps: int
    energy_ev: float
    max_force_ev_per_a: float
    device: Device
    positions_a: np.ndarray
    name: str
    path_energies_ev: tuple[float, ...] = ()
    reactant_positions_a: np.ndarray | None = None
    product_positions_a: np.ndarray | None = None


WorkflowProgressCallback = Callable[[WorkflowProgress], None]
