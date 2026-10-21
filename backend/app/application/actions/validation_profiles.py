from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class TransitionStateValidationProfile:
    profile_id: Literal["uma-ts-strict-v1"]
    stationary_fmax_ev_per_a: float
    imaginary_frequency_threshold_cm1: float
    maximum_hessian_asymmetry: float
    irc_path_fmax_ev_per_a: float
    irc_inner_fmax_ev_per_a: float
    irc_endpoint_fmax_ev_per_a: float


UMA_TS_STRICT_V1 = TransitionStateValidationProfile(
    profile_id="uma-ts-strict-v1",
    stationary_fmax_ev_per_a=0.05,
    imaginary_frequency_threshold_cm1=100.0,
    maximum_hessian_asymmetry=0.05,
    irc_path_fmax_ev_per_a=0.05,
    irc_inner_fmax_ev_per_a=0.01,
    irc_endpoint_fmax_ev_per_a=0.05,
)
