from typing import Literal

from pydantic import BaseModel, Field


class TransitionStateAlgorithmCapability(BaseModel):
    algorithm_id: str
    version: str
    display_name: str
    role: Literal["path_initializer", "path_optimizer"]
    available: bool = True
    unavailable_reason: str | None = None
    selector_parameter: Literal[
        "path_initializer_id",
        "path_optimizer_id",
    ]
    config_type: str
    capabilities: list[str] = Field(default_factory=list)
    accepts_unparameterized: bool
    accepted_parameterizations: list[str] | None = None
    produced_parameterization: str | None = None


class TransitionStateAlgorithmCapabilitiesResponse(BaseModel):
    items: list[TransitionStateAlgorithmCapability]
