from typing import Literal

from pydantic import BaseModel, Field


TaskName = Literal["omol"]
CalculatorEngine = Literal["uma", "xtb", "gaussian"]


class ModelInfo(BaseModel):
    name: str
    version: str
    engine: CalculatorEngine
    method: str
    checkpoint: str | None
    executable: str | None = None
    size_bytes: int | None
    available: bool
    recommended: bool = False
    archived: bool = False
    supported_tasks: list[TaskName]
    capabilities: list[str] = Field(default_factory=list)
    availability_reason: str | None = None
    note: str | None = None


class ModelListResponse(BaseModel):
    items: list[ModelInfo]
