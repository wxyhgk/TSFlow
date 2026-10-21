from dataclasses import dataclass
from pathlib import Path
import importlib.util
import os
from shutil import which

from app.contracts.models import ModelInfo, TaskName


BASE_TASKS: list[TaskName] = ["omol", "omat", "omc", "oc20", "odac"]
V12_TASKS: list[TaskName] = [*BASE_TASKS, "oc22", "oc25"]


@dataclass(frozen=True)
class ModelDefinition:
    name: str
    version: str
    engine: str
    method: str
    filename: str | None
    supported_tasks: list[TaskName]
    capabilities: list[str]
    executable: str | None = None
    recommended: bool = False
    archived: bool = False
    note: str | None = None


MODEL_DEFINITIONS = (
    ModelDefinition(
        "uma-s-1p2p1",
        "1.2.1",
        "uma",
        "UMA-S-1.2.1",
        "uma-s-1p2p1.pt",
        V12_TASKS,
        [
            "single_point",
            "geometry_optimization",
            "frequency_analysis",
            "transition_state_search",
            "irc",
        ],
        recommended=True,
    ),
    ModelDefinition(
        "uma-s-1p2",
        "1.2",
        "uma",
        "UMA-S-1.2",
        "uma-s-1p2.pt",
        V12_TASKS,
        [
            "single_point",
            "geometry_optimization",
            "frequency_analysis",
            "transition_state_search",
            "irc",
        ],
    ),
    ModelDefinition(
        "uma-s-1p1",
        "1.1",
        "uma",
        "UMA-S-1.1",
        "uma-s-1p1.pt",
        BASE_TASKS,
        [
            "single_point",
            "geometry_optimization",
            "frequency_analysis",
            "transition_state_search",
            "irc",
        ],
    ),
    ModelDefinition(
        "uma-m-1p1",
        "1.1",
        "uma",
        "UMA-M-1.1",
        "uma-m-1p1.pt",
        BASE_TASKS,
        [
            "single_point",
            "geometry_optimization",
            "frequency_analysis",
            "transition_state_search",
            "irc",
        ],
        note="Higher capacity; requires substantially more accelerator memory.",
    ),
    ModelDefinition(
        "uma-s-1",
        "1.0",
        "uma",
        "UMA-S-1.0",
        "uma-s-1.pt",
        BASE_TASKS,
        [
            "single_point",
            "geometry_optimization",
            "frequency_analysis",
            "transition_state_search",
            "irc",
        ],
        archived=True,
        note="Known extensivity bug; retained for reference only.",
    ),
    ModelDefinition(
        "xtb-gfn2",
        "6.7",
        "xtb",
        "GFN2-xTB",
        None,
        BASE_TASKS,
        [
            "single_point",
            "geometry_optimization",
            "frequency_analysis",
            "transition_state_search",
            "irc",
        ],
        note="Fast semiempirical calculator. Install the optional Python xtb package before use.",
    ),
    ModelDefinition(
        "gaussian-b3lyp-def2svp",
        "external",
        "gaussian",
        "B3LYP/def2SVP",
        None,
        BASE_TASKS,
        [
            "single_point",
            "geometry_optimization",
            "frequency_analysis",
            "transition_state_search",
            "irc",
        ],
        executable="g16",
        note="External Gaussian executable; use xTB or UMA for path screening and Gaussian for validation.",
    ),
)


class ModelRegistry:
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self._definitions = {item.name: item for item in MODEL_DEFINITIONS}

    def list_models(self) -> list[ModelInfo]:
        return [self._to_info(item) for item in MODEL_DEFINITIONS]

    def get(self, name: str) -> ModelInfo | None:
        definition = self._definitions.get(name)
        return self._to_info(definition) if definition else None

    def get_definition(self, name: str) -> ModelDefinition | None:
        return self._definitions.get(name)

    def _to_info(self, definition: ModelDefinition) -> ModelInfo:
        path = (
            self.checkpoint_dir / definition.filename if definition.filename else None
        )
        executable = self._executable(definition)
        available, reason = self._availability(definition, path, executable)
        return ModelInfo(
            name=definition.name,
            version=definition.version,
            engine=definition.engine,
            method=definition.method,
            checkpoint=str(path) if path else None,
            executable=executable,
            size_bytes=path.stat().st_size if path and path.is_file() else None,
            available=available,
            recommended=definition.recommended,
            archived=definition.archived,
            supported_tasks=[
                task for task in definition.supported_tasks if task == "omol"
            ],
            capabilities=definition.capabilities,
            availability_reason=reason,
            note=definition.note,
        )

    @staticmethod
    def _executable(definition: ModelDefinition) -> str | None:
        if definition.engine == "gaussian":
            configured = os.environ.get("GAUSSIAN_EXE")
            return configured or which(definition.executable or "g16")
        return None

    @staticmethod
    def _availability(
        definition: ModelDefinition,
        path: Path | None,
        executable: str | None,
    ) -> tuple[bool, str | None]:
        if definition.engine == "uma":
            if path is not None and path.is_file():
                return True, None
            return False, "UMA checkpoint is not available"
        if definition.engine == "xtb":
            if importlib.util.find_spec("xtb") is None:
                return False, "Python package 'xtb' is not installed"
            return True, None
        if executable:
            return True, None
        return False, "Gaussian executable was not found; set GAUSSIAN_EXE"
