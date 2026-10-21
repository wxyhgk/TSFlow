from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="UMA_",
        extra="ignore",
    )

    app_name: str = "UMA Atomistic Simulation API"
    environment: str = "development"
    log_level: str = "INFO"
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://localhost:5173",
        ]
    )
    checkpoint_dir: Path = PROJECT_ROOT / "model-assets" / "UMA" / "checkpoints"
    reference_dir: Path = PROJECT_ROOT / "model-assets" / "UMA" / "references"
    job_archive_dir: Path = PROJECT_ROOT / "data" / "jobs"
    job_database_path: Path = PROJECT_ROOT / "data" / "jobs.sqlite3"
    job_repository_backend: Literal["sqlite", "memory"] = "sqlite"
    artifact_dir: Path = PROJECT_ROOT / "data" / "artifacts"
    worker_enabled: bool = False
    # Keep one child process alive so UmaCalculatorFactory can reuse predictors.
    # A value of 0 means unlimited tasks until idle timeout or shutdown.
    worker_max_tasks_per_process: int = Field(default=0, ge=0, le=10_000)
    worker_idle_timeout_seconds: float = Field(default=1_800.0, gt=0, le=86_400)
    # On Apple Silicon, auto uses the validated hybrid path: MPS supplies the
    # forces during TRIC optimization and CPU recomputes the final result.
    optimization_device: Literal["auto", "cpu", "mps"] = "auto"


@lru_cache
def get_settings() -> Settings:
    return Settings()
