"""Runtime configuration (environment variables with sensible defaults)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    """All settings can be overridden via environment variables prefixed with ``COVTYPE_``."""

    artifacts_dir: Path = field(
        default_factory=lambda: Path(os.getenv("COVTYPE_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))
    )
    frontend_dir: Path = field(
        default_factory=lambda: Path(os.getenv("COVTYPE_FRONTEND_DIR", PROJECT_ROOT / "frontend"))
    )
    # "cpu" is the safe default for serving single requests; set "cuda" or "auto" if desired.
    device: str = field(default_factory=lambda: os.getenv("COVTYPE_DEVICE", "cpu"))
    cors_origins: list[str] = field(
        default_factory=lambda: _split_csv(os.getenv("COVTYPE_CORS_ORIGINS", "*"))
    )
    max_batch_size: int = field(default_factory=lambda: int(os.getenv("COVTYPE_MAX_BATCH_SIZE", "10000")))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
