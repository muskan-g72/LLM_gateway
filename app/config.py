from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


def _enabled(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    database_path: str
    groq_api_key: str
    groq_model: str
    gemini_api_key: str
    gemini_model: str
    provider_timeout_seconds: float
    force_primary_fail: bool


@lru_cache
def get_settings() -> Settings:
    timeout = float(os.getenv("PROVIDER_TIMEOUT_SECONDS", "30"))
    if timeout <= 0:
        raise ValueError("PROVIDER_TIMEOUT_SECONDS must be positive")

    return Settings(
        database_path=os.getenv("DATABASE_PATH", "gateway.db"),
        groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
        groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip(),
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite").strip(),
        provider_timeout_seconds=timeout,
        force_primary_fail=_enabled("FORCE_PRIMARY_FAIL"),
    )