"""Konfiguracja wczytywana ze zmiennych środowiskowych (plik .env)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    cache_dir: Path
    offline: bool
    finnhub_api_key: str | None
    fred_api_key: str | None
    sec_user_agent: str | None
    llm_provider: str
    gemini_model: str
    groq_model: str
    llm_rpm: float


def load_settings() -> Settings:
    load_dotenv()
    return Settings(
        cache_dir=Path(os.getenv("MARKETAGENTS_CACHE_DIR", ".cache")),
        offline=os.getenv("MARKETAGENTS_OFFLINE", "0") == "1",
        finnhub_api_key=os.getenv("FINNHUB_API_KEY") or None,
        fred_api_key=os.getenv("FRED_API_KEY") or None,
        sec_user_agent=os.getenv("SEC_USER_AGENT") or None,
        llm_provider=os.getenv("LLM_PROVIDER", "gemini").lower(),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash"),
        groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        llm_rpm=float(os.getenv("LLM_RPM") or os.getenv("GEMINI_RPM") or "10"),
    )
