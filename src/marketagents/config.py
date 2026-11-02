"""Konfiguracja wczytywana ze zmiennych środowiskowych (plik .env)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from marketagents.indicators import MovingAverages


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
    moving_averages: MovingAverages = MovingAverages()
    # Modele w kolejności preferencji, np. ("groq:openai/gpt-oss-120b", "gemini:gemini-3.8-flash").
    llm_models: tuple[str, ...] = ()
    # Czy pula ma używać tylko modeli z zaliczonym testem kwalifikacyjnym.
    require_qualified: bool = True
    results_dir: Path = Path("wyniki")


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
        moving_averages=MovingAverages.parse(
            os.getenv("SMA_WINDOWS", "20,50,200"), os.getenv("EMA_WINDOW", "20")
        ),
        llm_models=tuple(m.strip() for m in os.getenv("LLM_MODELS", "").split(",") if m.strip()),
        require_qualified=os.getenv("LLM_REQUIRE_QUALIFIED", "1") != "0",
        results_dir=Path(os.getenv("MARKETAGENTS_RESULTS_DIR", "wyniki")),
    )
