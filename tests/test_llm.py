from dataclasses import replace
from pathlib import Path

import pytest

from marketagents.config import Settings
from marketagents.llm import get_chat_model

SETTINGS = Settings(
    cache_dir=Path(".cache"),
    offline=True,
    finnhub_api_key=None,
    fred_api_key=None,
    sec_user_agent=None,
    llm_provider="gemini",
    gemini_model="gemini-3.5-flash",
    groq_model="openai/gpt-oss-120b",
    llm_rpm=0,
)


def test_groq_provider(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test")
    llm = get_chat_model(replace(SETTINGS, llm_provider="groq"))
    assert type(llm).__name__ == "ChatGroq"
    assert llm.model_name == "openai/gpt-oss-120b"


def test_unknown_provider():
    with pytest.raises(ValueError, match="gemini, groq"):
        get_chat_model(replace(SETTINGS, llm_provider="inny"))
