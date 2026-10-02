"""Połączenie z modelem językowym."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from marketagents.config import Settings


def get_chat_model(settings: Settings, temperature: float = 0.2) -> BaseChatModel:
    """Gemini przez Google AI Studio; klucz w zmiennej GOOGLE_API_KEY."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=settings.gemini_model, temperature=temperature)
