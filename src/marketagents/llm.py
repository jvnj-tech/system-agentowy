"""Połączenie z modelem językowym."""

from __future__ import annotations

import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter

from marketagents.config import Settings


def get_chat_model(settings: Settings, temperature: float = 0.2) -> BaseChatModel:
    """Gemini przez Google AI Studio; klucz w zmiennej GOOGLE_API_KEY.

    Darmowy plan ma limit zapytań na minutę, dlatego zapytania są równomiernie rozkładane
    w czasie (GEMINI_RPM), a pojedyncze zapytanie ma limit czasu.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    # Biblioteka google-genai ostrzega o funkcji, z której system nie korzysta.
    logging.getLogger("google_genai.models").setLevel(logging.ERROR)

    rate_limiter = None
    if settings.gemini_rpm > 0:
        rate_limiter = InMemoryRateLimiter(
            requests_per_second=settings.gemini_rpm / 60, check_every_n_seconds=0.2
        )
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        temperature=temperature,
        rate_limiter=rate_limiter,
        timeout=120,
        max_retries=3,
    )


def message_text(message) -> str:
    """Tekst odpowiedzi modelu; Gemini zwraca go czasem jako listę części."""
    content = message.content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return str(content)
