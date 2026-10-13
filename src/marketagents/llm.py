"""Połączenie z modelem językowym."""

from __future__ import annotations

import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter

from marketagents.config import Settings

PROVIDERS = ("gemini", "groq")


# Ponowienia po odrzuceniu zapytania z powodu limitu (błąd 429) lub chwilowej awarii.
MAX_RETRIES = 8


def get_chat_model(settings: Settings, temperature: float = 0.2) -> BaseChatModel:
    """Model wybrany zmienną LLM_PROVIDER: Gemini (GOOGLE_API_KEY) albo Groq (GROQ_API_KEY).

    Darmowe plany mają limity zapytań na minutę, dlatego zapytania są równomiernie rozkładane
    w czasie (LLM_RPM), a pojedyncze zapytanie ma limit czasu. Groq ogranicza też liczbę
    tokenów na minutę: czterech analityków pracujących równolegle potrafi ją przekroczyć,
    więc odrzucone zapytanie jest ponawiane (biblioteka czeka tyle, ile poda serwer).
    """
    rate_limiter = None
    if settings.llm_rpm > 0:
        rate_limiter = InMemoryRateLimiter(
            requests_per_second=settings.llm_rpm / 60, check_every_n_seconds=0.2
        )

    if settings.llm_provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=settings.groq_model,
            temperature=temperature,
            rate_limiter=rate_limiter,
            timeout=120,
            max_retries=MAX_RETRIES,
        )
    if settings.llm_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        # Biblioteka google-genai ostrzega o funkcji, z której system nie korzysta.
        logging.getLogger("google_genai.models").setLevel(logging.ERROR)
        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            temperature=temperature,
            rate_limiter=rate_limiter,
            timeout=120,
            max_retries=MAX_RETRIES,
        )
    raise ValueError(
        f"Nieznany dostawca modelu: {settings.llm_provider}; dostępne: {', '.join(PROVIDERS)}"
    )


def message_text(message) -> str:
    """Tekst odpowiedzi modelu; Gemini zwraca go czasem jako listę części."""
    content = message.content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return str(content)
