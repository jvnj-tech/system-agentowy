"""Połączenie z modelami językowymi.

System korzysta z listy modeli (LLM_MODELS) w kolejności preferencji. Darmowe plany mają
limity zapytań i tokenów na minutę i na dobę, dlatego zapytanie, którego model nie może
przyjąć (limit, przeciążenie serwera), trafia do następnego modelu z listy, a wyczerpany
model jest pomijany do czasu odnowienia limitu podanego przez serwer. Używane są tylko
modele, które przeszły test kwalifikacyjny (moduł qualification), żeby po przełączeniu
jakość raportu nie spadła poniżej przyjętego progu.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections.abc import Iterable
from pathlib import Path

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.rate_limiters import InMemoryRateLimiter
from pydantic import ConfigDict, PrivateAttr

from marketagents.config import Settings

PROVIDERS = ("gemini", "groq")
log = logging.getLogger(__name__)

# Ponowienia wewnątrz biblioteki dostawcy (chwilowe błędy sieci). Odrzucenia z powodu
# limitu obsługuje pula: limit minutowy odczekuje na tym samym modelu (żeby raport pisał
# jeden model), a limit dobowy i przeciążenie serwera przełączają na następny model.
MAX_RETRIES = 2
MINUTE_WAITS, MAX_WAIT_S = 3, 60
# Najdłuższa odpowiedź modelu w Groq (z tokenami rozumowania); raport ma zwykle ok. 2 tys.
GROQ_MAX_TOKENS = 3200
OVERLOAD_WAITS, OVERLOAD_WAIT_S = 2, 15
# Czas wyłączenia modelu, gdy serwer nie podał, kiedy limit się odnowi.
# „too_large”: zapytanie przekracza limit planu niezależnie od czasu (np. limit odpowiedzi
# na minutę mniejszy niż odpowiedź analityka), więc model jest pomijany dłużej.
DEFAULT_BLOCK_S = {"day": 3600, "overloaded": 300, "minute": 60, "too_large": 6 * 3600}

LIMIT_PATTERNS = re.compile(
    r"429|rate.?limit|RESOURCE_EXHAUSTED|quota|50[03]|UNAVAILABLE|INTERNAL|overloaded"
    r"|high demand",
    re.IGNORECASE,
)
RETRY_IN = re.compile(r"try again in (?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?", re.IGNORECASE)
RETRY_DELAY = re.compile(r"retry(?:Delay|_delay)['\"]?\s*[:=]\s*['\"]?([\d.]+)s", re.IGNORECASE)


class AllModelsExhausted(RuntimeError):
    """Żaden model z listy nie może teraz przyjąć zapytania."""

    def __init__(self, message: str, kinds: list[str] | None = None) -> None:
        super().__init__(message)
        self.kinds = kinds or []


def parse_spec(spec: str) -> tuple[str, str]:
    """„groq:openai/gpt-oss-120b” -> („groq”, „openai/gpt-oss-120b”)."""
    provider, _, model = spec.strip().partition(":")
    if provider not in PROVIDERS or not model:
        raise ValueError(
            f"Nieprawidłowy model „{spec}”; oczekiwany zapis dostawca:model, "
            f"dostawcy: {', '.join(PROVIDERS)}"
        )
    return provider, model


def limit_kind(error: Exception) -> str | None:
    """Rodzaj odrzucenia: „minute”, „day”, „overloaded” albo None (inny błąd)."""
    text = str(error)
    if not LIMIT_PATTERNS.search(text):
        return None
    if re.search(r"too large|exceed the enforced limit", text, re.IGNORECASE):
        return "too_large"
    if re.search(r"50[03]|UNAVAILABLE|INTERNAL|overloaded|high demand", text, re.IGNORECASE):
        return "overloaded"
    if re.search(r"per day|TPD|RPD|daily|PerDay", text, re.IGNORECASE):
        return "day"
    return "minute"


def retry_after(error: Exception) -> float | None:
    """Czas do odnowienia limitu podany przez serwer, w sekundach."""
    text = str(error)
    match = RETRY_IN.search(text)
    if match and any(match.groups()):
        hours, minutes, seconds = (float(g) if g else 0.0 for g in match.groups())
        return hours * 3600 + minutes * 60 + seconds
    match = RETRY_DELAY.search(text)
    return float(match.group(1)) if match else None


def make_model(spec: str, settings: Settings, temperature: float = 0.2) -> BaseChatModel:
    provider, model = parse_spec(spec)
    rate_limiter = None
    if settings.llm_rpm > 0:
        rate_limiter = InMemoryRateLimiter(
            requests_per_second=settings.llm_rpm / 60, check_every_n_seconds=0.2
        )
    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=model,
            temperature=temperature,
            rate_limiter=rate_limiter,
            timeout=120,
            max_retries=MAX_RETRIES,
            # Groq liczy do limitu tokenów na minutę polecenie i zarezerwowaną długość
            # odpowiedzi; bez tego limitu polecenie redaktora (ok. 4 tys. tokenów) z domyślną
            # rezerwą przekraczało 8 tys. tokenów darmowego planu.
            max_tokens=GROQ_MAX_TOKENS,
        )
    from langchain_google_genai import ChatGoogleGenerativeAI

    # Biblioteka google-genai ostrzega o funkcji, z której system nie korzysta.
    logging.getLogger("google_genai.models").setLevel(logging.ERROR)
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        rate_limiter=rate_limiter,
        timeout=120,
        max_retries=MAX_RETRIES,
    )


class ModelPool(BaseChatModel):
    """Model, który przekazuje zapytanie pierwszemu dostępnemu modelowi z listy.

    Stan wyłączeń jest zapisywany w pliku, więc po ponownym uruchomieniu aplikacji
    wyczerpany model nie jest od razu odpytywany. Nazwa modelu, który odpowiedział,
    trafia do response_metadata["model_spec"] odpowiedzi.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    specs: list[str]
    models: list[BaseChatModel]
    state_file: Path | None = None
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    _blocked: dict[str, float] = PrivateAttr(default_factory=dict)

    def model_post_init(self, context) -> None:
        self._blocked = self._load_state()

    @property
    def _llm_type(self) -> str:
        return "model-pool"

    def _load_state(self) -> dict[str, float]:
        if not self.state_file or not self.state_file.exists():
            return {}
        try:
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        now = time.time()
        return {k: float(v) for k, v in data.items() if float(v) > now and k in self.specs}

    def _save_state(self) -> None:
        if not self.state_file:
            return
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self.state_file.write_text(json.dumps(self._blocked), encoding="utf-8")
        except OSError:
            pass

    def block(self, spec: str, seconds: float) -> None:
        with self._lock:
            self._blocked[spec] = max(self._blocked.get(spec, 0), time.time() + seconds)
            self._save_state()

    def available(self) -> list[str]:
        now = time.time()
        with self._lock:
            return [s for s in self.specs if self._blocked.get(s, 0) <= now]

    def only(self, spec: str) -> ModelPool:
        """Pula z jednym modelem, ze wspólnym stanem wyłączeń (raport pisany jednym modelem)."""
        single = ModelPool(
            specs=[spec], models=[self.models[self.specs.index(spec)]], state_file=self.state_file
        )
        single._lock, single._blocked = self._lock, self._blocked
        return single

    def exhausted(self, errors: Iterable[str] = ()) -> AllModelsExhausted:
        return AllModelsExhausted(_exhausted_message(self.blocked_until(), errors))

    def blocked_until(self) -> dict[str, float]:
        now = time.time()
        with self._lock:
            return {s: t for s, t in self._blocked.items() if t > now}

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs,
    ) -> ChatResult:
        errors: list[str] = []
        kinds: list[str] = []
        for spec, model in zip(self.specs, self.models, strict=True):
            if spec not in self.available():
                continue
            for attempt in range(MINUTE_WAITS + 1):
                try:
                    # Bez wywołań zwrotnych: tokeny liczy odpowiedź puli, a nie model w środku.
                    message = model.invoke(messages, stop=stop, config={"callbacks": []})
                except Exception as exc:  # noqa: BLE001
                    kind = limit_kind(exc)
                    if kind is None:
                        raise
                    wait = retry_after(exc)
                    log.warning("%s: odrzucone (%s): %s", spec, kind, str(exc)[:300])
                    if kind == "minute" and attempt < MINUTE_WAITS and (wait or 0) <= MAX_WAIT_S:
                        time.sleep(wait or 10)
                        continue
                    # Przeciążenie serwera bywa chwilowe: dwie próby po krótkiej przerwie.
                    if kind == "overloaded" and attempt < OVERLOAD_WAITS:
                        time.sleep(OVERLOAD_WAIT_S)
                        continue
                    if kind == "too_large":
                        wait = None
                    self.block(spec, wait or DEFAULT_BLOCK_S[kind])
                    errors.append(f"{spec}: {kind}")
                    kinds.append(kind)
                    break
                else:
                    return _result(message, spec)
        raise AllModelsExhausted(_exhausted_message(self.blocked_until(), errors), kinds)


def _result(message: BaseMessage, spec: str) -> ChatResult:
    if isinstance(message, AIMessage):
        message = message.model_copy(
            update={"response_metadata": {**message.response_metadata, "model_spec": spec}}
        )
    return ChatResult(generations=[ChatGeneration(message=message)])


def _exhausted_message(blocked: dict[str, float], errors: Iterable[str]) -> str:
    if blocked:
        spec, until = min(blocked.items(), key=lambda kv: kv[1])
        when = time.strftime("%H:%M", time.localtime(until))
        return (
            "Wszystkie modele wyczerpały limity zapytań lub są przeciążone; najwcześniej "
            f"dostępny będzie {spec} około {when}."
        )
    return "Żaden model z listy LLM_MODELS nie jest dostępny (" + "; ".join(errors) + ")."


def configured_specs(settings: Settings) -> list[str]:
    """Modele z LLM_MODELS, a bez tej zmiennej jeden model z LLM_PROVIDER."""
    if settings.llm_models:
        return list(settings.llm_models)
    if settings.llm_provider == "groq":
        return [f"groq:{settings.groq_model}"]
    if settings.llm_provider == "gemini":
        return [f"gemini:{settings.gemini_model}"]
    raise ValueError(
        f"Nieznany dostawca modelu: {settings.llm_provider}; dostępne: {', '.join(PROVIDERS)}"
    )


def get_chat_model(
    settings: Settings, temperature: float = 0.2, require_qualified: bool | None = None
) -> BaseChatModel:
    """Pula modeli z konfiguracji. Z listy zostają tylko modele z zaliczonym testem
    kwalifikacyjnym; jeśli żaden nie był jeszcze testowany, używana jest cała lista."""
    from marketagents.qualification import qualified

    specs = configured_specs(settings)
    for spec in specs:
        parse_spec(spec)
    require = settings.require_qualified if require_qualified is None else require_qualified
    if require:
        specs = qualified(specs, settings.results_dir)
    if not specs:
        raise ValueError(
            "Żaden model z LLM_MODELS nie zaliczył testu kwalifikacyjnego; "
            "uruchom: uv run marketagents-eval kwalifikacja"
        )
    return ModelPool(
        specs=specs,
        models=[make_model(s, settings, temperature) for s in specs],
        state_file=settings.cache_dir / "limity_modeli.json",
    )


def model_name(message) -> str | None:
    """Model, który napisał odpowiedź (z puli modeli)."""
    return getattr(message, "response_metadata", {}).get("model_spec")


def message_text(message) -> str:
    """Tekst odpowiedzi modelu; Gemini zwraca go czasem jako listę części."""
    content = message.content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return str(content)
