from dataclasses import replace
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from marketagents import llm as llm_module
from marketagents.config import Settings
from marketagents.llm import (
    AllModelsExhausted,
    ModelPool,
    get_chat_model,
    limit_kind,
    model_name,
    retry_after,
)
from marketagents.qualification import evaluate_state, qualified, save_result

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

DAILY = (
    "Error code: 429 - Rate limit reached for model `openai/gpt-oss-120b` on tokens per day "
    "(TPD): Limit 200000, Used 198972, Requested 2150. Please try again in 8m4.704s."
)
MINUTE = (
    "Error code: 429 - Rate limit reached on tokens per minute (TPM). Please try again in 3.5s."
)


class FailingModel(FakeListChatModel):
    """Model zastępczy, który najpierw zgłasza podane błędy, a potem odpowiada."""

    errors: list[str] = []

    def _call(self, messages, *args, **kwargs):
        if self.errors:
            raise RuntimeError(self.errors.pop(0))
        return super()._call(messages, *args, **kwargs)


def pool(*models, state_file=None) -> ModelPool:
    return ModelPool(
        specs=[f"groq:m{i}" for i in range(len(models))],
        models=list(models),
        state_file=state_file,
    )


def test_groq_provider_without_model_list(monkeypatch, tmp_path):
    monkeypatch.setenv("GROQ_API_KEY", "test")
    llm = get_chat_model(replace(SETTINGS, llm_provider="groq", results_dir=tmp_path))
    assert llm.specs == ["groq:openai/gpt-oss-120b"]
    assert type(llm.models[0]).__name__ == "ChatGroq"


def test_unknown_provider_and_bad_spec():
    with pytest.raises(ValueError, match="gemini, groq"):
        get_chat_model(replace(SETTINGS, llm_provider="inny"))
    with pytest.raises(ValueError, match="dostawca:model"):
        get_chat_model(replace(SETTINGS, llm_models=("openai-gpt",)))


def test_limit_messages():
    assert limit_kind(RuntimeError(DAILY)) == "day"
    assert limit_kind(RuntimeError(MINUTE)) == "minute"
    assert limit_kind(RuntimeError("503 UNAVAILABLE: high demand")) == "overloaded"
    assert limit_kind(RuntimeError("504 DEADLINE_EXCEEDED")) == "overloaded"
    assert limit_kind(RuntimeError("Server disconnected without sending a response.")) == (
        "overloaded"
    )
    assert limit_kind(ValueError("zły argument")) is None
    assert retry_after(RuntimeError(DAILY)) == pytest.approx(484.704)
    assert retry_after(RuntimeError("{'retryDelay': '37s'}")) == 37


def test_pool_switches_model_after_daily_limit(tmp_path):
    first = FailingModel(responses=["A"], errors=[DAILY])
    second = FailingModel(responses=["B", "B2"])
    models = pool(first, second, state_file=tmp_path / "limity.json")
    answer = models.invoke("pytanie")
    assert answer.content == "B" and model_name(answer) == "groq:m1"
    assert models.available() == ["groq:m1"]
    # Kolejne zapytanie trafia od razu do drugiego modelu, także w nowej puli.
    assert models.invoke("pytanie").content == "B2"
    again = pool(FailingModel(responses=["A"]), second, state_file=tmp_path / "limity.json")
    assert again.available() == ["groq:m1"]


def test_short_minute_limit_waits_on_the_same_model(monkeypatch):
    waits = []
    monkeypatch.setattr(llm_module.time, "sleep", waits.append)
    models = pool(FailingModel(responses=["A"], errors=[MINUTE]), FailingModel(responses=["B"]))
    assert models.invoke("pytanie").content == "A"
    assert waits == [3.5]


def test_overloaded_server_is_retried_before_switching(monkeypatch):
    waits = []
    monkeypatch.setattr(llm_module.time, "sleep", waits.append)
    busy = "503 UNAVAILABLE: high demand"
    models = pool(FailingModel(responses=["A"], errors=[busy] * 3), FailingModel(responses=["B"]))
    assert models.invoke("pytanie").content == "B"
    assert waits == [15, 15]


def test_other_errors_are_not_hidden_and_exhaustion_is_reported():
    with pytest.raises(RuntimeError, match="zły format"):
        pool(FailingModel(responses=["A"], errors=["zły format"])).invoke("pytanie")
    with pytest.raises(AllModelsExhausted, match="najwcześniej dostępny"):
        pool(FailingModel(responses=["A"], errors=[DAILY])).invoke("pytanie")


def test_only_qualified_models_are_used(tmp_path):
    specs = ["groq:a", "gemini:b", "gemini:c"]
    assert qualified(specs, tmp_path) == specs  # nikt jeszcze nie był testowany
    save_result(tmp_path, "groq:a", {"zaliczony": True})
    save_result(tmp_path, "gemini:b", {"zaliczony": False})
    save_result(tmp_path, "gemini:c", {"zaliczony": None, "przerwany": "limit"})
    assert qualified(specs, tmp_path) == ["groq:a"]


def test_qualification_criteria():
    good = {
        "draft": "## Podsumowanie\nTekst.\n## Ograniczenia analizy\nBrak.",
        "analyses": {"a": "x", "b": "y"},
        "unverified": ["7,5"],
        "attempts": 2,
    }
    assert evaluate_state(good, expected_analysts=2, elapsed=120)["zaliczony"]
    empty_analysis = {**good, "analyses": {"a": "x"}}
    assert not evaluate_state(empty_analysis, 2, 120)["zaliczony"]
    too_many_marks = {**good, "flagged": ["Zdanie."], "uncited": ["Inne."]}
    assert not evaluate_state(too_many_marks, 2, 120)["zaliczony"]
    truncated = {**good, "draft": "## Podsumowanie\nTekst"}
    assert not evaluate_state(truncated, 2, 120)["zaliczony"]


def test_usage_counter_sees_each_answer_once_with_model_name():
    from marketagents.evaluation import UsageCounter

    counter = UsageCounter()
    models = pool(FailingModel(responses=["A"], errors=[DAILY]), FailingModel(responses=["B"]))
    models.invoke("pytanie", config={"callbacks": [counter]})
    assert counter.calls == 1
    assert counter.models == {"groq:m1": 1}


def test_request_too_large_for_the_plan_is_a_separate_kind():
    error = RuntimeError(
        "Error code: 429 - Request too large for model on output tokens per minute (OTPM): "
        "Limit 1000, Requested 2048. The request's expected output tokens exceed the enforced limit"
    )
    assert limit_kind(error) == "too_large"
    models = pool(FailingModel(responses=["A"], errors=[str(error)]))
    with pytest.raises(AllModelsExhausted) as info:
        models.invoke("pytanie")
    assert info.value.kinds == ["too_large"]


def test_groq_model_reserves_bounded_answer_length(monkeypatch, tmp_path):
    monkeypatch.setenv("GROQ_API_KEY", "test")
    llm = get_chat_model(replace(SETTINGS, llm_provider="groq", results_dir=tmp_path))
    assert llm.models[0].max_tokens == llm_module.GROQ_MAX_TOKENS


def test_qualification_resumes_after_daily_limit(monkeypatch):
    from marketagents import qualification

    calls = []

    def fake_report(graph, ticker, *args, **kwargs):
        calls.append(ticker)
        return {
            "draft": "## Podsumowanie\nA.\n## Ograniczenia analizy\nB.",
            "analyses": {"techniczna": "x"},
            "attempts": 1,
        }

    monkeypatch.setattr("marketagents.graph.run_report", fake_report)
    monkeypatch.setattr(qualification, "test_data", lambda s, t: None)
    monkeypatch.setattr(qualification, "_expected_analysts", lambda data: 1)
    monkeypatch.setattr("marketagents.graph.build_graph", lambda *args: None)
    monkeypatch.setattr("marketagents.llm.make_model", lambda spec, settings: None)
    monkeypatch.setattr("marketagents.llm.ModelPool", lambda **kw: None)
    done = {"AAPL": {"zaliczony": True}, "LULU": {"zaliczony": True}}
    result = qualification.qualify("groq:x", SETTINGS, ("AAPL", "LULU", "INTC"), done)
    assert calls == ["INTC"]
    assert result["zaliczony"] and set(result["spółki"]) == {"AAPL", "LULU", "INTC"}
