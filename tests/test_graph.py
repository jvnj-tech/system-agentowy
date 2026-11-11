from langchain_core.language_models.fake_chat_models import FakeListChatModel

from marketagents.facts import FUNDAMENTAL, MACRO, TECHNICAL, Fact
from marketagents.graph import LAST_SECTION, MAX_FIXES, MarketData, build_graph, run_report

TECH = [Fact("close", "kurs", 100.0, "USD"), Fact("rsi_14", "RSI", 55.56, "", 1, TECHNICAL)]
FUND = [Fact("fy_net_margin", "marża netto", 24.31, "%", 1, FUNDAMENTAL)]
MACROS = [Fact("macro_10y", "rentowność 10Y", 4.12, "%", 2, MACRO)]
# Wszystkie sekcje raportu: graf zgłasza brakującą sekcję jako błąd do poprawy.
HEADS = (
    "\n## Podsumowanie\nKrótko.\n## Analiza techniczna\nTak.\n## Analiza fundamentalna\nTak."
    "\n## Wiadomości\nBrak nowych wiadomości.\n## Otoczenie makroekonomiczne\nTak."
)
END = HEADS + f"\n{LAST_SECTION}\nTo nie jest rekomendacja."
NEWS = [{"published": "2026-10-01 14:00", "source": "Reuters", "headline": "Nowy produkt"}]


class RecordingLLM(FakeListChatModel):
    """Model zastępczy, który zapamiętuje otrzymane polecenia."""

    prompts: list[str] = []

    def _call(self, messages, *args, **kwargs):
        self.prompts.append(messages[-1].content)
        return super()._call(messages, *args, **kwargs)


def body(report: str) -> str:
    """Raport bez opisu zakresu danych dopisywanego przez kod."""
    return report.split("\n\nZakres danych")[0]


def run(responses: list[str], data: MarketData) -> tuple[dict, RecordingLLM]:
    llm = RecordingLLM(responses=responses, prompts=[])
    state = build_graph(llm, lambda ticker: data, review=False).invoke({"ticker": "TEST"})
    return state, llm


def test_only_technical_analysis_when_other_data_missing():
    state, llm = run(
        ["Analiza: RSI {{rsi_14}}.", "## Podsumowanie\nKurs {{close}}, RSI {{rsi_14}}." + END],
        MarketData(facts=TECH),
    )
    assert state["verified"]
    assert state["attempts"] == 1
    assert body(state["report"]) == "## Podsumowanie\nKurs 100,00 USD, RSI 55,6." + END
    assert list(state["analyses"]) == ["techniczna"]
    assert len(llm.prompts) == 2


def test_all_analysts_run_and_editor_sees_their_work():
    data = MarketData(facts=TECH + FUND + MACROS, news=NEWS)
    responses = ["Analiza A."] * 4 + ["Raport {{close}}." + END]
    state, llm = run(responses, data)
    assert set(state["analyses"]) == {
        "techniczna",
        "fundamentalna",
        "wiadomości",
        "makroekonomiczna",
    }
    assert state["verified"]

    assert len(llm.prompts) == 5
    analyst_prompts, editor_prompt = llm.prompts[:4], llm.prompts[4]
    fundamental = next(p for p in analyst_prompts if "analitykiem fundamentalnym" in p)
    assert "| {{fy_net_margin}} |" in fundamental and "| {{rsi_14}} |" not in fundamental
    news = next(p for p in analyst_prompts if "wiadomości rynkowych" in p)
    assert "[1] 2026-10-01, Reuters: Nowy produkt" in news
    assert editor_prompt.count("### Analiza:") == 4


def test_editor_fixes_invented_number():
    state, _ = run(
        ["Analiza.", "RSI wynosi 70,1." + END, "RSI wynosi {{rsi_14}}." + END],
        MarketData(facts=TECH),
    )
    assert state["verified"]
    assert state["attempts"] == 2
    assert body(state["report"]) == "RSI wynosi 55,6." + END
    # każda zweryfikowana wersja raportu jest zapisana razem z błędami weryfikatora
    first, second = state["versions"]
    assert "70,1" in first["draft"] and first["issues"]
    assert "{{rsi_14}}" in second["draft"] and second["issues"] == []


def test_unfixable_report_is_marked_after_max_attempts():
    responses = ["Analiza."] + ["RSI wynosi 70,1." + END] * (MAX_FIXES + 1)
    state, _ = run(responses, MarketData(facts=TECH))
    assert not state["verified"]
    assert state["attempts"] == MAX_FIXES + 1
    assert "70,1 [niezweryfikowane]" in state["report"]


def test_run_report_merges_updates_like_invoke():
    data = MarketData(facts=TECH + FUND, news=NEWS)
    responses = ["Analiza A."] * 3 + ["Raport {{close}}." + END]
    steps: list[str] = []
    llm = RecordingLLM(responses=responses, prompts=[])
    state = run_report(build_graph(llm, lambda ticker: data, review=False), "TEST", steps.append)
    assert set(state["analyses"]) == {"techniczna", "fundamentalna", "wiadomości"}
    assert body(state["report"]) == "Raport 100,00 USD." + END
    assert steps[0] == "load_data" and steps[-1] == "finalize"
    assert {"editor", "verifier", "macro_analyst"} <= set(steps)


def test_truncated_report_goes_back_to_editor():
    state, llm = run(
        ["Analiza.", "## Podsumowanie\nKurs {{close}} i linia", "Kurs {{close}}." + END],
        MarketData(facts=TECH),
    )
    assert state["verified"]
    assert state["attempts"] == 2
    assert "brak sekcji „## Ograniczenia analizy”" in llm.prompts[-1]


def test_technical_analyst_prompt_describes_moving_averages():
    _, llm = run(["Analiza.", "Raport {{close}}." + END], MarketData(facts=TECH))
    technical = llm.prompts[0]
    assert "wzajemny układ średnich" in technical
    assert "z opóźnieniem" in technical
    assert "bez rekomendacji" in technical


def test_wrong_interpretation_goes_back_to_editor_and_is_marked():
    wrong = "RSI {{rsi_14}} oznacza wykupienie rynku." + END
    responses = ["Analiza."] + [wrong] * (MAX_FIXES + 1)
    state, llm = run(responses, MarketData(facts=TECH))
    assert not state["verified"]
    assert "wykupienie oznacza RSI powyżej 70" in llm.prompts[-1]
    assert "rynku. [niezweryfikowana interpretacja]" in state["report"]
    assert 'class="fact"' in state["report_html"]


def test_analyst_output_goes_to_editor_without_separate_verification():
    state, llm = run(
        ["RSI wynosi 70,1.", "Raport {{rsi_14}}." + END],
        MarketData(facts=TECH),
    )
    # Analiza cząstkowa nie jest sprawdzana osobno: weryfikator sprawdza raport redaktora.
    assert state["analyses"]["techniczna"] == "RSI wynosi 70,1."
    assert len(llm.prompts) == 2
    assert state["verified"] and state["attempts"] == 1


def test_wrong_period_goes_back_to_editor_with_labelled_facts():
    facts = TECH + [Fact("change_1m", "zmiana 1M", -3.2, "%", 1, TECHNICAL)]
    state, llm = run(
        [
            "Analiza.",
            "W ostatnim roku kurs spadł o {{change_1m}}." + END,
            "W ostatnim miesiącu kurs spadł o {{change_1m}}." + END,
        ],
        MarketData(facts=facts),
    )
    assert state["verified"] and state["attempts"] == 2
    assert "okres „rok”" in llm.prompts[-1]
    assert "| -3,2% | spadek |" in llm.prompts[0]


class PromptLLM(FakeListChatModel):
    """Model zastępczy odpowiadający według treści polecenia (analitycy działają równolegle)."""

    responses: list[str] = ["-"]

    def _call(self, messages, *args, **kwargs):
        prompt = messages[-1].content
        if "Jesteś redaktorem" in prompt:
            return "Raport {{close}}." + END
        return "Analiza z liczbą 12,3."


def test_run_report_sums_issue_counts_like_invoke():
    data = MarketData(facts=TECH + FUND)
    graph = build_graph(PromptLLM(), lambda ticker: data, review=False)
    state = run_report(graph, "TEST")
    assert state["issue_counts"] == [0]
    invoked = graph.invoke({"ticker": "TEST"})
    assert state["issue_counts"] == invoked["issue_counts"]
    assert state["versions"] == invoked["versions"]


class ScriptedLLM(FakeListChatModel):
    """Model zastępczy z osobną kolejką odpowiedzi dla każdej roli (analitycy działają
    równolegle, więc kolejność wywołań nie jest ustalona)."""

    script: dict[str, list[str]] = {}
    prompts: list[str] = []

    def _call(self, messages, *args, **kwargs):
        prompt = messages[-1].content
        self.prompts.append(prompt)
        role = next(r for r in self.script if r in prompt)
        return self.script[role].pop(0)


NEWS_HEAD = "## Podsumowanie\nKrótko.\n## Analiza techniczna\nTak.\n## Wiadomości\n"
NEWS_END = f"\n{LAST_SECTION}\nTo nie jest rekomendacja."


def test_news_analyst_gets_numbered_news_and_price_changes_and_must_cite():
    facts = TECH + [Fact("change_1y", "zmiana 1R", 30.8, "%", 1, TECHNICAL)]
    news = NEWS + [{"published": "2026-10-02 09:00", "source": "CNBC", "headline": "Wyniki"}]
    sentence = "Spółka pokazała nowy produkt w tym tygodniu"
    llm = ScriptedLLM(
        responses=[""],
        prompts=[],
        script={
            "analitykiem technicznym": ["Analiza techniczna."],
            "wiadomości rynkowych": [sentence + "."],
            "redaktorem": [
                NEWS_HEAD + sentence + "." + NEWS_END,
                NEWS_HEAD + sentence + " [1]." + NEWS_END,
            ],
        },
    )
    graph = build_graph(llm, lambda t: MarketData(facts=facts, news=news), review=False)
    state = graph.invoke({"ticker": "T"})
    news_prompt = next(p for p in llm.prompts if "wiadomości rynkowych" in p)
    assert "[2] 2026-10-02, CNBC: Wyniki." in news_prompt
    assert "| {{change_1y}} |" in news_prompt
    assert state["attempts"] == 2
    assert "nie wskazuje źródła" in llm.prompts[-1]
    assert "tygodniu [1]." in state["report"]
    assert "1. Reuters, 2026-10-01: Nowy produkt" in state["report"]
    assert "Zakres danych" in state["report"]


def test_empty_model_answer_is_retried_and_then_skipped():
    state, llm = run(
        ["", "", "Raport {{close}}." + END],
        MarketData(facts=TECH),
    )
    # Pusta odpowiedź analityka: ponowienie, a potem pominięcie analizy.
    assert "techniczna" not in state.get("analyses", {})
    assert len(llm.prompts) == 3 and state["verified"]
    state, _ = run(["Analiza."] + [""] * 2 * (MAX_FIXES + 1), MarketData(facts=TECH))
    assert not state["verified"]
    assert body(state["report"]).startswith("Model językowy nie zwrócił treści raportu")


def test_report_restarts_on_next_model_after_daily_limit(tmp_path):
    from marketagents.graph import run_report_with_fallback
    from marketagents.llm import ModelPool

    class LimitedModel(FakeListChatModel):
        """Odpowiada, dopóki nie wyczerpie limitu dobowego."""

        left: int = 1

        def _call(self, messages, *args, **kwargs):
            if self.left <= 0:
                raise RuntimeError("Error code: 429 - tokens per day (TPD). Try again in 9m0s.")
            self.left -= 1
            return super()._call(messages, *args, **kwargs)

    first = LimitedModel(responses=["Analiza A."], left=1)
    second = FakeListChatModel(responses=["Analiza B.", "Raport B {{close}}." + END])
    pool = ModelPool(
        specs=["groq:a", "gemini:b"], models=[first, second], state_file=tmp_path / "limity.json"
    )
    restarts = []
    state = run_report_with_fallback(
        pool,
        lambda llm: build_graph(llm, lambda t: MarketData(facts=TECH), review=False),
        "TEST",
        on_restart=restarts.append,
    )
    assert restarts == ["groq:a"]
    assert state["model"] == "gemini:b" and state["restarts"] == 1
    # Cały raport, także analiza, pochodzi od drugiego modelu.
    assert state["analyses"] == {"techniczna": "Analiza B."}
    assert body(state["report"]).startswith("Raport B 100,00 USD.")
    assert pool.available() == ["gemini:b"]


def test_unverified_number_is_marked_only_where_it_was_typed():
    facts = TECH + [Fact("bb_width", "szerokość wstęg", 55.6, "%", 1, TECHNICAL)]
    typed = "RSI {{rsi_14}} i szerokość wstęg {{bb_width}}, czyli 55,6 %." + END
    state, _ = run(["Analiza."] + [typed] * (MAX_FIXES + 1), MarketData(facts=facts))
    report = body(state["report"])
    # RSI = 55,6 z karty nie dostaje oznaczenia, choć ma ten sam zapis co wpisana liczba.
    assert report.count("[niezweryfikowane]") == 1
    assert "czyli 55,6 [niezweryfikowane] %" in report


def test_missing_section_goes_back_to_editor_with_its_analysis():
    data = MarketData(facts=TECH + FUND)
    without_fundamentals = (
        "## Podsumowanie\nKrótko.\n## Analiza techniczna\nTak." + f"\n{LAST_SECTION}\nKoniec."
    )
    state, llm = run(
        ["Analiza A.", "Analiza A.", without_fundamentals, "Raport {{close}}." + END], data
    )
    assert state["verified"] and state["attempts"] == 2
    assert "brak sekcji „## Analiza fundamentalna”" in llm.prompts[-1]
    assert "Analizy cząstkowe brakujących sekcji" in llm.prompts[-1]
    assert "### Analiza: fundamentalna" in llm.prompts[-1]
    assert "### Analiza: techniczna" not in llm.prompts[-1]


def test_editor_review_finds_section_error_and_report_is_fixed_once():
    from marketagents.graph import parse_review

    assert parse_review("BRAK UWAG") == []
    assert parse_review("- [Wiadomości] „Akcje nie rosną”: sprzeczne z kartą") == [
        "[Wiadomości] „Akcje nie rosną”: sprzeczne z kartą"
    ]
    contradiction = "Raport {{close}}, akcje od miesięcy nie rosną." + END
    fixed = "Raport {{close}}." + END
    llm = RecordingLLM(
        responses=[
            "Analiza.",
            contradiction,
            "- [Podsumowanie] „akcje od miesięcy nie rosną”: sprzeczne z kartą faktów",
            fixed,
        ],
        prompts=[],
    )
    state = build_graph(llm, lambda t: MarketData(facts=TECH)).invoke({"ticker": "T"})
    assert "Sprawdź poniższy raport" in llm.prompts[2]
    assert "Redaktor (sprawdzenie sekcji)" in llm.prompts[3]
    assert state["verified"] and state["reviewed"]
    assert state["review_issues"] == [
        "[Podsumowanie] „akcje od miesięcy nie rosną”: sprzeczne z kartą faktów"
    ]
    assert body(state["report"]).startswith("Raport 100,00 USD.")
    assert len(llm.prompts) == 4  # bez drugiego sprawdzenia przez redaktora


def test_review_fix_does_not_use_verifier_fix_limit():
    wrong = "RSI wynosi 70,1." + END
    llm = RecordingLLM(
        responses=["Analiza.", "Raport {{close}}." + END, "- [Podsumowanie] „Raport”: ogólnik"]
        + [wrong] * (MAX_FIXES + 1),
        prompts=[],
    )
    state = build_graph(llm, lambda t: MarketData(facts=TECH)).invoke({"ticker": "T"})
    # Pierwsza wersja, poprawka po uwagach redaktora i trzy poprawki błędów weryfikatora.
    assert not state["verified"] and state["attempts"] == MAX_FIXES + 2
    assert len(llm.prompts) == 3 + MAX_FIXES + 1


def test_review_enabled_keeps_verifier_fix_limit():
    llm = RecordingLLM(
        responses=["Analiza."] + ["RSI wynosi 70,1." + END] * (MAX_FIXES + 2), prompts=[]
    )
    state = build_graph(llm, lambda t: MarketData(facts=TECH)).invoke({"ticker": "T"})
    # Raport nie przeszedł weryfikacji, więc nie było przeglądu ani poprawki po nim.
    assert not state["verified"] and state["attempts"] == MAX_FIXES + 1
    assert len(llm.prompts) == 2 + MAX_FIXES


def test_editor_review_without_remarks_finishes():
    llm = RecordingLLM(responses=["Analiza.", "Raport {{close}}." + END, "BRAK UWAG"], prompts=[])
    state = build_graph(llm, lambda t: MarketData(facts=TECH)).invoke({"ticker": "T"})
    assert state["verified"] and state["review_issues"] == [] and len(llm.prompts) == 3
