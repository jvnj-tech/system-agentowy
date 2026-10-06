from langchain_core.language_models.fake_chat_models import FakeListChatModel

from marketagents.facts import FUNDAMENTAL, MACRO, TECHNICAL, Fact
from marketagents.graph import LAST_SECTION, MAX_FIXES, MarketData, build_graph, run_report

TECH = [Fact("close", "kurs", 100.0, "USD"), Fact("rsi_14", "RSI", 55.56, "", 1, TECHNICAL)]
FUND = [Fact("fy_net_margin", "marża netto", 24.31, "%", 1, FUNDAMENTAL)]
MACROS = [Fact("macro_10y", "rentowność 10Y", 4.12, "%", 2, MACRO)]
END = f"\n{LAST_SECTION}\nTo nie jest rekomendacja."
NEWS = [{"published": "2026-10-01 14:00", "source": "Reuters", "headline": "Nowy produkt"}]


class RecordingLLM(FakeListChatModel):
    """Model zastępczy, który zapamiętuje otrzymane polecenia."""

    prompts: list[str] = []

    def _call(self, messages, *args, **kwargs):
        self.prompts.append(messages[-1].content)
        return super()._call(messages, *args, **kwargs)


def run(responses: list[str], data: MarketData) -> tuple[dict, RecordingLLM]:
    llm = RecordingLLM(responses=responses, prompts=[])
    state = build_graph(llm, lambda ticker: data).invoke({"ticker": "TEST"})
    return state, llm


def test_only_technical_analysis_when_other_data_missing():
    state, llm = run(
        ["Analiza: RSI {{rsi_14}}.", "## Podsumowanie\nKurs {{close}}, RSI {{rsi_14}}." + END],
        MarketData(facts=TECH),
    )
    assert state["verified"]
    assert state["attempts"] == 1
    assert state["report"] == "## Podsumowanie\nKurs 100,00 USD, RSI 55,6." + END
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
    assert "Reuters: Nowy produkt" in news
    assert editor_prompt.count("### Analiza:") == 4


def test_editor_fixes_invented_number():
    state, _ = run(
        ["Analiza.", "RSI wynosi 70,1." + END, "RSI wynosi {{rsi_14}}." + END],
        MarketData(facts=TECH),
    )
    assert state["verified"]
    assert state["attempts"] == 2
    assert state["report"] == "RSI wynosi 55,6." + END


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
    state = run_report(build_graph(llm, lambda ticker: data), "TEST", steps.append)
    assert set(state["analyses"]) == {"techniczna", "fundamentalna", "wiadomości"}
    assert state["report"] == "Raport 100,00 USD." + END
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


def test_analyst_output_is_verified_and_fixed_before_editor():
    state, llm = run(
        ["RSI wynosi 70,1.", "RSI wynosi {{rsi_14}}.", "Raport {{rsi_14}}." + END],
        MarketData(facts=TECH),
    )
    assert state["analyses"]["techniczna"] == "RSI wynosi {{rsi_14}}."
    assert state["analyst_fixes"] == 1
    assert "Weryfikator znalazł w Twojej analizie" in llm.prompts[1]
    assert state["verified"] and state["attempts"] == 1
