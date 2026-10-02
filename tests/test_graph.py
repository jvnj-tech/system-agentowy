from langchain_core.language_models.fake_chat_models import FakeListChatModel

from marketagents.facts import Fact
from marketagents.graph import MAX_FIXES, build_graph

FACTS = [Fact("close", "kurs", 100.0, "USD"), Fact("rsi_14", "RSI", 55.56, "", 1)]


def load_facts(ticker: str) -> list[Fact]:
    assert ticker == "TEST"
    return FACTS


def run(responses: list[str]) -> tuple[dict, FakeListChatModel]:
    llm = FakeListChatModel(responses=responses)
    state = build_graph(llm, load_facts).invoke({"ticker": "TEST"})
    return state, llm


def test_report_passes_verification_first_time():
    state, _ = run(["Analiza: RSI {{rsi_14}}.", "## Podsumowanie\nKurs {{close}}, RSI {{rsi_14}}."])
    assert state["verified"]
    assert state["attempts"] == 1
    assert state["report"] == "## Podsumowanie\nKurs 100,00 USD, RSI 55,6."
    assert state["analyses"] == {"techniczna": "Analiza: RSI {{rsi_14}}."}


def test_editor_fixes_invented_number():
    state, _ = run(["Analiza.", "RSI wynosi 70,1.", "RSI wynosi {{rsi_14}}."])
    assert state["verified"]
    assert state["attempts"] == 2
    assert state["report"] == "RSI wynosi 55,6."


def test_unfixable_report_is_marked_after_max_attempts():
    responses = ["Analiza."] + ["RSI wynosi 70,1."] * (MAX_FIXES + 1)
    state, _ = run(responses)
    assert not state["verified"]
    assert state["attempts"] == MAX_FIXES + 1
    assert "70,1 [niezweryfikowane]" in state["report"]
