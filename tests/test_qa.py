from langchain_core.language_models.fake_chat_models import FakeListChatModel

from marketagents.facts import TECHNICAL, Fact, FactSheet
from marketagents.qa import MAX_HISTORY, answer_question


class RecordingLLM(FakeListChatModel):
    prompts: list[str] = []

    def _call(self, messages, *args, **kwargs):
        self.prompts.append(messages[-1].content)
        return super()._call(messages, *args, **kwargs)


FACTS = FactSheet(
    [Fact("close", "kurs", 100.0, "USD"), Fact("rsi_14", "RSI", 55.56, "", 1, TECHNICAL)]
)
REPORT = "## Podsumowanie\nKurs 100,00 USD, RSI 55,6."


def ask(responses: list[str], history=None):
    llm = RecordingLLM(responses=responses, prompts=[])
    answer = answer_question(llm, "TEST", "Ile wynosi RSI?", REPORT, FACTS, history)
    return answer, llm


def test_answer_uses_fact_references():
    answer, llm = ask(["RSI(14) wynosi {{rsi_14}}."])
    assert answer.verified
    assert answer.text == "RSI(14) wynosi 55,6."
    prompt = llm.prompts[0]
    assert REPORT in prompt and "| {{rsi_14}} |" in prompt
    assert prompt.rstrip().endswith("Pytanie: Ile wynosi RSI?")


def test_invented_number_is_fixed():
    answer, llm = ask(["RSI wynosi 70,1.", "RSI wynosi {{rsi_14}}."])
    assert answer.verified
    assert answer.text == "RSI wynosi 55,6."
    assert "Liczba „70,1”" in llm.prompts[1]


def test_unfixable_number_is_marked():
    answer, llm = ask(["RSI wynosi 70,1.", "RSI wynosi 70,1."])
    assert not answer.verified
    assert answer.text == "RSI wynosi 70,1 [niezweryfikowane]."
    assert len(llm.prompts) == 2


def test_history_is_limited():
    history = [(f"pytanie {i}", f"odpowiedź {i}") for i in range(MAX_HISTORY + 2)]
    _, llm = ask(["Nie wiem."], history)
    prompt = llm.prompts[0]
    assert "Pytanie: pytanie 0\n" not in prompt
    assert f"Pytanie: pytanie {MAX_HISTORY + 1}\nOdpowiedź: odpowiedź {MAX_HISTORY + 1}" in prompt


NEWS = [
    {
        "published": "2026-10-14 08:00",
        "source": "Yahoo",
        "headline": "Nvidia invests in Doctolib",
        "summary": "",
        "url": "https://x.example/1",
    }
]


def test_answer_cites_news_and_names_must_come_from_the_source():
    llm = RecordingLLM(
        responses=[
            "Spółka zainwestowała w Doctolib i Mistral [1].",
            "Spółka zainwestowała w Doctolib [1].",
        ],
        prompts=[],
    )
    answer = answer_question(llm, "NVDA", "Co nowego?", REPORT, FACTS, news=NEWS)
    assert "[1] 2026-10-14, Yahoo: Nvidia invests in Doctolib." in llm.prompts[0]
    assert "„Mistral”" in llm.prompts[1]
    assert answer.verified
    assert "Doctolib [[1]](https://x.example/1)." in answer.text
    assert "1. [Yahoo, 2026-10-14: Nvidia invests in Doctolib](https://x.example/1)" in answer.text


def test_forecast_in_answer_is_marked_when_not_fixed():
    answer, _ = ask(["Kurs ma przestrzeń do dalszych wzrostów."] * 2)
    assert not answer.verified
    assert "[niezweryfikowana interpretacja]" in answer.text
