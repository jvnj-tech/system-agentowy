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
