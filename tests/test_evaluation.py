import random

from langchain_core.language_models.fake_chat_models import (
    FakeListChatModel,
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage

from marketagents.evaluation import (
    OUTSIDE_QUESTION,
    QUESTIONS,
    WITH_CARD,
    WITHOUT_CARD,
    UsageCounter,
    append_record,
    ask_questions,
    inject_errors,
    injection_test,
    load_records,
    main,
    markdown_table,
    report_record,
    report_without_fact_card,
    summarize_answers,
    summarize_injections,
    summarize_reports,
    unsupported_numbers,
)
from marketagents.facts import FUNDAMENTAL, TECHNICAL, Fact, FactSheet
from marketagents.graph import LAST_SECTION, MarketData, build_graph, run_report
from marketagents.verifier import verify

FACTS = [
    Fact("as_of", "data ostatniej sesji", "2026-10-02"),
    Fact("close", "kurs zamknięcia", 254.63, "USD"),
    Fact("sma_200", "średnia krocząca SMA(200)", 221.4, "USD", 2, TECHNICAL),
    Fact("change_1m", "zmiana kursu w ostatnim miesiącu", -3.24, "%", 1, TECHNICAL),
    Fact("change_1y", "zmiana kursu w ostatnim roku", 12.5, "%", 1, TECHNICAL),
    Fact("rsi_14", "RSI(14)", 54.2, "", 1, TECHNICAL),
    Fact("fy_net_margin", "marża netto", 24.31, "%", 1, FUNDAMENTAL),
]
SHEET = FactSheet(FACTS)
# Wszystkie sekcje raportu: graf zgłasza brakującą sekcję jako błąd do poprawy.
HEADS = (
    "\n## Podsumowanie\nKrótko.\n## Analiza techniczna\nTak.\n## Analiza fundamentalna\nTak."
    "\n## Wiadomości\nTak.\n## Otoczenie makroekonomiczne\nTak."
)
END = HEADS + f"\n{LAST_SECTION}\nTo nie jest rekomendacja."
DRAFT = (
    "## Podsumowanie\nStan na {{as_of}}: kurs {{close}} USD jest powyżej SMA(200) {{sma_200}}. "
    "W ostatnim miesiącu kurs spadł o {{change_1m}}. W ciągu roku wzrósł o {{change_1y}}. "
    "RSI(14) wynosi {{rsi_14}}, co oznacza strefę neutralną. "
    "Marża netto to {{fy_net_margin}}." + END
)


class RecordingLLM(FakeListChatModel):
    prompts: list[str] = []

    def _call(self, messages, *args, **kwargs):
        self.prompts.append(messages[-1].content)
        return super()._call(messages, *args, **kwargs)


def test_usage_counter_counts_calls_and_tokens():
    usage = UsageCounter()
    message = AIMessage(
        "x", usage_metadata={"input_tokens": 10, "output_tokens": 3, "total_tokens": 13}
    )
    GenericFakeChatModel(messages=iter([message])).invoke("a", config={"callbacks": [usage]})
    assert usage.calls == 1
    assert usage.tokens() == (10, 3)
    assert UsageCounter().tokens() == (None, None)


def test_record_of_graph_run(tmp_path):
    responses = ["Analiza.", "Analiza.", "Raport {{close}}, RSI 70,1." + END, DRAFT]
    llm = FakeListChatModel(responses=responses)
    data = MarketData(facts=list(FACTS))
    usage = UsageCounter()
    graph = build_graph(llm, lambda ticker: data, review=False)
    state = run_report(graph, "TEST", config={"callbacks": [usage]})
    record = report_record("TEST", state, 12.34, usage, "model")

    assert usage.calls == 4  # dwóch analityków i dwie wersje redakcji
    assert record["attempts"] == 2 and record["verified"]
    assert record["flagged"] == 0
    assert record["issue_counts"] == [1, 0]
    assert record["input_tokens"] is None
    assert record["unsupported_numbers"] == []

    path = tmp_path / "wyniki.jsonl"
    append_record(record, path)
    append_record(record, path)
    assert [r["ticker"] for r in load_records(path)] == ["TEST", "TEST"]


def test_unsupported_numbers_compares_with_any_fact():
    text = "Kurs 254,63 USD na tle NASDAQ-100, 2.10.2026 r., RSI 54 i 61,7. W 2026 r. i w 2019 r."
    assert unsupported_numbers(text, SHEET) == ["2019", "61,7"]


def test_variant_without_fact_card_gets_plain_table():
    llm = RecordingLLM(
        responses=["Analiza: kurs 254,63 USD.", "Raport: kurs 254,64 USD."], prompts=[]
    )
    data = MarketData(facts=[f for f in FACTS if f.group != FUNDAMENTAL])
    state = report_without_fact_card(llm, "TEST", data)
    assert len(llm.prompts) == 2
    assert all("{{" not in p for p in llm.prompts)
    assert "| kurs zamknięcia | 254,63 USD |" in llm.prompts[0]
    assert unsupported_numbers(state["report"], SHEET) == ["254,64"]


def test_injected_errors_are_real_and_detected():
    assert verify(DRAFT, SHEET).ok
    mutations = inject_errors(DRAFT, SHEET, random.Random(1))
    assert {m.kind for m in mutations} == {"liczba", "inny fakt", "kierunek", "okres", "rok"}
    for m in mutations:
        assert m.text != DRAFT
        assert not verify(m.text, SHEET).ok, m


def test_number_and_unit_are_not_duplicated():
    mutations = inject_errors("Kurs {{close}} USD." + END, SHEET, random.Random(0))
    number = next(m for m in mutations if m.kind == "liczba")
    assert "USD USD" not in number.text


def test_injection_test_with_judge():
    records = [
        {"ticker": "A", "variant": WITH_CARD, "draft": DRAFT, "facts": SHEET.to_records()},
        {"ticker": "B", "variant": WITHOUT_CARD, "draft": "x", "facts": SHEET.to_records()},
    ]
    judge = FakeListChatModel(responses=["POPRAWNY"] + ["BŁĄD\nfragment"] * 5)
    rows = injection_test(records, seed=1, llm=judge)
    assert [r["kind"] for r in rows][0] == "bez błędu"
    assert len(rows) == 6
    assert rows[0] == {
        "ticker": "A",
        "kind": "bez błędu",
        "before": "",
        "after": "",
        "verifier": False,
        "judge": False,
    }
    assert all(r["verifier"] and r["judge"] for r in rows[1:])

    summary = summarize_injections(rows)
    assert summary[0] == {
        "rodzaj": "bez błędu",
        "przypadki": 1,
        "weryfikator [%]": 0.0,
        "sędzia [%]": 0.0,
    }
    assert summary[1]["weryfikator [%]"] == 100.0


def test_summary_and_cli(tmp_path, capsys):
    path = tmp_path / "wyniki.jsonl"
    base = {
        "ticker": "A",
        "elapsed_s": 10.0,
        "llm_calls": 6,
        "input_tokens": None,
        "output_tokens": None,
        "unsupported_numbers": [],
        "facts": [],
        "draft": "",
    }
    append_record(
        base
        | {
            "variant": WITH_CARD,
            "attempts": 1,
            "verified": True,
            "issue_counts": [0],
            "unverified": 0,
            "flagged": 0,
        },
        path,
    )
    append_record(
        base
        | {
            "variant": WITH_CARD,
            "attempts": 3,
            "verified": False,
            "issue_counts": [4, 2, 1],
            "unverified": 1,
            "flagged": 0,
        },
        path,
    )
    append_record(base | {"variant": WITHOUT_CARD, "unsupported_numbers": ["1,5"]}, path)

    rows = summarize_reports(load_records(path))
    card, plain = rows
    assert card["zweryfikowane za 1. razem [%]"] == 50.0
    assert card["śr. problemy w 1. wersji"] == 2.0
    assert card["śr. tokeny wejściowe"] is None
    assert plain["raporty z liczbą spoza karty [%]"] == 100.0
    assert "| wariant |" in markdown_table(rows)

    main(["--wyniki", str(path), "podsumowanie"])
    assert "bez-karty" in capsys.readouterr().out


def test_words_are_flipped_only_where_they_agree_with_the_data():
    text = (
        "Kurs nie wzrósł o {{change_1y}}. "
        "RSI {{rsi_14}} jest daleko od strefy wykupienia. "
        "Kurs wzrósł w roku o {{change_1y}}, a w miesiącu spadł o {{change_1m}}." + END
    )
    kinds = {m.kind for m in inject_errors(text, SHEET, random.Random(0))}
    assert "kierunek" not in kinds and "okres" not in kinds


def test_flipped_rsi_zone_contradicts_the_data():
    text = "RSI(14) wynosi {{rsi_14}}, co oznacza strefę neutralną." + END
    word = next(m for m in inject_errors(text, SHEET, random.Random(0)) if m.kind == "kierunek")
    assert (word.before, word.after) == ("neutralną", "wykupienia")
    assert not verify(word.text, SHEET).ok


def test_questions_are_asked_about_saved_report_and_refusals_counted():
    record = {
        "ticker": "TEST",
        "report": "## Podsumowanie\nKurs jest powyżej SMA(200).",
        "facts": SHEET.to_records(),
        "news": [
            {"source": "Reuters", "published": "2026-10-14 08:00", "headline": "Nowy produkt"}
        ],
        "company": "Test Inc.",
    }
    answers = [
        "Według [1] spółka pokazała nowy produkt."
        if group != OUTSIDE_QUESTION
        else "Raport nie zawiera tej informacji."
        for group, _ in QUESTIONS
    ]
    llm = FakeListChatModel(responses=answers)
    rows = ask_questions(llm, record)
    assert len(rows) == len(QUESTIONS)
    assert all(r["zweryfikowana"] for r in rows)
    summary = {r["grupa"]: r for r in summarize_answers(rows)}
    assert summary[OUTSIDE_QUESTION]["odmowy [%]"] == 100.0
    assert summary["z raportu"]["odmowy [%]"] == 0.0
    assert "Źródła wiadomości" in rows[0]["odpowiedź"]


def test_question_command_uses_latest_saved_report(tmp_path, monkeypatch):
    results = tmp_path / "raporty.jsonl"
    record = {"ticker": "TEST", "variant": WITH_CARD, "report": "Raport.", "facts": []}
    append_record(record | {"report": "Stary raport."}, results)
    append_record(record, results)
    seen = []
    monkeypatch.setattr(
        "marketagents.evaluation.ask_questions", lambda llm, r: seen.append(r["report"]) or []
    )
    monkeypatch.setattr("marketagents.llm.get_chat_model", lambda settings: None)
    main(["--wyniki", str(results), "pytania", "--odpowiedzi", str(tmp_path / "p.jsonl")])
    assert seen == ["Raport."]
