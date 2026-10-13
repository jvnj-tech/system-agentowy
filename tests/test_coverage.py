from marketagents.coverage import coverage_note
from marketagents.facts import Fact, FactSheet


def test_coverage_note_lists_dates_staleness_and_missing_data():
    facts = FactSheet(
        [
            Fact("as_of", "data", "2026-10-02"),
            Fact("fy_end", "koniec roku", "2025-12-27"),
            Fact("fy_eps", "EPS", -0.06, "USD"),
            Fact("macro_10y_date", "data", "2026-09-30"),
            Fact("macro_inflation_date", "data", "2026-08-01"),
        ]
    )
    news = [{"published": "2026-10-13 09:00"}, {"published": "2026-10-07 12:00"}]
    note = coverage_note(facts, news)
    assert "notowania do 2026-10-02" in note
    assert "dane sprzed 10 miesięcy" in note
    assert "od 2026-08-01 do 2026-09-30" in note
    assert "2 z okresu 2026-10-07–2026-10-13" in note
    assert "o 11 dni starsze niż najnowsze wiadomości" in note
    assert "wskaźnik C/Z (zysk na akcję nie jest dodatni)" in note


def test_coverage_note_without_news_and_recent_prices():
    facts = FactSheet([Fact("as_of", "data", "2026-10-12")])
    note = coverage_note(facts, [])
    assert "brak wiadomości o spółce" in note and "Uwaga" not in note
