import pandas as pd
import pytest

from marketagents.facts import Fact, FactSheet, format_number, technical_facts
from marketagents.indicators import compute_all


def test_format_number_polish_style():
    assert format_number(1234567.891, 2) == "1 234 567,89"
    assert format_number(-3.25, 1) == "-3,2"
    assert format_number(61.0, 1) == "61,0"


def test_fact_formatting_with_units():
    assert Fact("a", "", 12.345, "USD").formatted() == "12,35 USD"
    assert Fact("b", "", 4.81, "%", 1).formatted() == "4,8%"
    assert Fact("c", "", "2026-10-01").formatted() == "2026-10-01"


def test_fact_sheet_rejects_duplicate_ids():
    with pytest.raises(ValueError):
        FactSheet([Fact("x", "", 1.0), Fact("x", "", 2.0)])


def test_fact_sheet_round_trip():
    sheet = FactSheet([Fact("x", "opis", 1.5, "USD"), Fact("d", "data", "2026-01-02")])
    restored = FactSheet.from_records(sheet.to_records())
    assert [f.id for f in restored] == ["x", "d"]
    assert "{{x}}" in restored.as_prompt_table()


def test_technical_facts_from_prices(close):
    prices = pd.DataFrame({"close": close})
    facts = FactSheet(technical_facts("TEST", prices, compute_all(prices)))
    assert facts["close"].value == pytest.approx(close.iloc[-1])
    expected = (close.iloc[-1] / close.iloc[-22] - 1) * 100
    assert facts["change_1m"].value == pytest.approx(expected)
    assert facts["as_of"].value == close.index[-1].strftime("%Y-%m-%d")
    assert {"rsi_14", "macd", "sma_200", "dist_sma_200"} <= {f.id for f in facts}


def test_technical_facts_skip_indicators_without_history(close):
    prices = pd.DataFrame({"close": close.iloc[:60]})
    ids = {f.id for f in technical_facts("TEST", prices, compute_all(prices))}
    assert "sma_50" in ids
    assert "sma_200" not in ids and "change_1y" not in ids
