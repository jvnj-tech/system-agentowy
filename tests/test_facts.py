import pandas as pd
import pytest

from marketagents.facts import (
    FUNDAMENTAL,
    MACRO,
    TECHNICAL,
    Fact,
    FactSheet,
    format_number,
    fundamental_facts,
    macro_facts,
    technical_facts,
)
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


def test_fundamental_facts():
    annual = pd.DataFrame(
        {
            "revenue": [80e9, 100e9],
            "net_income": [16e9, 25e9],
            "operating_income": [20e9, 30e9],
            "total_assets": [200e9, 250e9],
            "total_liabilities": [120e9, 150e9],
            "stockholders_equity": [80e9, 100e9],
            "eps_diluted": [4.0, 5.0],
        },
        index=pd.to_datetime(["2024-12-31", "2025-12-31"]),
    )
    facts = FactSheet(fundamental_facts(annual, close=150.0))
    assert facts["fy_end"].value == "2025-12-31"
    assert facts["fy_revenue"].formatted() == "100,0 mld USD"
    assert facts["fy_revenue_growth"].value == pytest.approx(25.0)
    assert facts["fy_net_margin"].value == pytest.approx(25.0)
    assert facts["fy_liabilities_to_assets"].value == pytest.approx(60.0)
    assert facts["fy_roe"].value == pytest.approx(25.0)
    assert facts["pe_ratio"].value == pytest.approx(30.0)
    assert "fy_gross_margin" not in facts
    assert all(f.group == FUNDAMENTAL for f in facts)


def test_fundamental_facts_empty():
    assert fundamental_facts(pd.DataFrame()) == []


def test_macro_facts():
    months = pd.date_range("2024-01-01", periods=14, freq="MS")
    macro = pd.DataFrame(
        {
            "cpi": [100.0 + i for i in range(14)],
            "fed_funds_rate": [4.33] * 14,
            "unemployment_rate": [4.1] * 13 + [None],
        },
        index=months,
    )
    facts = FactSheet(macro_facts(macro))
    assert facts["macro_inflation"].value == pytest.approx((113 / 101 - 1) * 100)
    assert facts["macro_inflation_date"].value == "2025-02-01"
    assert facts["macro_unemployment_date"].value == "2025-01-01"
    assert "macro_10y" not in facts


def test_group_selection_keeps_general_facts():
    sheet = FactSheet(
        [
            Fact("ticker", "", "X"),
            Fact("a", "", 1.0, group=TECHNICAL),
            Fact("b", "", 2.0, group=MACRO),
        ]
    )
    assert [f.id for f in sheet.group(MACRO)] == ["ticker", "b"]
