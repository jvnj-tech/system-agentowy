import numpy as np
import pandas as pd
import pytest

from marketagents.analytics import (
    drawdown,
    macro_history,
    margins,
    monthly_returns,
    relative_performance,
    rolling_volatility,
)


def series(values, start="2024-01-01", freq="B"):
    return pd.Series(
        values, index=pd.date_range(start, periods=len(values), freq=freq), dtype=float
    )


def test_relative_performance_starts_at_100_on_common_dates():
    close = series([10, 11, 12, 15])
    benchmark = series([200, 220, 180], start="2024-01-02")
    rel = relative_performance(close, benchmark)
    assert list(rel.index) == list(benchmark.index)
    assert rel["spółka"].tolist() == pytest.approx([100, 1200 / 11, 1500 / 11])
    assert rel["indeks"].tolist() == pytest.approx([100, 110, 90])


def test_drawdown_from_running_peak():
    dd = drawdown(series([100, 120, 90, 130, 117]))
    assert dd.tolist() == pytest.approx([0, 0, -25, 0, -10])


def test_monthly_returns_table():
    close = series([100, 110, 99, 99], start="2023-11-30", freq="ME")
    table = monthly_returns(close)
    assert list(table.columns) == list(range(1, 13))
    assert list(table.index) == [2023, 2024]
    assert table.loc[2023, 12] == pytest.approx(10)
    assert table.loc[2024, 1] == pytest.approx(-10)
    assert table.loc[2024, 2] == pytest.approx(0)
    assert np.isnan(table.loc[2023, 11])


def test_rolling_volatility_is_annualized_std_of_log_returns(close):
    vol = rolling_volatility(close, window=30)
    assert vol.iloc[:30].isna().all()
    expected = np.log(close.iloc[-31:]).diff().dropna().std() * np.sqrt(252) * 100
    assert vol.iloc[-1] == pytest.approx(expected)


def test_constant_price_has_zero_volatility():
    assert rolling_volatility(series([50.0] * 40), window=30).iloc[-1] == pytest.approx(0)


def test_margins():
    annual = pd.DataFrame(
        {"revenue": [200.0, 250.0], "gross_profit": [80.0, 100.0], "net_income": [20.0, 50.0]},
        index=pd.to_datetime(["2024-09-28", "2025-09-27"]),
    )
    result = margins(annual)
    assert list(result.columns) == ["marża brutto", "marża netto"]
    assert result["marża brutto"].tolist() == pytest.approx([40, 40])
    assert result["marża netto"].tolist() == pytest.approx([10, 20])
    assert margins(annual.drop(columns="revenue")).empty


def test_macro_history_inflation_year_over_year():
    months = pd.date_range("2024-01-01", periods=14, freq="MS")
    cpi = pd.Series(np.linspace(300, 313, 14), index=months)
    daily = pd.Series([4.0, 4.2], index=pd.to_datetime(["2025-02-03", "2025-02-28"]))
    macro = pd.concat({"cpi": cpi, "treasury_10y": daily}, axis=1)
    result = macro_history(macro)
    assert list(result.columns) == ["rentowność 10Y", "inflacja CPI r/r"]
    last = result.loc["2025-02-28"]
    assert last["inflacja CPI r/r"] == pytest.approx((313 / 301 - 1) * 100)
    assert last["rentowność 10Y"] == pytest.approx(4.2)
    assert macro_history(pd.DataFrame()).empty
