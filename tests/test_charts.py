import pandas as pd

from marketagents.charts import fundamentals_chart, technical_chart
from marketagents.indicators import compute_all


def test_technical_chart_has_all_panels(close):
    prices = pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1}
    )
    fig = technical_chart(prices, compute_all(prices))
    names = [trace.name for trace in fig.data]
    for name in ["Kurs", "SMA(20)", "SMA(50)", "SMA(200)", "RSI(14)", "MACD", "Histogram MACD"]:
        assert name in names
    rsi_levels = sorted(shape.y0 for shape in fig.layout.shapes)
    assert rsi_levels == [30, 70]


def test_fundamentals_chart_in_billions():
    annual = pd.DataFrame(
        {"revenue": [200e9, 250e9], "net_income": [50e9, 60e9]},
        index=pd.to_datetime(["2024-09-28", "2025-09-27"]),
    )
    fig = fundamentals_chart(annual)
    assert [trace.name for trace in fig.data] == ["Przychody", "Zysk netto"]
    assert list(fig.data[0].x) == ["2024", "2025"]
    assert list(fig.data[0].y) == [200, 250]


def test_volume_panel_under_candles(close):
    prices = pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 5}
    )
    fig = technical_chart(prices, compute_all(prices))
    volume = next(t for t in fig.data if t.name == "Wolumen")
    assert volume.yaxis == "y2"


def test_additional_charts(close):
    from marketagents.analytics import (
        drawdown,
        macro_history,
        monthly_returns,
        relative_performance,
        rolling_volatility,
    )
    from marketagents.charts import (
        drawdown_chart,
        macro_chart,
        margins_chart,
        monthly_heatmap,
        relative_chart,
        volatility_chart,
    )

    rel = relative_chart(relative_performance(close, close * 2), "TEST", "NASDAQ-100")
    assert [t.name for t in rel.data] == ["TEST", "NASDAQ-100"]
    assert drawdown_chart(drawdown(close)).data[0].y.max() <= 0

    heatmap = monthly_heatmap(monthly_returns(close)).data[0]
    assert heatmap.zmid == 0 and len(heatmap.x) == 12

    assert len(volatility_chart(rolling_volatility(close), 30).data) == 1
    margins = pd.DataFrame(
        {"marża brutto": [40.0, 41.0]}, index=pd.to_datetime(["2024-09-28", "2025-09-27"])
    )
    assert list(margins_chart(margins).data[0].x) == ["2024", "2025"]
    macro = pd.DataFrame(
        {"fed_funds_rate": [4.0, 4.25]}, index=pd.date_range("2025-01-01", periods=2, freq="MS")
    )
    assert macro_chart(macro_history(macro)).data[0].name == "stopa Fed"
