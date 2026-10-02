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
