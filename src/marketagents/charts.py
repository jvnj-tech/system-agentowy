"""Interaktywne wykresy Plotly dla interfejsu."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Kolory serii w stałej kolejności (dobrane pod kątem czytelności przy daltonizmie).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
UP, DOWN = "#1baf7a", "#e34948"
BAND = "rgba(128, 128, 128, 0.15)"
GUIDE = "rgba(128, 128, 128, 0.6)"


def technical_chart(prices: pd.DataFrame, indicators: pd.DataFrame) -> go.Figure:
    """Świece z SMA i wstęgami Bollingera, pod nimi RSI(14) i MACD na wspólnej osi czasu."""
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=("", "RSI(14)", "MACD(12, 26, 9)"),
    )
    x = prices.index

    fig.add_trace(
        go.Scatter(
            x=x,
            y=indicators["bb_upper"],
            name="Bollinger(20, 2)",
            line={"width": 0},
            legendgroup="bb",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=indicators["bb_lower"],
            name="Bollinger(20, 2)",
            line={"width": 0},
            fill="tonexty",
            fillcolor=BAND,
            legendgroup="bb",
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Candlestick(
            x=x,
            open=prices["open"],
            high=prices["high"],
            low=prices["low"],
            close=prices["close"],
            name="Kurs",
            increasing={"line": {"color": UP}},
            decreasing={"line": {"color": DOWN}},
        ),
        row=1,
        col=1,
    )
    for column, name, color in [
        ("sma_20", "SMA(20)", BLUE),
        ("sma_50", "SMA(50)", ORANGE),
        ("sma_200", "SMA(200)", AQUA),
    ]:
        fig.add_trace(
            go.Scatter(x=x, y=indicators[column], name=name, line={"color": color, "width": 2}),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=indicators["rsi_14"],
            name="RSI(14)",
            line={"color": BLUE, "width": 2},
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    for level in (30, 70):
        fig.add_hline(y=level, line={"color": GUIDE, "dash": "dash", "width": 1}, row=2, col=1)

    hist = indicators["macd_hist"]
    fig.add_trace(
        go.Bar(
            x=x,
            y=hist,
            name="Histogram MACD",
            showlegend=False,
            marker={"color": [UP if v >= 0 else DOWN for v in hist.fillna(0)]},
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=indicators["macd"],
            name="MACD",
            line={"color": BLUE, "width": 2},
            showlegend=False,
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=indicators["macd_signal"],
            name="Sygnał",
            line={"color": ORANGE, "width": 2},
            showlegend=False,
        ),
        row=3,
        col=1,
    )

    fig.update_yaxes(title_text="kurs (USD)", row=1, col=1)
    fig.update_yaxes(range=[0, 100], tickvals=[30, 50, 70], row=2, col=1)
    fig.update_xaxes(rangeslider_visible=False, rangebreaks=[{"bounds": ["sat", "mon"]}])
    fig.update_layout(
        height=760,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.0, "yanchor": "bottom", "x": 0},
        separators=", ",
    )
    return fig


def fundamentals_chart(annual: pd.DataFrame) -> go.Figure:
    """Przychody i zysk netto z rocznych sprawozdań, w mld USD."""
    years = annual.index.year.astype(str)
    fig = go.Figure()
    for column, name, color in [
        ("revenue", "Przychody", BLUE),
        ("net_income", "Zysk netto", ORANGE),
    ]:
        if column in annual:
            fig.add_trace(
                go.Bar(
                    x=years,
                    y=annual[column] / 1e9,
                    name=name,
                    marker={"color": color},
                    hovertemplate="%{y:,.1f} mld USD<extra>" + name + "</extra>",
                )
            )
    fig.update_layout(
        barmode="group",
        bargap=0.25,
        bargroupgap=0.05,
        height=360,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        yaxis_title="mld USD",
        xaxis={"title": "rok obrotowy (koniec)", "type": "category"},
        legend={"orientation": "h", "y": 1.1, "x": 0},
        separators=", ",
    )
    return fig
