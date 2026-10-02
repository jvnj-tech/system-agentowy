"""Interaktywne wykresy Plotly dla interfejsu."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Kolory serii w stałej kolejności (dobrane pod kątem czytelności przy daltonizmie).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
UP, DOWN = "#1baf7a", "#e34948"
# Skala rozbieżna dla zwrotów: czerwień (spadek), szary środek (zero), niebieski (wzrost).
DIVERGING = [[0.0, DOWN], [0.5, "#8c8c88"], [1.0, BLUE]]
BAND = "rgba(128, 128, 128, 0.15)"
GUIDE = "rgba(128, 128, 128, 0.6)"


def technical_chart(prices: pd.DataFrame, indicators: pd.DataFrame) -> go.Figure:
    """Świece z SMA i wstęgami Bollingera, pod nimi wolumen, RSI(14) i MACD na wspólnej osi."""
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        row_heights=[0.5, 0.14, 0.18, 0.18],
        subplot_titles=("", "Wolumen", "RSI(14)", "MACD(12, 26, 9)"),
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

    rising = prices["close"] >= prices["open"]
    fig.add_trace(
        go.Bar(
            x=x,
            y=prices["volume"],
            name="Wolumen",
            showlegend=False,
            marker={"color": [UP if r else DOWN for r in rising], "opacity": 0.6},
            hovertemplate="%{y:,.0f}<extra>wolumen</extra>",
        ),
        row=2,
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
        row=3,
        col=1,
    )
    for level in (30, 70):
        fig.add_hline(y=level, line={"color": GUIDE, "dash": "dash", "width": 1}, row=3, col=1)

    hist = indicators["macd_hist"]
    fig.add_trace(
        go.Bar(
            x=x,
            y=hist,
            name="Histogram MACD",
            showlegend=False,
            marker={"color": [UP if v >= 0 else DOWN for v in hist.fillna(0)]},
        ),
        row=4,
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
        row=4,
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
        row=4,
        col=1,
    )

    fig.update_yaxes(title_text="kurs (USD)", row=1, col=1)
    fig.update_yaxes(range=[0, 100], tickvals=[30, 50, 70], row=3, col=1)
    fig.update_xaxes(rangeslider_visible=False, rangebreaks=[{"bounds": ["sat", "mon"]}])
    fig.update_layout(
        height=880,
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


def _layout(fig: go.Figure, height: int, yaxis_title: str) -> go.Figure:
    fig.update_layout(
        height=height,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        hovermode="x unified",
        yaxis_title=yaxis_title,
        legend={"orientation": "h", "y": 1.0, "yanchor": "bottom", "x": 0},
        separators=", ",
    )
    return fig


def relative_chart(relative: pd.DataFrame, ticker: str, benchmark: str) -> go.Figure:
    """Kurs spółki i indeksu znormalizowane do 100 na początku okresu."""
    fig = go.Figure()
    for column, name, color in [("spółka", ticker, BLUE), ("indeks", benchmark, ORANGE)]:
        fig.add_trace(
            go.Scatter(
                x=relative.index,
                y=relative[column],
                name=name,
                line={"color": color, "width": 2},
                hovertemplate="%{y:.1f}",
            )
        )
    fig.add_hline(y=100, line={"color": GUIDE, "dash": "dash", "width": 1})
    return _layout(fig, 360, "wartość (początek okresu = 100)")


def drawdown_chart(values: pd.Series) -> go.Figure:
    """Obsunięcie kursu od szczytu w %."""
    fig = go.Figure(
        go.Scatter(
            x=values.index,
            y=values,
            name="obsunięcie",
            fill="tozeroy",
            line={"color": DOWN, "width": 2},
            fillcolor="rgba(227, 73, 72, 0.25)",
            hovertemplate="%{y:.1f}%",
        )
    )
    fig.update_layout(showlegend=False)
    return _layout(fig, 300, "spadek od szczytu (%)")


MONTHS = ["sty", "lut", "mar", "kwi", "maj", "cze", "lip", "sie", "wrz", "paź", "lis", "gru"]


def monthly_heatmap(table: pd.DataFrame) -> go.Figure:
    """Miesięczne stopy zwrotu w układzie rok × miesiąc."""
    limit = float(table.abs().max().max()) if table.notna().any().any() else 1.0
    # round(..) + 0.0 zamienia „-0,0” na „0,0”.
    text = table.map(lambda v: "" if pd.isna(v) else f"{round(v, 1) + 0.0:.1f}".replace(".", ","))
    fig = go.Figure(
        go.Heatmap(
            z=table.values,
            x=MONTHS,
            y=table.index.astype(str),
            colorscale=DIVERGING,
            zmid=0,
            zmin=-limit,
            zmax=limit,
            text=text.values,
            texttemplate="%{text}",
            xgap=2,
            ygap=2,
            colorbar={"title": "%"},
            hovertemplate="%{x} %{y}: %{z:.1f}%<extra></extra>",
        )
    )
    fig.update_yaxes(type="category", autorange="reversed")
    fig.update_layout(
        height=max(240, 34 * len(table) + 80),
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        separators=", ",
    )
    return fig


def volatility_chart(values: pd.Series, window: int) -> go.Figure:
    """Zmienność krocząca w ujęciu rocznym, w %."""
    fig = go.Figure(
        go.Scatter(
            x=values.index,
            y=values,
            name=f"zmienność {window} sesji",
            line={"color": BLUE, "width": 2},
            hovertemplate="%{y:.1f}%",
        )
    )
    fig.update_layout(showlegend=False)
    return _layout(fig, 300, "zmienność roczna (%)")


def _lines(frame: pd.DataFrame, x, yaxis_title: str, height: int) -> go.Figure:
    fig = go.Figure()
    for column, color in zip(frame.columns, [BLUE, ORANGE, AQUA], strict=False):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=frame[column],
                name=column,
                mode="lines+markers" if len(frame) < 30 else "lines",
                line={"color": color, "width": 2},
                marker={"size": 8},
                connectgaps=True,
                hovertemplate="%{y:.1f}%",
            )
        )
    return _layout(fig, height, yaxis_title)


def margins_chart(values: pd.DataFrame) -> go.Figure:
    """Marże brutto, operacyjna i netto w kolejnych latach obrotowych."""
    fig = _lines(values, values.index.year.astype(str), "% przychodów", 340)
    fig.update_xaxes(type="category", title_text="rok obrotowy (koniec)")
    return fig


def macro_chart(values: pd.DataFrame) -> go.Figure:
    """Stopa Fed, rentowność 10-letnich obligacji i inflacja CPI r/r w czasie."""
    return _lines(values, values.index, "%", 340)
