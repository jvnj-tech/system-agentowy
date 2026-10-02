"""Implementacje wskaźników technicznych.

Definicje:
- SMA: średnia arytmetyczna z n ostatnich cen zamknięcia.
- EMA: średnia wykładnicza z wagą alfa = 2 / (n + 1), inicjowana pierwszą wartością szeregu.
- RSI: wskaźnik Wildera; pierwsza średnia zysków i strat to średnia arytmetyczna z n zmian,
  kolejne są wygładzane wzorem avg_t = (avg_{t-1} * (n - 1) + x_t) / n.
- MACD: EMA(12) - EMA(26), linia sygnału EMA(9) z linii MACD, histogram = MACD - sygnał.
- Wstęgi Bollingera: SMA(20) +/- k * odchylenie standardowe populacji z 20 okresów.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _check_window(window: int) -> None:
    if window < 1:
        raise ValueError("Okno musi być dodatnie")


def sma(close: pd.Series, window: int) -> pd.Series:
    _check_window(window)
    return close.rolling(window, min_periods=window).mean()


def ema(close: pd.Series, window: int) -> pd.Series:
    _check_window(window)
    return close.ewm(span=window, adjust=False, min_periods=window).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    _check_window(window)
    delta = close.diff().to_numpy()
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    out = np.full(len(close), np.nan)
    if len(close) <= window:
        return pd.Series(out, index=close.index, name="rsi")

    avg_gain = gains[1 : window + 1].mean()
    avg_loss = losses[1 : window + 1].mean()
    out[window] = _rsi_value(avg_gain, avg_loss)
    for i in range(window + 1, len(close)):
        avg_gain = (avg_gain * (window - 1) + gains[i]) / window
        avg_loss = (avg_loss * (window - 1) + losses[i]) / window
        out[i] = _rsi_value(avg_gain, avg_loss)
    return pd.Series(out, index=close.index, name="rsi")


def _rsi_value(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    if fast >= slow:
        raise ValueError("Okres szybkiej średniej musi być krótszy niż wolnej")
    # Linia MACD istnieje od chwili, gdy obie średnie mają pełne okno;
    # linia sygnału jest liczona tylko z tych wartości.
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def bollinger_bands(close: pd.Series, window: int = 20, k: float = 2.0) -> pd.DataFrame:
    _check_window(window)
    mid = sma(close, window)
    std = close.rolling(window, min_periods=window).std(ddof=0)
    return pd.DataFrame({"bb_mid": mid, "bb_upper": mid + k * std, "bb_lower": mid - k * std})


def compute_all(prices: pd.DataFrame) -> pd.DataFrame:
    """Zestaw wskaźników używany w raporcie, liczony z kolumny close."""
    close = prices["close"]
    return pd.concat(
        [
            sma(close, 20).rename("sma_20"),
            sma(close, 50).rename("sma_50"),
            sma(close, 200).rename("sma_200"),
            ema(close, 20).rename("ema_20"),
            rsi(close, 14).rename("rsi_14"),
            macd(close),
            bollinger_bands(close),
        ],
        axis=1,
    )
