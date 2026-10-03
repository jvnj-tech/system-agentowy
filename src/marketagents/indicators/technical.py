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

from dataclasses import dataclass

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


@dataclass(frozen=True)
class MovingAverages:
    """Okna średnich kroczących: krótkie, średnie i długie dla SMA oraz okno EMA."""

    sma: tuple[int, int, int] = (20, 50, 200)
    ema: int = 20

    def __post_init__(self) -> None:
        short, mid, long = self.sma
        if not 2 <= short < mid < long:
            raise ValueError("Okna SMA muszą rosnąć: krótkie < średnie < długie (co najmniej 2)")
        if self.ema < 2:
            raise ValueError("Okno EMA musi wynosić co najmniej 2")

    @property
    def long(self) -> int:
        return self.sma[-1]

    @classmethod
    def parse(cls, sma: str, ema: str) -> MovingAverages:
        """Okna z tekstu, np. sma="20,50,200", ema="20" (zmienne SMA_WINDOWS, EMA_WINDOW)."""
        windows = tuple(int(x) for x in sma.split(","))
        if len(windows) != 3:
            raise ValueError("SMA_WINDOWS musi zawierać trzy okna, np. 20,50,200")
        return cls(windows, int(ema))


def compute_all(prices: pd.DataFrame, ma: MovingAverages | None = None) -> pd.DataFrame:
    """Zestaw wskaźników używany w raporcie, liczony z kolumny close."""
    ma = ma or MovingAverages()
    close = prices["close"]
    return pd.concat(
        [
            *(sma(close, n).rename(f"sma_{n}") for n in ma.sma),
            ema(close, ma.ema).rename(f"ema_{ma.ema}"),
            rsi(close, 14).rename("rsi_14"),
            macd(close),
            bollinger_bands(close),
        ],
        axis=1,
    )
