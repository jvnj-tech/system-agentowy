"""Notowania dzienne OHLCV z Yahoo Finance (biblioteka yfinance)."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd

from marketagents.data.cache import DiskCache

COLUMNS = ["open", "high", "low", "close", "volume"]


def normalize_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    """Sprowadza ramkę z yfinance do kolumn open/high/low/close/volume z indeksem dat."""
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Brak kolumn w danych cenowych: {missing}")
    df = df[COLUMNS].dropna(subset=["close"])
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "date"
    return df.sort_index()


# Notowania z ostatniej sesji są dostępne po zamknięciu rynku, więc wystarczy raz na kilka godzin.
MAX_AGE = timedelta(hours=6)


def _download(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(
        ticker, start=start, end=end, auto_adjust=True, progress=False, multi_level_index=False
    )
    if raw.empty:
        raise ValueError(f"yfinance nie zwrócił danych dla {ticker}")
    return raw


def get_prices(
    cache: DiskCache, ticker: str, start: str = "2015-01-01", end: str | None = None
) -> pd.DataFrame:
    """Ceny skorygowane o splity i dywidendy (auto_adjust)."""
    key = f"{ticker}_{start}_{end or 'latest'}"
    # Zakres z datą końcową się nie zmienia; „latest” jest pobierany ponownie po MAX_AGE.
    max_age = None if end else MAX_AGE
    return cache.frame(
        "prices", key, lambda: normalize_ohlcv(_download(ticker, start, end)), max_age
    )
