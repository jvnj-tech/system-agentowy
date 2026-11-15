"""Notowania dzienne OHLCV z Yahoo Finance (biblioteka yfinance)."""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

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
    # Kolumna splitów („Stock Splits” przy actions=True) jest potrzebna do przeliczenia EPS
    # ze sprawozdania sprzed splitu; ceny z auto_adjust są już o splity skorygowane.
    keep = COLUMNS + (["splits"] if "stock_splits" in df.columns else [])
    df = df.rename(columns={"stock_splits": "splits"}) if "stock_splits" in df.columns else df
    df = df[keep].dropna(subset=["close"])
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "date"
    return df.sort_index()


# Notowania z ostatniej sesji są dostępne po zamknięciu rynku, więc wystarczy raz na kilka godzin.
MAX_AGE = timedelta(hours=6)


def _download(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        actions=True,
        progress=False,
        multi_level_index=False,
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
    prices = cache.frame(
        "prices", key, lambda: normalize_ohlcv(_download(ticker, start, end)), max_age
    )
    return completed_sessions(prices)


NEW_YORK = ZoneInfo("America/New_York")
# Sesja na giełdzie Nasdaq kończy się o 16:00 czasu nowojorskiego; z zapasem na publikację
# kursu zamknięcia wiersz z bieżącego dnia jest uznawany za kompletny od 16:30.
SESSION_CLOSED = (16, 30)


def completed_sessions(prices: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    """Bez wiersza z trwającej sesji: w czasie sesji yfinance podaje w nim bieżący kurs,
    który w raporcie zostałby opisany jako kurs zamknięcia."""
    now = (now or datetime.now(NEW_YORK)).astimezone(NEW_YORK)
    if prices.empty or prices.index[-1].date() != now.date():
        return prices
    if (now.hour, now.minute) >= SESSION_CLOSED:
        return prices
    return prices.iloc[:-1]


def splits_after(prices: pd.DataFrame, date: str | pd.Timestamp) -> float:
    """Łączny współczynnik splitów po danej dacie (np. 25 dla splitu 25:1, 1 bez splitów)."""
    if "splits" not in prices.columns:
        return 1.0
    after = prices.loc[prices.index > pd.Timestamp(date), "splits"]
    after = after[after > 0]
    return float(after.prod()) if not after.empty else 1.0
