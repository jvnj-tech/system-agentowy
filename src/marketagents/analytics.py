"""Miary do wykresów liczone z notowań, sprawozdań i danych makro, bez modelu językowego."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def relative_performance(close: pd.Series, benchmark: pd.Series) -> pd.DataFrame:
    """Kurs spółki i indeksu na wspólnych sesjach, oba znormalizowane do 100 na początku."""
    both = pd.concat({"spółka": close, "indeks": benchmark}, axis=1, join="inner").dropna()
    return both / both.iloc[0] * 100


def drawdown(close: pd.Series) -> pd.Series:
    """Spadek od najwyższego dotychczasowego kursu zamknięcia, w %."""
    return (close / close.cummax() - 1) * 100


def monthly_returns(close: pd.Series) -> pd.DataFrame:
    """Miesięczne stopy zwrotu w % w układzie rok (wiersze) × miesiąc 1–12 (kolumny)."""
    month_end = close.resample("ME").last()
    returns = month_end.pct_change().dropna() * 100
    table = pd.DataFrame(
        {"rok": returns.index.year, "miesiąc": returns.index.month, "zwrot": returns.values}
    )
    pivot = table.pivot(index="rok", columns="miesiąc", values="zwrot")
    return pivot.reindex(columns=range(1, 13))


def rolling_volatility(close: pd.Series, window: int = 30) -> pd.Series:
    """Zmienność krocząca: odchylenie standardowe dziennych log-zwrotów × √252, w %."""
    log_returns = np.log(close / close.shift(1))
    return log_returns.rolling(window, min_periods=window).std() * np.sqrt(TRADING_DAYS) * 100


def margins(annual: pd.DataFrame) -> pd.DataFrame:
    """Marże brutto, operacyjna i netto w % przychodów dla kolejnych lat obrotowych."""
    if "revenue" not in annual:
        return pd.DataFrame()
    out = pd.DataFrame(index=annual.index)
    for column, name in [
        ("gross_profit", "marża brutto"),
        ("operating_income", "marża operacyjna"),
        ("net_income", "marża netto"),
    ]:
        if column in annual:
            out[name] = annual[column] / annual["revenue"] * 100
    return out.dropna(how="all")


def macro_history(macro: pd.DataFrame) -> pd.DataFrame:
    """Stopa Fed, rentowność 10-letnich obligacji i inflacja CPI r/r w % (koniec miesiąca)."""
    series: dict[str, pd.Series] = {}
    if "fed_funds_rate" in macro:
        series["stopa Fed"] = macro["fed_funds_rate"].dropna().resample("ME").last()
    if "treasury_10y" in macro:
        series["rentowność 10Y"] = macro["treasury_10y"].dropna().resample("ME").last()
    if "cpi" in macro:
        cpi = macro["cpi"].dropna().resample("ME").last()
        series["inflacja CPI r/r"] = cpi.pct_change(12, fill_method=None) * 100
    if not series:
        return pd.DataFrame()
    return pd.concat(series, axis=1, sort=True).dropna(how="all")
