"""Dane fundamentalne z SEC EDGAR (API XBRL companyfacts).

SEC wymaga nagłówka User-Agent z nazwą i adresem e-mail (zmienna SEC_USER_AGENT).
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import requests

from marketagents.data.cache import DiskCache

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
# Nowe sprawozdanie roczne pojawia się raz w roku, ale jego termin nie jest znany z góry.
MAX_AGE = timedelta(days=7)

# Nazwa wskaźnika -> pojęcia us-gaap w kolejności preferencji (spółki raportują różnie,
# a jedna spółka potrafi zmienić pojęcie między latami, np. NVIDIA w 2023 r.).
CONCEPTS: dict[str, list[str]] = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
    ],
    "net_income": ["NetIncomeLoss"],
    "operating_income": ["OperatingIncomeLoss"],
    "gross_profit": ["GrossProfit"],
    "cost_of_revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold"],
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    "liabilities_and_equity": ["LiabilitiesAndStockholdersEquity"],
    "stockholders_equity": ["StockholdersEquity"],
    "total_equity": ["StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "eps_diluted": ["EarningsPerShareDiluted"],
}


# Pozycje bilansowe (stan na dzień), pozostałe są przepływami za okres.
BALANCE_SHEET = frozenset(
    {
        "total_assets",
        "total_liabilities",
        "liabilities_and_equity",
        "stockholders_equity",
        "total_equity",
    }
)


def _get_json(url: str, user_agent: str | None) -> dict:
    if not user_agent:
        raise RuntimeError("Ustaw SEC_USER_AGENT, np. 'Imię Nazwisko adres@email.com'")
    resp = requests.get(url, headers={"User-Agent": user_agent}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def cik_for_ticker(cache: DiskCache, ticker: str, user_agent: str | None) -> int:
    table = cache.json("sec", "company_tickers", lambda: _get_json(TICKERS_URL, user_agent))
    for row in table.values():
        if row["ticker"].upper() == ticker.upper():
            return int(row["cik_str"])
    raise KeyError(f"Nie znaleziono CIK dla {ticker}")


def company_title(cache: DiskCache, ticker: str, user_agent: str | None) -> str | None:
    """Nazwa spółki z tabeli SEC (np. „LULULEMON ATHLETICA INC.”) albo None."""
    try:
        table = cache.json("sec", "company_tickers", lambda: _get_json(TICKERS_URL, user_agent))
    except Exception:  # noqa: BLE001
        return None
    return next(
        (row["title"] for row in table.values() if row["ticker"].upper() == ticker.upper()), None
    )


def annual_facts(companyfacts: dict) -> pd.DataFrame:
    """Wartości roczne z raportów 10-K: wiersz = rok obrotowy, kolumna = wskaźnik.

    Bierze tylko okresy pełnoroczne (fp == FY), a przy kilku zgłoszeniach tego samego
    okresu zostawia najnowsze (korekty w późniejszych raportach). Pojęcia z listy są łączone
    okres po okresie: brak wartości w pojęciu preferowanym uzupełnia następne pojęcie.
    Zysk brutto i zobowiązania, których spółka nie podaje wprost, są wyliczane
    z przychodów i kosztu sprzedaży oraz z sumy bilansowej i kapitału.
    """
    gaap = companyfacts.get("facts", {}).get("us-gaap", {})
    columns: dict[str, pd.Series] = {}
    for name, candidates in CONCEPTS.items():
        for concept in candidates:
            units = gaap.get(concept, {}).get("units", {})
            rows = units.get("USD") or units.get("USD/shares") or []
            rows = [r for r in rows if r.get("form") == "10-K" and r.get("fp") == "FY"]
            if name not in BALANCE_SHEET:
                # Pozycje przepływowe: tylko okresy ok. 12-miesięczne.
                rows = [r for r in rows if _is_full_year(r)]
            if not rows:
                continue
            df = pd.DataFrame(rows)
            df["end"] = pd.to_datetime(df["end"])
            df = df.sort_values("filed").drop_duplicates("end", keep="last")
            series = df.set_index("end")["val"].astype(float)
            columns[name] = columns[name].combine_first(series) if name in columns else series
    if not columns:
        return pd.DataFrame()
    out = pd.DataFrame(columns).sort_index()
    out.index.name = "period_end"
    if {"revenue", "cost_of_revenue"} <= set(out):
        derived = out["revenue"] - out["cost_of_revenue"]
        out["gross_profit"] = out.get("gross_profit", derived).combine_first(derived)
    if "liabilities_and_equity" in out:
        equity = out.get("total_equity", pd.Series(dtype=float))
        if "stockholders_equity" in out:
            equity = equity.combine_first(out["stockholders_equity"])
        derived = out["liabilities_and_equity"] - equity
        out["total_liabilities"] = out.get("total_liabilities", derived).combine_first(derived)
    return out.drop(
        columns=["cost_of_revenue", "liabilities_and_equity", "total_equity"], errors="ignore"
    )


def _is_full_year(row: dict) -> bool:
    if "start" not in row:
        return False
    days = (pd.Timestamp(row["end"]) - pd.Timestamp(row["start"])).days
    return 350 <= days <= 380


def get_fundamentals(cache: DiskCache, ticker: str, user_agent: str | None) -> pd.DataFrame:
    cik = cik_for_ticker(cache, ticker, user_agent)
    facts = cache.json(
        "sec",
        f"companyfacts_{cik}",
        lambda: _get_json(FACTS_URL.format(cik=cik), user_agent),
        MAX_AGE,
    )
    return annual_facts(facts)
