"""Dane fundamentalne z SEC EDGAR (API XBRL companyfacts).

SEC wymaga nagłówka User-Agent z nazwą i adresem e-mail (zmienna SEC_USER_AGENT).
"""

from __future__ import annotations

import pandas as pd
import requests

from marketagents.data.cache import DiskCache

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

# Nazwa wskaźnika -> pojęcia us-gaap w kolejności preferencji (spółki raportują różnie).
CONCEPTS: dict[str, list[str]] = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ],
    "net_income": ["NetIncomeLoss"],
    "operating_income": ["OperatingIncomeLoss"],
    "gross_profit": ["GrossProfit"],
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    "stockholders_equity": ["StockholdersEquity"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "eps_diluted": ["EarningsPerShareDiluted"],
}


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


def annual_facts(companyfacts: dict) -> pd.DataFrame:
    """Wartości roczne z raportów 10-K: wiersz = rok obrotowy, kolumna = wskaźnik.

    Bierze tylko okresy pełnoroczne (fp == FY), a przy kilku zgłoszeniach tego samego
    okresu zostawia najnowsze (korekty w późniejszych raportach).
    """
    gaap = companyfacts.get("facts", {}).get("us-gaap", {})
    columns: dict[str, pd.Series] = {}
    for name, candidates in CONCEPTS.items():
        for concept in candidates:
            units = gaap.get(concept, {}).get("units", {})
            rows = units.get("USD") or units.get("USD/shares") or []
            rows = [r for r in rows if r.get("form") == "10-K" and r.get("fp") == "FY"]
            if name not in {"total_assets", "total_liabilities", "stockholders_equity"}:
                # Pozycje przepływowe: tylko okresy ok. 12-miesięczne.
                rows = [r for r in rows if _is_full_year(r)]
            if not rows:
                continue
            df = pd.DataFrame(rows)
            df["end"] = pd.to_datetime(df["end"])
            df = df.sort_values("filed").drop_duplicates("end", keep="last")
            columns[name] = df.set_index("end")["val"].astype(float)
            break
    if not columns:
        return pd.DataFrame()
    out = pd.DataFrame(columns).sort_index()
    out.index.name = "period_end"
    return out


def _is_full_year(row: dict) -> bool:
    if "start" not in row:
        return False
    days = (pd.Timestamp(row["end"]) - pd.Timestamp(row["start"])).days
    return 350 <= days <= 380


def get_fundamentals(cache: DiskCache, ticker: str, user_agent: str | None) -> pd.DataFrame:
    cik = cik_for_ticker(cache, ticker, user_agent)
    facts = cache.json(
        "sec", f"companyfacts_{cik}", lambda: _get_json(FACTS_URL.format(cik=cik), user_agent)
    )
    return annual_facts(facts)
