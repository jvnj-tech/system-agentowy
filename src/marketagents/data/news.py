"""Newsy o spółce z Finnhub (endpoint company-news, darmowy klucz API)."""

from __future__ import annotations

import re
from datetime import date, timedelta

import pandas as pd
import requests

from marketagents.data.cache import DiskCache

URL = "https://finnhub.io/api/v1/company-news"
NEWS_COLUMNS = ["published", "source", "headline", "summary", "url"]
# Newsy z bieżącego dnia przybywają, więc lista jest pobierana ponownie po kilku godzinach.
MAX_AGE = timedelta(hours=6)


def normalize_news(items: list[dict]) -> pd.DataFrame:
    if not items:
        return pd.DataFrame(columns=NEWS_COLUMNS)
    df = pd.DataFrame(items)
    df["published"] = pd.to_datetime(df["datetime"], unit="s", utc=True)
    df = df.drop_duplicates("url").sort_values("published", ascending=False)
    return df[NEWS_COLUMNS].reset_index(drop=True)


def get_news(
    cache: DiskCache, ticker: str, api_key: str | None, days: int = 7, until: date | None = None
) -> pd.DataFrame:
    until = until or date.today()
    since = until - timedelta(days=days)

    def fetch() -> list[dict]:
        if not api_key:
            raise RuntimeError("Ustaw FINNHUB_API_KEY (darmowy klucz z finnhub.io)")
        params = {
            "symbol": ticker,
            "from": since.isoformat(),
            "to": until.isoformat(),
            "token": api_key,
        }
        resp = requests.get(URL, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    return normalize_news(cache.json("news", f"{ticker}_{since}_{until}", fetch, MAX_AGE))


# Człony nazw spółek pomijane przy dopasowaniu („Apple Inc.” -> „Apple”).
LEGAL_SUFFIXES = re.compile(
    r"\b(?:inc|incorporated|corp|corporation|co|company|plc|ltd|limited|holdings?|group"
    r"|class [abc]|the)\b\.?,?",
    re.IGNORECASE,
)


def company_aliases(ticker: str, name: str | None) -> list[str]:
    """Słowa, po których rozpoznaje się wiadomość o spółce: symbol i początek nazwy."""
    aliases = [ticker.upper()]
    if name:
        core = re.sub(r"[,.]", " ", LEGAL_SUFFIXES.sub(" ", name)).split()
        if core:
            aliases.append(core[0])
            if len(core) > 1:
                aliases.append(" ".join(core[:2]))
    return aliases


def mentions_company(item: dict, aliases: list[str]) -> bool:
    """Czy nagłówek lub streszczenie wymienia spółkę (symbol: wielkimi literami)."""
    text = f"{item.get('headline', '')} {item.get('summary', '')}"
    ticker, *names = aliases
    if re.search(rf"(?<![A-Za-z]){re.escape(ticker)}(?![A-Za-z])", text):
        return True
    return any(re.search(rf"(?<!\w){re.escape(n)}(?!\w)", text, re.IGNORECASE) for n in names)


def relevant_news(items: list[dict], ticker: str, name: str | None) -> list[dict]:
    """Tylko wiadomości, które wymieniają spółkę.

    Finnhub zwraca dla symbolu także artykuły ogólnorynkowe i o innych spółkach; bez tego
    filtra analityk wiadomości opisywał je jako dotyczące analizowanej spółki.
    """
    aliases = company_aliases(ticker, name)
    return [item for item in items if mentions_company(item, aliases)]
