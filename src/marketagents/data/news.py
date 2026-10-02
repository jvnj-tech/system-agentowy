"""Newsy o spółce z Finnhub (endpoint company-news, darmowy klucz API)."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import requests

from marketagents.data.cache import DiskCache

URL = "https://finnhub.io/api/v1/company-news"
NEWS_COLUMNS = ["published", "source", "headline", "summary", "url"]


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

    return normalize_news(cache.json("news", f"{ticker}_{since}_{until}", fetch))
