"""Wskaźniki makroekonomiczne z FRED (Federal Reserve Bank of St. Louis)."""

from __future__ import annotations

import pandas as pd
import requests

from marketagents.data.cache import DiskCache

URL = "https://api.stlouisfed.org/fred/series/observations"

# Identyfikatory serii FRED używane w raporcie.
SERIES: dict[str, str] = {
    "fed_funds_rate": "FEDFUNDS",
    "cpi": "CPIAUCSL",
    "treasury_10y": "DGS10",
    "unemployment_rate": "UNRATE",
}


def normalize_observations(payload: dict) -> pd.Series:
    """FRED oznacza brak wartości kropką; takie obserwacje są pomijane."""
    obs = payload.get("observations", [])
    rows = [(o["date"], float(o["value"])) for o in obs if o.get("value") not in (".", "", None)]
    s = pd.Series(dict(rows), dtype=float)
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def get_series(
    cache: DiskCache, series_id: str, api_key: str | None, start: str = "2015-01-01"
) -> pd.Series:
    def fetch() -> dict:
        if not api_key:
            raise RuntimeError("Ustaw FRED_API_KEY (darmowy klucz z fred.stlouisfed.org)")
        params = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": start,
        }
        resp = requests.get(URL, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    return normalize_observations(cache.json("macro", f"{series_id}_{start}", fetch))


def get_macro(cache: DiskCache, api_key: str | None, start: str = "2015-01-01") -> pd.DataFrame:
    """Wszystkie serie z SERIES w jednej ramce (różne częstotliwości, więc z lukami)."""
    return pd.DataFrame(
        {name: get_series(cache, sid, api_key, start) for name, sid in SERIES.items()}
    )
