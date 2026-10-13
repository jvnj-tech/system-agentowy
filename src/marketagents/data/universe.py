"""Lista spółek z indeksu NASDAQ-100.

Skład indeksu zmienia się (coroczna rewizja w grudniu i zmiany w ciągu roku), dlatego
lista jest pobierana z serwisu Nasdaq i odświeżana raz na dobę. Plik nasdaq100.csv
z datą stanu w nagłówku jest kopią zapasową na wypadek braku dostępu do serwisu.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from importlib import resources

import requests

from marketagents.data.cache import DiskCache

URL = "https://api.nasdaq.com/api/quote/list-type/nasdaq100"
MAX_AGE = timedelta(hours=12)
# Serwis odrzuca zapytania bez nagłówków przeglądarki.
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
# Nazwa spółki bez opisu papieru wartościowego („Apple Inc. Common Stock” -> „Apple Inc.”).
SECURITY_SUFFIX = re.compile(
    r"\s+(?:Class [A-C]\s+)?(?:Common Stock|Common Shares|Ordinary Shares|Capital Stock"
    r"|American Depositary Shares|Depositary Shares|New York Registry Shares"
    r"|Subordinate Voting Shares)\b.*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Universe:
    companies: list[dict[str, str]]
    as_of: str
    source: str


def clean_name(name: str) -> str:
    return SECURITY_SUFFIX.sub("", name).strip()


def parse_nasdaq(payload: dict) -> Universe:
    data = payload["data"]
    rows = data["data"]["rows"]
    companies = sorted(
        ({"ticker": r["symbol"].upper(), "name": clean_name(r["companyName"])} for r in rows),
        key=lambda c: c["ticker"],
    )
    if len(companies) < 90:
        raise ValueError(f"Niepełna lista NASDAQ-100: {len(companies)} spółek")
    return Universe(companies, _iso_date(data.get("date", "")), "api.nasdaq.com")


def _iso_date(text: str) -> str:
    """„Oct 13, 2026 12:25 PM” -> „2026-10-13 12:25 czasu nowojorskiego”."""
    for pattern, out in (("%b %d, %Y %I:%M %p", "%Y-%m-%d %H:%M"), ("%b %d, %Y", "%Y-%m-%d")):
        try:
            parsed = datetime.strptime(text.strip(), pattern)
        except ValueError:
            continue
        suffix = " czasu nowojorskiego" if "%H" in out else ""
        return parsed.strftime(out) + suffix
    return text


def _download() -> dict:
    resp = requests.get(URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


def snapshot() -> Universe:
    """Lista z pliku nasdaq100.csv dołączonego do programu (stan z nagłówka)."""
    text = resources.files("marketagents.data").joinpath("nasdaq100.csv").read_text("utf-8")
    header = next((line for line in text.splitlines() if line.startswith("#")), "")
    as_of = header.removeprefix("# Stan na:").strip()
    lines = [line for line in text.splitlines() if line and not line.startswith("#")]
    return Universe(list(csv.DictReader(lines)), as_of, "plik nasdaq100.csv")


def current(cache: DiskCache | None = None) -> Universe:
    """Skład indeksu z bieżącego dnia; bez dostępu do serwisu kopia z pliku."""
    if cache is None:
        return snapshot()
    try:
        return parse_nasdaq(cache.json("universe", "nasdaq100", _download, MAX_AGE))
    except Exception:  # noqa: BLE001
        return snapshot()


def nasdaq100(cache: DiskCache | None = None) -> list[dict[str, str]]:
    """Spółki (ticker, nazwa) z bieżącego składu indeksu."""
    return current(cache).companies


def company_name(ticker: str, cache: DiskCache | None = None) -> str | None:
    """Nazwa spółki z listy NASDAQ-100 albo None dla symbolu spoza listy."""
    return next((c["name"] for c in nasdaq100(cache) if c["ticker"] == ticker.upper()), None)
