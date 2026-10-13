"""Lista spółek, dla których system generuje raporty."""

from __future__ import annotations

import csv
from importlib import resources


def nasdaq100() -> list[dict[str, str]]:
    """Spółki z pliku nasdaq100.csv (ticker, nazwa).

    Skład indeksu zmienia się co roku, dlatego plik zawiera datę stanu w nagłówku
    i powinien być uzupełniany z oficjalnej listy publikowanej przez Nasdaq.
    """
    text = resources.files("marketagents.data").joinpath("nasdaq100.csv").read_text("utf-8")
    lines = [line for line in text.splitlines() if line and not line.startswith("#")]
    return list(csv.DictReader(lines))


def company_name(ticker: str) -> str | None:
    """Nazwa spółki z listy NASDAQ-100 albo None dla symbolu spoza listy."""
    return next((c["name"] for c in nasdaq100() if c["ticker"] == ticker.upper()), None)
