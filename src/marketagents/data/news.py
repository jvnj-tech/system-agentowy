"""Newsy o spółce z Finnhub (endpoint company-news, darmowy klucz API)."""

from __future__ import annotations

import re
from datetime import date, timedelta

import pandas as pd
import requests

from marketagents.data.cache import DiskCache

URL = "https://finnhub.io/api/v1/company-news"
NEWS_COLUMNS = ["published", "source", "headline", "summary", "url"]
# Newsy z bieżącego dnia przybywają i mogą dotyczyć nagłych zdarzeń, więc lista jest
# pobierana ponownie po godzinie.
MAX_AGE = timedelta(hours=1)


def normalize_news(items: list[dict]) -> pd.DataFrame:
    if not items:
        return pd.DataFrame(columns=NEWS_COLUMNS)
    df = pd.DataFrame(items)
    df["published"] = pd.to_datetime(df["datetime"], unit="s", utc=True)
    df = df.drop_duplicates("url").sort_values("published", ascending=False)
    return df[NEWS_COLUMNS].reset_index(drop=True)


def get_news(
    cache: DiskCache, ticker: str, api_key: str | None, days: int = 30, until: date | None = None
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


# Najwyżej tyle wiadomości trafia do analizy (limit długości poleceń dla modeli).
NEWS_LIMIT = 15
# Waga aktualności maleje o połowę co tyle dni.
HALF_LIFE_DAYS = 14
# Zdarzenia rozpoznawane po słowach kluczowych w nagłówku i ich waga w ocenie ważności.
EVENTS = [
    (
        2.0,
        re.compile(
            r"\b(?:earnings|results|revenue|profit|guidance|outlook|forecast|quarter"
            r"|acqui\w*|merger|takeover|buyout|lawsuit|sues|sued|antitrust|probe"
            r"|investigation|settlement|ceo|chief executive|resign\w*|dividend|buyback"
            r"|repurchase|split|bankrupt\w*|recall)\b",
            re.IGNORECASE,
        ),
    ),
    (
        1.0,
        re.compile(
            r"\b(?:upgrade\w*|downgrade\w*|price target|rating|partnership|deal|contract"
            r"|launch\w*|unveil\w*|layoffs?|job cuts|approval|approved|regulator\w*|tariff\w*)\b",
            re.IGNORECASE,
        ),
    ),
]
# Nagłówki o podobieństwie słów co najmniej takim uznaje się za to samo zdarzenie.
SAME_EVENT = 0.35
STOPWORDS = {
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "is", "its", "of", "on", "or",
    "the", "to", "with", "after", "over", "says", "new",
}  # fmt: skip


def _published(item: dict) -> pd.Timestamp:
    return pd.Timestamp(item["published"])


def _words(item: dict) -> set[str]:
    words = re.findall(r"[a-z0-9]+", str(item.get("headline", "")).lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def _similar(a: set[str], b: set[str]) -> bool:
    """Podobieństwo Jaccarda zbiorów słów dwóch nagłówków."""
    if not a or not b:
        return False
    return len(a & b) / len(a | b) >= SAME_EVENT


def importance(item: dict, aliases: list[str], coverage: int) -> float:
    """Ważność wiadomości: rodzaj zdarzenia, spółka w nagłówku, liczba serwisów o tym samym."""
    headline = str(item.get("headline", ""))
    event = max((w for w, pattern in EVENTS if pattern.search(headline)), default=0.0)
    in_headline = 1.0 if mentions_company({"headline": headline}, aliases) else 0.0
    return event + in_headline + 0.5 * min(coverage, 2)


def select_news(
    items: list[dict], ticker: str, name: str | None, limit: int = NEWS_LIMIT
) -> list[dict]:
    """Najwyżej limit wiadomości o największej ocenie ważność × aktualność, bez powtórzeń.

    Ocena końcowa to (1 + ważność) · 0,5^(wiek / HALF_LIFE_DAYS), gdzie wiek liczony jest
    w dniach względem najnowszej wiadomości. Z kilku wiadomości o tym samym zdarzeniu
    zostaje ta z najwyższą oceną. Wynik jest posortowany od najnowszych.
    """
    if not items:
        return []
    aliases = company_aliases(ticker, name)
    words = [_words(item) for item in items]
    newest = max(_published(item) for item in items)
    scored = []
    for i, item in enumerate(items):
        coverage = sum(
            1
            for j, other in enumerate(items)
            if j != i and other.get("source") != item.get("source") and _similar(words[i], words[j])
        )
        age = (newest - _published(item)).total_seconds() / 86400
        score = (1 + importance(item, aliases, coverage)) * 0.5 ** (age / HALF_LIFE_DAYS)
        scored.append((score, i))
    chosen: list[int] = []
    for _, i in sorted(scored, key=lambda pair: (-pair[0], pair[1])):
        if any(_similar(words[i], words[j]) for j in chosen):
            continue
        chosen.append(i)
        if len(chosen) == limit:
            break
    return sorted((items[i] for i in chosen), key=_published, reverse=True)
