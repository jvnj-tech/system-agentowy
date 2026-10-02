"""Weryfikacja liczb w raporcie względem karty faktów.

Działa bez modelu językowego, więc jej wynik jest powtarzalny. Sprawdza, czy:
1. każde odwołanie {{identyfikator}} wskazuje istniejący fakt,
2. poza odwołaniami w tekście nie ma liczb, których nie ma w karcie faktów
   (z dokładnością do zaokrąglenia), z wyjątkiem dozwolonych parametrów wskaźników.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from marketagents.facts import FactSheet

PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z0-9_]+)\s*\}\}")
# Liczba w zapisie polskim lub angielskim, z opcjonalnym separatorem tysięcy (spacja).
NUMBER = re.compile(
    r"(?<![\w.,])[-−+]?\d{1,3}(?:[  ]\d{3})+(?:[.,]\d+)?(?![\w])"
    r"|(?<![\w.,])[-−+]?\d+(?:[.,]\d+)?(?![\w])"
)
ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
LIST_MARKER = re.compile(r"^\s*\d+[.)]\s", re.MULTILINE)

# Okresy wskaźników, które mogą wystąpić w tekście jako nazwy, np. „RSI(14)”, „SMA 200”.
DEFAULT_ALLOWED = frozenset({2, 9, 12, 14, 20, 26, 50, 52, 200})


@dataclass
class Verification:
    rendered: str
    unknown_ids: list[str] = field(default_factory=list)
    unverified_numbers: list[str] = field(default_factory=list)
    checked_numbers: int = 0

    @property
    def ok(self) -> bool:
        return not self.unknown_ids and not self.unverified_numbers

    def issues(self) -> list[str]:
        out = [f"Nieznany identyfikator faktu: {{{{{i}}}}}" for i in self.unknown_ids]
        out += [
            f"Liczba „{n}” nie pochodzi z karty faktów; zastąp ją odwołaniem {{{{identyfikator}}}}"
            for n in self.unverified_numbers
        ]
        return out


def parse_number(text: str) -> float:
    cleaned = text.replace("−", "-").replace(" ", "").replace(" ", "")
    return float(cleaned.replace(",", "."))


def _matches_fact(value: float, facts: FactSheet) -> bool:
    for fact in facts:
        if isinstance(fact.value, str):
            continue
        tolerance = 0.5 * 10 ** (-fact.decimals) + 1e-9
        # Spadek bywa opisany słownie („spadek o 3,2%”), więc liczy się też wartość bezwzględna.
        if abs(value - fact.value) <= tolerance or abs(value - abs(fact.value)) <= tolerance:
            return True
        # Liczba zaokrąglona przez autora do mniejszej liczby miejsc po przecinku.
        if float(value).is_integer() and abs(abs(value) - abs(fact.value)) < 0.5:
            return True
    return False


def verify(text: str, facts: FactSheet, allowed: frozenset[int] = DEFAULT_ALLOWED) -> Verification:
    unknown = sorted({m.group(1) for m in PLACEHOLDER.finditer(text) if m.group(1) not in facts})

    stripped = LIST_MARKER.sub(" ", ISO_DATE.sub(" ", PLACEHOLDER.sub(" ", text)))
    unverified: list[str] = []
    checked = 0
    for match in NUMBER.finditer(stripped):
        raw = match.group(0)
        value = parse_number(raw)
        checked += 1
        if value.is_integer() and int(value) in allowed:
            continue
        if not _matches_fact(value, facts):
            unverified.append(raw)

    return Verification(
        rendered=render(text, facts),
        unknown_ids=unknown,
        unverified_numbers=unverified,
        checked_numbers=checked,
    )


def render(text: str, facts: FactSheet) -> str:
    def substitute(match: re.Match) -> str:
        fact_id = match.group(1)
        return facts[fact_id].formatted() if fact_id in facts else f"[brak danych: {fact_id}]"

    return PLACEHOLDER.sub(substitute, text)


def mark_unverified(rendered: str, numbers: list[str]) -> str:
    """Oznacza w tekście liczby, których nie udało się zweryfikować."""
    for raw in sorted(set(numbers), key=len, reverse=True):
        rendered = re.sub(
            rf"(?<![\w.,]){re.escape(raw)}(?![\w])", f"{raw} [niezweryfikowane]", rendered
        )
    return rendered
