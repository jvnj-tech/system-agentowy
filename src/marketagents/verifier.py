"""Weryfikacja liczb w raporcie względem karty faktów.

Działa bez modelu językowego, więc jej wynik jest powtarzalny. Sprawdza, czy:
1. każde odwołanie {{identyfikator}} wskazuje istniejący fakt,
2. poza odwołaniami w tekście nie ma liczb, których nie ma w karcie faktów
   (z dokładnością do zaokrąglenia), z wyjątkiem dozwolonych parametrów wskaźników.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html import escape

from marketagents.facts import FactSheet

PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z0-9_]+)\s*\}\}")
# Model czasem otacza odwołanie znakami ` jak kod; przy podstawianiu są one usuwane.
PLACEHOLDER_IN_CODE = re.compile(r"`\{\{\s*([A-Za-z0-9_]+)\s*\}\}`")
# Liczba w zapisie polskim lub angielskim, z opcjonalnym separatorem tysięcy (spacja).
# Pomijane są liczby w złożeniach słownych, np. „10-letnie obligacje”.
NUMBER = re.compile(
    r"(?<![\w.,])[-−+]?\d{1,3}(?:[  ]\d{3})+(?:[.,]\d+)?(?![\w])"
    r"|(?<![\w.,])[-−+]?\d+(?:[.,]\d+)?(?![\w]|-[^\W\d])"
)
ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
LIST_MARKER = re.compile(r"^\s*\d+[.)]\s", re.MULTILINE)

# Okresy wskaźników, które mogą wystąpić w tekście jako nazwy, np. „RSI(14)”, „SMA 200”.
DEFAULT_ALLOWED = frozenset({2, 9, 12, 14, 20, 26, 50, 52, 200})
# Okna średnich ustawiane przez użytkownika są odczytywane z identyfikatorów faktów (sma_30).
PERIOD_IN_ID = re.compile(r"^(?:sma|ema|dist_sma)_(\d+)$")


def indicator_periods(facts: FactSheet) -> frozenset[int]:
    return frozenset(int(m.group(1)) for f in facts if (m := PERIOD_IN_ID.match(f.id)))


@dataclass
class Verification:
    rendered: str
    unknown_ids: list[str] = field(default_factory=list)
    unverified_numbers: list[str] = field(default_factory=list)
    wrong_years: list[str] = field(default_factory=list)
    checked_numbers: int = 0

    @property
    def ok(self) -> bool:
        return not self.unknown_ids and not self.unverified_numbers and not self.wrong_years

    @property
    def to_mark(self) -> list[str]:
        """Wartości, które trzeba oznaczyć w tekście, jeśli nie zostaną poprawione."""
        return self.unverified_numbers + self.wrong_years

    def issues(self) -> list[str]:
        out = [f"Nieznany identyfikator faktu: {{{{{i}}}}}" for i in self.unknown_ids]
        out += [
            f"Liczba „{n}” nie pochodzi z karty faktów; zastąp ją odwołaniem {{{{identyfikator}}}}"
            for n in self.unverified_numbers
        ]
        out += [
            f"Rok {y} nie odpowiada żadnej dacie w karcie faktów; "
            "sprawdź okres, którego dotyczy zdanie"
            for y in self.wrong_years
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


# Rok w tekście, np. „w 2025 r.” (bez dat ISO, które są usuwane wcześniej).
YEAR = re.compile(r"(?<![\w.,-])(?:19|20)\d{2}(?![\w,.]?\d)")


def fact_years(facts: FactSheet) -> frozenset[int]:
    """Lata, o których może mówić raport: z dat w karcie faktów oraz rok wcześniejszy
    (zmiany rok do roku i notowania z 52 tygodni obejmują poprzedni rok)."""
    years: set[int] = set()
    for fact in facts:
        if isinstance(fact.value, str) and ISO_DATE.fullmatch(fact.value):
            year = int(fact.value[:4])
            years |= {year, year - 1}
    return frozenset(years)


def verify(text: str, facts: FactSheet, allowed: frozenset[int] = DEFAULT_ALLOWED) -> Verification:
    allowed = allowed | indicator_periods(facts)
    unknown = sorted({m.group(1) for m in PLACEHOLDER.finditer(text) if m.group(1) not in facts})

    stripped = LIST_MARKER.sub(" ", ISO_DATE.sub(" ", PLACEHOLDER.sub(" ", text)))
    years = fact_years(facts)
    wrong_years = [m.group(0) for m in YEAR.finditer(stripped) if int(m.group(0)) not in years]
    stripped = YEAR.sub(" ", stripped)
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
        wrong_years=wrong_years,
        checked_numbers=checked,
    )


# Jednostka dopisana przez model zaraz po odwołaniu, które już ją zawiera („{{close}} USD”).
TRAILING_UNIT = re.compile(r"\{\{\s*([A-Za-z0-9_]+)\s*\}\}(\s*(?:USD|%|mld USD))?(?![\w])")


def render(text: str, facts: FactSheet, html: bool = False) -> str:
    """Podstawia wartości w miejsce odwołań. W wersji HTML liczba ma podpowiedź ze źródłem."""

    def substitute(match: re.Match) -> str:
        fact_id, trailing = match.group(1), match.group(2) or ""
        if fact_id not in facts:
            return f"[brak danych: {fact_id}]{trailing}"
        fact = facts[fact_id]
        if trailing and trailing.strip() == fact.unit:
            trailing = ""
        value = fact.formatted()
        if html:
            hint = escape(f"{fact.label}: {fact.source}" if fact.source else fact.label)
            value = f'<span class="fact" title="{hint}">{escape(value)}</span>'
        return value + trailing

    return TRAILING_UNIT.sub(substitute, PLACEHOLDER_IN_CODE.sub(r"{{\1}}", text))


HTML_TAG = re.compile(r"(<[^>]*>)")


def mark_unverified(rendered: str, numbers: list[str]) -> str:
    """Oznacza liczby, których nie udało się zweryfikować (z pominięciem znaczników HTML)."""
    parts = HTML_TAG.split(rendered)
    for raw in sorted(set(numbers), key=len, reverse=True):
        # Bez dopasowań wewnątrz dat w rodzaju 2026-10-02.
        pattern = re.compile(rf"(?<![\w.,])(?<!\d-){re.escape(raw)}(?![\w]|-\d)")
        parts = [
            p if HTML_TAG.fullmatch(p) else pattern.sub(f"{raw} [niezweryfikowane]", p)
            for p in parts
        ]
    return "".join(parts)
