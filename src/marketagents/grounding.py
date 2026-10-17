"""Kontrola treści bez liczb: zgodność zdań o wiadomościach ze źródłem i wnioski bez podstaw.

Reguły działają bez modelu językowego, więc są powtarzalne, ale sprawdzają tylko to,
co da się uchwycić w samym tekście:

1. Zdanie, które wskazuje wiadomość [n], może wymieniać tylko te nazwy własne (firmy,
   osoby, produkty, instytucje), które występują w nagłówku lub streszczeniu cytowanych
   wiadomości albo w nazwie analizowanej spółki. Nazwa spoza źródła oznacza treść
   dopisaną przez model. Reguła nie rozumie sensu zdania: prawdziwa nazwa użyta
   w zmyślonym kontekście przejdzie.
2. Zdanie nie może zawierać prognozy ani wniosku o przyszłym zachowaniu kursu
   („dalszy wzrost”, „przestrzeń do spadków”, „ryzyko korekty”), chyba że jest to
   opinia przypisana autorowi cytowanej wiadomości („według [2] …”).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

CITATION = re.compile(r"\[(\d{1,2})\]")
PLACEHOLDER = re.compile(r"\{\{\s*[A-Za-z0-9_]+\s*\}\}")
SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
# Słowo pisane wielką literą lub w całości wielkimi literami (np. „Microsoftem”, „LSEG”).
PROPER = re.compile(r"(?<![\w&])[A-ZĄĆĘŁŃÓŚŹŻ][\w&’'-]*")
# Nazwy, które nie wymagają potwierdzenia w źródle: nazwy krajów i instytucji pisane
# po polsku inaczej niż w angielskich wiadomościach oraz skróty ogólne.
COMMON = {
    "usa",
    "stany",
    "zjednoczone",
    "stanów",
    "zjednoczonych",
    "ue",
    "unia",
    "europa",
    "europejska",
    "europejskiej",
    "chiny",
    "chin",
    "chinach",
    "japonia",
    "japonii",
    "korea",
    "korei",
    "tajwan",
    "tajwanu",
    "francja",
    "francji",
    "francuskich",
    "francuskie",
    "niemcy",
    "niemiec",
    "wielka",
    "brytania",
    "brytanii",
    "brytyjskich",
    "brytyjskie",
    "indie",
    "indii",
    "fed",
    "rezerwa",
    "federalna",
    "federalnej",
    "cpi",
    "ai",
    "si",
    "ceo",
    "cfo",
    "etf",
    "ipo",
    "it",
    "pkb",
    "gdp",
    "nasdaq",
    "s&p",
    "dow",
    "wall",
    "street",
    "rsi",
    "macd",
    "sma",
    "ema",
    "eps",
    "roe",
    "c/z",
    "p/e",
    "q1",
    "q2",
    "q3",
    "q4",
    "ii",
    "iii",
    "iv",
    "usd",
    "eur",
    "mld",
    "mln",
}
# Opinia przypisana autorowi wiadomości jest dozwolona także wtedy, gdy dotyczy przyszłości.
ATTRIBUTION = re.compile(
    r"\b(?:według|zdaniem|ocen\w*|uważa\w*|twierdz\w*|przewiduj\w*|prognozuj\w*|"
    r"szacuj\w*|spodziewa\w*|oczekuj\w*|analityk\w*|analiz\w*|autor\w*|artykuł\w*)\b",
    re.IGNORECASE,
)
FORECAST = re.compile(
    r"\bdalsz\w*\s+(?:wzrost\w*|spad\w*|osłabieni\w*|umocnieni\w*|zwyżk\w*|zniżk\w*|"
    r"ruch\w*|ożywieni\w*|wzrostow\w*|spadkow\w*|kontynuacj\w*)"
    r"|\bprzestrze\w*\s+(?:do|dla)\s+(?:dalsz\w*\s+)?(?:wzrost\w*|spadk\w*|spadków|ruch\w*)"
    r"|\bryzyk\w*\s+(?:ewentualnej\s+)?korekt\w*"
    r"|\bw\s+najbliższ\w*\s+(?:okresie|czasie|dniach|tygodniach|miesiącach|sesjach)"
    r"|\bpotencja\w*\s+(?:do\s+)?(?:dalsz\w*\s+)?(?:wzrost\w*|odbici\w*|odwróceni\w*|spadk\w*)"
    r"|\b(?:zapowiada|zwiastuje)\w*"
    r"|\b(?:wzrośnie|wzrosną|spadnie|spadną|odbije|odbiją)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ContentIssue:
    sentence: str
    message: str


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE.split(text) if s.strip() and not s.lstrip().startswith("#")]


def _short(sentence: str) -> str:
    plain = re.sub(r"\s+", " ", sentence)
    return plain if len(plain) <= 90 else plain[:87] + "..."


def stem(word: str) -> str:
    """Początek słowa bez polskiej końcówki („Microsoftem” -> „microsof”)."""
    word = word.lower().strip("’'-")
    return word if len(word) <= 4 else word[: max(4, len(word) - 3)]


def proper_names(sentence: str) -> list[str]:
    """Nazwy własne w zdaniu (bez pierwszego słowa, odwołań i numerów źródeł)."""
    text = CITATION.sub(" ", PLACEHOLDER.sub(" ", sentence))
    names = []
    for i, match in enumerate(PROPER.finditer(text)):
        word = match.group(0)
        first = i == 0 and not text[: match.start()].strip()
        if first and not word.isupper():
            continue
        if word.lower() in COMMON or len(word) < 2:
            continue
        names.append(word)
    return names


def check_news_content(text: str, news: list[dict], allowed: list[str] = ()) -> list[ContentIssue]:
    """Nazwy własne w zdaniach z numerem źródła, których nie ma w cytowanych wiadomościach."""
    issues = []
    allowed_stems = {stem(w) for name in allowed for w in re.findall(r"[\w&]+", name)}
    for sentence in sentences(text):
        cited = [int(n) for n in CITATION.findall(sentence) if 1 <= int(n) <= len(news)]
        if not cited:
            continue
        source = " ".join(
            f"{news[n - 1].get('headline', '')} {news[n - 1].get('summary', '')} "
            f"{news[n - 1].get('source', '')}"
            for n in cited
        ).lower()
        missing = [
            w
            for w in proper_names(sentence)
            if stem(w) not in allowed_stems and stem(w) not in source
        ]
        if missing:
            numbers = "".join(f"[{n}]" for n in dict.fromkeys(cited))
            issues.append(
                ContentIssue(
                    sentence,
                    f"Zdanie „{_short(sentence)}” wymienia {', '.join(f'„{w}”' for w in missing)}, "
                    f"czego nie ma w cytowanych wiadomościach {numbers}; opisz tylko to, co "
                    "podaje źródło, albo usuń zdanie",
                )
            )
    return issues


LIMITATIONS = "## Ograniczenia analizy"
NEGATION = re.compile(r"\bnie\b", re.IGNORECASE)


def check_forecasts(text: str) -> list[ContentIssue]:
    """Prognozy i wnioski o przyszłym kursie, nieprzypisane autorowi wiadomości.

    Pomijana jest sekcja ograniczeń (opisuje, czego raport nie uwzględnia, np. „zmian
    w najbliższym okresie”) i zdania, w których sformułowanie stoi po przeczeniu.
    """
    issues = []
    start = text.find(LIMITATIONS)
    if start >= 0:
        end = text.find("\n## ", start + len(LIMITATIONS))
        text = text[:start] + (text[end:] if end >= 0 else "")
    for sentence in sentences(text):
        match = FORECAST.search(sentence)
        if not match or NEGATION.search(sentence[: match.start()]):
            continue
        if CITATION.search(sentence) and ATTRIBUTION.search(sentence):
            continue
        issues.append(
            ContentIssue(
                sentence,
                f"Zdanie „{_short(sentence)}” zawiera prognozę lub wniosek bez podstaw w danych "
                f"(„{match.group(0)}”); opisz stan wynikający z karty faktów albo usuń wniosek",
            )
        )
    return issues
