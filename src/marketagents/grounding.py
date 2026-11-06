"""Kontrola treści bez liczb: zgodność zdań o wiadomościach ze źródłem i wnioski bez podstaw.

Reguły działają bez modelu językowego, więc są powtarzalne, ale sprawdzają tylko to,
co da się uchwycić w samym tekście:

1. Zdanie, które wskazuje wiadomość [n], może wymieniać tylko te nazwy własne (firmy,
   osoby, produkty, instytucje), które występują w nagłówku lub streszczeniu cytowanych
   wiadomości albo w nazwie analizowanej spółki. Nazwa spoza źródła oznacza treść
   dopisaną przez model. Nazwa jest porównywana z całymi słowami źródła z dokładnością
   do polskiej końcówki. Reguła nie rozumie sensu zdania: prawdziwa nazwa użyta
   w zmyślonym kontekście przejdzie.
2. Zdanie nie może zawierać prognozy ani wniosku o przyszłym zachowaniu kursu
   („dalszy wzrost”, „przestrzeń do spadków”, „ryzyko korekty”), chyba że jest to
   opinia przypisana autorowi cytowanej wiadomości („według [2] …”).
3. Twierdzenie o rekordzie („rekordowe przychody”, „najwyższy w historii”) musi wskazywać
   wiadomość, bo karta faktów obejmuje tylko 52 tygodnie notowań i dwa lata obrotowe.
4. Liczba, krotność ani okres nie mogą być zapisane słowami („o cztery procent”,
   „dwukrotnie”, „od pięciu sesji”), z wyjątkiem okresów z karty faktów.
"""

from __future__ import annotations

import os
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
    r"szacuj\w*|spodziewa\w*|oczekuj\w*|analityk\w*|autor\w*|artykuł\w*)\b",
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


# Liczba zapisana słowami przy jednostce („o cztery procent”, „pięćdziesiąt milionów dolarów”):
# wartości z wiadomości nie da się sprawdzić, więc ma być opisana bez liczby.
NUMBER_WORD = (
    r"(?:jed(?:en|na|no|nego)|dw(?:a|ie|óch|u)|trz(?:y|ech)|czter(?:y|ech)|pię(?:ć|ciu)|"
    r"sze(?:ść|ściu)|siedem|siedmiu|osiem|ośmiu|dziewię(?:ć|ciu)|dziesię(?:ć|ciu)|"
    r"\w*naście|\w*naściu|\w*dzieści|\w*dziesiąt|\w*dziesięciu|sto|stu|dwieście|trzysta|czterysta|\w*set|półtora|pół|"
    r"tysiąc\w*|tysięcy|milion\w*|miliard\w*|bilion\w*)"
)
SPELLED_NUMBER = re.compile(
    rf"\b{NUMBER_WORD}(?:\s+{NUMBER_WORD})*\s+(?:procent\w*|punkt\w*\s+procent\w*|"
    r"dolar\w*|euro|akcji|miejsc\w*|tysięcy|tysiąc\w*|milion\w*|miliard\w*|bilion\w*|razy)\b",
    re.IGNORECASE,
)


# Okres zapisany słowami („od pięciu sesji”) musi być okresem z karty faktów: 3 miesiące,
# 12 miesięcy, 52 tygodnie albo dwa lata obrotowe (zmiana przychodów rok do roku i rok wcześniej).
SPELLED_PERIOD = re.compile(
    rf"\b{NUMBER_WORD}(?:\s+{NUMBER_WORD})*\s+(?:kolejn\w*\s+|ostatni\w*\s+)?"
    r"(?:sesj\w*|dni|dzie\w*|tygod\w*|miesi\w*|lat\b|lata\b|latach|kwarta\w*)",
    re.IGNORECASE,
)
CARD_PERIOD = re.compile(
    r"^(?:trzy|trzech)\s+(?:\w+\s+)?miesi|^dwunast\w*\s+(?:\w+\s+)?miesi"
    r"|^pięćdziesi\w*\s+dw\w*\s+(?:\w+\s+)?tygod|^dw\w*\s+(?:\w+\s+)?la",
    re.IGNORECASE,
)
# Krotność i ułamek zapisane słowem („dwukrotnie”, „podwoił”, „o połowę”) też są liczbą.
MULTIPLE = re.compile(
    r"\b(?:dwu|trzy|cztero|pięcio|sześcio|siedmio|ośmio|dziewięcio|dziesięcio|stu)krotn\w*"
    r"|\bpo(?:dw|tr)o(?:i|j)\w*|\bo\s+połowę\b",
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


def _base(word: str) -> str:
    """Słowo bez końcówki dopisanej po apostrofie („Apple’a” -> „apple”)."""
    return re.split(r"[’']", word.lower())[0].strip("-")


def same_word(name: str, word: str) -> bool:
    """Czy nazwa z raportu jest odmianą słowa ze źródła („Intela” i „Intel”).

    Słowa muszą mieć wspólny początek, a różnić się mogą tylko końcówką: słowo ze źródła
    najwyżej dwiema ostatnimi literami, nazwa w raporcie najwyżej trzema. Dzięki temu
    „Nvidii” pasuje do „Nvidia”, a „Intel” nie pasuje do „interest”.
    """
    name, word = _base(name), _base(word)
    if name == word:
        return True
    common = len(os.path.commonprefix([name, word]))
    return common >= 3 and common >= len(word) - 2 and common >= len(name) - 3


def source_words(text: str) -> set[str]:
    """Słowa źródła, także części nazw pisanych z łącznikiem („Coca-Cola”, „Nvidia-backed”)."""
    words = set(re.findall(r"[\w&’'-]+", text.lower()))
    return words | {part for w in words for part in re.split(r"[-’']", w) if part}


def known(name: str, words: set[str]) -> bool:
    if any(same_word(name, w) for w in words):
        return True
    parts = [p for p in name.split("-") if p]
    return len(parts) > 1 and all(any(same_word(p, w) for w in words) for p in parts)


def news_names(news: list[dict]) -> set[str]:
    """Słowa pisane w wiadomościach zawsze wielką literą, czyli nazwy („Nvidia”, „Reuters”)."""
    text = " ".join(
        f"{item.get('headline', '')} {item.get('summary', '')} {item.get('source', '')}"
        for item in news
    )
    words = re.findall(r"[\w&’'-]+", text)
    lower = {w.lower() for w in words if w[0].islower()}
    return {w.lower() for w in words if w[0].isupper() and len(w) > 2} - lower


def proper_names(sentence: str, first_names: set[str] = frozenset()) -> list[str]:
    """Nazwy własne w zdaniu (bez odwołań i numerów źródeł).

    Pierwsze słowo zdania jest pisane wielką literą także wtedy, gdy nie jest nazwą
    („Spółka”, „Według”), dlatego jest sprawdzane tylko wtedy, gdy jest skrótem
    albo nazwą występującą w którejś wiadomości z listy (first_names).
    """
    text = CITATION.sub(" ", PLACEHOLDER.sub(" ", sentence))
    names = []
    for i, match in enumerate(PROPER.finditer(text)):
        word = match.group(0)
        first = i == 0 and not text[: match.start()].strip()
        if first and not word.isupper() and not known(word, first_names):
            continue
        if word.lower() in COMMON or len(word) < 2:
            continue
        names.append(word)
    return names


def check_news_content(text: str, news: list[dict], allowed: list[str] = ()) -> list[ContentIssue]:
    """Nazwy własne w zdaniach z numerem źródła, których nie ma w cytowanych wiadomościach."""
    issues = []
    allowed_words = source_words(" ".join(allowed))
    first_names = news_names(news)
    for sentence in sentences(text):
        cited = [int(n) for n in CITATION.findall(sentence) if 1 <= int(n) <= len(news)]
        if not cited:
            continue
        source = source_words(
            " ".join(
                f"{news[n - 1].get('headline', '')} {news[n - 1].get('summary', '')} "
                f"{news[n - 1].get('source', '')}"
                for n in cited
            )
        )
        missing = [
            w
            for w in proper_names(sentence, first_names)
            if not known(w, allowed_words) and not known(w, source)
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
# Przeczenie przy czasowniku opisującym zakres analizy („nie uwzględnia zmian w najbliższym
# okresie”, „nie przesądza, czy kurs wzrośnie”) albo czasownik „pomija”; zaprzeczona
# prognoza („kurs nie spadnie”) nadal jest prognozą.
NEGATION = re.compile(
    r"\bnie\s+(?:uwzględni\w*|obejmuj\w*|opisuj\w*|analizuj\w*|zawier\w*|bierze|biorą|dotycz\w*"
    r"|przewiduj\w*|prognozuj\w*|przesądz\w*|ocenia\w*)"
    r"|\bpomij\w*",
    re.IGNORECASE,
)
# Sekcja ograniczeń z natury wymienia to, czego raport nie obejmuje („ryzyko korekty”,
# „zmiany w najbliższym okresie”), więc sprawdzane są w niej tylko prognozy wprost.
FUTURE = re.compile(
    r"\b(?:zapowiada|zwiastuje)\w*|\b(?:wzrośnie|wzrosną|spadnie|spadną|odbije|odbiją)\b",
    re.IGNORECASE,
)


def check_forecasts(text: str) -> list[ContentIssue]:
    """Prognozy i wnioski o przyszłym kursie, nieprzypisane autorowi wiadomości.

    W sekcji ograniczeń sprawdzane są tylko prognozy wprost („wzrośnie”, „zapowiada”),
    bo sekcja opisuje, czego raport nie uwzględnia (np. „zmian w najbliższym okresie”).
    Pomijane są zdania, w których sformułowanie stoi po przeczeniu czasownika opisującego
    zakres analizy (np. „nie uwzględnia”).
    """
    issues = []
    limits = ""
    start = text.find(LIMITATIONS)
    if start >= 0:
        end = text.find("\n## ", start + len(LIMITATIONS))
        limits = text[start:end] if end >= 0 else text[start:]
        text = text[:start] + (text[end:] if end >= 0 else "")
    found = [(s, FORECAST.search(s)) for s in sentences(text)]
    found += [(s, FUTURE.search(s)) for s in sentences(limits)]
    for sentence, match in found:
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


# Rekord albo poziom „w historii”: karta faktów obejmuje tylko 52 tygodnie notowań
# i dwa lata obrotowe, więc takie twierdzenie może pochodzić wyłącznie z wiadomości.
RECORD = re.compile(
    r"\brekord\w*|\bhistoryczn\w*\s+(?:maksim|minim|szczyt|poziom)\w*|\bw\s+historii\b"
    r"|\bnajwyższ\w*\s+od\s+lat\b|\bnajniższ\w*\s+od\s+lat\b",
    re.IGNORECASE,
)


def check_records(text: str) -> list[ContentIssue]:
    """Twierdzenia o rekordach bez numeru wiadomości, z której pochodzą."""
    issues = []
    for sentence in sentences(text):
        match = RECORD.search(sentence)
        if match and not CITATION.search(sentence):
            issues.append(
                ContentIssue(
                    sentence,
                    f"Zdanie „{_short(sentence)}” podaje rekord („{match.group(0)}”), którego "
                    "nie da się sprawdzić w karcie faktów; usuń to twierdzenie albo wskaż "
                    "wiadomość, która je podaje",
                )
            )
    return issues


def check_spelled_numbers(text: str) -> list[ContentIssue]:
    """Liczby zapisane słowami, którymi model omija zakaz wpisywania liczb."""
    issues = []
    for sentence in sentences(text):
        periods = [m for m in SPELLED_PERIOD.finditer(sentence) if not CARD_PERIOD.match(m[0])]
        match = SPELLED_NUMBER.search(sentence) or MULTIPLE.search(sentence)
        match = match or (periods[0] if periods else None)
        if match:
            issues.append(
                ContentIssue(
                    sentence,
                    f"Zdanie „{_short(sentence)}” podaje liczbę słowami („{match.group(0)}”), "
                    "której nie ma w karcie faktów; opisz zjawisko bez liczby albo użyj "
                    "odwołania do karty",
                )
            )
    return issues
