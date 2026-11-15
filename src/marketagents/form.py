"""Reguły formy raportu: błędy, które nie dotyczą zgodności liczby z kartą faktów,
ale w przeglądzie raportów powtarzały się mimo przejścia przez weryfikator.

Każda reguła odpowiada rodzajowi błędu zaobserwowanemu w raportach zbioru projektowego
(spółki, na których dobierano reguły) i rozpoznawalnemu po formie tekstu:
- znak minus przy słowie „spadek” („spadek o -0,21 p.p.”),
- numer wiadomości przy wartości z karty faktów („wzrost o {{change_1y}} [5]”),
- opis wewnętrznych etapów systemu w treści raportu („karta faktów”, „analiza cząstkowa”),
- odnośnik, który nie jest numerem wiadomości („[ADBE]”, „[cite: 6]”),
- etykieta lub ocena z karty wklejona bez odmiany („odnotowano zmiana kursu”,
  „co odpowiada strefa neutralna”),
- kropka dziesiętna zamiast przecinka („3.4%”),
- ogólnik w miejscu liczby z wiadomości („płynność opisana w raporcie źródłowym”),
- wiadomość zestawiona z kartą jako „rozbieżność”, choć dotyczy innego dnia lub okresu,
- angielskie słowa i skróty („facility”, „6,12B USD”).

Treści zdania bez liczb (np. czy artykuł rzeczywiście to podaje) reguły nie oceniają.
"""

from __future__ import annotations

import re

from marketagents.facts import FactSheet
from marketagents.grounding import ContentIssue, sentences

PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z0-9_]+)\s*\}\}")

DECREASE = re.compile(
    r"\b(?:spad\w*|spadł\w*|zmniejsz\w*|obniż\w*|zniżk\w*|stracił\w*|zmalał\w*)\b"
    r"[^.,;{}\[\]]{0,40}?\bo\s+\{\{\s*([A-Za-z0-9_]+)\s*\}\}",
    re.IGNORECASE,
)
# Wartość z karty, po której (ewentualnie po nawiasie „(wzrost)” lub dopowiedzeniu
# „, co oznacza wzrost”) stoją numery wiadomości.
CARD_THEN_CITATION = re.compile(
    r"\{\{\s*([A-Za-z0-9_]+)\s*\}\}(?:\s*\([^)]*\))?"
    r"(?:,?\s+co\s+(?:[^\W\d]+\s+){0,3}?[^\W\d]+)?[\s,]*((?:\[\d{1,2}\][\s,]*)+)"
)
# Identyfikatory, które nazywają spółkę, a nie podają wartości.
NAME_IDS = frozenset({"ticker", "company"})
INTERNAL = re.compile(
    r"\bkart\w*\s+fakt\w*|\banaliz\w*\s+cząstkow\w*|\b(?:dan|materiał)\w*\s+cząstkow\w*"
    r"|\bdostarczon\w*\s+(?:dan|materiał|analiz)\w*|\bodwołani\w*\s+do\s+fakt\w*",
    re.IGNORECASE,
)
BAD_CITATION = re.compile(r"\[(?!\d{1,2}\])(?!\s*\])[^\]\n]{1,40}\](?!\()")
UNINFLECTED = re.compile(
    r"\b(?:zanotował\w*|odnotował\w*|odnotowano|zanotowano|wystąpiła|osiągnęł\w*|wykazał\w*)"
    r"\s+(?:zmiana|marża)\b"
    r"|\bstref(?:ie|ę)\s+\w+na\b"
    r"|\b(?:odpowiada|oznacza|wskazuje\s+na|przyjmuje)\s+strefa\b"
    r"|\b(?:wskazuje\s+na|oznacza)\s+typowa\b"
    r"|\boznacza\s+(?:przy|w)\s+(?:dolnej|górnej|środku|minimum|maksimum)\b"
    r"|\b(?:między|pomiędzy)\s+(?:górna|dolna)\s+wstęga\b"
    r"|\bśredniej\s+wykładnicza\b"
    r"|\b(?:oznacza|odzwierciedla|charakteryzuje\s+się)\s+(?:zmiana|marża|linia)\b"
)
# Ta sama nazwa wielkości przed i po czasowniku („ROE wyniosła rentowność kapitału
# własnego (ROE) na poziomie …”).
REPEATED_LABEL = re.compile(
    r"(?P<l>\b[^\W\d][\w() ]{10,60}?)\s+(?:wynosi|wyniosł\w*|wyniósł|osiągnęł\w*)\s+(?P=l)",
    re.IGNORECASE,
)
CITATIONS = re.compile(r"(?:\[\d{1,2}\])+")
DECIMAL_POINT = re.compile(
    r"(?<![\w.])\d+\.\d+(?=\s*(?:%|USD|\$|mld|mln|bln|proc|punkt|p\.\s?p))", re.IGNORECASE
)
VAGUE = re.compile(
    r"\bokreślon\w*\s+w\s+(?:źródle|artykule|raporcie|wiadomości)"
    r"|\bopisan\w*\s+w\s+(?:raporcie|źródle|artykule)\s+źródłow\w*"
    r"|\bna\s+konkretny\s+dzień"
    r"|\bokreślon\w*\s+(?:poziom\w*|stop\w*|wartoś\w*|kwot\w*|cen\w*)"
    r"|\bna\s+dany\s+rok",
    re.IGNORECASE,
)
DISCREPANCY = re.compile(
    r"\brozbieżn\w*|\bodmienn\w*\s+niż\b|\bróżni\w*\s+się\s+od\b", re.IGNORECASE
)
ENGLISH = re.compile(
    r"\b(?:facility|facilities|profit|beat|guidance|outlook|revenue|earnings|shares|stock"
    r"|upgrade|downgrade|buyback|billion|million|trillion|yield|deal|growth|selloff)\b"
    r"|(?<![\w.])\d+(?:[.,]\d+)?\s?(?:[BMK]|bn|[bmk](?<=\d[bmk]))\b(?!\w)"
)


def _short(sentence: str) -> str:
    plain = re.sub(r"\s+", " ", sentence).strip()
    return plain if len(plain) <= 90 else plain[:87] + "..."


def _value(facts: FactSheet, fact_id: str) -> float | None:
    if fact_id not in facts:
        return None
    value = facts[fact_id].value
    return float(value) if isinstance(value, (int, float)) else None


def check_form(text: str, facts: FactSheet) -> list[ContentIssue]:
    """Błędy formy, po jednym komunikacie na zdanie (pierwsza pasująca reguła)."""
    issues: list[ContentIssue] = []
    for sentence in sentences(text):
        message = _first_issue(sentence, facts)
        if message:
            issues.append(ContentIssue(sentence, message.replace("<S>", _short(sentence))))
    return issues


def _first_issue(sentence: str, facts: FactSheet) -> str | None:
    if (match := INTERNAL.search(sentence)) is not None:
        return (
            f"Zdanie „<S>” opisuje sposób przygotowania raportu („{match.group(0)}”); "
            "czytelnik nie zna tych pojęć, więc napisz o danych (np. „dane roczne”, "
            "„brak sprawozdania”) albo usuń zdanie"
        )
    if (match := BAD_CITATION.search(sentence)) is not None:
        return (
            f"Odnośnik „{match.group(0)}” w zdaniu „<S>” nie jest numerem wiadomości; "
            "wpisz numer z listy wiadomości, np. [2], albo usuń nawias"
        )
    for match in CARD_THEN_CITATION.finditer(sentence):
        if match.group(1) not in NAME_IDS:
            refs = re.sub(r"[\s,]+", "", match.group(2))
            return (
                f"W zdaniu „<S>” wartość z karty {{{{{match.group(1)}}}}} ma numer "
                f"wiadomości {refs}; wartości z karty nie pochodzą z artykułów, więc usuń przy "
                "niej numer wiadomości (numer zostaje tylko przy informacji z artykułu)"
            )
    for match in DECREASE.finditer(sentence):
        value = _value(facts, match.group(1))
        if value is not None and value < 0:
            return (
                f"W zdaniu „<S>” wartość {{{{{match.group(1)}}}}} ma znak minus, a zdanie "
                "już mówi o spadku („spadek o -…”); napisz „zmiana o …” albo pomiń słowo "
                "„spadek”"
            )
    if DISCREPANCY.search(sentence) and (PLACEHOLDER.search(sentence) or "kart" in sentence):
        return (
            "Zdanie „<S>” zestawia wiadomość z wartością z karty jako rozbieżność; artykuły "
            "opisują inne dni lub okresy niż karta, więc nie porównuj ich, tylko podaj "
            "informację z artykułu z jego numerem"
        )
    if (match := VAGUE.search(sentence)) is not None:
        return (
            f"Zdanie „<S>” zastępuje liczbę ogólnikiem („{match.group(0)}”); przytocz "
            "wartość tak jak w cytowanej wiadomości, z jej numerem, albo usuń tę część zdania"
        )
    words = re.findall(r"[^\W\d_]+", CITATIONS.sub(" ", PLACEHOLDER.sub(" ", sentence)))
    if CITATIONS.search(sentence) and len(words) <= 1:
        return (
            "Zdanie „<S>” nie zawiera żadnej informacji poza numerem wiadomości; "
            "napisz, co podaje wiadomość, albo usuń zdanie"
        )
    if (match := REPEATED_LABEL.search(sentence)) is not None:
        return (
            f"W zdaniu „<S>” ta sama nazwa wielkości stoi dwa razy („{match.group('l')}”); "
            "usuń powtórzenie"
        )
    if (match := UNINFLECTED.search(sentence)) is not None:
        return (
            f"W zdaniu „<S>” nazwa lub ocena z karty stoi bez odmiany („{match.group(0)}”); "
            "przeredaguj zdanie, odmieniając ją zgodnie z resztą zdania"
        )
    if (match := DECIMAL_POINT.search(sentence)) is not None:
        return (
            f"Liczba „{match.group(0)}” w zdaniu „<S>” ma kropkę dziesiętną; w polskim "
            "tekście zapisz ją z przecinkiem"
        )
    if (match := ENGLISH.search(sentence)) is not None:
        return (
            f"Zdanie „<S>” zawiera angielskie słowo lub skrót („{match.group(0)}”); "
            "napisz to po polsku (np. „mld USD” zamiast „B”)"
        )
    return None
