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
   „dwukrotnie”, „od pięciu sesji”), z wyjątkiem okresów z karty faktów i liczb, które
   w tej samej wartości podaje cytowana wiadomość („pięciu bilionów dolarów [2]” przy
   „$5 Trillion”). Tak samo liczba zapisana cyframi w zdaniu z numerem wiadomości.
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
# Początki nazw geograficznych i ogólnych, które po polsku brzmią inaczej niż w angielskich
# wiadomościach („Ameryce Północnej” przy „North America”, „Izraelu” przy „Israel”).
COMMON_PREFIXES = (
    "amery",
    "america",
    "północ",
    "połud",
    "zachod",
    "wschod",
    "azj",
    "afryk",
    "izrael",
    "rosj",
    "ukrain",
    "kanad",
    "meksyk",
    "brazyl",
    "wietnam",
    "bank",
)
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


# Liczba zapisana słowami przy jednostce („o cztery procent”, „pięćdziesiąt milionów dolarów”)
# jest dozwolona tylko wtedy, gdy tę samą wartość podaje cytowana wiadomość.
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
# 12 miesięcy, 52 tygodnie, dwa lata obrotowe (zmiana przychodów rok do roku i rok wcześniej)
# albo trzy lata (historia, na tle której oceniana jest odległość od SMA(200)).
SPELLED_PERIOD = re.compile(
    rf"\b{NUMBER_WORD}(?:\s+{NUMBER_WORD})*\s+(?:kolejn\w*\s+|ostatni\w*\s+)?"
    r"(?:sesj\w*|dni|dzie\w*|tygod\w*|miesi\w*|lat\b|lata\b|latach|kwarta\w*)",
    re.IGNORECASE,
)
CARD_PERIOD = re.compile(
    r"^(?:trzy|trzech)\s+(?:\w+\s+)?miesi|^dwunast\w*\s+(?:\w+\s+)?miesi"
    r"|^pięćdziesi\w*\s+dw\w*\s+(?:\w+\s+)?tygod|^(?:dw|trz)\w*\s+(?:\w+\s+)?la",
    re.IGNORECASE,
)
# Krotność i ułamek zapisane słowem („dwukrotnie”, „podwoił”, „o połowę”) też są liczbą.
MULTIPLE = re.compile(
    r"\b(?:dwu|trzy|cztero|pięcio|sześcio|siedmio|ośmio|dziewięcio|dziesięcio|stu)krotn\w*"
    r"|\bpo(?:dw|tr)o(?:i|j)\w*|\bo\s+połowę\b",
    re.IGNORECASE,
)


# Liczbę z cytowanej wiadomości wolno przytoczyć, jeśli ta sama wartość stoi w źródle
# („pięciu bilionów dolarów” albo „5 bln USD” przy „$5 Trillion”).
_PL_WORDS = {
    **dict.fromkeys(("jeden", "jedna", "jedno", "jednego", "jednej"), 1),
    **dict.fromkeys(("dwa", "dwie", "dwóch", "dwu"), 2),
    **dict.fromkeys(("trzy", "trzech"), 3),
    **dict.fromkeys(("cztery", "czterech"), 4),
    **dict.fromkeys(("pięć", "pięciu"), 5),
    **dict.fromkeys(("sześć", "sześciu"), 6),
    **dict.fromkeys(("siedem", "siedmiu"), 7),
    **dict.fromkeys(("osiem", "ośmiu"), 8),
    **dict.fromkeys(("dziewięć", "dziewięciu"), 9),
    **dict.fromkeys(("dziesięć", "dziesięciu"), 10),
    **{
        f"{stem}{end}": 11 + i
        for i, stem in enumerate(
            ("jedena", "dwuna", "trzyna", "czterna", "piętna", "szesna", "siedemna", "osiemna")
            + ("dziewiętna",)
        )
        for end in ("ście", "stu")
    },
    "dwanaście": 12,
    **dict.fromkeys(("dwadzieścia", "dwudziestu"), 20),
    **dict.fromkeys(("trzydzieści", "trzydziestu"), 30),
    **dict.fromkeys(("czterdzieści", "czterdziestu"), 40),
    **{
        f"{stem}dziesi{end}": 10 * value
        for stem, value in (("pięć", 5), ("sześć", 6), ("siedem", 7), ("osiem", 8), ("dziewięć", 9))
        for end in ("ąt", "ęciu")
    },
    **dict.fromkeys(("sto", "stu"), 100),
    **dict.fromkeys(("dwieście", "dwustu"), 200),
    **dict.fromkeys(("trzysta", "trzystu"), 300),
    **dict.fromkeys(("czterysta", "czterystu"), 400),
    **dict.fromkeys(("pięćset", "pięciuset"), 500),
    **dict.fromkeys(("sześćset", "sześciuset"), 600),
    **dict.fromkeys(("siedemset", "siedmiuset"), 700),
    **dict.fromkeys(("osiemset", "ośmiuset"), 800),
    **dict.fromkeys(("dziewięćset", "dziewięciuset"), 900),
    "pół": 0.5,
    "półtora": 1.5,
}
_PL_SCALE = (("tysi", 3), ("milion", 6), ("miliard", 9), ("bilion", 12))
_PL_MULTIPLE = {"dwu": 2, "trzy": 3, "cztero": 4, "pięcio": 5, "sześcio": 6, "siedmio": 7}
_PL_MULTIPLE |= {"ośmio": 8, "dziewięcio": 9, "dziesięcio": 10, "stu": 100}
_EN_WORDS = {
    word: i
    for i, word in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
    )
}
_EN_TENS = "thirty forty fifty sixty seventy eighty ninety".split()
_EN_WORDS |= {word: 10 * i for i, word in enumerate(_EN_TENS, 3)}
_EN_WORDS |= {"hundred": 100, "half": 0.5, "halved": 0.5}
_EN_WORDS |= dict.fromkeys(("double", "doubled", "doubles", "doubling", "twice"), 2)
_EN_WORDS |= dict.fromkeys(("triple", "tripled", "triples", "tripling"), 3)
_EN_SCALE = {"thousand": 3, "k": 3, "million": 6, "mln": 6, "m": 6, "billion": 9, "bn": 9, "b": 9}
_EN_SCALE |= {"trillion": 12, "tn": 12, "t": 12}
_EN_WORD = "|".join(sorted(_EN_WORDS, key=len, reverse=True))
SOURCE_NUMBER = re.compile(
    rf"(?<![\w.])(\d+(?:,\d{{3}})*(?:\.\d+)?|\b(?:{_EN_WORD})\b)(?:[\s-]+({_EN_WORD})\b)?"
    r"(?:\s*(thousand|million|billion|trillion|bn|tn|mln|[kmbt])\b)?(\s*(?:%|percent\b))?",
    re.IGNORECASE,
)
REPORT_NUMBER = re.compile(
    r"(?<![\w.,])(\d{1,3}(?:[ \u00a0\u202f]\d{3})+(?:,\d+)?|\d+(?:[.,]\d+)?)"
    # Jednostka przedziału („15,0–16,0 mld USD”) dotyczy też jego początku.
    r"(?=(?:\s*[-‐‑–—]\s*\d+(?:[.,]\d+)?)?\s*(%|bln|mld|mln|tys|bilion\w*|miliard\w*|milion\w*|tysi\w*)?)"
    r"(?![.,]?\d)",
    re.IGNORECASE,
)


def _pl_scale(word: str) -> int:
    lower = word.lower()
    return next((exp for stem, exp in _PL_SCALE if lower.startswith(stem)), 0)


def source_values(source: str) -> list[tuple[float, bool]]:
    """Wartości liczb z wiadomości, zapisanych cyframi albo słowami angielskimi, i to,
    czy są procentami."""
    values = []
    for m in SOURCE_NUMBER.finditer(source):
        first, second, scale = m.group(1).lower(), (m.group(2) or "").lower(), m.group(3)
        if first[0].isdigit():
            value = float(first.replace(",", ""))
            if second in _EN_SCALE and not scale:
                scale = second
        elif first in _EN_WORDS:
            value = _EN_WORDS[first] + (_EN_WORDS.get(second, 0) if first.endswith("ty") else 0)
            if second in _EN_SCALE and not scale:
                scale = second
        else:
            continue
        values.append((value * 10 ** _EN_SCALE.get((scale or "").lower(), 0), bool(m.group(4))))
    return values


def spelled_value(phrase: str) -> float | None:
    """Wartość liczby zapisanej słowami po polsku („pięciu bilionów” = 5·10¹²)."""
    words = re.findall(r"\w+", phrase.lower())
    for prefix, value in _PL_MULTIPLE.items():
        if words and words[0].startswith(prefix + "krotn"):
            return float(value)
    if words and re.match(r"po(?:dw|tr)o", words[0]):
        return 2.0 if words[0][2] == "d" else 3.0
    if words[-2:] == ["o", "połowę"]:
        return 0.5
    total = 0.0
    for word in words:
        if word in _PL_WORDS:
            total += _PL_WORDS[word]
        elif _pl_scale(word):
            return (total or 1) * 10 ** _pl_scale(word)
    return total or None


def in_source(
    value: float, values: list[tuple[float, bool]], scale: int = 0, percent: bool = False
) -> bool:
    """Czy wartość stoi w źródle, z dokładnością do zaokrąglenia przez autora tekstu.
    Procent musi odpowiadać procentowi w źródle."""
    tolerance = (0.5 if float(value / 10**scale).is_integer() else 0.05) * 10**scale
    return any(abs(value - v) <= tolerance + 1e-9 and p == percent for v, p in values)


def cited_source(sentence: str, news: list[dict]) -> str:
    """Nagłówki i streszczenia wiadomości, których numery stoją w zdaniu."""
    cited = [int(n) for n in CITATION.findall(sentence) if 1 <= int(n) <= len(news)]
    source = " ".join(
        f"{news[n - 1].get('headline', '')} {news[n - 1].get('summary', '')}" for n in cited
    )
    # Spacja po kropce dziesiętnej w źródle („$9. 1 billion”) to błąd zapisu, nie dwie liczby.
    return re.sub(r"(?<=\d)\. (?=\d)", ".", source)


def strip_news_numbers(sentence: str, news: list[dict]) -> str:
    """Usuwa ze zdania liczby, które w tej samej wartości podaje cytowana wiadomość."""
    values = source_values(cited_source(sentence, news))
    if not values:
        return sentence

    def drop(match: re.Match) -> str:
        unit = match.group(2) or ""
        scale = {"bln": 12, "mld": 9, "mln": 6, "tys": 3}.get(unit.lower(), _pl_scale(unit))
        digits = re.sub(r"[ \u00a0\u202f]", "", match.group(1))
        value = float(digits.replace(",", ".")) * 10**scale
        quoted = in_source(value, values, scale, unit == "%")
        return " " * len(match.group(1)) if quoted else match.group(0)

    return REPORT_NUMBER.sub(drop, sentence)


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
    for match in PROPER.finditer(text):
        word = match.group(0)
        before = text[: match.start()]
        # Początek zdania także po znacznikach Markdown („*Uwaga:*”) i po dwukropku.
        first = not re.sub(r"[\s*_>#~`-]", "", before) or bool(re.search(r":[\s*_]*$", before))
        if first and not word.isupper() and not known(word, first_names):
            continue
        lower = word.lower()
        if lower in COMMON or lower.startswith(COMMON_PREFIXES) or len(word) < 2:
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


def _quoted(phrase: str, values: list[tuple[float, bool]]) -> bool:
    value = spelled_value(phrase)
    scale = max((_pl_scale(w) for w in phrase.split()), default=0)
    percent = "procent" in phrase.lower()
    return value is not None and in_source(value, values, scale, percent)


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


def check_spelled_numbers(text: str, news: list[dict] | None = None) -> list[ContentIssue]:
    """Liczby zapisane słowami, którymi model omija zakaz wpisywania liczb.

    Liczba, którą w tej samej wartości podaje cytowana wiadomość, jest dozwolona."""
    issues = []
    for sentence in sentences(text):
        values = source_values(cited_source(sentence, news)) if news else []
        found = [m for m in SPELLED_PERIOD.finditer(sentence) if not CARD_PERIOD.match(m[0])]
        found = [*SPELLED_NUMBER.finditer(sentence), *MULTIPLE.finditer(sentence), *found]
        found = [m for m in found if not _quoted(m.group(0), values)]
        match = found[0] if found else None
        if match:
            issues.append(
                ContentIssue(
                    sentence,
                    f"Zdanie „{_short(sentence)}” podaje liczbę słowami („{match.group(0)}”), "
                    + (
                        "której nie ma w karcie faktów ani w cytowanych wiadomościach; podaj "
                        "wartość taką jak w wiadomości albo opisz zjawisko bez liczby"
                        if values
                        else "której nie ma w karcie faktów; opisz zjawisko bez liczby albo "
                        "użyj odwołania do karty"
                    ),
                )
            )
    return issues
