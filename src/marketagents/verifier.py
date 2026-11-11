"""Weryfikacja liczb w raporcie względem karty faktów.

Działa bez modelu językowego, więc jej wynik jest powtarzalny. Sprawdza, czy:
1. każde odwołanie {{identyfikator}} wskazuje istniejący fakt,
2. w tekście nie ma liczb wpisanych poza odwołaniami, z wyjątkiem okresów stojących przy
   nazwie wskaźnika (np. „RSI(14)”), okresów z karty faktów i okien wskaźników przed
   jednostką czasu („z 52 tygodni”, „średnia z 50 sesji”) oraz nazw indeksów. Ta sama liczba
   w innym miejscu („wzrósł o 20%”) jest błędem, a liczba w złożeniu słownym
   („5-procentowy”) tylko wtedy nie jest, gdy jest okresem albo pochodzi z opisu faktu
   („10-letnich”). Liczba wpisana wprost jest błędem nawet wtedy, gdy równa się wartości z
   karty, bo nie wiadomo, której wielkości dotyczy: przy dopuszczeniu liczb zgodnych z
   faktem z tego samego zdania test wstrzykniętych błędów pokazał, że podmiana na wartość
   sąsiedniego faktu (np. SMA(50) zamiast SMA(20)) przechodzi niezauważona. Zgodność z kartą
   służy tylko do podpowiedzi, którego odwołania użyć przy poprawce,
3. lata w tekście odpowiadają datom z karty faktów, a data w formacie ISO występuje w karcie
   (liczba z jednostką, np. „2025 USD”, jest sprawdzana jako liczba, a nie jako rok),
4. słowny opis liczby i okres zmiany zgadzają się z kartą (moduł interpretation),
5. każde zdanie o wiadomościach wskazuje źródło numerem z listy newsów, np. [2], a numer
   w dowolnej sekcji raportu odpowiada wiadomości z listy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html import escape

from marketagents.facts import Fact, FactSheet
from marketagents.grounding import (
    check_forecasts,
    check_news_content,
    check_records,
    check_spelled_numbers,
    strip_news_numbers,
)
from marketagents.interpretation import InterpretationIssue, check_interpretations

PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z0-9_]+)\s*\}\}")
# Model czasem otacza odwołanie znakami ` jak kod; przy podstawianiu są one usuwane.
PLACEHOLDER_IN_CODE = re.compile(r"`\{\{\s*([A-Za-z0-9_]+)\s*\}\}`")
# Liczba w zapisie polskim lub angielskim, z opcjonalnym separatorem tysięcy (spacja).
# Liczby w złożeniach słownych, np. „10-letnie obligacje”, sprawdza osobno wzorzec COMPOUND.
NUMBER = re.compile(
    r"(?<![\w.,])[-−+]?\d{1,3}(?:[  ]\d{3})+(?:[.,]\d+)?(?![\w])"
    r"|(?<![\w.,])[-−+]?\d+(?:[.,]\d+)?(?![\w]|-[^\W\d])"
)
# Liczba w złożeniu słownym („10-letnich”, „5-procentowy”); dopuszczalna tylko wtedy, gdy jest
# okresem albo liczbą z opisu faktu. Pomijane są nazwy formularzy SEC („10-K”, „10-Q”).
COMPOUND = re.compile(r"(?<![\w.,−+-])\d+(?=-[^\W\d])(?!-[KQ]\b)")
# Łącznik zapisany innym znakiem („10‑letnich”, „10–letnich”) jest traktowany jak „-”.
DASH_IN_WORD = re.compile(r"(?<=\d)[\u2010\u2011\u2012\u2013\u2014](?=[^\W\d_])")
ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
# Odwołanie do wiadomości z listy newsów, np. [3].
CITATION = re.compile(r"\[(\d{1,2})\]")
NEWS_SECTION = "## Wiadomości"
LIST_MARKER = re.compile(r"^\s*\d+[.)]\s", re.MULTILINE)
INDEX_NAME = re.compile(
    r"NASDAQ[-\s\u00a0\u202f]?100|S&P[-\s\u00a0\u202f]?500|Russell[-\s\u00a0\u202f]?2000",
    re.IGNORECASE,
)
# Koniec zdania: znak końca i odstęp albo nowy wiersz. Kropka po skrócie („ok. 61”, „np. RSI”)
# nie kończy zdania, żeby podpowiedź wskazała fakt, przy którym stoi liczba.
SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")
ABBREVIATION = re.compile(
    r"(?<![\w.])(?:ok|np|m\.in|tj|tzn|tzw|wg|ang|por|zob|ca|mln|mld|tys|pkt|proc|r|ds|itd|itp"
    r"|nr|dot|godz|mies|tyg|prof|dr|mgr|inż|św|ul|wyd|red|zł)\.$",
    re.IGNORECASE,
)

# Okresy wskaźników, które mogą wystąpić w tekście jako nazwy, np. „RSI(14)”, „SMA 200”.
DEFAULT_ALLOWED = frozenset({2, 9, 12, 14, 20, 26, 50, 52, 200})
# Okres jest dopuszczany tylko przy nazwie wskaźnika („RSI(14)”, „SMA 50 i 200”,
# „Bollingera (20, 2)”) albo przed jednostką czasu („z 52 tygodni”, „3 miesiące”).
# Bez tego warunku liczba równa okresowi przechodziłaby także jako wartość („wzrósł o 20%”).
# Przed jednostką czasu liczba musi być okresem z karty faktów albo oknem wskaźnika
# („średnia z 50 sesji”), inaczej przechodziłby zmyślony okres („rośnie od 9 sesji”).
INDICATOR_NAME = r"(?:RSI|SMA|EMA|MACD|Bollinger\w*|sygnał\w*|średni\w*)"
PERIOD_NUMBER = r"\d+(?![.,]\d|\s*%)"
NAMED_PERIODS = re.compile(
    rf"(?<!\w){INDICATOR_NAME}\s*(?:\(\s*\d+(?:\s*,\s*\d+)*\s*\)"
    rf"|{PERIOD_NUMBER}(?:(?:\s*,\s*|\s+(?:i|oraz)\s+){PERIOD_NUMBER})*)",
    re.IGNORECASE,
)
PERIOD_BEFORE_UNIT = re.compile(
    r"(?<![\w.,])(\d+)(\s*(?:sesj|dni\b|dzie|tyg|miesi|mies\.|lat|rok|kwarta|odchyle))",
    re.IGNORECASE,
)
WINDOW_BEFORE = re.compile(
    rf"(?:{INDICATOR_NAME}|wstęg\w*)\S*(?:\s+\w+)?\s+z\s+(?:ostatnich\s+)?$", re.IGNORECASE
)
BANDS_BEFORE = re.compile(r"wstęg|Bollinger", re.IGNORECASE)
# Okresy zmian w karcie faktów: liczba i początek jednostki („3 miesiącach” -> (3, „mie”)).
# Zmiana z ostatniego roku bywa opisana jako zmiana z 12 miesięcy, karta obejmuje dwa lata
# obrotowe (zmiana przychodów rok do roku i rok wcześniej), a odległość od SMA(200) jest
# oceniana na tle ostatnich 3 lat.
STANDARD_PERIODS = frozenset({(3, "mie"), (12, "mie"), (52, "tyg"), (2, "lat"), (3, "lat")})
# Liczba z opisu lub oceny faktu, do którego odwołuje się zdanie, zapisana z tą samą
# jednostką („0% = minimum, 100% = maksimum”, „na tle ostatnich 3 lat”).
DESCRIBED_NUMBER = re.compile(r"(?<![\w.,])(\d+)\s*(%|[^\W\d_]{3})")
# Zero jako poziom odniesienia („histogram powyżej 0”), a nie wartość.
ZERO_LEVEL = re.compile(
    r"\b((?:powyżej|poniżej|ponad|nad|pod|wokół|blisko|linii|poziomu|okolicach)\s+)0"
    r"(?![.,]?\d|\s*%)",
    re.IGNORECASE,
)
# Numer w nazwie produktu z cytowanej wiadomości („iPhone 18”), a nie wartość.
NAMED_NUMBER = re.compile(r"(?<![\w.,])([^\W\d_][\w-]*)\s+(\d+)(?![.,]?\d|\s*%)")
# Liczba sklejona z jednostką („20mld”, „5x”) też jest liczbą wpisaną wprost.
GLUED = re.compile(
    r"(?<![\w.,−+-])\d+(?:[.,]\d+)?(?=(?:mld|mln|tys|usd|proc|pkt|pp|x)\b)", re.IGNORECASE
)
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
    # Daty w formacie ISO, których nie ma w karcie faktów (np. zmyślona data wydarzenia).
    wrong_dates: list[str] = field(default_factory=list)
    interpretations: list[InterpretationIssue] = field(default_factory=list)
    checked_numbers: int = 0
    # Liczba wpisana wprost -> fakt o tej wartości (podpowiedź dla autora tekstu).
    suggestions: dict[str, str] = field(default_factory=dict)
    # Liczby, które powtarzają wartość faktu, do którego odwołuje się to samo zdanie.
    repeated: list[str] = field(default_factory=list)
    # Liczba wpisana wprost -> zdanie, w którym stoi (żeby autor wiedział, gdzie poprawić).
    contexts: dict[str, str] = field(default_factory=dict)
    # Zdania o wiadomościach bez numeru źródła i numery spoza listy newsów.
    uncited: list[str] = field(default_factory=list)
    bad_citations: list[str] = field(default_factory=list)
    news_count: int = 0

    @property
    def ok(self) -> bool:
        return not (
            self.unknown_ids
            or self.unverified_numbers
            or self.wrong_years
            or self.wrong_dates
            or self.interpretations
            or self.uncited
            or self.bad_citations
        )

    @property
    def flagged_sentences(self) -> list[str]:
        """Zdania z błędną interpretacją liczby (do oznaczenia, jeśli nie zostaną poprawione)."""
        return [i.sentence for i in self.interpretations]

    @property
    def to_mark(self) -> list[str]:
        """Wartości, które trzeba oznaczyć w tekście, jeśli nie zostaną poprawione."""
        return self.unverified_numbers + self.wrong_years + self.wrong_dates

    def issues(self) -> list[str]:
        out = [f"Nieznany identyfikator faktu: {{{{{i}}}}}" for i in self.unknown_ids]
        # Ta sama liczba powtórzona w zdaniu daje jeden komunikat.
        for n in dict.fromkeys(self.unverified_numbers):
            fact_id = self.suggestions.get(n)
            where = f" w zdaniu „{_short(self.contexts[n])}”" if n in self.contexts else ""
            if n in self.repeated:
                out.append(
                    f"Liczba „{n}”{where} powtarza bez odwołania wartość {{{{{fact_id}}}}}, "
                    "która już stoi w zdaniu; usuń ją"
                )
            elif fact_id:
                out.append(
                    f"Liczba „{n}”{where} jest wpisana bez odwołania; jeśli to wartość "
                    f"{{{{{fact_id}}}}}, wstaw w jej miejsce odwołanie {{{{{fact_id}}}}}"
                )
            else:
                out.append(
                    f"Liczba „{n}”{where} nie pochodzi z karty faktów; "
                    "wstaw odwołanie do faktu z karty, przytocz wartość tak jak w cytowanej "
                    "wiadomości razem z jej numerem albo opisz zjawisko bez liczby"
                )
        out += [
            f"Rok {y} nie odpowiada żadnej dacie w karcie faktów; "
            "sprawdź okres, którego dotyczy zdanie"
            for y in self.wrong_years
        ]
        out += [
            f"Data {d} nie występuje w karcie faktów; wstaw odwołanie do daty z karty "
            "albo usuń datę"
            for d in self.wrong_dates
        ]
        out += [i.message for i in self.interpretations]
        out += [
            f"Zdanie „{_short(s)}” nie wskazuje źródła; dodaj numer wiadomości, "
            "na której się opiera"
            for s in self.uncited
        ]
        out += [
            f"Numer [{n}] nie odpowiada żadnej wiadomości (dostępne: 1–{self.news_count})"
            for n in self.bad_citations
        ]
        return out


def _short(sentence: str) -> str:
    plain = re.sub(r"\s+", " ", sentence).strip()
    return plain if len(plain) <= 90 else plain[:87] + "..."


def news_part(text: str) -> str:
    """Sekcja wiadomości raportu (od nagłówka do następnego nagłówka) albo pusty tekst."""
    start = text.find(NEWS_SECTION)
    if start < 0:
        return ""
    body = text[start + len(NEWS_SECTION) :]
    end = body.find("\n## ")
    return body if end < 0 else body[:end]


# Wiersze bez treści do potwierdzenia: nagłówek, pogrubiony tytuł akapitu, wprowadzenie
# do listy („Najważniejsze wątki:”) i informacja o braku wiadomości.
NOT_A_CLAIM = re.compile(
    r"\s*(?:#|\*\*[^*]+\*\*:?\s*$|[^\n]*:\s*$|brak\s+(?:\w+\s+){0,2}(?:wiadomości|newsów)\b)"
    r"|\W*$",
    re.IGNORECASE,
)


def check_citations(text: str, count: int) -> tuple[list[str], list[str]]:
    """Zdania bez numeru źródła oraz numery spoza listy (1..count)."""
    uncited, bad = [], []
    for sentence in sentences(text):
        # Zdanie z odwołaniem do karty (np. porównanie ze zmianą kursu) ma źródło w karcie.
        if NOT_A_CLAIM.match(sentence) or PLACEHOLDER.search(sentence):
            continue
        numbers = CITATION.findall(sentence)
        if not numbers:
            uncited.append(sentence.strip())
        bad += [n for n in numbers if not 1 <= int(n) <= count]
    return uncited, sorted(set(bad), key=int)


def parse_number(text: str) -> float:
    cleaned = text.replace("−", "-").replace(" ", "").replace(" ", "")
    return float(cleaned.replace(",", "."))


def matches_fact(value: float, fact: Fact) -> bool:
    """Czy liczba może być zapisem wartości faktu (do podpowiedzi, którego odwołania użyć)."""
    if isinstance(fact.value, str):
        return False
    tolerance = 0.5 * 10 ** (-fact.decimals) + 1e-9
    # Spadek bywa opisany słownie („spadek o 3,2%”), więc liczy się też wartość bezwzględna.
    if abs(value - fact.value) <= tolerance or abs(value - abs(fact.value)) <= tolerance:
        return True
    # Liczba zaokrąglona przez autora do mniejszej liczby miejsc po przecinku.
    return float(value).is_integer() and abs(abs(value) - abs(fact.value)) < 0.5


def _unit_key(unit: str) -> str:
    key = unit.strip().lower()[:3]
    return "lat" if key == "let" else key


def label_periods(facts: FactSheet) -> frozenset[tuple[int, str]]:
    """Okresy z opisów faktów, np. (52, „tyg”) z „najwyższy kurs z 52 tygodni”."""
    return STANDARD_PERIODS | frozenset(
        (int(m.group(1)), _unit_key(m.group(2)))
        for fact in facts
        for m in re.finditer(r"(\d+)[\s-]*([^\W\d_]{3})", fact.label)
    )


def strip_periods(
    text: str, allowed: frozenset[int], periods: frozenset[tuple[int, str]] = STANDARD_PERIODS
) -> str:
    """Usuwa okresy z dozwolonego zbioru, gdy stoją przy nazwie wskaźnika, oraz okresy przed
    jednostką czasu, gdy są okresem z karty faktów albo oknem wskaźnika. Pozostałe liczby,
    także równe okresom, zostają do sprawdzenia."""

    def drop(match: re.Match) -> str:
        return re.sub(
            r"\d+", lambda n: " " if int(n.group(0)) in allowed else n.group(0), match.group(0)
        )

    text = NAMED_PERIODS.sub(drop, text)

    def drop_period(match: re.Match) -> str:
        number, key = int(match.group(1)), _unit_key(match.group(2))
        before = text[max(0, match.start() - 60) : match.start()]
        window = number in allowed and (
            WINDOW_BEFORE.search(before) or (key == "odc" and BANDS_BEFORE.search(before))
        )
        return " " + match.group(2) if (number, key) in periods or window else match.group(0)

    return PERIOD_BEFORE_UNIT.sub(drop_period, text)


def described_numbers(facts: list[Fact]) -> frozenset[tuple[int, str]]:
    """Liczby z jednostką z opisów i ocen faktów, np. (100, „%”), (3, „lat”)."""
    return frozenset(
        (int(m.group(1)), _unit_key(m.group(2)) if m.group(2) != "%" else "%")
        for fact in facts
        for m in DESCRIBED_NUMBER.finditer(f"{fact.label} {fact.assessment or ''}")
    )


def strip_described(text: str, pairs: frozenset[tuple[int, str]]) -> str:
    """Usuwa liczby, które powtarzają opis faktu z tą samą jednostką, i zero jako poziom."""

    def drop(match: re.Match) -> str:
        unit = match.group(2)
        key = "%" if unit == "%" else _unit_key(unit)
        return " " + unit if (int(match.group(1)), key) in pairs else match.group(0)

    return ZERO_LEVEL.sub(r"\1 ", DESCRIBED_NUMBER.sub(drop, text))


def strip_product_numbers(text: str, source: str) -> str:
    """Usuwa numery z nazw występujących w cytowanych wiadomościach („iPhone 18”)."""

    def drop(match: re.Match) -> str:
        name = f"{match.group(1)} {match.group(2)}".lower()
        return match.group(1) + " " if name in source else match.group(0)

    return NAMED_NUMBER.sub(drop, text)


def numbers_to_check(
    text: str, allowed: frozenset[int], periods: frozenset[tuple[int, str]] = STANDARD_PERIODS
) -> list[str]:
    """Liczby wpisane wprost w kolejności występowania, po usunięciu okresów i dat."""
    text = DASH_IN_WORD.sub("-", text)
    text = strip_periods(text, allowed, periods)
    found = [(m.start(), m.group(0)) for m in GLUED.finditer(text)]
    text = GLUED.sub(lambda m: " " * len(m.group(0)), text)
    found += [(m.start(), m.group(0)) for m in NUMBER.finditer(text)]
    found += [
        (m.start(), m.group(0)) for m in COMPOUND.finditer(text) if int(m.group(0)) not in allowed
    ]
    return [raw for _, raw in sorted(found)]


def label_numbers(facts: FactSheet) -> frozenset[int]:
    """Liczby całkowite z opisów faktów, np. „3 miesiącach”, „10-letnich obligacji”."""
    return frozenset(int(raw) for fact in facts for raw in re.findall(r"\d+", fact.label))


def sentences(text: str) -> list[str]:
    """Podział tekstu na zdania; kropka w liczbie („3.2”) i po skrócie nie kończy zdania."""
    out, start = [], 0
    for match in SENTENCE_END.finditer(text):
        head = text[start : match.start()]
        if "\n" not in match.group(0) and ABBREVIATION.search(head):
            continue
        out.append(head)
        start = match.end()
    out.append(text[start:])
    return [s for s in out if s.strip()]


# Rok w tekście, np. „w 2025 r.” (bez dat ISO, które są usuwane wcześniej).
# Liczba z jednostką („2025 USD”, „2030 mld”) jest wartością, a nie rokiem.
YEAR = re.compile(
    r"(?<![\w.,-])(?:19|20)\d{2}(?![\w,.]?\d)"
    r"(?!\s*(?:%|USD|\$|mld|mln|tys|pkt|punkt|p\.\s?p|proc|dolar|akcj))",
    re.IGNORECASE,
)


def fact_years(facts: FactSheet) -> frozenset[int]:
    """Lata, o których może mówić raport: z dat w karcie faktów oraz rok wcześniejszy
    (zmiany rok do roku i notowania z 52 tygodni obejmują poprzedni rok)."""
    years: set[int] = set()
    for fact in facts:
        if isinstance(fact.value, str) and ISO_DATE.fullmatch(fact.value):
            year = int(fact.value[:4])
            years |= {year, year - 1}
    return frozenset(years)


def fact_dates(facts: FactSheet) -> frozenset[str]:
    """Daty zapisane w karcie faktów (ostatnia sesja, koniec roku obrotowego, odczyty makro)."""
    return frozenset(
        f.value for f in facts if isinstance(f.value, str) and ISO_DATE.fullmatch(f.value)
    )


def verify(
    text: str,
    facts: FactSheet,
    allowed: frozenset[int] = DEFAULT_ALLOWED,
    news_count: int | None = None,
    news: list[dict] | None = None,
    names: tuple[str, ...] = (),
) -> Verification:
    """news_count: liczba wiadomości na liście; gdy podana, w całym raporcie sprawdzane są
    numery źródeł, a w sekcji wiadomości także to, czy każde zdanie wskazuje źródło.
    news: lista wiadomości; zdania z numerem źródła mogą wymieniać tylko nazwy z cytowanych
    wiadomości lub z nazw spółki (names). Prognozy bez podstaw są sprawdzane zawsze."""
    if news is not None and news_count is None:
        news_count = len(news)
    allowed = allowed | indicator_periods(facts) | label_numbers(facts)
    periods = label_periods(facts)
    unknown = sorted({m.group(1) for m in PLACEHOLDER.finditer(text) if m.group(1) not in facts})
    years = fact_years(facts)
    dates = fact_dates(facts)

    wrong_years: list[str] = []
    wrong_dates: list[str] = []
    unverified: list[str] = []
    suggestions: dict[str, str] = {}
    repeated: list[str] = []
    contexts: dict[str, str] = {}
    checked = 0
    previous: list[Fact] = []
    for sentence in sentences(LIST_MARKER.sub(" ", text)):
        referenced = [facts[i] for i in PLACEHOLDER.findall(sentence) if i in facts]
        # Objaśnienie skali bywa osobnym zdaniem po zdaniu z odwołaniem.
        described = described_numbers(referenced + previous)
        previous = referenced
        wrong_dates += [
            d for d in ISO_DATE.findall(PLACEHOLDER.sub(" ", sentence)) if d not in dates
        ]
        stripped = INDEX_NAME.sub(" ", ISO_DATE.sub(" ", PLACEHOLDER.sub(" ", sentence)))
        stripped = strip_described(stripped, described)
        cited = [int(n) for n in CITATION.findall(sentence) if news and 1 <= int(n) <= len(news)]
        news_years: set[int] = set()
        if cited:
            source = " ".join(
                f"{news[n - 1].get('headline', '')} {news[n - 1].get('summary', '')}" for n in cited
            ).lower()
            stripped = strip_product_numbers(stripped, re.sub(r"\s+", " ", source))
            stripped = strip_news_numbers(stripped, news)
            # Rok podany w cytowanej wiadomości („2027 Growth Plans”) też ma źródło.
            news_years = {int(y) for y in re.findall(r"\b(?:19|20)\d{2}\b", source)}
        stripped = CITATION.sub(" ", stripped)
        wrong_years += [
            m.group(0) for m in YEAR.finditer(stripped) if int(m.group(0)) not in years | news_years
        ]
        for raw in numbers_to_check(YEAR.sub(" ", stripped), allowed, periods):
            checked += 1
            unverified.append(raw)
            contexts.setdefault(raw, sentence.strip())
            value = parse_number(raw)
            same = next((f for f in referenced if matches_fact(value, f)), None)
            # W zdaniu z numerem wiadomości liczba pochodzi zwykle z wiadomości, a nie z karty.
            other = same or (
                None if cited else next((f for f in facts if matches_fact(value, f)), None)
            )
            if other is not None:
                suggestions[raw] = other.id
            if same is not None:
                repeated.append(raw)

    uncited: list[str] = []
    bad: list[str] = []
    content = check_forecasts(text) + check_spelled_numbers(text, news) + check_records(text)
    if news_count is not None:
        # Przy pustej liście nie ma czego cytować, a każdy numer jest błędny.
        part = news_part(text)
        if part.strip() and news_count:
            uncited, _ = check_citations(part, news_count)
        # Numer spoza listy i nazwa spoza cytowanej wiadomości są błędem w każdej sekcji.
        bad = sorted({n for n in CITATION.findall(text) if not 1 <= int(n) <= news_count}, key=int)
        if news:
            content += check_news_content(text, news, names)
    interpretations = check_interpretations(text, facts)
    flagged = {i.sentence for i in interpretations}
    interpretations += [
        InterpretationIssue(c.sentence, c.message) for c in content if c.sentence not in flagged
    ]

    return Verification(
        rendered=render(text, facts),
        unknown_ids=unknown,
        unverified_numbers=unverified,
        wrong_years=wrong_years,
        wrong_dates=wrong_dates,
        interpretations=interpretations,
        checked_numbers=checked,
        suggestions=suggestions,
        repeated=repeated,
        contexts=contexts,
        uncited=uncited,
        bad_citations=bad,
        news_count=news_count or 0,
    )


def cite(text: str, news: list[dict]) -> str:
    """Numery wiadomości [n] jako krótkie odnośniki i lista źródeł pod sekcją wiadomości.

    Wiadomości są numerowane od nowa w kolejności pierwszego cytowania, a pod sekcją
    wiadomości pojawia się lista: źródło, data i nagłówek z odnośnikiem do artykułu.
    """
    order: list[int] = []
    for match in CITATION.finditer(text):
        n = int(match.group(1))
        if 1 <= n <= len(news) and n not in order:
            order.append(n)
    number = {n: i for i, n in enumerate(order, start=1)}

    def link(match: re.Match) -> str:
        n = int(match.group(1))
        if n not in number:
            return match.group(0)
        url = news[n - 1].get("url")
        return f"[[{number[n]}]]({url})" if url else f"[{number[n]}]"

    cited = CITATION.sub(link, text)
    if not order:
        return cited
    lines = []
    for n in order:
        item = news[n - 1]
        date = str(item.get("published", ""))[:10]
        headline = re.sub(r"[\[\]]", "", str(item.get("headline", ""))).strip()
        label = f"{item.get('source', 'źródło')}, {date}: {headline}"
        url = item.get("url")
        lines.append(f"{number[n]}. [{label}]({url})" if url else f"{number[n]}. {label}")
    sources = "\n\n*Źródła wiadomości:*\n\n" + "\n".join(lines) + "\n"
    start = cited.find(NEWS_SECTION)
    if start < 0:
        return cited.rstrip() + sources
    end = cited.find("\n## ", start + len(NEWS_SECTION))
    if end < 0:
        return cited.rstrip() + sources
    return cited[:end].rstrip() + sources + cited[end:]


SOURCE_MARKER = " [brak źródła]"


# Jednostka dopisana przez model zaraz po odwołaniu, które już ją zawiera („{{close}} USD”).
TRAILING_UNIT = re.compile(r"\{\{\s*([A-Za-z0-9_]+)\s*\}\}(\s*(?:USD|%|mld USD|p\.p\.))?(?![\w])")


def render(text: str, facts: FactSheet, html: bool = False) -> str:
    """Podstawia wartości w miejsce odwołań.

    W wersji HTML liczba ma podpowiedź ze źródłem w atrybucie data-zrodlo, którą pokazuje CSS
    po kliknięciu lub najechaniu; tekst podpowiedzi nie trafia do treści raportu.
    """

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
            value = f'<span class="fact" tabindex="0" data-zrodlo="{hint}">{escape(value)}</span>'
        return value + trailing

    return TRAILING_UNIT.sub(substitute, PLACEHOLDER_IN_CODE.sub(r"{{\1}}", text))


HTML_TAG = re.compile(r"(<[^>]*>)")


def mark_unverified(rendered: str, numbers: list[str]) -> str:
    """Oznacza liczby, których nie udało się zweryfikować (z pominięciem znaczników HTML)."""
    parts = HTML_TAG.split(rendered)
    for raw in sorted(set(numbers), key=len, reverse=True):
        # Bez dopasowań wewnątrz dat w rodzaju 2026-10-02 i części innej liczby („61” w „61,2”).
        pattern = re.compile(rf"(?<![\w.,])(?<!\d-){re.escape(raw)}(?![\w]|-\d|[.,]\d)")
        parts = [
            p if HTML_TAG.fullmatch(p) else pattern.sub(f"{raw} [niezweryfikowane]", p)
            for p in parts
        ]
    return "".join(parts)
