"""Weryfikacja liczb w raporcie względem karty faktów.

Działa bez modelu językowego, więc jej wynik jest powtarzalny. Sprawdza, czy:
1. każde odwołanie {{identyfikator}} wskazuje istniejący fakt,
2. w tekście nie ma liczb wpisanych poza odwołaniami, z wyjątkiem okresów wskaźników
   (np. „RSI(14)”), liczb z opisów faktów i nazw indeksów. Liczba wpisana wprost jest
   błędem nawet wtedy, gdy równa się wartości z karty, bo nie wiadomo, której wielkości
   dotyczy: przy dopuszczeniu liczb zgodnych z faktem z tego samego zdania test
   wstrzykniętych błędów pokazał, że podmiana na wartość sąsiedniego faktu (np. SMA(50)
   zamiast SMA(20)) przechodzi niezauważona. Zgodność z kartą służy tylko do podpowiedzi,
   którego odwołania użyć przy poprawce,
3. lata w tekście odpowiadają datom z karty faktów,
4. słowny opis liczby i okres zmiany zgadzają się z kartą (moduł interpretation),
5. każde zdanie o wiadomościach wskazuje źródło numerem z listy newsów, np. [2].
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html import escape

from marketagents.facts import Fact, FactSheet
from marketagents.interpretation import InterpretationIssue, check_interpretations

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
# Odwołanie do wiadomości z listy newsów, np. [3].
CITATION = re.compile(r"\[(\d{1,2})\]")
NEWS_SECTION = "## Wiadomości"
LIST_MARKER = re.compile(r"^\s*\d+[.)]\s", re.MULTILINE)
INDEX_NAME = re.compile(r"NASDAQ[- ]?100|S&P[- ]?500|Russell[- ]?2000", re.IGNORECASE)
# Koniec zdania: znak końca i odstęp albo nowy wiersz. Kropka po skrócie („ok. 61”, „np. RSI”)
# nie kończy zdania, żeby podpowiedź wskazała fakt, przy którym stoi liczba.
SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")
ABBREVIATION = re.compile(
    r"(?<![\w.])(?:ok|np|m\.in|tj|tzn|tzw|wg|ang|por|zob|ca|mln|mld|tys|pkt|proc|r)\.$",
    re.IGNORECASE,
)

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
    interpretations: list[InterpretationIssue] = field(default_factory=list)
    checked_numbers: int = 0
    # Liczba wpisana wprost -> fakt o tej wartości (podpowiedź dla autora tekstu).
    suggestions: dict[str, str] = field(default_factory=dict)
    # Liczby, które powtarzają wartość faktu, do którego odwołuje się to samo zdanie.
    repeated: list[str] = field(default_factory=list)
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
        return self.unverified_numbers + self.wrong_years

    def issues(self) -> list[str]:
        out = [f"Nieznany identyfikator faktu: {{{{{i}}}}}" for i in self.unknown_ids]
        for n in self.unverified_numbers:
            fact_id = self.suggestions.get(n)
            if n in self.repeated:
                out.append(
                    f"Liczba „{n}” powtarza bez odwołania wartość {{{{{fact_id}}}}}, "
                    "która już stoi w zdaniu; usuń ją"
                )
            elif fact_id:
                out.append(
                    f"Liczba „{n}” jest wpisana bez odwołania; jeśli to wartość "
                    f"{{{{{fact_id}}}}}, wstaw w jej miejsce odwołanie {{{{{fact_id}}}}}"
                )
            else:
                out.append(
                    f"Liczba „{n}” nie pochodzi z karty faktów; "
                    "zastąp ją odwołaniem {{identyfikator}}"
                )
        out += [
            f"Rok {y} nie odpowiada żadnej dacie w karcie faktów; "
            "sprawdź okres, którego dotyczy zdanie"
            for y in self.wrong_years
        ]
        out += [i.message for i in self.interpretations]
        out += [
            f"Zdanie „{_short(s)}” nie wskazuje źródła; dodaj numer wiadomości, np. [1]"
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


def check_citations(text: str, count: int) -> tuple[list[str], list[str]]:
    """Zdania bez numeru źródła oraz numery spoza listy (1..count)."""
    uncited, bad = [], []
    for sentence in sentences(text):
        if sentence.lstrip().startswith("#") or len(sentence.split()) < 4:
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


def verify(
    text: str,
    facts: FactSheet,
    allowed: frozenset[int] = DEFAULT_ALLOWED,
    news_count: int | None = None,
    whole_text_is_news: bool = False,
) -> Verification:
    """news_count: liczba wiadomości na liście; gdy podana, sprawdzane są numery źródeł
    w sekcji wiadomości raportu albo (whole_text_is_news) w całym tekście analizy newsów."""
    allowed = allowed | indicator_periods(facts) | label_numbers(facts)
    unknown = sorted({m.group(1) for m in PLACEHOLDER.finditer(text) if m.group(1) not in facts})
    years = fact_years(facts)

    wrong_years: list[str] = []
    unverified: list[str] = []
    suggestions: dict[str, str] = {}
    repeated: list[str] = []
    checked = 0
    for sentence in sentences(LIST_MARKER.sub(" ", text)):
        referenced = [facts[i] for i in PLACEHOLDER.findall(sentence) if i in facts]
        stripped = INDEX_NAME.sub(" ", ISO_DATE.sub(" ", PLACEHOLDER.sub(" ", sentence)))
        stripped = CITATION.sub(" ", stripped)
        wrong_years += [m.group(0) for m in YEAR.finditer(stripped) if int(m.group(0)) not in years]
        for match in NUMBER.finditer(YEAR.sub(" ", stripped)):
            raw = match.group(0)
            checked += 1
            # Okres wskaźnika to liczba całkowita bez znaku („SMA 50”), a nie np. „50,0”.
            if raw.isdigit() and int(raw) in allowed:
                continue
            unverified.append(raw)
            value = parse_number(raw)
            same = next((f for f in referenced if matches_fact(value, f)), None)
            other = same or next((f for f in facts if matches_fact(value, f)), None)
            if other is not None:
                suggestions[raw] = other.id
            if same is not None:
                repeated.append(raw)

    uncited: list[str] = []
    bad: list[str] = []
    if news_count is not None:
        part = text if whole_text_is_news else news_part(text)
        if part.strip():
            uncited, bad = check_citations(part, news_count)

    return Verification(
        rendered=render(text, facts),
        unknown_ids=unknown,
        unverified_numbers=unverified,
        wrong_years=wrong_years,
        interpretations=check_interpretations(text, facts),
        checked_numbers=checked,
        suggestions=suggestions,
        repeated=repeated,
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
