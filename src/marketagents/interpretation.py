"""Sprawdzanie słownej interpretacji liczb w raporcie.

Weryfikator liczb pilnuje, aby wartości pochodziły z karty faktów, ale nie wykrywa zdań,
w których opis przeczy liczbie, np. „kurs wzrósł o {{change_1m}}” przy ujemnej zmianie.
Reguły są celowo zachowawcze: zdanie jest sprawdzane tylko wtedy, gdy jednoznacznie
odnosi się do jednej wartości, a opis nie jest zanegowany ani złagodzony.

W ten sam sposób sprawdzany jest okres zmiany: zdanie „w ciągu roku kurs spadł
o {{change_1m}}” przypisuje zmianie z ostatniego miesiąca niewłaściwy okres.

Oceny, które wymagają porównania kilku wartości (położenie kursu we wstęgach i w zakresie
z 52 tygodni, układ średnich, odległość od średniej na tle historii, szerokość wstęg,
MACD względem linii sygnału), wyznacza kod przy tworzeniu karty faktów. Reguły poniżej
sprawdzają, czy słowa w tekście się z nimi zgadzają. Sprawdzane są też zdania, które
opisują zmianę w czasie (wzrost, utrzymywanie się, kontynuację) wielkości podanej w karcie
jednym odczytem, bez faktu o jej zmianie, oraz odwołania stojące przy nazwie innego
wskaźnika („SMA(50) wynosi {{sma_200}}”).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from marketagents.facts import FactSheet
from marketagents.sentences import split as split_sentences

PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z0-9_]+)\s*\}\}")

# Fakty, których znak określa kierunek zmiany.
DIRECTIONAL = re.compile(r"^(?:change_\w+|fy_\w+_growth(?:_prev)?|macro_\w+_change_12m)$")
AVERAGE = re.compile(r"^(?:sma|ema)_\d+$")
DISTANCE = re.compile(r"^dist_sma_\d+$")

UP = re.compile(
    r"(?<!nie )\b(?:wzrost(?!ow)\w*|wzr[oó]s[łl]\w*|rośnie|rosną|zyskał\w*|zwyżkował\w*"
    r"|podrożał\w*|zwiększ\w*|umocni\w*|podwyższ\w*|podwyżk\w*|przyrost\w*)",
    re.IGNORECASE,
)
DOWN = re.compile(
    r"(?<!nie )\b(?:spad(?!kow)\w*|stracił\w*|zniżkował\w*|potaniał\w*|zmalał\w*"
    r"|zmniejsz\w*|obniż\w*|osłab\w*|skurcz\w*|przecen\w*)",
    re.IGNORECASE,
)
ABOVE = re.compile(r"\b(?:powyżej|ponad|nad)\b", re.IGNORECASE)
BELOW = re.compile(r"\b(?:poniżej|pod)\b", re.IGNORECASE)
OVERBOUGHT = re.compile(r"(?<!nie )\bwykupi[eo]n\w*", re.IGNORECASE)
OVERSOLD = re.compile(r"(?<!nie )\bwyprzedan\w*", re.IGNORECASE)
# Sformułowania łagodzące lub opisujące odległość od progu („zbliża się do wykupienia”).
HEDGE = re.compile(
    r"\b(?:nie|brak\w*|zbliż\w*|blisko|pobliż\w*|granic\w*|okolic\w*|daleko|poza|poniżej|powyżej)\b",
    re.IGNORECASE,
)
RSI_LOW, RSI_HIGH = 30, 70
NEGATION = re.compile(r"\bnie\b", re.IGNORECASE)

MONTH, QUARTER, YEAR = "miesiąc", "3 miesiące", "rok"
# Okres, którego dotyczy fakt. Sprawdzane są zmiany; pozostałe fakty z okresem służą tylko
# do pominięcia zdań, w których stoją obok siebie wielkości z różnych okresów.
PERIODS = {
    "change_1m": MONTH,
    "change_3m": QUARTER,
    "change_1y": YEAR,
    "fy_revenue_growth": YEAR,
    "high_52w": YEAR,
    "low_52w": YEAR,
}
PERIOD_CHECKED = frozenset({"change_1m", "change_3m", "change_1y", "fy_revenue_growth"})
# Wielkości podawane jednym odczytem (poziom, a nie zmiana).
LEVEL = re.compile(
    r"^(?:macro_(?:fed_rate|10y|unemployment|inflation)|fy_(?:revenue|net_income|ocf|eps|roe"
    r"|net_margin|operating_margin|gross_margin|liabilities_to_assets)|pe_ratio)$"
)
# Słowa opisujące zmianę w czasie lub jej trwanie.
CHANGE_WORDS = re.compile(
    r"\b(?:utrzymuj\w*|pozostaj\w*|nadal|wciąż|kontynu\w*|stabilizuj\w*|ustabiliz\w*"
    r"|przyspiesz\w*|spowalni\w*|zwalnia\w*|rosnąc\w*|malejąc\w*|rośnie|rosną|spada|spadają"
    r"|wzrost\w*|wzrósł|wzrosł\w*|spadek|spadk\w*|spadł\w*|zmalał\w*|poprawi\w*|pogorsz\w*)\b",
    re.IGNORECASE,
)
# Słowa, które przypisują zmianie trwanie dłuższe niż jeden okres.
PERSISTENCE = re.compile(r"\b(?:kontynu\w*|nadal|wciąż|utrzymuj\w*|kolejn\w*)\b", re.IGNORECASE)
# Kolejność ma znaczenie: „3 miesiące” i „12 miesięcy” są usuwane przed szukaniem „miesiąca”.
PERIOD_WORDS = [
    (QUARTER, r"(?:3|trzy|trzech)[\s-]+miesi\w*|trzymiesięczn\w*|kwarta\w*"),
    (YEAR, r"rok|roku|rokiem|roczn\w*|(?:12|dwunast\w*)[\s-]+miesi\w*|52[\s-]+tygod\w*|r/r"),
    (MONTH, r"miesiąc|miesiąca|miesiącu|miesiącem|miesięczn\w*"),
]


@dataclass(frozen=True)
class InterpretationIssue:
    sentence: str
    message: str


def sentences(text: str) -> list[str]:
    return [s.strip() for s in split_sentences(text)]


def _short(sentence: str) -> str:
    plain = re.sub(r"\s+", " ", sentence)
    return plain if len(plain) <= 90 else plain[:87] + "..."


def _issue(sentence: str, fact_id: str, facts: FactSheet, why: str) -> InterpretationIssue:
    value = facts[fact_id].formatted()
    message = (
        f"Opis „{_short(sentence)}” nie zgadza się z wartością {{{{{fact_id}}}}} = {value}: "
        f"{why}; popraw interpretację"
    )
    return InterpretationIssue(sentence, message)


def _numeric(facts: FactSheet, fact_id: str) -> float | None:
    if fact_id not in facts:
        return None
    value = facts[fact_id].value
    return None if isinstance(value, str) else float(value)


def _check_direction(
    sentence: str, refs: list[str], facts: FactSheet
) -> InterpretationIssue | None:
    directional = [r for r in refs if DIRECTIONAL.match(r) and _numeric(facts, r) is not None]
    if len(directional) == 1:
        return _direction_issue(sentence, sentence, directional[0], facts)
    if not directional:
        return None
    # Kilka zmian w jednym zdaniu („w roku wzrósł o …, a w miesiącu spadł o …”): kierunek
    # sprawdzany jest w każdym członie zdania, który zawiera dokładnie jedną zmianę.
    for clause in CLAUSE.split(sentence):
        in_clause = [r for r in PLACEHOLDER.findall(clause) if r in directional]
        if len(in_clause) == 1:
            issue = _direction_issue(clause, sentence, in_clause[0], facts)
            if issue:
                return issue
    return None


def _direction_issue(
    part: str, sentence: str, fact_id: str, facts: FactSheet
) -> InterpretationIssue | None:
    value = _numeric(facts, fact_id)
    up, down = bool(UP.search(part)), bool(DOWN.search(part))
    if up and not down and value < 0:
        return _issue(sentence, fact_id, facts, "zdanie opisuje wzrost, a wartość jest ujemna")
    if down and not up and value > 0:
        return _issue(sentence, fact_id, facts, "zdanie opisuje spadek, a wartość jest dodatnia")
    return None


def _check_rsi(sentence: str, refs: list[str], facts: FactSheet) -> InterpretationIssue | None:
    value = _numeric(facts, "rsi_14")
    if "rsi_14" not in refs or value is None or HEDGE.search(sentence):
        return None
    if OVERBOUGHT.search(sentence) and value <= RSI_HIGH:
        why = f"wykupienie oznacza RSI powyżej {RSI_HIGH}"
        return _issue(sentence, "rsi_14", facts, why)
    if OVERSOLD.search(sentence) and value >= RSI_LOW:
        why = f"wyprzedanie oznacza RSI poniżej {RSI_LOW}"
        return _issue(sentence, "rsi_14", facts, why)
    return None


def _check_position(sentence: str, refs: list[str], facts: FactSheet) -> InterpretationIssue | None:
    """Kurs powyżej lub poniżej średniej: zdanie z jedną średnią (lub odległością od niej)."""
    averages = [r for r in refs if AVERAGE.match(r) or DISTANCE.match(r)]
    if len(averages) != 1 or "kurs" not in sentence.lower():
        return None
    above, below = bool(ABOVE.search(sentence)), bool(BELOW.search(sentence))
    if above == below:
        return None
    fact_id = averages[0]
    if DISTANCE.match(fact_id):
        diff = _numeric(facts, fact_id)
    else:
        close, average = _numeric(facts, "close"), _numeric(facts, fact_id)
        diff = None if close is None or average is None else close - average
    if diff is None:
        return None
    if above and diff < 0:
        return _issue(sentence, fact_id, facts, "kurs jest poniżej tej średniej")
    if below and diff > 0:
        return _issue(sentence, fact_id, facts, "kurs jest powyżej tej średniej")
    return None


# Zmiana w czasie dla wielkości podanej jednym odczytem: poziom -> fakt o jego zmianie.
CHANGE_FACT = {
    "fy_revenue": "fy_revenue_growth",
    "fy_net_income": "fy_net_income_growth",
    "macro_fed_rate": "macro_fed_rate_change_12m",
    "macro_10y": "macro_10y_change_12m",
    "macro_unemployment": "macro_unemployment_change_12m",
    "macro_inflation": "macro_inflation_change_12m",
}


def _check_change_claim(
    sentence: str, refs: list[str], facts: FactSheet
) -> InterpretationIssue | None:
    """Opis zmiany („rośnie”, „utrzymuje się”) przy wielkości podanej jednym odczytem."""
    if any(DIRECTIONAL.match(r) for r in refs):
        return None
    levels = [r for r in refs if LEVEL.match(r) and r in facts]
    match = CHANGE_WORDS.search(sentence)
    if not levels or not match:
        return None
    fact_id = levels[0]
    hint = next((CHANGE_FACT[r] for r in levels if CHANGE_FACT.get(r) in facts), None)
    why = f"słowo „{match.group(0)}” opisuje zmianę w czasie, a karta podaje jeden odczyt" + (
        f"; zmianę opisuje fakt {{{{{hint}}}}}" if hint else "; nie opisuj zmiany"
    )
    return _issue(sentence, fact_id, facts, why)


def _check_persistence(
    sentence: str, refs: list[str], facts: FactSheet
) -> InterpretationIssue | None:
    """„Kontynuacja wzrostu” na podstawie zmiany z jednego roku obrotowego."""
    if "fy_revenue_growth" not in refs or "fy_revenue_growth_prev" in refs:
        return None
    match = PERSISTENCE.search(sentence)
    if not match or NEGATION.search(sentence):
        return None
    why = f"„{match.group(0)}” wymaga porównania z poprzednim rokiem"
    if "fy_revenue_growth_prev" in facts:
        why += "; użyj także {{fy_revenue_growth_prev}}"
    return _issue(sentence, "fy_revenue_growth", facts, why)


# Części zdania: oceny położenia sprawdzane są w obrębie jednego członu, bo zdanie bywa
# o kilku wskaźnikach naraz („…powyżej dolnej wstęgi, w górnej części zakresu z 52 tygodni”).
CLAUSE = re.compile(
    r"[,;–—]|\s-\s|\boraz\b|\ba\b|\bnatomiast\b|\bjednak\w*\b|\bprzy\s+czym\b", re.IGNORECASE
)
PARENS = re.compile(r"\([^)]*\)")
NEAR = re.compile(
    r"\b(?:blisk\w*|bliżej|pobliż\w*|przy|okolic\w*|częś\w*|połow\w*|granic\w*|sąsiedztw\w*)\b",
    re.IGNORECASE,
)
LOW_SIDE = re.compile(r"\b(?:doln\w*|minim\w*|dołk\w*|najniższ\w*|dna)\b", re.IGNORECASE)
HIGH_SIDE = re.compile(r"\b(?:górn\w*|maksim\w*|szczyt\w*|najwyższ\w*)\b", re.IGNORECASE)
MIDDLE = re.compile(r"\bśrod(?:ek|ka|ku|kow\w*)\b", re.IGNORECASE)
# Słowo, które nazywa granicę stojącą przy jej odwołaniu („przy maksimum wynoszącym
# {{high_52w}}”, „najwyższy kurs zamknięcia z 52 tygodni wynosi {{high_52w}}”), nie opisuje
# położenia kursu.
BOUNDARY_NAME = re.compile(
    r"\b(?:doln\w*|minim\w*|najniższ\w*|górn\w*|maksim\w*|najwyższ\w*)\s+(?:[^\s{}]+\s+){0,6}"
    r"\{\{\s*(?:high_52w|low_52w|bb_upper|bb_lower)\s*\}\}",
    re.IGNORECASE,
)
BANDS = re.compile(r"wstęg|bollinger|\{\{\s*bb_", re.IGNORECASE)
RANGE = re.compile(r"52|zakres\w*|roczn\w*|\{\{\s*(?:high|low)_52w|\{\{\s*range_52w", re.IGNORECASE)


def clauses(sentence: str) -> list[str]:
    return [c for c in CLAUSE.split(PARENS.sub(" ", sentence)) if c.strip()]


def _zone_issue(clause: str, sentence: str, fact_id: str, facts: FactSheet):
    position = _numeric(facts, fact_id)
    if position is None or NEGATION.search(clause):
        return None
    clause = BOUNDARY_NAME.sub(" ", clause)
    low, high = bool(LOW_SIDE.search(clause)), bool(HIGH_SIDE.search(clause))
    zone = facts[fact_id].assessment
    if MIDDLE.search(clause) and not (low or high):
        if not 30 <= position <= 70:
            return _issue(sentence, fact_id, facts, f"kurs nie jest w środku ({zone})")
        return None
    if not NEAR.search(clause) or low == high:
        return None
    if low and position > 50:
        return _issue(sentence, fact_id, facts, f"kurs jest bliżej górnej granicy ({zone})")
    if high and position < 50:
        return _issue(sentence, fact_id, facts, f"kurs jest bliżej dolnej granicy ({zone})")
    return None


def _check_zones(sentence: str, refs: list[str], facts: FactSheet) -> InterpretationIssue | None:
    """Położenie kursu we wstęgach Bollingera i w zakresie z 52 tygodni."""
    for clause in clauses(sentence):
        if BANDS.search(clause) and "bb_position" in facts:
            issue = _zone_issue(clause, sentence, "bb_position", facts)
        elif RANGE.search(clause) and "range_52w_position" in facts:
            issue = _zone_issue(clause, sentence, "range_52w_position", facts)
        else:
            continue
        if issue:
            return issue
    return None


SHORTER = re.compile(r"\bkrótsz\w*", re.IGNORECASE)
LONGER = re.compile(r"\bdłuższ\w*", re.IGNORECASE)
HIGHER = re.compile(r"\b(?:powyżej|ponad|nad|wyżej|wyższ\w*)\b", re.IGNORECASE)
LOWER = re.compile(r"\b(?:poniżej|pod|niżej|niższ\w*)\b", re.IGNORECASE)


def _check_ma_order(sentence: str, refs: list[str], facts: FactSheet) -> InterpretationIssue | None:
    """„Krótsza średnia nad dłuższą” tylko przy uporządkowanym układzie średnich."""
    if "ma_order" not in facts:
        return None
    order = facts["ma_order"].assessment
    for clause in clauses(sentence):
        if not (SHORTER.search(clause) and LONGER.search(clause)) or NEGATION.search(clause):
            continue
        above, below = bool(HIGHER.search(clause)), bool(LOWER.search(clause))
        if above and not below and "wzrostowy" not in order:
            return _issue(sentence, "ma_order", facts, f"układ średnich: {order}")
        if below and not above and "spadkowy" not in order:
            return _issue(sentence, "ma_order", facts, f"układ średnich: {order}")
    return None


SIGNAL = re.compile(r"sygnał\w*|\{\{\s*macd_signal", re.IGNORECASE)


def _check_macd(sentence: str, refs: list[str], facts: FactSheet) -> InterpretationIssue | None:
    """Linia MACD powyżej lub poniżej linii sygnału zgodnie ze znakiem histogramu."""
    hist = _numeric(facts, "macd_hist")
    if hist is None or not hist:
        return None
    for clause in clauses(sentence):
        if not (SIGNAL.search(clause) and re.search(r"macd", clause, re.IGNORECASE)):
            continue
        if NEGATION.search(clause):
            continue
        above, below = bool(HIGHER.search(clause)), bool(LOWER.search(clause))
        if above and not below and hist < 0:
            return _issue(sentence, "macd_hist", facts, "linia MACD jest poniżej linii sygnału")
        if below and not above and hist > 0:
            return _issue(sentence, "macd_hist", facts, "linia MACD jest powyżej linii sygnału")
    return None


LARGE = re.compile(
    r"\b(?:duż\w*|znaczn\w*|wyraźn\w*|silne\w*|mocn\w*|rozstaw\w*|oddal\w*)\b", re.IGNORECASE
)


def _check_distance(sentence: str, refs: list[str], facts: FactSheet) -> InterpretationIssue | None:
    """Duża odległość od długiej średniej tylko wtedy, gdy jest nietypowa na tle historii."""
    distance = [r for r in refs if DISTANCE.match(r) and r in facts]
    if not distance or NEGATION.search(sentence):
        return None
    fact_id = distance[0]
    assessment = facts[fact_id].assessment
    match = LARGE.search(sentence)
    sized = "odległoś" in assessment
    if match and sized and "nietypowo duż" not in assessment:
        why = f"„{match.group(0)}” sugeruje dużą odległość, a ocena to: {assessment}"
        return _issue(sentence, fact_id, facts, why)
    return None


# Przymiotniki i czasowniki oceniające szerokość; rzeczownik „szerokość” sam niczego nie ocenia
# („szerokość wstęg wynosi …”), więc nie jest tu dopasowywany.
NARROW = re.compile(
    r"\b(?:wąsk(?:i|ie|a|ą|ich|imi|im)|węż(?:sz\w*)|zwęż\w*|ściśnięt\w*)\b", re.IGNORECASE
)
WIDE = re.compile(r"\b(?:szerok(?:i|ie|a|ą|ich|imi|im)|szersz\w*|rozszerz\w*)\b", re.IGNORECASE)


def _check_band_width(
    sentence: str, refs: list[str], facts: FactSheet
) -> InterpretationIssue | None:
    if "bb_width" not in facts:
        return None
    assessment = facts["bb_width"].assessment
    for clause in clauses(sentence):
        if not BANDS.search(clause) or NEGATION.search(clause):
            continue
        if NARROW.search(clause) and "węższe" not in assessment:
            return _issue(sentence, "bb_width", facts, assessment)
        if WIDE.search(clause) and "szersze" not in assessment:
            return _issue(sentence, "bb_width", facts, assessment)
    return None


# Nazwa wskaźnika tuż przed odwołaniem („SMA(50) wynosi {{sma_50}}”) i fakty, do których
# może się ono odnosić. Odwołanie do innego wskaźnika („SMA(50) wynosi {{sma_200}}”) podaje
# wartość z karty, ale nie tej wielkości, o której mówi zdanie.
INDICATOR_MENTION = re.compile(
    r"\b(?P<avg>SMA|EMA)\s*\(?\s*(?P<periods>\d+(?:\s*(?:,|i|oraz)\s*\d+)*)\s*\)?"
    r"|\b(?P<rsi>RSI)\b|\b(?P<signal>sygnał\w*)|\b(?P<hist>histogram\w*)|\b(?P<macd>MACD)\b"
    r"|\b(?P<upper>górn\w*)\s+wstęg\w*|\b(?P<lower>doln\w*)\s+wstęg\w*",
    re.IGNORECASE,
)
INDICATOR_FACT = re.compile(r"^(?:(?:dist_)?sma_\d+|ema_\d+|rsi_14|macd\w*|bb_(?:upper|lower))$")
MAX_GAP = 30
NAME_JOIN = re.compile(r"\s*(?:,|i|oraz)\s*", re.IGNORECASE)


def _mentioned_facts(match: re.Match) -> tuple[str, set[str]]:
    """Nazwa wskaźnika z tekstu i identyfikatory faktów, które mogą jej odpowiadać."""
    if match["avg"]:
        kind = match["avg"].lower()
        periods = re.findall(r"\d+", match["periods"])
        ids = {f"{kind}_{n}" for n in periods}
        if kind == "sma":
            ids |= {f"dist_sma_{n}" for n in periods}
        return match.group(0).strip(), ids
    if match["rsi"]:
        return "RSI", {"rsi_14"}
    if match["signal"]:
        return "linia sygnału", {"macd_signal"}
    if match["hist"]:
        return "histogram MACD", {"macd_hist"}
    if match["macd"]:
        return "MACD", {"macd", "macd_signal", "macd_hist"}
    if match["upper"]:
        return "górna wstęga", {"bb_upper"}
    return "dolna wstęga", {"bb_lower"}


def _check_indicator_ref(
    sentence: str, refs: list[str], facts: FactSheet
) -> InterpretationIssue | None:
    """Odwołanie stojące zaraz po nazwie wskaźnika musi dotyczyć tego wskaźnika."""
    start = 0
    for ref in PLACEHOLDER.finditer(sentence):
        before, start = sentence[start : ref.start()], ref.end()
        fact_id = ref.group(1)
        if fact_id not in facts or not INDICATOR_FACT.match(fact_id):
            continue
        mentions = list(INDICATOR_MENTION.finditer(before))
        if not mentions or len(before) - mentions[-1].end() > MAX_GAP:
            continue
        # Kilka nazw wymienionych razem („RSI i MACD wynoszą {{rsi_14}} i {{macd}}”).
        named = [mentions.pop()]
        while mentions and NAME_JOIN.fullmatch(before[mentions[-1].end() : named[-1].start()]):
            named.append(mentions.pop())
        found = [_mentioned_facts(m) for m in reversed(named)]
        name = " i ".join(n for n, _ in found)
        expected = set().union(*(ids for _, ids in found))
        if fact_id not in expected:
            why = f"odwołanie stoi przy nazwie „{name}”, a dotyczy innego wskaźnika"
            return _issue(sentence, fact_id, facts, why)
    return None


# Nazwy wielkości finansowych i makroekonomicznych oraz przedrostki identyfikatorów faktów,
# które mogą im odpowiadać. Nazwy wieloznaczne („rentowność”, „zysk”) dopuszczają kilka
# wielkości. Kolejność ma znaczenie: dłuższa nazwa stoi przed krótszą.
QUANTITIES = [
    (r"cen\w*\s*(?:do|/)\s*zysk\w*|\bC/Z\b|\bP/E\b", ("pe_ratio",)),
    (r"(?:stop\w*\s+)?bezroboci\w*", ("macro_unemployment",)),
    (
        r"stop\w*\s+(?:procentow\w*|funduszy\s+federaln\w*|Fed\w*|referencyjn\w*)"
        r"|\bRezerw\w*\s+Federaln\w*|\bFed\b",
        ("macro_fed_rate",),
    ),
    (r"obligacj\w*", ("macro_10y",)),
    (r"inflacj\w*|\bCPI\b|cen\w*\s+konsument\w*|cen\w*\s+konsumpcyjn\w*", ("macro_inflation",)),
    (r"rentowno\w*\s+kapitał\w*|\bROE\b", ("fy_roe",)),
    (r"marż\w*\s+(?:zysku\s+)?netto", ("fy_net_margin",)),
    (r"marż\w*\s+(?:zysku\s+)?operacyjn\w*", ("fy_operating_margin",)),
    (r"marż\w*\s+(?:zysku\s+)?brutto", ("fy_gross_margin",)),
    (r"marż\w*", ("fy_net_margin", "fy_operating_margin", "fy_gross_margin")),
    (
        r"zysk\w*\s+na\s+(?:jedn\w*\s+)?akcj\w*|\bEPS\b",
        ("fy_eps", "eps_diluted", "eps_filed"),
    ),
    (r"(?:zysk|wynik|doch[oó]d)\w*\s+netto", ("fy_net_income", "fy_net_margin")),
    (r"zysk\w*\s+operacyjn\w*", ("fy_operating_margin",)),
    (r"zysk\w*\s+brutto", ("fy_gross_margin",)),
    (
        r"zysk\w*",
        ("fy_net_income", "fy_net_margin", "fy_operating_margin", "fy_gross_margin", "fy_eps"),
    ),
    (r"rentowno\w*", ("macro_10y", "fy_roe", "fy_net_margin", "fy_operating_margin")),
    (r"przych[oó]d\w*|sprzedaż\w*", ("fy_revenue",)),
    (r"przepływ\w*", ("fy_ocf",)),
    (r"zadłużeni\w*|zobowiąza\w*", ("fy_liabilities_to_assets",)),
]
QUANTITY_MENTION = re.compile(
    "|".join(f"(?P<q{i}>(?<![\\w/])(?:{pattern}))" for i, (pattern, _) in enumerate(QUANTITIES)),
    re.IGNORECASE,
)
QUANTITY_PREFIXES = tuple(p for _, prefixes in QUANTITIES for p in prefixes)
QUANTITY_GAP = 40


def _quantity_prefixes(match: re.Match) -> tuple[str, ...]:
    return next(QUANTITIES[int(k[1:])][1] for k, v in match.groupdict().items() if v)


def _check_quantity_ref(
    sentence: str, refs: list[str], facts: FactSheet
) -> InterpretationIssue | None:
    """Odwołanie stojące zaraz po nazwie wielkości („stopa bezrobocia wynosi …”) musi
    dotyczyć tej wielkości. Zdanie, które wcześniej nazywa wielkość z odwołania, nie jest
    oceniane, bo odwołanie może stać dalej od swojej nazwy („zysk netto wzrósł szybciej niż
    przychody ({{fy_net_income_growth}} wobec {{fy_revenue_growth}})”)."""
    start = 0
    for ref in PLACEHOLDER.finditer(sentence):
        before, start = sentence[start : ref.start()], ref.end()
        fact_id = ref.group(1)
        if fact_id not in facts or not fact_id.startswith(QUANTITY_PREFIXES):
            continue
        mentions = list(QUANTITY_MENTION.finditer(before))
        if not mentions or len(before) - mentions[-1].end() > QUANTITY_GAP:
            continue
        # Kilka nazw wymienionych razem („stopa bezrobocia i inflacja wynoszą … i …”).
        named = [mentions.pop()]
        while mentions and NAME_JOIN.fullmatch(before[mentions[-1].end() : named[-1].start()]):
            named.append(mentions.pop())
        expected = {p for m in named for p in _quantity_prefixes(m)}
        if fact_id.startswith(tuple(expected)):
            continue
        earlier = QUANTITY_MENTION.finditer(sentence[: ref.start()])
        if any(fact_id.startswith(_quantity_prefixes(m)) for m in earlier):
            continue
        name = " i ".join(m.group(0) for m in reversed(named))
        why = f"odwołanie stoi przy nazwie „{name}”, a dotyczy innej wielkości"
        return _issue(sentence, fact_id, facts, why)
    return None


def periods_in(words: str) -> list[str]:
    """Okresy wymienione w tekście, np. [„rok”] dla „w ostatnim roku”."""
    found = []
    for period, pattern in PERIOD_WORDS:
        regex = re.compile(rf"(?<![\w])(?:{pattern})(?![\w])", re.IGNORECASE)
        if regex.search(words):
            found.append(period)
            words = regex.sub(" ", words)
    return found


def _check_period(sentence: str, refs: list[str], facts: FactSheet) -> InterpretationIssue | None:
    """Okres zmiany: zdanie z wielkościami z jednego okresu, bez przeczenia."""
    ids = [r for r in refs if r in facts and r in PERIODS]
    if len({PERIODS[r] for r in ids}) != 1 or NEGATION.search(sentence):
        return None
    checked = [r for r in ids if r in PERIOD_CHECKED]
    if not checked:
        return None
    fact_id, expected = checked[0], PERIODS[checked[0]]
    found = periods_in(PLACEHOLDER.sub(" ", sentence))
    if found and expected not in found:
        why = f"zdanie podaje okres „{found[0]}”, a wartość dotyczy okresu „{expected}”"
        return _issue(sentence, fact_id, facts, why)
    return None


def label(fact_id: str, facts: FactSheet) -> str | None:
    """Ocena słowna faktu wyznaczona przez kod (kolumna „ocena” w karcie dla modelu)."""
    if fact_id in facts and facts[fact_id].assessment:
        return facts[fact_id].assessment
    value = _numeric(facts, fact_id)
    if value is None:
        return None
    if DIRECTIONAL.match(fact_id):
        return "wzrost" if value > 0 else "spadek" if value < 0 else "bez zmian"
    if fact_id == "rsi_14":
        if value > RSI_HIGH:
            return "wykupienie"
        return "wyprzedanie" if value < RSI_LOW else "strefa neutralna"
    if fact_id in ("macd", "macd_hist"):
        hist = _numeric(facts, "macd_hist")
        if not hist:
            return None
        side = "powyżej" if hist > 0 else "poniżej"
        return f"linia MACD {side} linii sygnału"
    if DISTANCE.match(fact_id):
        diff = value
    elif AVERAGE.match(fact_id):
        close = _numeric(facts, "close")
        diff = None if close is None else close - value
    else:
        return None
    if not diff:
        return None
    return "kurs powyżej średniej" if diff > 0 else "kurs poniżej średniej"


def prompt_table(facts: FactSheet) -> str:
    """Karta faktów dla modelu z oceną słowną, której tekst ma się trzymać."""
    lines = ["| identyfikator | opis | wartość | ocena |", "| --- | --- | --- | --- |"]
    for f in facts:
        lines.append(
            f"| {{{{{f.id}}}}} | {f.label} | {f.formatted()} | {label(f.id, facts) or ''} |"
        )
    return "\n".join(lines)


# Słowa opisu faktu, które nie nazywają wielkości („zmiana kursu w ostatnim roku”).
LABEL_STOPWORDS = {
    "w",
    "z",
    "ze",
    "na",
    "od",
    "do",
    "i",
    "o",
    "a",
    "oraz",
    "dla",
    "po",
    "przed",
    "przy",
    "ostatnim",
    "ostatnich",
    "ostatniej",
    "ostatni",
    "ostatnie",
    "poprzednim",
    "roku",
    "rok",
    "lat",
    "dni",
    "sesji",
    "usa",
    "względem",
    "ciągu",
    "miesięcy",
    "miesiącach",
    "miesiącu",
}
# Opisy faktów, z których pochodzą nazwy wielkości (jak w karcie faktów); średnie kroczące
# mają okna ustawiane przez użytkownika, dlatego są rozpoznawane po początku identyfikatora.
CANONICAL_LABELS = {
    "close": "kurs zamknięcia",
    "high_52w": "najwyższy kurs zamknięcia z 52 tygodni",
    "low_52w": "najniższy kurs zamknięcia z 52 tygodni",
    "change_1m": "zmiana kursu w ostatnim miesiącu",
    "change_3m": "zmiana kursu w ostatnich 3 miesiącach",
    "change_1y": "zmiana kursu w ostatnim roku",
    "rsi_14": "RSI(14)",
    "macd": "linia MACD(12, 26)",
    "macd_signal": "linia sygnału MACD(9)",
    "macd_hist": "histogram MACD",
    "bb_upper": "górna wstęga Bollingera(20, 2)",
    "bb_lower": "dolna wstęga Bollingera(20, 2)",
    "dist_sma_200": "odległość kursu od SMA(200)",
    "bb_position": "położenie kursu między wstęgami Bollingera (0% = dolna, 100% = górna)",
    "bb_width": "szerokość wstęg Bollingera względem SMA(20)",
    "range_52w_position": "położenie kursu w zakresie z 52 tygodni (0% = minimum, 100% = maksimum)",
    "fy_revenue": "przychody w ostatnim roku obrotowym",
    "fy_net_income": "zysk netto w ostatnim roku obrotowym",
    "fy_ocf": "przepływy operacyjne w ostatnim roku obrotowym",
    "fy_revenue_growth": "zmiana przychodów rok do roku",
    "fy_revenue_growth_prev": "zmiana przychodów rok do roku w poprzednim roku obrotowym",
    "fy_net_income_growth": "zmiana zysku netto rok do roku",
    "fy_net_margin": "marża netto",
    "fy_operating_margin": "marża operacyjna",
    "fy_gross_margin": "marża brutto",
    "fy_liabilities_to_assets": "zobowiązania do aktywów",
    "fy_roe": "rentowność kapitału własnego (ROE)",
    "fy_eps": "rozwodniony zysk na akcję (EPS)",
    "pe_ratio": "cena do zysku (C/Z) z ostatniego roku",
    "macro_fed_rate": "stopa funduszy federalnych",
    "macro_fed_rate_change_12m": "zmiana stopy funduszy federalnych w ciągu 12 miesięcy",
    "macro_10y": "rentowność 10-letnich obligacji USA",
    "macro_10y_change_12m": "zmiana rentowności 10-letnich obligacji USA w ciągu 12 miesięcy",
    "macro_unemployment": "stopa bezrobocia w USA",
    "macro_inflation": "inflacja CPI rok do roku",
    "macro_inflation_change_12m": "zmiana inflacji CPI rok do roku w ciągu 12 miesięcy",
}
CANONICAL_PREFIXES = {"sma_": "średnia krocząca SMA", "ema_": "średnia wykładnicza EMA"}
# Inne nazwy kursu akcji („notowania wzrosły o …”, „cena akcji spadła o …”).
PRICE_SYNONYMS = {"noto", "akcj", "cena", "ceny", "cen"}
# Fakty, których nazwa musi zawierać słowo główne opisu: „SMA(200) wynosi {{dist_sma_200}}”
# podaje odległość od średniej jako jej wartość.
HEAD_STEMS = {
    "dist_": {"odle", "odda"},
    "bb_width": {"szer", "wąsk"},
    "macd_signal": {"sygn"},
    "macd_hist": {"hist"},
}
CHANGE_VERBS = {"wzró", "wzro", "spad", "zmie", "zysk", "stra", "obni", "podn"}
# Zdanie nawiązujące do poprzedniego („Wskaźnik ten wzrósł o …”, „jej zmiana wynosi …”).
POINTER = re.compile(r"\b(?:ten|ta|tego|tej|jej|jego|ich|tych|tym)\b", re.IGNORECASE)
NAME_WORD = re.compile(r"[^\W\d_][\w/]*", re.UNICODE)


def _stems(text: str) -> set[str]:
    words = (w.lower() for w in NAME_WORD.findall(text))
    # Cztery pierwsze litery obejmują polskie odmiany („kurs”, „kursem”, „kursu”).
    return {w[:4] for w in words if w not in LABEL_STOPWORDS and len(w) >= 2}


def _check_unnamed_ref(
    sentence: str, refs: list[str], facts: FactSheet
) -> InterpretationIssue | None:
    """Wartość liczbowa musi mieć w zdaniu nazwę wielkości, której dotyczy, np. „kurs jest
    poniżej {{sma_20}}, {{sma_50}} i {{ema_20}}” nie mówi, czym są te trzy kwoty. Nazwą jest
    dowolne słowo z opisu faktu („średnia”, „SMA”, „zysk”, „przychody”), więc wyliczenie
    po jednej nazwie („SMA(20) wynosi {{sma_20}}, a SMA(50) {{sma_50}}”) jest poprawne."""
    if POINTER.search(sentence):
        return None
    words = _stems(PLACEHOLDER.sub(" ", sentence))
    for fact_id in dict.fromkeys(refs):
        if fact_id not in facts or isinstance(facts[fact_id].value, str):
            continue
        name = CANONICAL_LABELS.get(fact_id) or next(
            (v for k, v in CANONICAL_PREFIXES.items() if fact_id.startswith(k)), None
        )
        if name is None:
            continue
        stems = _stems(name)
        if "kurs" in stems:
            stems |= PRICE_SYNONYMS
        if fact_id.startswith("change_"):
            # Podmiot bywa pominięty: „Kurs spadł w miesiącu o …, a w ciągu roku wzrósł o …”.
            stems |= CHANGE_VERBS
        head = next((h for k, h in HEAD_STEMS.items() if fact_id.startswith(k)), None)
        if head and re.search(rf"\bo\s+\{{\{{\s*{fact_id}\b", sentence):
            head = None  # „kurs jest powyżej SMA(200) o {{dist_sma_200}}”
        if not stems & words or (head and not head & words):
            name = facts[fact_id].label
            return InterpretationIssue(
                sentence,
                f"W zdaniu „{_short(sentence)}” wartość {{{{{fact_id}}}}} nie ma nazwy "
                f"wielkości, której dotyczy ({name}); napisz, czego dotyczy wartość, "
                f"np. „{name} wynosi {{{{{fact_id}}}}}”",
            )
    return None


# Reguły dla zdań z odwołaniami oraz reguły, które sprawdzają opis także bez odwołania
# (np. „kurs jest blisko dolnej wstęgi” bez podania wartości).
CHECKS_WITH_REFS = (
    _check_indicator_ref,
    _check_quantity_ref,
    _check_direction,
    _check_rsi,
    _check_position,
    _check_period,
    _check_change_claim,
    _check_persistence,
    _check_distance,
    _check_unnamed_ref,
)
CHECKS_ANY = (_check_zones, _check_ma_order, _check_macd, _check_band_width)


def check_interpretations(text: str, facts: FactSheet) -> list[InterpretationIssue]:
    issues: list[InterpretationIssue] = []
    for sentence in sentences(text):
        if sentence.lstrip().startswith("#"):
            continue
        refs = PLACEHOLDER.findall(sentence)
        checks = (CHECKS_WITH_REFS + CHECKS_ANY) if refs else CHECKS_ANY
        for check in checks:
            issue = check(sentence, refs, facts)
            if issue:
                issues.append(issue)
                break
    return issues


MARKER = " [niezweryfikowana interpretacja]"


def mark_sentences(text: str, flagged: list[str], marker: str = MARKER) -> str:
    """Dopisuje oznaczenie po zdaniach, których nie udało się poprawić."""
    for sentence in dict.fromkeys(flagged):
        text = text.replace(sentence, sentence + marker, 1)
    return text
