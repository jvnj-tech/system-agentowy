"""Sprawdzanie słownej interpretacji liczb w raporcie.

Weryfikator liczb pilnuje, aby wartości pochodziły z karty faktów, ale nie wykrywa zdań,
w których opis przeczy liczbie, np. „kurs wzrósł o {{change_1m}}” przy ujemnej zmianie.
Reguły są celowo zachowawcze: zdanie jest sprawdzane tylko wtedy, gdy jednoznacznie
odnosi się do jednej wartości, a opis nie jest zanegowany ani złagodzony.

W ten sam sposób sprawdzany jest okres zmiany: zdanie „w ciągu roku kurs spadł
o {{change_1m}}” przypisuje zmianie z ostatniego miesiąca niewłaściwy okres.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from marketagents.facts import FactSheet

PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z0-9_]+)\s*\}\}")
SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")

# Fakty, których znak określa kierunek zmiany.
DIRECTIONAL = re.compile(r"^(?:change_\w+|fy_revenue_growth)$")
AVERAGE = re.compile(r"^(?:sma|ema)_\d+$")
DISTANCE = re.compile(r"^dist_sma_\d+$")

UP = re.compile(
    r"(?<!nie )\b(?:wzrost(?!ow)\w*|wzr[oó]s[łl]\w*|rośnie|rosną|zyskał\w*|zwyżkował\w*"
    r"|podrożał\w*)",
    re.IGNORECASE,
)
DOWN = re.compile(
    r"(?<!nie )\b(?:spad(?!kow)\w*|stracił\w*|zniżkował\w*|potaniał\w*|zmalał\w*)",
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
    return [s.strip() for s in SENTENCE.split(text) if s.strip()]


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
    if len(directional) != 1:
        return None
    fact_id = directional[0]
    value = _numeric(facts, fact_id)
    up, down = bool(UP.search(sentence)), bool(DOWN.search(sentence))
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
    value = _numeric(facts, fact_id)
    if value is None:
        return None
    if DIRECTIONAL.match(fact_id):
        return "wzrost" if value > 0 else "spadek" if value < 0 else None
    if fact_id == "rsi_14":
        if value > RSI_HIGH:
            return "wykupienie"
        return "wyprzedanie" if value < RSI_LOW else "strefa neutralna"
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


def check_interpretations(text: str, facts: FactSheet) -> list[InterpretationIssue]:
    issues: list[InterpretationIssue] = []
    for sentence in sentences(text):
        refs = PLACEHOLDER.findall(sentence)
        if not refs:
            continue
        for check in (_check_direction, _check_rsi, _check_position, _check_period):
            issue = check(sentence, refs, facts)
            if issue:
                issues.append(issue)
                break
    return issues


MARKER = " [niezweryfikowana interpretacja]"


def mark_sentences(text: str, flagged: list[str]) -> str:
    """Dopisuje oznaczenie po zdaniach, których interpretacji nie udało się poprawić."""
    for sentence in dict.fromkeys(flagged):
        text = text.replace(sentence, sentence + MARKER, 1)
    return text
