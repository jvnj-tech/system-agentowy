"""Podział tekstu raportu na zdania, wspólny dla weryfikatora i jego reguł.

Kropka po skrócie („ok. 61”, „np. RSI”), po inicjale („Gregory K. Peters”) i po skrócie
formy prawnej („Microsoft Corp. ogłosił”) nie kończy zdania. Bez tego zdanie z wiadomości
rozpadało się na dwie części, z których druga nie miała numeru źródła.
"""

from __future__ import annotations

import re

SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")
ABBREVIATION = re.compile(
    r"(?<![\w.])(?:ok|np|m\.in|tj|tzn|tzw|wg|ang|por|zob|ca|mln|mld|tys|pkt|proc|r|ds|itd|itp"
    r"|nr|dot|godz|mies|tyg|prof|dr|mgr|inż|św|ul|wyd|red|zł"
    r"|corp|inc|co|ltd|plc|jr|sr|st)\.$",
    re.IGNORECASE,
)
# Inicjał imienia: pojedyncza wielka litera z kropką po słowie zaczynającym się wielką literą.
INITIAL = re.compile(r"(?<![\w.])[A-ZĄĆĘŁŃÓŚŹŻ][\w’'-]+\s+[A-ZĄĆĘŁŃÓŚŹŻ]\.$")


def spans(text: str) -> list[tuple[int, str]]:
    """Zdania tekstu z pozycją początku (bez pustych); nowy wiersz zawsze kończy zdanie."""
    out, start = [], 0
    for match in SENTENCE_END.finditer(text):
        head = text[start : match.start()]
        if "\n" not in match.group(0) and (ABBREVIATION.search(head) or INITIAL.search(head)):
            continue
        out.append((start, head))
        start = match.end()
    out.append((start, text[start:]))
    return [(i, sentence) for i, sentence in out if sentence.strip()]


def split(text: str) -> list[str]:
    """Zdania tekstu (bez pustych); nowy wiersz zawsze kończy zdanie."""
    return [sentence for _, sentence in spans(text)]
