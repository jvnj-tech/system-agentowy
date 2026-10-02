"""Karta faktów: jedyne źródło liczb, które mogą trafić do raportu.

Agenci nie przepisują liczb, tylko wstawiają w tekst odwołania w postaci {{identyfikator}}.
Wartości są podstawiane dopiero przy składaniu raportu, a weryfikator sprawdza,
czy w tekście nie pojawiła się żadna liczba spoza karty.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fact:
    id: str
    label: str
    value: float | str
    unit: str = ""
    decimals: int = 2

    def formatted(self) -> str:
        if isinstance(self.value, str):
            return self.value
        text = format_number(self.value, self.decimals)
        if not self.unit:
            return text
        return f"{text}{self.unit}" if self.unit == "%" else f"{text} {self.unit}"


def format_number(value: float, decimals: int) -> str:
    """Zapis polski: przecinek dziesiętny, spacja niełamliwa jako separator tysięcy."""
    text = f"{value:,.{decimals}f}"
    return text.replace(",", " ").replace(".", ",")


class FactSheet:
    def __init__(self, facts: list[Fact]) -> None:
        self._facts: dict[str, Fact] = {}
        for fact in facts:
            if fact.id in self._facts:
                raise ValueError(f"Powtórzony identyfikator faktu: {fact.id}")
            self._facts[fact.id] = fact

    def __contains__(self, fact_id: str) -> bool:
        return fact_id in self._facts

    def __getitem__(self, fact_id: str) -> Fact:
        return self._facts[fact_id]

    def __iter__(self):
        return iter(self._facts.values())

    def __len__(self) -> int:
        return len(self._facts)

    def as_prompt_table(self) -> str:
        lines = ["| identyfikator | opis | wartość |", "| --- | --- | --- |"]
        lines += [f"| {{{{{f.id}}}}} | {f.label} | {f.formatted()} |" for f in self]
        return "\n".join(lines)

    def to_records(self) -> list[dict]:
        return [asdict(f) for f in self]

    @classmethod
    def from_records(cls, records: list[dict]) -> FactSheet:
        return cls([Fact(**r) for r in records])


def _pct_change(close: pd.Series, sessions: int) -> float | None:
    if len(close) <= sessions:
        return None
    return float((close.iloc[-1] / close.iloc[-1 - sessions] - 1) * 100)


def technical_facts(ticker: str, prices: pd.DataFrame, indicators: pd.DataFrame) -> list[Fact]:
    """Fakty z notowań i wskaźników technicznych na ostatnią dostępną sesję."""
    close = prices["close"]
    last = indicators.iloc[-1]
    year = close.iloc[-252:]
    facts = [
        Fact("ticker", "symbol spółki", ticker),
        Fact("as_of", "data ostatniej sesji", close.index[-1].strftime("%Y-%m-%d")),
        Fact("close", "kurs zamknięcia", float(close.iloc[-1]), "USD"),
        Fact("high_52w", "najwyższy kurs zamknięcia z 52 tygodni", float(year.max()), "USD"),
        Fact("low_52w", "najniższy kurs zamknięcia z 52 tygodni", float(year.min()), "USD"),
    ]
    for sessions, fid, label in [
        (21, "change_1m", "zmiana kursu w ostatnim miesiącu"),
        (63, "change_3m", "zmiana kursu w ostatnich 3 miesiącach"),
        (252, "change_1y", "zmiana kursu w ostatnim roku"),
    ]:
        change = _pct_change(close, sessions)
        if change is not None:
            facts.append(Fact(fid, label, change, "%", 1))

    indicator_facts = [
        ("sma_20", "średnia krocząca SMA(20)", "USD", 2),
        ("sma_50", "średnia krocząca SMA(50)", "USD", 2),
        ("sma_200", "średnia krocząca SMA(200)", "USD", 2),
        ("ema_20", "średnia wykładnicza EMA(20)", "USD", 2),
        ("rsi_14", "RSI(14)", "", 1),
        ("macd", "linia MACD(12, 26)", "", 2),
        ("macd_signal", "linia sygnału MACD(9)", "", 2),
        ("macd_hist", "histogram MACD", "", 2),
        ("bb_upper", "górna wstęga Bollingera(20, 2)", "USD", 2),
        ("bb_lower", "dolna wstęga Bollingera(20, 2)", "USD", 2),
    ]
    for fid, label, unit, decimals in indicator_facts:
        value = last.get(fid)
        if value is not None and np.isfinite(value):
            facts.append(Fact(fid, label, float(value), unit, decimals))

    sma_200 = last.get("sma_200")
    if sma_200 is not None and np.isfinite(sma_200):
        distance = (close.iloc[-1] / sma_200 - 1) * 100
        facts.append(Fact("dist_sma_200", "odległość kursu od SMA(200)", float(distance), "%", 1))
    return facts
