"""Karta faktów: jedyne źródło liczb, które mogą trafić do raportu.

Agenci nie przepisują liczb, tylko wstawiają w tekst odwołania w postaci {{identyfikator}}.
Wartości są podstawiane dopiero przy składaniu raportu, a weryfikator sprawdza,
czy w tekście nie pojawiła się żadna liczba spoza karty.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from marketagents.data.macro import SERIES
from marketagents.indicators import MovingAverages

# Grupy faktów: każdy analityk dostaje fakty ze swojej grupy oraz fakty ogólne.
TECHNICAL, FUNDAMENTAL, MACRO = "technical", "fundamental", "macro"


@dataclass(frozen=True)
class Fact:
    id: str
    label: str
    value: float | str
    unit: str = ""
    decimals: int = 2
    group: str = ""
    # Skąd pochodzi wartość i jak ją obliczono (pokazywane w interfejsie przy liczbie).
    source: str = ""

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

    def group(self, *names: str) -> FactSheet:
        """Fakty z wybranych grup oraz fakty ogólne (bez grupy)."""
        return FactSheet([f for f in self if not f.group or f.group in names])

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


def technical_facts(
    ticker: str, prices: pd.DataFrame, indicators: pd.DataFrame, ma: MovingAverages | None = None
) -> list[Fact]:
    """Fakty z notowań i wskaźników technicznych na ostatnią dostępną sesję."""
    ma = ma or MovingAverages()
    close = prices["close"]
    last = indicators.iloc[-1]
    year = close.iloc[-252:]
    as_of = close.index[-1].strftime("%Y-%m-%d")
    prices_src = f"Yahoo Finance, ceny zamknięcia do {as_of}"
    facts = [
        Fact("ticker", "symbol spółki", ticker),
        Fact("as_of", "data ostatniej sesji", as_of, source="Yahoo Finance"),
        Fact("close", "kurs zamknięcia", float(close.iloc[-1]), "USD", source=prices_src),
        Fact(
            "high_52w",
            "najwyższy kurs zamknięcia z 52 tygodni",
            float(year.max()),
            "USD",
            2,
            TECHNICAL,
            f"{prices_src}; maksimum z 252 sesji",
        ),
        Fact(
            "low_52w",
            "najniższy kurs zamknięcia z 52 tygodni",
            float(year.min()),
            "USD",
            2,
            TECHNICAL,
            f"{prices_src}; minimum z 252 sesji",
        ),
    ]
    for sessions, fid, label in [
        (21, "change_1m", "zmiana kursu w ostatnim miesiącu"),
        (63, "change_3m", "zmiana kursu w ostatnich 3 miesiącach"),
        (252, "change_1y", "zmiana kursu w ostatnim roku"),
    ]:
        change = _pct_change(close, sessions)
        if change is not None:
            source = f"{prices_src}; zmiana z {sessions} sesji"
            facts.append(Fact(fid, label, change, "%", 1, TECHNICAL, source))

    indicator_facts = [
        *(
            (f"sma_{n}", f"średnia krocząca SMA({n})", "USD", 2, f"średnia z {n} sesji")
            for n in ma.sma
        ),
        (
            f"ema_{ma.ema}",
            f"średnia wykładnicza EMA({ma.ema})",
            "USD",
            2,
            f"średnia wykładnicza, alfa = 2/({ma.ema}+1)",
        ),
        ("rsi_14", "RSI(14)", "", 1, "RSI z 14 sesji, wygładzanie Wildera"),
        ("macd", "linia MACD(12, 26)", "", 2, "EMA(12) - EMA(26)"),
        ("macd_signal", "linia sygnału MACD(9)", "", 2, "EMA(9) linii MACD"),
        ("macd_hist", "histogram MACD", "", 2, "MACD - linia sygnału"),
        ("bb_upper", "górna wstęga Bollingera(20, 2)", "USD", 2, "SMA(20) + 2 odchylenia"),
        ("bb_lower", "dolna wstęga Bollingera(20, 2)", "USD", 2, "SMA(20) - 2 odchylenia"),
    ]
    for fid, label, unit, decimals, method in indicator_facts:
        value = last.get(fid)
        if value is not None and np.isfinite(value):
            source = f"{prices_src}; {method}"
            facts.append(Fact(fid, label, float(value), unit, decimals, TECHNICAL, source))

    long = ma.long
    sma_long = last.get(f"sma_{long}")
    if sma_long is not None and np.isfinite(sma_long):
        distance = (close.iloc[-1] / sma_long - 1) * 100
        facts.append(
            Fact(
                f"dist_sma_{long}",
                f"odległość kursu od SMA({long})",
                float(distance),
                "%",
                1,
                TECHNICAL,
                f"{prices_src}; kurs / SMA({long}) - 1",
            )
        )
    return facts


def _get(row: pd.Series | None, col: str) -> float | None:
    if row is None:
        return None
    value = row.get(col)
    return float(value) if value is not None and np.isfinite(value) else None


def _ratio(num: float, den: float) -> float | None:
    if num is None or den is None or den == 0:
        return None
    return float(num / den * 100)


def fundamental_facts(annual: pd.DataFrame, close: float | None = None) -> list[Fact]:
    """Fakty z ostatniego rocznego sprawozdania (10-K) i porównanie z poprzednim rokiem."""
    if annual.empty:
        return []
    last = annual.iloc[-1]
    prev = annual.iloc[-2] if len(annual) > 1 else None
    fy_end = annual.index[-1].strftime("%Y-%m-%d")
    filing_src = f"SEC EDGAR, sprawozdanie roczne (10-K) za rok obrotowy do {fy_end}"
    facts = [
        Fact(
            "fy_end",
            "koniec ostatniego roku obrotowego",
            fy_end,
            group=FUNDAMENTAL,
            source="SEC EDGAR",
        )
    ]

    for col, fid, label in [
        ("revenue", "fy_revenue", "przychody w ostatnim roku obrotowym"),
        ("net_income", "fy_net_income", "zysk netto w ostatnim roku obrotowym"),
        ("operating_cash_flow", "fy_ocf", "przepływy operacyjne w ostatnim roku obrotowym"),
    ]:
        value = _get(last, col)
        if value is not None:
            facts.append(Fact(fid, label, value / 1e9, "mld USD", 1, FUNDAMENTAL, filing_src))

    revenue, prev_revenue = _get(last, "revenue"), _get(prev, "revenue")
    derived = [
        (
            "fy_revenue_growth",
            "zmiana przychodów rok do roku",
            _ratio(revenue, prev_revenue),
            -100,
            "przychody / przychody z poprzedniego roku - 1",
        ),
        (
            "fy_net_margin",
            "marża netto",
            _ratio(_get(last, "net_income"), revenue),
            0,
            "zysk netto / przychody",
        ),
        (
            "fy_operating_margin",
            "marża operacyjna",
            _ratio(_get(last, "operating_income"), revenue),
            0,
            "zysk operacyjny / przychody",
        ),
        (
            "fy_gross_margin",
            "marża brutto",
            _ratio(_get(last, "gross_profit"), revenue),
            0,
            "zysk brutto / przychody",
        ),
        (
            "fy_liabilities_to_assets",
            "zobowiązania do aktywów",
            _ratio(_get(last, "total_liabilities"), _get(last, "total_assets")),
            0,
            "zobowiązania ogółem / aktywa ogółem",
        ),
        (
            "fy_roe",
            "rentowność kapitału własnego (ROE)",
            _ratio(_get(last, "net_income"), _get(last, "stockholders_equity")),
            0,
            "zysk netto / kapitał własny",
        ),
    ]
    for fid, label, value, shift, formula in derived:
        if value is not None:
            source = f"{filing_src}; {formula}"
            facts.append(Fact(fid, label, value + shift, "%", 1, FUNDAMENTAL, source))

    eps = _get(last, "eps_diluted")
    if eps is not None:
        label = "rozwodniony zysk na akcję (EPS)"
        facts.append(Fact("fy_eps", label, eps, "USD", 2, FUNDAMENTAL, filing_src))
        if close is not None and eps > 0:
            facts.append(
                Fact(
                    "pe_ratio",
                    "cena do zysku (C/Z) z ostatniego roku",
                    close / eps,
                    "",
                    1,
                    FUNDAMENTAL,
                    f"kurs zamknięcia (Yahoo Finance) / EPS ({filing_src})",
                )
            )
    return facts


def macro_facts(macro: pd.DataFrame) -> list[Fact]:
    """Ostatnie dostępne odczyty wskaźników makroekonomicznych z datami."""
    facts: list[Fact] = []
    levels = [
        ("fed_funds_rate", "macro_fed_rate", "stopa funduszy federalnych"),
        ("treasury_10y", "macro_10y", "rentowność 10-letnich obligacji USA"),
        ("unemployment_rate", "macro_unemployment", "stopa bezrobocia w USA"),
    ]
    for col, fid, label in levels:
        if col in macro:
            series = macro[col].dropna()
            if not series.empty:
                date = series.index[-1].strftime("%Y-%m-%d")
                source = f"FRED, seria {SERIES[col]}, odczyt z {date}"
                facts.append(Fact(fid, label, float(series.iloc[-1]), "%", 2, MACRO, source))
                facts.append(
                    Fact(f"{fid}_date", f"data odczytu: {label}", date, group=MACRO, source="FRED")
                )
    if "cpi" in macro:
        cpi = macro["cpi"].dropna()
        if len(cpi) > 12:
            inflation = (cpi.iloc[-1] / cpi.iloc[-13] - 1) * 100
            date = cpi.index[-1].strftime("%Y-%m-%d")
            source = f"FRED, seria {SERIES['cpi']}, odczyt z {date}; zmiana CPI z 12 miesięcy"
            facts.append(
                Fact(
                    "macro_inflation",
                    "inflacja CPI rok do roku",
                    float(inflation),
                    "%",
                    1,
                    MACRO,
                    source,
                )
            )
            facts.append(
                Fact(
                    "macro_inflation_date",
                    "data odczytu: inflacja CPI",
                    date,
                    group=MACRO,
                    source="FRED",
                )
            )
    return facts
