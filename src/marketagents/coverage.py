"""Opis zakresu danych dopisywany przez kod do sekcji ograniczeń raportu.

Model językowy nie zna dat danych ani przyczyn ich braku, więc ten fragment raportu
powstaje bez niego: z dat w karcie faktów i z listy wiadomości.
"""

from __future__ import annotations

from datetime import date

from marketagents.facts import FactSheet

# Fakty, których brak warto zgłosić czytelnikowi (identyfikator -> nazwa w raporcie).
EXPECTED = {
    "fy_revenue": "przychody",
    "fy_revenue_growth": "zmiana przychodów rok do roku",
    "fy_gross_margin": "marża brutto",
    "fy_operating_margin": "marża operacyjna",
    "fy_net_margin": "marża netto",
    "fy_liabilities_to_assets": "zobowiązania do aktywów",
    "fy_roe": "ROE",
    "pe_ratio": "wskaźnik C/Z",
    "macro_fed_rate": "stopa funduszy federalnych",
    "macro_10y": "rentowność obligacji 10-letnich",
    "macro_inflation": "inflacja CPI",
    "macro_unemployment": "stopa bezrobocia",
}
# Po tylu dniach różnicy między najnowszą wiadomością a ostatnią sesją raport ostrzega,
# że opis kursu może nie uwzględniać zdarzeń z wiadomości.
STALE_PRICES_DAYS = 3
NOTE_PREFIX = "Zakres danych (opis wygenerowany automatycznie): "


def _date(value: object) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _months_between(earlier: date, later: date) -> int:
    return (later.year - earlier.year) * 12 + later.month - earlier.month


def _missing_reason(fact_id: str, facts: FactSheet) -> str:
    if fact_id == "pe_ratio" and "fy_eps" in facts and float(facts["fy_eps"].value) <= 0:
        return " (zysk na akcję nie jest dodatni)"
    if fact_id == "fy_roe" and "fy_equity_negative" in facts:
        return " (ujemny kapitał własny)"
    return ""


def coverage_note(facts: FactSheet, news: list[dict]) -> str:
    """Akapit o zakresie i aktualności danych oraz o brakujących wielkościach."""
    parts = []
    as_of = _date(facts["as_of"].value) if "as_of" in facts else None
    if as_of:
        parts.append(f"notowania do {as_of.isoformat()}")
    fy_end = _date(facts["fy_end"].value) if "fy_end" in facts else None
    if fy_end:
        text = f"sprawozdanie roczne za rok obrotowy zakończony {fy_end.isoformat()}"
        if as_of and _months_between(fy_end, as_of) >= 6:
            months = _months_between(fy_end, as_of)
            text += (
                f" (dane sprzed {months} miesięcy; późniejsze wyniki kwartalne nie są uwzględnione)"
            )
        parts.append(text)
    macro_dates = sorted(
        d
        for f in facts
        if f.id.startswith("macro_") and f.id.endswith("_date")
        if (d := _date(f.value))
    )
    if macro_dates:
        first, last = macro_dates[0].isoformat(), macro_dates[-1].isoformat()
        span = first if first == last else f"od {first} do {last}"
        parts.append(f"odczyty makroekonomiczne {span}")
    news_dates = sorted(d for n in news if (d := _date(n.get("published"))))
    if news_dates:
        parts.append(
            f"wiadomości o spółce: {len(news)} z okresu "
            f"{news_dates[0].isoformat()}–{news_dates[-1].isoformat()}"
        )
    else:
        parts.append("brak wiadomości o spółce")

    note = NOTE_PREFIX + "; ".join(parts) + "."
    if as_of and news_dates and (news_dates[-1] - as_of).days > STALE_PRICES_DAYS:
        days = (news_dates[-1] - as_of).days
        note += (
            f" Uwaga: notowania są o {days} dni starsze niż najnowsze wiadomości, więc opis "
            "kursu może nie uwzględniać zdarzeń, o których piszą media."
        )
    missing = [
        name + _missing_reason(fact_id, facts)
        for fact_id, name in EXPECTED.items()
        if fact_id not in facts
    ]
    if missing:
        note += " Brak danych: " + ", ".join(missing) + "."
    return note
