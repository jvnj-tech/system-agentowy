"""Reguły formy raportu (przypadki z przeglądu raportów zbioru projektowego, 14.11.2026)."""

import pandas as pd

from marketagents.facts import Fact, FactSheet, cpi_yearly, fundamental_facts, macro_facts
from marketagents.form import check_form
from marketagents.sentences import split
from marketagents.verifier import render, verify

FACTS = FactSheet(
    [
        Fact("ticker", "symbol spółki", "AAPL"),
        Fact("close", "kurs zamknięcia", 356.27, "USD", 2),
        Fact("change_1y", "zmiana kursu w ostatnim roku", 30.7, "%", 1),
        Fact("macro_fed_rate_change_12m", "zmiana stopy", -0.21, "p.p.", 2),
        Fact("rsi_14", "RSI(14)", 60.9, "", 1, assessment="strefa neutralna"),
    ]
)


def messages(text: str) -> list[str]:
    return [i.message for i in check_form(text, FACTS)]


def test_decrease_with_negative_value():
    assert messages("Stopa zanotowała spadek o {{macro_fed_rate_change_12m}}.")
    assert not messages("Stopa zmieniła się o {{macro_fed_rate_change_12m}}.")
    # Wartość dodatnia po „wzrost o” nie jest błędem formy.
    assert not messages("Kurs wzrósł o {{change_1y}}.")
    # „Spadek” z innego członu zdania nie dotyczy tej wartości.
    assert not messages("Kurs spadł, a stopa zmieniła się o {{macro_fed_rate_change_12m}}.")


def test_card_value_with_news_number():
    assert messages("Kurs wzrósł o {{change_1y}} [5].")
    assert messages("Zmiana wyniosła {{change_1y}} (wzrost) [1][3].")
    assert messages("Zmiana wyniosła {{change_1y}}, co oznacza wzrost [2].")
    # Numer przy informacji z artykułu, a nie przy wartości z karty.
    assert not messages("Kurs wzrósł o {{change_1y}}, a według autorów artykułu popyt rośnie [2].")
    # Nazwa spółki z karty nie jest wartością.
    assert not messages("Spółka {{ticker}} ogłosiła wyniki [3].")


def test_internal_mechanisms():
    assert messages("Karta faktów nie zawiera wskaźnika C/Z.")
    assert messages("Pominięto sekcję, dla której nie przygotowano analizy cząstkowej.")
    assert messages("Raport opiera się na dostarczonych danych.")
    assert not messages("Raport opiera się na danych rocznych zamiast kwartalnych.")


def test_citation_that_is_not_a_number():
    assert messages("Kurs wzrósł o {{change_1y}} [{{ticker}}].")
    assert messages("Sprzedaż wzrosła [cite: 6].")
    assert not messages("Sprzedaż wzrosła [6].")


def test_uninflected_label_or_assessment():
    assert messages("Kurs zanotował zmiana kursu w ostatnim roku {{change_1y}}.")
    assert messages("RSI wynosi {{rsi_14}}, co odpowiada strefa neutralna.")
    assert messages("RSI wynosi {{rsi_14}} w strefie neutralna.")
    assert not messages("RSI wynosi {{rsi_14}}, co odpowiada strefie neutralnej.")


def test_decimal_point_vague_phrase_and_english():
    assert messages("Hershey oferuje 3.4% [1].")
    assert not messages("Hershey oferuje 3,4% [1].")
    assert messages("Spółka planuje zakończyć rok z płynnością opisaną w raporcie źródłowym [2].")
    assert messages("Termin publikacji wyznaczono na konkretny dzień listopada [6].")
    assert messages("Incydent miał miejsce w facility w Indiach [3].")
    assert messages("Przychody wyniosły 6,12B USD [4].")


def test_discrepancy_with_card():
    assert messages("Artykuł podaje 25,94 USD, co różni się od kursu {{close}}.")
    assert not messages("Analitycy różnią się w ocenach [3].")


def test_verify_reports_form_issues():
    text = "## Podsumowanie\nKurs wzrósł o {{change_1y}} [1]."
    news = [{"headline": "Apple", "summary": "Apple rises"}]
    assert not verify(text, FACTS, news=news).ok


def test_render_does_not_double_dot_after_unit():
    assert render("Spadek o {{macro_fed_rate_change_12m}}.", FACTS) == "Spadek o -0,21 p.p."
    html = render("Zmiana {{macro_fed_rate_change_12m}}.", FACTS, html=True)
    assert html.endswith("p.p.</span>")


def test_initials_and_company_suffix_do_not_end_sentence():
    text = "Dyrektor Gregory K. Peters sprzedał akcje [3]. Microsoft Corp. ogłosił umowę [1]."
    assert split(text) == [
        "Dyrektor Gregory K. Peters sprzedał akcje [3].",
        "Microsoft Corp. ogłosił umowę [1].",
    ]


def test_inflation_compares_same_month_year_ago():
    months = pd.date_range("2024-09-01", "2025-10-01", freq="MS")
    cpi = pd.Series([100 + i for i in range(len(months))], index=months, dtype=float)
    # Brak odczytu za październik 2024: inflacja za październik 2025 nie może być policzona.
    cpi = cpi.drop(pd.Timestamp("2024-10-01"))
    yearly = cpi_yearly(cpi)
    assert yearly.index[-1] == pd.Timestamp("2025-09-01")
    assert round(yearly.iloc[-1], 2) == round((112 / 100 - 1) * 100, 2)


def test_macro_change_needs_reading_from_same_month():
    months = pd.date_range("2024-09-01", "2025-10-01", freq="MS")
    rate = pd.Series(4.0, index=months).drop(pd.Timestamp("2024-10-01"))
    facts = FactSheet(macro_facts(pd.DataFrame({"unemployment_rate": rate})))
    assert "macro_unemployment" in facts
    assert "macro_unemployment_change_12m" not in facts


def test_eps_is_adjusted_for_split_after_filing():
    annual = pd.DataFrame(
        {"revenue": [10e9], "net_income": [2e9], "eps_diluted": [100.0]},
        index=pd.to_datetime(["2025-12-31"]),
    )
    annual["eps_filed"] = pd.to_datetime(["2026-02-10"])
    facts = FactSheet(fundamental_facts(annual, close=160.0, split_factor=lambda date: 25.0))
    assert facts["fy_eps"].value == 4.0
    assert round(facts["pe_ratio"].value, 1) == 40.0
    assert "split" in facts["fy_eps"].source


def test_empty_sentence_repeated_label_and_uninflected_after_verb():
    assert messages("Spółka {{ticker}} [1].")
    assert not messages("Spółka {{ticker}} opublikowała wyniki [1].")
    assert messages(
        "Rentowność kapitału własnego (ROE) wyniosła rentowność kapitału własnego (ROE) "
        "na poziomie {{change_1y}}."
    )
    assert messages("Histogram wynosi {{change_1y}}, co oznacza linia MACD poniżej sygnału.")
