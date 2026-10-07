from marketagents.facts import Fact, FactSheet
from marketagents.verifier import mark_unverified, parse_number, render, verify

FACTS = FactSheet(
    [
        Fact("close", "kurs", 1234.56, "USD"),
        Fact("rsi_14", "RSI", 61.23, "", 1),
        Fact("change_1m", "zmiana", -3.24, "%", 1),
        Fact("as_of", "data", "2026-10-01"),
    ]
)


def test_parse_number_formats():
    assert parse_number("1 234,56") == 1234.56
    assert parse_number("1 234.56") == 1234.56
    assert parse_number("−3,2") == -3.2


def test_placeholders_only_pass():
    result = verify("Kurs {{close}} na {{as_of}}, RSI(14) = {{rsi_14}}.", FACTS)
    assert result.ok
    assert result.rendered == "Kurs 1 234,56 USD na 2026-10-01, RSI(14) = 61,2."


def test_unknown_placeholder_is_reported():
    result = verify("Wolumen {{volume}}.", FACTS)
    assert result.unknown_ids == ["volume"]
    assert "[brak danych: volume]" in result.rendered


def test_invented_number_is_reported():
    result = verify("RSI wynosi 75,3, a kurs {{close}}.", FACTS)
    assert result.unverified_numbers == ["75,3"]
    assert not result.ok


def test_every_number_outside_references_is_reported():
    text = "Kurs 1 234,56 USD, RSI 61,2, spadek o 3,2% w miesiąc, SMA(200), 52 tygodnie."
    result = verify(text, FACTS)
    assert result.unverified_numbers == ["1 234,56", "61,2", "3,2"]
    assert result.suggestions == {"1 234,56": "close", "61,2": "rsi_14", "3,2": "change_1m"}
    assert "wstaw w jej miejsce odwołanie {{rsi_14}}" in result.issues()[1]


def test_value_of_neighbouring_fact_is_reported():
    # 61 to zaokrąglone RSI wpisane przy kursie, a nie przy RSI.
    result = verify("Kurs {{close}}, a RSI 61.", FACTS)
    assert result.unverified_numbers == ["61"]
    assert result.suggestions == {"61": "rsi_14"}


def test_repeated_value_of_referenced_fact_should_be_removed():
    result = verify("RSI(14) wynosi {{rsi_14}} (ok. 61). Kurs {{close}}.", FACTS)
    assert result.repeated == ["61"]
    assert "usuń ją" in result.issues()[0]


def test_indicator_periods_are_allowed_only_as_plain_integers():
    assert verify("SMA 50 i SMA(200), MACD(12, 26, 9).", FACTS).ok
    assert verify("RSI 50,0 i -14.", FACTS).unverified_numbers == ["50,0", "-14"]


def test_index_names_are_not_numbers():
    assert verify("Kurs {{close}} na tle NASDAQ-100 i S&P 500.", FACTS).ok


def test_numbers_from_fact_labels_are_allowed():
    facts = FactSheet([Fact("change_3m", "zmiana kursu w ostatnich 3 miesiącach", 5.0, "%", 1)])
    assert verify("W ciągu 3 miesięcy kurs wzrósł o {{change_3m}}.", facts).ok
    assert not verify("W ciągu 4 miesięcy kurs wzrósł o {{change_3m}}.", facts).ok


def test_list_markers_and_dates_are_ignored():
    text = "1. Kurs {{close}}\n2. Stan na 2026-10-01"
    assert verify(text, FACTS).ok


def test_mark_unverified():
    rendered = render("RSI 75,3 i 75,30.", FACTS)
    assert mark_unverified(rendered, ["75,3"]) == "RSI 75,3 [niezweryfikowane] i 75,30."
    marked = mark_unverified("RSI 61,2 to ok. 61.", ["61"])
    assert marked == "RSI 61,2 to ok. 61 [niezweryfikowane]."


def test_placeholders_in_code_ticks_render_as_plain_text():
    result = verify("Kurs `{{close}}`, RSI `{{ rsi_14 }}`.", FACTS)
    assert result.ok
    assert result.rendered == "Kurs 1 234,56 USD, RSI 61,2."


def test_unit_repeated_after_placeholder_is_not_duplicated():
    result = verify("Kurs {{close}} USD, zmiana {{change_1m}}%, RSI {{rsi_14}} USD.", FACTS)
    assert result.rendered == "Kurs 1 234,56 USD, zmiana -3,2%, RSI 61,2 USD."


def test_number_in_compound_word_is_ignored():
    assert verify("Rentowność 10-letnich obligacji i 3-miesięczny trend.", FACTS).ok
    assert not verify("Wzrost o 10-15%.", FACTS).ok


def test_mark_unverified_skips_dates():
    text = "Na dzień 2026-10-01 RSI wynosi 10."
    assert mark_unverified(text, ["10"]) == "Na dzień 2026-10-01 RSI wynosi 10 [niezweryfikowane]."


def test_render_html_adds_source_hint():
    facts = FactSheet([Fact("rsi_14", "RSI(14)", 61.23, "", 1, source="Yahoo Finance")])
    html = render("RSI {{rsi_14}}.", facts, html=True)
    assert html == (
        'RSI <span class="fact" tabindex="0" data-zrodlo="RSI(14): Yahoo Finance">61,2</span>.'
    )


def test_mark_unverified_ignores_html_attributes():
    html = '<span class="fact" tabindex="0" data-zrodlo="RSI(14): 14 sesji">61,2</span> i 14.'
    assert mark_unverified(html, ["14"]).endswith("i 14 [niezweryfikowane].")
    assert 'data-zrodlo="RSI(14): 14 sesji"' in mark_unverified(html, ["14"])


def test_years_must_match_dates_in_fact_sheet():
    facts = FactSheet(
        [Fact("as_of", "data", "2026-10-01"), Fact("fy_end", "rok obrotowy", "2025-09-27")]
    )
    assert verify("W 2025 r. i w 2024 r. przychody rosły.", facts).ok
    result = verify("W 2021 r. spółka miała rekordowy rok.", facts)
    assert result.wrong_years == ["2021"]
    assert result.unverified_numbers == []
    assert "Rok 2021" in result.issues()[0]
    assert mark_unverified(result.rendered, result.to_mark).startswith("W 2021 [niezweryfikowane]")
