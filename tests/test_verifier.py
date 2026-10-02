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


def test_numbers_matching_facts_are_accepted():
    text = "Kurs 1 234,56 USD, RSI 61,2 (ok. 61), spadek o 3,2% w miesiąc, SMA(200), 52 tygodnie."
    assert verify(text, FACTS).ok


def test_rounding_tolerance_follows_fact_precision():
    assert not verify("RSI 61,4", FACTS).ok
    assert verify("RSI 61,2", FACTS).ok


def test_list_markers_and_dates_are_ignored():
    text = "1. Kurs {{close}}\n2. Stan na 2026-10-01"
    assert verify(text, FACTS).ok


def test_mark_unverified():
    rendered = render("RSI 75,3 i 75,30.", FACTS)
    assert mark_unverified(rendered, ["75,3"]) == "RSI 75,3 [niezweryfikowane] i 75,30."


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
