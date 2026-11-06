from marketagents.facts import Fact, FactSheet
from marketagents.interpretation import (
    MARKER,
    check_interpretations,
    label,
    mark_sentences,
    periods_in,
    prompt_table,
)

FACTS = FactSheet(
    [
        Fact("close", "kurs", 100.0, "USD"),
        Fact("change_1m", "zmiana 1M", -3.2, "%", 1),
        Fact("change_1y", "zmiana 1R", 12.5, "%", 1),
        Fact("rsi_14", "RSI", 45.0, "", 1),
        Fact("sma_200", "SMA(200)", 110.0, "USD"),
        Fact("sma_50", "SMA(50)", 95.0, "USD"),
        Fact("dist_sma_200", "odległość", -9.1, "%", 1),
    ]
)


def messages(text: str) -> list[str]:
    return [i.message for i in check_interpretations(text, FACTS)]


def test_direction_contradicting_the_sign_is_reported():
    issues = messages("W ostatnim miesiącu kurs wzrósł o {{change_1m}}.")
    assert len(issues) == 1 and "{{change_1m}}" in issues[0] and "wzrost" in issues[0]
    assert messages("Przychody spadły, a kurs w roku zmienił się o {{change_1y}}.")


def test_correct_direction_and_ambiguous_sentences_pass():
    assert not messages("W ostatnim miesiącu kurs spadł o {{change_1m}}.")
    assert not messages("Kurs nie wzrósł, zmiana wyniosła {{change_1m}}.")
    # dwie wartości o różnych znakach w jednym zdaniu -- bez oceny
    assert not messages("Kurs wzrósł w roku o {{change_1y}}, ale spadł w miesiąc o {{change_1m}}.")
    # przymiotnik „wzrostowy” nie opisuje kierunku tej konkretnej zmiany
    assert not messages("Mimo trendu wzrostowego zmiana z miesiąca to {{change_1m}}.")


def test_rsi_zones():
    assert messages("RSI wynosi {{rsi_14}}, co oznacza wykupienie rynku.")
    assert messages("RSI na poziomie {{rsi_14}} wskazuje, że akcje są wyprzedane.")
    assert not messages("RSI {{rsi_14}} jest daleko od strefy wykupienia.")
    assert not messages("RSI {{rsi_14}} nie wskazuje na wykupienie.")


def test_price_position_against_moving_average():
    assert messages("Kurs {{close}} pozostaje powyżej SMA(200) wynoszącej {{sma_200}}.")
    assert messages("Kurs jest powyżej średniej, odległość od SMA(200) to {{dist_sma_200}}.")
    assert not messages("Kurs {{close}} znajduje się poniżej SMA(200) ({{sma_200}}).")
    assert not messages("Kurs jest powyżej SMA(50) ({{sma_50}}).")
    # dwie średnie w jednym zdaniu -- porównanie średnich, nie kursu
    assert not messages("SMA(50) {{sma_50}} jest poniżej SMA(200) {{sma_200}}.")


def test_mark_sentences():
    text = "Kurs wzrósł o {{change_1m}}. RSI {{rsi_14}}."
    flagged = [i.sentence for i in check_interpretations(text, FACTS)]
    assert (
        mark_sentences(text, flagged)
        == f"Kurs wzrósł o {{{{change_1m}}}}.{MARKER} RSI {{{{rsi_14}}}}."
    )


def test_period_words():
    assert periods_in("w ostatnich 3 miesiącach") == ["3 miesiące"]
    assert periods_in("w ciągu 12 miesięcy, czyli r/r") == ["rok"]
    assert periods_in("w ostatnim miesiącu i w skali roku") == ["rok", "miesiąc"]


def test_period_of_change_must_match_fact():
    issues = messages("W ostatnim roku kurs spadł o {{change_1m}}.")
    assert len(issues) == 1 and "okres „rok”" in issues[0] and "„miesiąc”" in issues[0]
    assert messages("W ciągu 3 miesięcy kurs wzrósł o {{change_1y}}.")
    assert not messages("W ostatnim miesiącu kurs spadł o {{change_1m}}.")
    assert not messages("W ciągu 12 miesięcy kurs wzrósł o {{change_1y}}.")
    assert not messages("Kurs spadł o {{change_1m}}.")
    # wielkości z różnych okresów w jednym zdaniu -- bez oceny
    assert not messages("W roku kurs zyskał {{change_1y}}, a w ostatnim miesiącu {{change_1m}}.")


def test_labels_in_prompt_table():
    assert label("change_1m", FACTS) == "spadek"
    assert label("change_1y", FACTS) == "wzrost"
    assert label("rsi_14", FACTS) == "strefa neutralna"
    assert label("sma_200", FACTS) == "kurs poniżej średniej"
    assert label("sma_50", FACTS) == "kurs powyżej średniej"
    assert label("dist_sma_200", FACTS) == "kurs poniżej średniej"
    assert label("close", FACTS) is None
    macd = FactSheet([Fact("macd", "MACD", 1.2), Fact("macd_hist", "histogram", -0.3)])
    assert label("macd", macd) == "linia MACD poniżej linii sygnału"
    assert label("macd_hist", macd) == "linia MACD poniżej linii sygnału"
    table = prompt_table(FACTS)
    assert "| {{change_1m}} | zmiana 1M | -3,2% | spadek |" in table
    assert "| {{close}} | kurs | 100,00 USD |  |" in table


POSITIONS = FactSheet(
    [
        Fact("close", "kurs", 332.9, "USD"),
        Fact("bb_upper", "górna wstęga", 346.2, "USD"),
        Fact("bb_lower", "dolna wstęga", 317.5, "USD"),
        Fact("bb_position", "wstęgi", 53.6, "%", 0, assessment="w środku przedziału"),
        Fact("high_52w", "maks.", 140.9, "USD"),
        Fact("low_52w", "min.", 33.6, "USD"),
        Fact("range_52w_position", "zakres", 63.1, "%", 0, assessment="w górnej części"),
        Fact("ma_order", "układ", "SMA(50) > SMA(200) > SMA(20)", assessment="układ mieszany"),
        Fact("macd_hist", "histogram", 0.14, "", 2),
        Fact("macd", "MACD", -2.66, "", 2),
        Fact("macd_signal", "sygnał", -2.80, "", 2),
        Fact(
            "dist_sma_200",
            "odległość",
            -6.0,
            "%",
            1,
            assessment="kurs poniżej średniej, odległość typowa na tle ostatnich 3 lat",
        ),
        Fact("bb_width", "szerokość", 8.0, "%", 1, assessment="typowa szerokość wstęg"),
        Fact("macro_inflation", "inflacja", 3.7, "%", 1),
        Fact("macro_inflation_change_12m", "zmiana inflacji", 0.8, "p.p.", 2),
        Fact("fy_net_income", "zysk netto", 112.0, "mld USD", 1),
        Fact("fy_revenue", "przychody", 416.2, "mld USD", 1),
        Fact("fy_revenue_growth", "zmiana przychodów", 6.4, "%", 1),
        Fact("fy_revenue_growth_prev", "zmiana rok wcześniej", 2.0, "%", 1),
    ]
)


def found(text: str) -> list[str]:
    return [i.message for i in check_interpretations(text, POSITIONS)]


def test_position_within_bands_and_52_week_range():
    assert found("Kurs {{close}} znajduje się bliżej dolnej wstęgi Bollingera.")
    assert found("Cena jest w dolnej części zakresu z 52 tygodni.")
    # w jednym zdaniu dwa człony o różnych przedziałach: oceniany jest każdy osobno
    assert found(
        "Kurs jest nieco powyżej dolnej wstęgi {{bb_lower}} i daleko od górnej {{bb_upper}}, "
        "handlując w dolnej części 52-tygodniowego zakresu."
    )
    assert not found("Kurs oscyluje w środku wstęg Bollingera(20, 2).")
    assert not found("Kurs jest w górnej części zakresu z 52 tygodni.")
    # wymienienie obu granic bez oceny położenia
    assert not found("Górna wstęga {{bb_upper}} i dolna wstęga {{bb_lower}} wyznaczają zakres.")
    assert not found("Najniższy kurs z 52 tygodni wyniósł {{low_52w}}, a najwyższy {{high_52w}}.")


def test_moving_average_order_and_macd():
    assert found("Krótsza średnia znajduje się poniżej dłuższej, co potwierdza słabość.")
    assert found("Układ, w którym krótsze średnie leżą nad dłuższymi, wskazuje na siłę.")
    assert not found("Krótsza średnia nie leży nad dłuższą.")
    assert found("Linia MACD {{macd}} jest poniżej linii sygnału {{macd_signal}}.")
    assert not found("Linia MACD {{macd}} jest nieco wyższa od linii sygnału {{macd_signal}}.")


def test_distance_and_band_width_follow_assessment():
    assert found("Odległość od SMA(200) {{dist_sma_200}} oznacza wyraźne oddalenie.")
    assert not found("Kurs jest poniżej SMA(200), odległość wynosi {{dist_sma_200}}.")
    assert found("Wstęgi Bollingera są wąskie, co oznacza małą zmienność.")
    assert found("Wstęgi Bollingera są szersze niż zwykle.")
    # rzeczownik „szerokość” nie jest oceną
    assert not found("Szerokość wstęg Bollingera wynosi {{bb_width}} i jest typowa.")
    assert not found("Kurs znajduje się między wstęgami Bollingera.")


def test_change_claims_need_a_change_fact():
    issues = found("Inflacja utrzymuje się na poziomie {{macro_inflation}}.")
    assert issues and "{{macro_inflation_change_12m}}" in issues[0]
    assert found("Fundamenty są solidne, z rosnącym zyskiem netto {{fy_net_income}}.")
    assert found("Spółka odnotowała solidny wzrost przychodów {{fy_revenue}}.")
    # zmiana podana faktem o zmianie albo stan bez opisu zmiany
    assert not found(
        "Inflacja wynosi {{macro_inflation}}, o {{macro_inflation_change_12m}} więcej niż rok temu."
    )
    assert not found("Przychody wyniosły {{fy_revenue}} przy wzroście o {{fy_revenue_growth}}.")
    assert not found("Inflacja CPI wynosi {{macro_inflation}}.")


def test_continuation_needs_previous_year():
    assert found("Przychody wzrosły o {{fy_revenue_growth}}, co oznacza kontynuację wzrostu.")
    assert not found(
        "Przychody wzrosły o {{fy_revenue_growth}} po {{fy_revenue_growth_prev}} rok wcześniej, "
        "co oznacza kontynuację wzrostu."
    )


def test_direction_words_beyond_rise_and_fall():
    from marketagents.facts import Fact, FactSheet
    from marketagents.interpretation import check_interpretations

    facts = FactSheet(
        [
            Fact("fy_revenue_growth", "zmiana przychodów r/r", -4.0, "%", 1),
            Fact("macro_fed_rate_change_12m", "zmiana stopy w ciągu 12 miesięcy", 0.5, "p.p.", 2),
        ]
    )
    assert check_interpretations("Przychody zwiększyły się o {{fy_revenue_growth}}.", facts)
    assert check_interpretations("Fed obniżył stopy o {{macro_fed_rate_change_12m}}.", facts)
    assert not check_interpretations("Przychody zmniejszyły się o {{fy_revenue_growth}}.", facts)


def test_reference_must_match_the_indicator_named_before_it():
    from marketagents.facts import Fact, FactSheet
    from marketagents.interpretation import check_interpretations

    facts = FactSheet(
        [
            Fact("close", "kurs", 100.0, "USD"),
            Fact("sma_50", "SMA(50)", 95.0, "USD", 2),
            Fact("sma_200", "SMA(200)", 90.0, "USD", 2),
            Fact("dist_sma_200", "odległość od SMA(200)", 11.1, "%", 1),
            Fact("rsi_14", "RSI(14)", 55.0, "", 1),
            Fact("macd", "MACD", 1.2, "", 2),
            Fact("macd_signal", "sygnał", 0.8, "", 2),
            Fact("bb_upper", "górna wstęga", 110.0, "USD", 2),
            Fact("bb_lower", "dolna wstęga", 90.0, "USD", 2),
        ]
    )
    assert check_interpretations("SMA(50) wynosi {{sma_200}}.", facts)
    assert check_interpretations("RSI wynosi {{macd}}.", facts)
    assert check_interpretations("Górna wstęga Bollingera leży na {{bb_lower}}.", facts)
    ok = (
        "Kurs {{close}} jest powyżej SMA(50) {{sma_50}} i SMA(200) na poziomie {{sma_200}}. "
        "Średnie SMA 50 i 200 wynoszą {{sma_50}} i {{sma_200}}. "
        "Odległość od SMA(200) wynosi {{dist_sma_200}}, a RSI {{rsi_14}}. "
        "Linia MACD {{macd}} jest nad linią sygnału {{macd_signal}}. "
        "RSI i MACD wynoszą {{rsi_14}} i {{macd}}."
    )
    assert not check_interpretations(ok, facts)
