from marketagents.facts import Fact, FactSheet
from marketagents.interpretation import MARKER, check_interpretations, mark_sentences

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
