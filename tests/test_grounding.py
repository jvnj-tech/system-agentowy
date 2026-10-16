from marketagents.grounding import check_forecasts, check_news_content, proper_names, stem

NEWS = [
    {
        "source": "SeekingAlpha",
        "headline": "Microsoft's Maia 300 cuts Nvidia dependence",
        "summary": "Scott Guthrie discussed the chip with Deirdre Bosa.",
    },
    {"source": "Yahoo", "headline": "Intel CEO Lip-Bu Tan on Chinese AI models", "summary": ""},
]


def issues(text: str) -> list[str]:
    return [i.message for i in check_news_content(text, NEWS, ["INTC", "Intel Corporation"])]


def test_names_must_appear_in_cited_news():
    # nazwa spoza cytowanej wiadomości: zmyślona treść albo pomylone źródło
    assert issues("Intel podpisał umowę z Amazonem na dostawy procesorów [1].")
    assert issues("Według analizy z Yahoo chip zmniejsza zależność od Nvidii [1].")
    # odmiana polskich końcówek i nazwa analizowanej spółki
    assert not issues("Microsoft przedstawił chip Maia, który zmniejsza zależność od Nvidii [1].")
    assert not issues("Według Guthriego chip Maia jest konkurencją dla Nvidii [1].")
    assert not issues("Prezes Intela, Lip-Bu Tan, ocenił chińskie modele AI [2].")
    # nazwy ogólne i zdania bez numeru źródła nie są sprawdzane
    assert not issues("Chiny i USA rywalizują w AI, a Fed utrzymuje stopy [2].")
    assert not issues("Zarząd spółki Amazon nie skomentował sprawy.")


def test_proper_name_extraction():
    assert proper_names("Nvidia ogłosiła współpracę z Danaher i LSEG [1].") == ["Danaher", "LSEG"]
    assert stem("Microsoftem") == "microsof" and stem("Intela") == "inte"


def test_forecasts_without_attribution():
    assert check_forecasts("Kurs jest blisko dolnej wstęgi, co ogranicza przestrzeń do spadków.")
    assert check_forecasts("Zmiana {{change_1m}} wskazuje na dalszy wzrost.")
    assert check_forecasts("Duża odległość od średniej oznacza ryzyko korekty.")
    assert check_forecasts("Wąskie wstęgi oznaczają brak wahań w najbliższym okresie.")
    # opinia przypisana źródłu i opis stanu są dozwolone
    assert not check_forecasts("Według [2] akcje mają potencjał wzrostu.")
    assert not check_forecasts(
        "Analityk Dan Ives ocenił, że spadek jest przesadzony i zapowiada dalszy wzrost [1]."
    )
    assert not check_forecasts("Kurs wzrósł w ostatnim miesiącu o {{change_1m}}.")
