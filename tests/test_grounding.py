from marketagents.grounding import check_forecasts, check_news_content, proper_names, same_word

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


def test_first_word_is_checked_when_it_is_a_name_from_the_news():
    # „Microsoft” jest nazwą z wiadomości [1], więc na początku zdania z [2] jest błędem
    assert issues("Microsoft przedstawił własny chip [2].")
    assert not issues("Microsoft przedstawił własny chip [1].")
    # zwykłe słowo na początku zdania nie jest nazwą
    assert not issues("Prezes ocenił chińskie modele AI [2].")


def test_proper_name_extraction():
    assert proper_names("Nvidia ogłosiła współpracę z Danaher i LSEG [1].") == ["Danaher", "LSEG"]
    assert same_word("Microsoftem", "microsoft") and same_word("Intela", "intel")
    assert same_word("Nvidii", "nvidia") and same_word("Apple’a", "apple")
    # wspólny początek nie wystarcza, gdy słowa różnią się czymś więcej niż końcówką
    assert not same_word("Intel", "interest") and not same_word("Ford", "fordham")


def test_name_matching_uses_whole_words_of_the_source():
    news = [{"source": "Reuters", "headline": "Interest rates stay high", "summary": ""}]
    # „Intel” nie może przejść tylko dlatego, że źródło zawiera słowo „interest”
    assert check_news_content("Akcje spółki Intel tracą przez stopy [1].", news)
    headline = "Nvidia-backed CoreWeave and Coca-Cola deal"
    news = [{"source": "CNBC", "headline": headline, "summary": ""}]
    assert not check_news_content("Firma CoreWeave wspierana przez Nvidię [1].", news)
    assert not check_news_content("Umowa z Coca-Colą [1].", news)


def test_names_are_checked_in_every_section():
    from marketagents.facts import Fact, FactSheet
    from marketagents.verifier import verify

    facts = FactSheet([Fact("close", "kurs", 100.0, "USD")])
    text = "## Podsumowanie\nSpółka podpisała umowę z Amazonem [1].\n## Wiadomości\nChip Maia [1]."
    result = verify(text, facts, news=NEWS, names=("INTC", "Intel Corporation"))
    assert any("Amazonem" in i for i in result.issues())


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
    # sekcja ograniczeń i przeczenia opisują, czego raport nie obejmuje
    assert not check_forecasts("Dane nie uwzględniają zmian w najbliższym okresie.")
    # zaprzeczona prognoza nadal jest prognozą
    assert check_forecasts("Kurs nie spadnie w najbliższym okresie.")
    assert check_forecasts("Nic nie zapowiada dalszego wzrostu.")
    assert not check_forecasts("## Ograniczenia analizy\nRaport pomija ryzyko korekty.")
    assert not check_forecasts(
        "## Ograniczenia analizy\nRaport nie uwzględnia zmian w najbliższym okresie "
        "i nie przesądza, czy kurs wzrośnie."
    )
    # prognoza wprost w sekcji ograniczeń nadal jest prognozą
    assert check_forecasts("## Ograniczenia analizy\nKurs wzrośnie po publikacji wyników.")
    # słowo „analiza” nie przypisuje prognozy autorowi wiadomości
    assert check_forecasts("Analiza wiadomości [1] wskazuje na dalszy wzrost.")


def test_numbers_written_as_words():
    from marketagents.grounding import check_spelled_numbers

    assert check_spelled_numbers("Sprzedaż spadła o cztery procent [4].")
    assert check_spelled_numbers("Spadek akcji o ponad pięćdziesiąt procent [7].")
    assert check_spelled_numbers("Nvidia przekaże dwieście pięćdziesiąt milionów dolarów [3].")
    assert not check_spelled_numbers("Spółka ogłosiła dwa nowe produkty [1].")
    assert not check_spelled_numbers("Sprzedaż wyraźnie spadła [4].")
    assert check_spelled_numbers("Zysk wzrósł dwukrotnie [2].")
    assert check_spelled_numbers("Spółka podwoiła sprzedaż chipów [1].")
    assert check_spelled_numbers("Popyt spadł o połowę [3].")
    assert not check_spelled_numbers("Spółka wielokrotnie podkreślała popyt [1].")
    assert check_spelled_numbers("Popyt jest trzy razy większy [1].")
    # okres słowami: tylko okresy z karty faktów
    assert check_spelled_numbers("Kurs rośnie od pięciu sesji.")
    assert check_spelled_numbers("Akcje tracą od trzech tygodni.")
    assert not check_spelled_numbers("W ostatnich trzech miesiącach kurs wzrósł o {{change_3m}}.")
    assert not check_spelled_numbers("Przychody rosną od dwóch lat obrotowych.")


def test_records_need_a_news_source():
    from marketagents.grounding import check_records

    assert check_records("Kurs osiągnął rekordowy poziom.")
    assert check_records("Przychody są najwyższe w historii spółki.")
    assert not check_records("Spółka podała rekordowe przychody [2].")
    assert not check_records("Kurs jest blisko maksimum z 52 tygodni.")
