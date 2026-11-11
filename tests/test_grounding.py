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
    # zwykłe słowo na początku zdania nie jest nazwą, także po kursywie i dwukropku
    assert not issues("Prezes ocenił chińskie modele AI [2].")
    assert not issues("*Błąd w artykule:* Choć chip jest nowy, opis jest niepełny [1].")
    # polskie nazwy geograficzne i ogólne są pomijane
    assert not issues("Według Bank of America sprzedaż w Ameryce Północnej i Izraelu rośnie [2].")


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


NEWS_WITH_NUMBERS = [
    {"headline": "Apple Is Near a $5 Trillion Market Cap", "summary": "", "source": "Yahoo"},
    {
        "headline": "Lululemon sales fall, first decline in four quarters",
        "summary": "Sales fell 3% to $2.5 billion.",
        "source": "CNBC",
    },
]


def test_spelled_number_from_cited_news_is_allowed():
    from marketagents.grounding import check_spelled_numbers

    news = NEWS_WITH_NUMBERS
    assert not check_spelled_numbers(
        "Kapitalizacja zbliżyła się do pięciu bilionów dolarów [1].", news
    )
    assert not check_spelled_numbers(
        "To pierwszy spadek od czterech kolejnych kwartałach [2].", news
    )
    assert not check_spelled_numbers("Sprzedaż spadła o trzy procent [2].", news)
    # Inna wartość, inna skala albo brak numeru wiadomości to nadal błąd.
    assert check_spelled_numbers(
        "Kapitalizacja zbliżyła się do pięciu miliardów dolarów [1].", news
    )
    assert check_spelled_numbers("To pierwszy spadek od pięciu kolejnych kwartałach [2].", news)
    assert check_spelled_numbers("Kapitalizacja zbliżyła się do pięciu bilionów dolarów.", news)
    assert check_spelled_numbers("Sprzedaż spadła o trzy procent [1].", news)
    assert check_spelled_numbers("Sprzedaż spadła o cztery procent [2].", news)


def test_number_from_cited_news_is_not_checked_against_card():
    from marketagents.facts import FactSheet
    from marketagents.verifier import verify

    facts = FactSheet([])
    news = NEWS_WITH_NUMBERS
    assert (
        verify("Sprzedaż spadła o 3% do 2,5 mld USD [2].", facts, news=news).unverified_numbers
        == []
    )
    assert verify("Kapitalizacja zbliżyła się do 5 bln USD [1].", facts, news=news).ok
    assert verify(
        "Kapitalizacja zbliżyła się do 5 mld USD [1].", facts, news=news
    ).unverified_numbers == ["5"]
    assert verify("Sprzedaż spadła o 7% [2].", facts, news=news).unverified_numbers == ["7"]
    # Procent musi odpowiadać procentowi w źródle („four quarters” to nie 4%).
    assert verify("Sprzedaż spadła o 4% [2].", facts, news=news).unverified_numbers == ["4"]
    assert verify("Sprzedaż spadła o 3%.", facts, news=news).unverified_numbers == ["3"]


def test_number_with_thousands_separator_from_cited_news():
    from marketagents.facts import FactSheet
    from marketagents.verifier import verify

    news = [
        {
            "headline": "Intel Q3",
            "summary": "Revenue US$16,513 million, net income US$3,320 million.",
        },
        {"headline": "Starbucks plans 40,000 Stores", "summary": ""},
    ]
    facts = FactSheet([])
    text = "Sprzedaż wyniosła 16 513 mln USD, a zysk netto 3 320 mln USD [1]."
    assert verify(text, facts, news=news).unverified_numbers == []
    assert verify("Sieć ma liczyć 40 000 sklepów [2].", facts, news=news).unverified_numbers == []
    assert verify("Sieć ma liczyć 45 000 sklepów [2].", facts, news=news).unverified_numbers == [
        "45 000"
    ]


def test_range_and_typo_in_cited_news():
    from marketagents.facts import Fact, FactSheet
    from marketagents.verifier import verify

    news = [
        {
            "headline": "Intel guidance",
            "summary": "Calling for US$15.0 billion to US$16.0 billion.",
        },
        {"headline": "Starbucks", "summary": "Revenue fell 4% to $9. 1 billion."},
    ]
    facts = FactSheet([Fact("fy_ocf", "przepływy operacyjne", 4.7, "mld USD", 1)])
    text = "Prognoza przychodów to 15,0‑16,0 mld USD [1]."
    assert verify(text, facts, news=news).unverified_numbers == []
    assert verify("Przychody spadły o 4% do 9,1 mld USD [2].", facts, news=news).ok
    # W zdaniu z numerem wiadomości nie ma podpowiedzi zaokrąglonego faktu z karty.
    result = verify("Spółka otworzy 5 nowych sklepów [2].", facts, news=news)
    assert result.unverified_numbers == ["5"] and result.suggestions == {}


def test_year_from_cited_news_is_allowed():
    from marketagents.facts import FactSheet
    from marketagents.verifier import verify

    news = [{"headline": "Starbucks Unveils 2027 Growth Plans", "summary": ""}]
    assert verify("Spółka przedstawiła plany na 2027 rok [1].", FactSheet([]), news=news).ok
    assert verify(
        "Spółka przedstawiła plany na 2028 rok [1].", FactSheet([]), news=news
    ).wrong_years
    assert verify("Spółka przedstawiła plany na 2027 rok.", FactSheet([]), news=news).wrong_years
