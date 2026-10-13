# System agentowy do analizy danych rynkowych

System oparty na modelach językowych, który dla wybranej spółki z indeksu NASDAQ-100
łączy dane cenowe, fundamentalne, newsy i wskaźniki makroekonomiczne, a następnie
generuje raport analityczny z interaktywnymi wykresami i pozwala zadawać pytania o jego treść.

## Stan prac

- [x] Szkielet projektu, testy i lint
- [x] Warstwa danych z pamięcią podręczną (ceny, fundamenty, newsy, makro)
- [x] Wskaźniki techniczne z testami
- [x] Graf agentów (LangGraph): czterech analityków, redaktor i weryfikator liczb
- [x] Interfejs (Streamlit + Plotly) z asystentem pytań o raport
- [x] Ewaluacja jakości raportów (`marketagents-eval`)
- [ ] Ewaluacja: backtesting wskaźników

## Źródła danych

| Dane | Źródło | Klucz API |
| --- | --- | --- |
| Notowania dzienne OHLCV | Yahoo Finance (`yfinance`) | nie |
| Sprawozdania finansowe (10-K) | SEC EDGAR, API XBRL `companyfacts` | nie, wymagany `SEC_USER_AGENT` |
| Newsy | Finnhub, `company-news` | tak, darmowy |
| Makro (stopa Fed, CPI, rentowność 10Y, bezrobocie) | FRED | tak, darmowy |

## Uruchomienie

Wymagany jest [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env   # uzupełnij klucze
uv run marketagents-fetch AAPL MSFT NVDA
uv run marketagents-report AAPL --out raport_AAPL.md
```

Interfejs w przeglądarce:

```bash
uv run streamlit run src/marketagents/app.py
```

W panelu bocznym wybiera się spółkę z listy NASDAQ-100 lub wpisuje symbol. Lista to skład
indeksu z danego dnia pobierany z serwisu Nasdaq (odświeżany co 12 godzin); bez dostępu
do serwisu używana jest kopia z pliku `data/nasdaq100.csv` z datą stanu w nagłówku. Aplikacja ma trzy
zakładki: raport (ze statusem weryfikacji liczb i kartą faktów), interaktywne wykresy
oraz pytania, w których asystent odpowiada na pytania o wygenerowany raport. Wygenerowany
raport jest przechowywany w sesji, więc nie powstaje ponownie przy każdej interakcji.

Wykresy powstają z danych bez udziału modelu językowego:

- świece ze średnimi SMA (domyślnie 20/50/200) i wstęgami Bollingera, wolumen,
  RSI(14) z poziomami 30/70, MACD,
- kurs spółki na tle NASDAQ-100 (ETF QQQ), oba znormalizowane do 100 na początku okresu,
- obsunięcie od szczytu i zmienność krocząca z 30 sesji w skali roku,
- mapa miesięcznych stóp zwrotu (rok × miesiąc),
- przychody, zysk netto oraz marże brutto, operacyjna i netto ze sprawozdań 10-K,
- stopa Fed, rentowność 10-letnich obligacji i inflacja CPI r/r.

Wykres, dla którego brakuje danych (np. bez klucza FRED), jest pomijany.

Okna średnich kroczących (trzy SMA i jedna EMA) ustawiają zmienne `SMA_WINDOWS`
i `EMA_WINDOW` w pliku `.env`; trafiają one do wykresów, karty faktów i raportu.

Raport wymaga modelu językowego. Modele podaje się w zmiennej `LLM_MODELS` w kolejności
preferencji, w zapisie `dostawca:model`, np.
`groq:openai/gpt-oss-120b,gemini:gemini-3.8-flash` (klucze `GROQ_API_KEY` z konsoli Groq
i `GOOGLE_API_KEY` z Google AI Studio). Darmowe plany mają limity na minutę i na dobę
(np. 200 tys. tokenów dziennie dla gpt-oss-120b w Groq, czyli ok. 10 raportów). Limit
minutowy system odczekuje na tym samym modelu, a po wyczerpaniu limitu dobowego albo przy
dłuższym przeciążeniu serwera raport zaczyna się od nowa na następnym modelu z listy, tak
żeby cały raport napisał jeden model. Wyczerpany model jest pomijany do czasu odnowienia
limitu podanego przez serwer (stan w `.cache/limity_modeli.json`). Aplikacja pokazuje,
który model napisał raport. Asystent pytań przełącza model przy pojedynczym pytaniu.
Bez `LLM_MODELS` używany jest jeden model z `LLM_PROVIDER` (`GEMINI_MODEL` albo `GROQ_MODEL`).

Do puli trafiają tylko modele, które zaliczyły test kwalifikacyjny:

```bash
uv run marketagents-eval kwalifikacja                      # modele z LLM_MODELS
uv run marketagents-eval kwalifikacja groq:qwen/qwen3.8-27b --spolki AAPL INTC
```

Test uruchamia raporty dla AAPL (wzrost kursu), LULU (spadek), INTC (straty, brak C/Z)
i SBUX (ujemny kapitał własny) na tych samych zapisanych danych (`wyniki/kwalifikacja_dane/`)
i wymaga, by każdy agent zwrócił niepustą odpowiedź, raport był kompletny, po poprawkach
zostało w nim najwyżej jedno oznaczenie, a raport powstał w ciągu 10 minut. Wyniki trafiają
do `wyniki/kwalifikacja.json`. Dopóki żaden model z listy nie był testowany, lista nie jest
zawężana; zmienna `LLM_REQUIRE_QUALIFIED=0` wyłącza ten warunek.
Brak klucza do Finnhub, FRED lub `SEC_USER_AGENT` nie przerywa pracy: pomijana jest tylko
odpowiednia część analizy.

Pobrane dane trafiają do katalogu `.cache/`. Notowania i wiadomości są pobierane ponownie
po 6 godzinach, dane makro po dobie, a sprawozdania z SEC po tygodniu; gdy pobranie się nie
uda, używane są zapisane dane, a ich data trafia do raportu. Po ustawieniu
`MARKETAGENTS_OFFLINE=1` system korzysta wyłącznie z zapisanych danych, co pozwala powtarzać
testy na tym samym zbiorze.

## Jak powstaje raport

```
                       /-> analityk techniczny     -\
dane -> karta faktów -+--> analityk fundamentalny --+-> redaktor -> weryfikator -> raport
                       +-> analityk newsów         -+      ^             |
                       \-> analityk makro          -/      +-- poprawki -+
```

Analitycy działają równolegle i każdy dostaje tylko fakty ze swojej dziedziny.
Szósty agent, asystent pytań, odpowiada na pytania o gotowy raport na podstawie raportu
i karty faktów, z tymi samymi zasadami odwołań i weryfikacją liczb.

Liczby liczy kod, a model językowy tylko je interpretuje. Agenci nie wpisują wartości
do tekstu, lecz odwołania do karty faktów w postaci `{{rsi_14}}`, które są podstawiane
przy składaniu raportu. Weryfikator sprawdza, czy:

- każde odwołanie istnieje, a w tekście nie ma liczby wpisanej poza odwołaniem (poza okresami
  wskaźników, np. „RSI(14)”). Liczba wpisana wprost jest błędem nawet wtedy, gdy zgadza się
  z kartą, bo nie wiadomo, której wielkości dotyczy; autor dostaje podpowiedź, którego
  odwołania użyć,
- lata wymienione w tekście odpowiadają datom z karty faktów,
- opis jest zgodny z liczbą: kierunek zmiany (wzrost/spadek), strefy RSI (wykupienie powyżej
  70, wyprzedanie poniżej 30) oraz położenie kursu względem średniej kroczącej,
- okres zmiany jest zgodny z kartą: zmiana z ostatniego miesiąca nie może być opisana
  jako zmiana roczna,
- opisy położenia i układu zgadzają się z ocenami kodu: położenie kursu we wstęgach
  Bollingera i w zakresie z 52 tygodni, układ średnich, linia MACD względem linii sygnału,
  szerokość wstęg oraz to, czy odległość od długiej średniej jest duża na tle 3 lat,
- zmiana w czasie („rośnie”, „utrzymuje się”, „kontynuacja”) jest opisana tylko przy
  wielkości, dla której karta zawiera fakt o zmianie (np. zmianę stopy Fed w ciągu 12 miesięcy),
- każde zdanie w sekcji wiadomości wskazuje numer wiadomości z listy, np. [2]; w gotowym
  raporcie numer zamienia się w odnośnik do artykułu.

Karta faktów, którą dostaje model, ma kolumnę „ocena” z oceną słowną wyznaczoną przez kod
(kierunek zmiany, strefa RSI, położenie kursu, układ średnich, szerokość wstęg).
Analityk wiadomości dostaje tylko wiadomości, które wymieniają spółkę (nazwę lub symbol),
oraz zmiany kursu z karty, żeby mógł wskazać, gdy opinia z artykułu przeczy danym.
Na końcu raportu kod dopisuje zakres danych: daty notowań, sprawozdania rocznego, odczytów
makro i wiadomości, ostrzeżenie, gdy notowania są starsze niż wiadomości, oraz listę
brakujących wielkości.

Każda analiza cząstkowa jest sprawdzana od razu i analityk może ją raz poprawić. Gotowy
raport w razie błędów wraca do redaktora z ich listą (najwyżej dwa razy), a to, czego nie
udało się poprawić, jest oznaczane w raporcie jako „[niezweryfikowane]”,
„[niezweryfikowana interpretacja]” lub „[brak źródła]”. Twierdzeń bez liczb i ocen
(np. o popycie konsumentów) weryfikator nie sprawdza. Każda liczba w raporcie ma w aplikacji podpowiedź
ze źródłem danych i sposobem obliczenia, widoczną po kliknięciu liczby lub najechaniu na nią.

## Ewaluacja

Każdy raport wygenerowany w aplikacji jest dopisywany do pliku `wyniki/raporty.jsonl`
(czas, wywołania modelu, tokeny, poprawki analityków, liczba wersji redakcji, problemy
w kolejnych wersjach, oznaczone liczby i zdania). Polecenie `marketagents-eval` zbiera
z niego wyniki:

```bash
uv run marketagents-eval raporty AAPL MSFT NVDA          # raporty z kartą faktów
uv run marketagents-eval raporty AAPL MSFT --bez-karty   # model sam przepisuje liczby
uv run marketagents-eval podsumowanie                    # tabela wyników
uv run marketagents-eval bledy --sedzia --csv bledy.csv  # wstrzyknięte błędy
```

Wariant bez karty faktów pokazuje, ile liczb spoza danych pojawia się, gdy model sam
przepisuje wartości i nikt ich nie sprawdza. Test wstrzykniętych błędów zmienia
w zweryfikowanych raportach po jednej liczbie, opisie, okresie lub roku i sprawdza,
ile zmian wykrywa weryfikator, a ile model językowy w roli sędziego (opcja `--sedzia`).

## Testy

```bash
uv run pytest
uv run ruff check .
```

Testy grafu korzystają z modelu zastępczego, więc nie wymagają klucza API.
Wskaźniki techniczne są sprawdzane na wartościach liczonych ręcznie, na niezależnej
implementacji krok po kroku oraz na wynikach biblioteki `ta`.

## Struktura

```
src/marketagents/
  config.py              konfiguracja ze zmiennych środowiskowych
  cli.py                 polecenia: pobieranie danych i generowanie raportu
  facts.py               karta faktów
  verifier.py            weryfikacja liczb, lat i numerów źródeł w raporcie
  interpretation.py      zgodność opisu z liczbami i ocenami kodu
  coverage.py            opis zakresu danych dopisywany do raportu
  evaluation.py          ewaluacja: wyniki raportów, wariant bez karty, wstrzyknięte błędy
  graph.py               graf agentów (LangGraph)
  prompts.py             instrukcje dla agentów
  llm.py                 pula modeli (Gemini, Groq) z przełączaniem po wyczerpaniu limitu
  qualification.py       test kwalifikacyjny modeli do puli
  qa.py                  asystent pytań o raport
  analytics.py           miary do wykresów (obsunięcie, zmienność, zwroty miesięczne, marże)
  charts.py              wykresy Plotly
  app.py                 interfejs Streamlit
  data/
    cache.py             pamięć podręczna na dysku
    prices.py            notowania (yfinance)
    fundamentals.py      dane finansowe (SEC EDGAR)
    news.py              newsy (Finnhub)
    macro.py             dane makro (FRED)
    universe.py          lista spółek
  indicators/
    technical.py         SMA, EMA, RSI, MACD, wstęgi Bollingera
tests/
```
