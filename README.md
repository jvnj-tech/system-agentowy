# System agentowy do analizy danych rynkowych

System oparty na modelach językowych, który dla wybranej spółki z indeksu NASDAQ-100
łączy dane cenowe, fundamentalne, newsy i wskaźniki makroekonomiczne, a następnie
generuje raport analityczny z interaktywnymi wykresami i pozwala zadawać pytania o jego treść.

## Stan prac

- [x] Szkielet projektu, testy i lint
- [x] Warstwa danych z pamięcią podręczną (ceny, fundamenty, newsy, makro)
- [x] Wskaźniki techniczne z testami
- [x] Graf agentów (LangGraph): czterech analityków, redaktor i weryfikator liczb
- [ ] Interfejs (Streamlit + Plotly)
- [ ] Ewaluacja: backtesting i jakość raportów

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

Raport wymaga klucza `GOOGLE_API_KEY` (darmowy, z Google AI Studio). Model ustawia się
zmienną `GEMINI_MODEL`. Brak klucza do Finnhub, FRED lub `SEC_USER_AGENT` nie przerywa
pracy: pomijana jest tylko odpowiednia część analizy.

Pobrane dane trafiają do katalogu `.cache/`. Po ustawieniu `MARKETAGENTS_OFFLINE=1`
system korzysta wyłącznie z zapisanych danych, co pozwala powtarzać testy na tym samym zbiorze.

## Jak powstaje raport

```
                       /-> analityk techniczny     -\
dane -> karta faktów -+--> analityk fundamentalny --+-> redaktor -> weryfikator -> raport
                       +-> analityk newsów         -+      ^             |
                       \-> analityk makro          -/      +-- poprawki -+
```

Analitycy działają równolegle i każdy dostaje tylko fakty ze swojej dziedziny.

Liczby liczy kod, a model językowy tylko je interpretuje. Agenci nie wpisują wartości
do tekstu, lecz odwołania do karty faktów w postaci `{{rsi_14}}`, które są podstawiane
przy składaniu raportu. Weryfikator sprawdza, czy każde odwołanie istnieje i czy w tekście
nie pojawiła się liczba spoza karty. Jeśli tak, raport wraca do redaktora z listą błędów
(najwyżej dwa razy), a liczby, których nie udało się poprawić, są oznaczane w raporcie.

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
  verifier.py            weryfikacja liczb w raporcie
  graph.py               graf agentów (LangGraph)
  prompts.py             instrukcje dla agentów
  llm.py                 połączenie z modelem (Gemini)
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
