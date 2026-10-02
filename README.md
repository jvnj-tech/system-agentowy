# System agentowy do analizy danych rynkowych

System oparty na modelach językowych, który dla wybranej spółki z indeksu NASDAQ-100
łączy dane cenowe, fundamentalne, newsy i wskaźniki makroekonomiczne, a następnie
generuje raport analityczny z interaktywnymi wykresami i pozwala zadawać pytania o jego treść.

## Stan prac

- [x] Szkielet projektu, testy i lint
- [x] Warstwa danych z pamięcią podręczną (ceny, fundamenty, newsy, makro)
- [x] Wskaźniki techniczne z testami
- [ ] Graf agentów (LangGraph) z weryfikacją liczb w raporcie
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
```

Pobrane dane trafiają do katalogu `.cache/`. Po ustawieniu `MARKETAGENTS_OFFLINE=1`
system korzysta wyłącznie z zapisanych danych, co pozwala powtarzać testy na tym samym zbiorze.

## Testy

```bash
uv run pytest
uv run ruff check .
```

Wskaźniki techniczne są sprawdzane na wartościach liczonych ręcznie, na niezależnej
implementacji krok po kroku oraz na wynikach biblioteki `ta`.

## Struktura

```
src/marketagents/
  config.py              konfiguracja ze zmiennych środowiskowych
  cli.py                 pobieranie danych z linii poleceń
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
