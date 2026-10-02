"""Polecenia uruchamiane z linii poleceń.

uv run marketagents-fetch AAPL MSFT NVDA   pobranie danych do pamięci podręcznej
uv run marketagents-report AAPL            wygenerowanie raportu dla spółki
"""

from __future__ import annotations

import argparse
import time

from marketagents.config import load_settings
from marketagents.data.cache import DiskCache
from marketagents.data.fundamentals import get_fundamentals
from marketagents.data.macro import get_macro
from marketagents.data.news import get_news
from marketagents.data.prices import get_prices
from marketagents.facts import fundamental_facts, macro_facts, technical_facts
from marketagents.indicators import compute_all


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Pobranie danych dla wskazanych spółek.")
    parser.add_argument("tickers", nargs="+")
    parser.add_argument("--start", default="2015-01-01")
    args = parser.parse_args(argv)

    settings = load_settings()
    cache = DiskCache(settings.cache_dir, offline=settings.offline)

    macro = get_macro(cache, settings.fred_api_key, args.start)
    print(f"makro: {len(macro)} obserwacji, serie: {', '.join(macro.columns)}")

    for ticker in args.tickers:
        prices = get_prices(cache, ticker, args.start)
        indicators = compute_all(prices).iloc[-1]
        fundamentals = get_fundamentals(cache, ticker, settings.sec_user_agent)
        news = get_news(cache, ticker, settings.finnhub_api_key)
        print(
            f"{ticker}: {len(prices)} notowań do {prices.index[-1].date()}, "
            f"RSI(14) = {indicators['rsi_14']:.1f}, "
            f"{len(fundamentals)} lat danych finansowych, {len(news)} newsów"
        )


STEP_NAMES = {
    "load_data": "pobieranie danych",
    "technical_analyst": "analiza techniczna",
    "fundamental_analyst": "analiza fundamentalna",
    "news_analyst": "analiza newsów",
    "macro_analyst": "analiza makro",
    "editor": "redakcja raportu",
    "verifier": "weryfikacja liczb",
    "finalize": "składanie raportu",
}


def load_market_data(settings, cache: DiskCache, ticker: str):
    """Dane dla raportu. Brak klucza do źródła pomija tylko odpowiednią część analizy."""
    from marketagents.graph import MarketData

    prices = get_prices(cache, ticker)
    close = float(prices["close"].iloc[-1])
    facts = technical_facts(ticker, prices, compute_all(prices))
    data = MarketData(facts=facts)

    sources = {
        "fundamenty": lambda: fundamental_facts(
            get_fundamentals(cache, ticker, settings.sec_user_agent), close
        ),
        "makro": lambda: macro_facts(get_macro(cache, settings.fred_api_key)),
    }
    for name, load in sources.items():
        try:
            data.facts += load()
        except Exception as exc:  # noqa: BLE001
            print(f"Pomijam {name}: {exc}")
    try:
        news = get_news(cache, ticker, settings.finnhub_api_key)
        news["published"] = news["published"].dt.strftime("%Y-%m-%d %H:%M")
        data.news = news.to_dict("records")
    except Exception as exc:  # noqa: BLE001
        print(f"Pomijam newsy: {exc}")
    return data


def report(argv: list[str] | None = None) -> None:
    from marketagents.graph import build_graph
    from marketagents.llm import get_chat_model

    parser = argparse.ArgumentParser(description="Wygenerowanie raportu dla spółki.")
    parser.add_argument("ticker")
    parser.add_argument("--out", help="ścieżka pliku .md z raportem")
    args = parser.parse_args(argv)

    settings = load_settings()
    cache = DiskCache(settings.cache_dir, offline=settings.offline)

    print("Pobieranie danych i generowanie raportu...", flush=True)
    graph = build_graph(get_chat_model(settings), lambda t: load_market_data(settings, cache, t))
    state: dict = {"ticker": args.ticker.upper()}
    for update in graph.stream(state, stream_mode="updates"):
        for node, values in update.items():
            print(f"[{time.strftime('%H:%M:%S')}] gotowe: {STEP_NAMES.get(node, node)}", flush=True)
            for key, value in (values or {}).items():
                if key == "analyses":
                    state.setdefault("analyses", {}).update(value)
                else:
                    state[key] = value
    print()

    status = "zweryfikowany" if state["verified"] else "z niezweryfikowanymi liczbami"
    print(state["report"])
    print(f"\n---\nRaport {status}, liczba prób redakcji: {state['attempts']}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(state["report"])


if __name__ == "__main__":
    main()
