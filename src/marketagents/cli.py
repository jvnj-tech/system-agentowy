"""Polecenia uruchamiane z linii poleceń.

uv run marketagents-fetch AAPL MSFT NVDA   pobranie danych do pamięci podręcznej
uv run marketagents-report AAPL            wygenerowanie raportu dla spółki
"""

from __future__ import annotations

import argparse

from marketagents.config import load_settings
from marketagents.data.cache import DiskCache
from marketagents.data.fundamentals import get_fundamentals
from marketagents.data.macro import get_macro
from marketagents.data.news import get_news
from marketagents.data.prices import get_prices
from marketagents.facts import Fact, technical_facts
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


def report(argv: list[str] | None = None) -> None:
    from marketagents.graph import build_graph
    from marketagents.llm import get_chat_model

    parser = argparse.ArgumentParser(description="Wygenerowanie raportu dla spółki.")
    parser.add_argument("ticker")
    parser.add_argument("--out", help="ścieżka pliku .md z raportem")
    args = parser.parse_args(argv)

    settings = load_settings()
    cache = DiskCache(settings.cache_dir, offline=settings.offline)

    def load_facts(ticker: str) -> list[Fact]:
        prices = get_prices(cache, ticker)
        return technical_facts(ticker, prices, compute_all(prices))

    graph = build_graph(get_chat_model(settings), load_facts)
    state = graph.invoke({"ticker": args.ticker.upper()})

    status = "zweryfikowany" if state["verified"] else "z niezweryfikowanymi liczbami"
    print(state["report"])
    print(f"\n---\nRaport {status}, liczba prób redakcji: {state['attempts']}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(state["report"])


if __name__ == "__main__":
    main()
