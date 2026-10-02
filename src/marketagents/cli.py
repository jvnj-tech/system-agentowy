"""Pobranie i zapisanie w pamięci podręcznej wszystkich danych dla wskazanych spółek.

Przykład: uv run marketagents-fetch AAPL MSFT NVDA
"""

from __future__ import annotations

import argparse

from marketagents.config import load_settings
from marketagents.data.cache import DiskCache
from marketagents.data.fundamentals import get_fundamentals
from marketagents.data.macro import get_macro
from marketagents.data.news import get_news
from marketagents.data.prices import get_prices
from marketagents.indicators import compute_all


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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


if __name__ == "__main__":
    main()
