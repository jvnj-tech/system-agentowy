import pandas as pd
import pytest

from marketagents.data.cache import CacheMiss, DiskCache
from marketagents.data.fundamentals import annual_facts, cik_for_ticker
from marketagents.data.macro import normalize_observations
from marketagents.data.news import normalize_news
from marketagents.data.prices import normalize_ohlcv
from marketagents.data.universe import nasdaq100


def test_cache_fetches_once_and_then_reads_from_disk(tmp_path):
    cache = DiskCache(tmp_path)
    calls = []

    def fetch():
        calls.append(1)
        return {"a": 1}

    assert cache.json("src", "k", fetch) == {"a": 1}
    assert cache.json("src", "k", fetch) == {"a": 1}
    assert len(calls) == 1

    df = pd.DataFrame({"x": [1.0, 2.0]})
    cache.frame("src", "f", lambda: df)
    pd.testing.assert_frame_equal(cache.frame("src", "f", lambda: None), df)


def test_offline_cache_raises_on_miss(tmp_path):
    with pytest.raises(CacheMiss):
        DiskCache(tmp_path, offline=True).json("src", "missing", lambda: {})


def test_normalize_ohlcv():
    raw = pd.DataFrame(
        {
            "Open": [2.0, 1.0],
            "High": [2.5, 1.5],
            "Low": [1.5, 0.5],
            "Close": [2.2, 1.2],
            "Volume": [200, 100],
        },
        index=pd.DatetimeIndex(["2024-01-03", "2024-01-02"], tz="America/New_York"),
    )
    df = normalize_ohlcv(raw)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.tolist() == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert df.index.tz is None


def test_normalize_ohlcv_requires_columns():
    with pytest.raises(ValueError):
        normalize_ohlcv(pd.DataFrame({"Close": [1.0]}))


def _fact(start, end, val, filed, form="10-K", fp="FY"):
    return {"start": start, "end": end, "val": val, "filed": filed, "form": form, "fp": fp}


def test_annual_facts_keeps_full_year_10k_and_latest_filing():
    payload = {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            _fact("2022-01-01", "2022-12-31", 100.0, "2023-02-01"),
                            # Korekta tego samego roku w późniejszym raporcie.
                            _fact("2022-01-01", "2022-12-31", 105.0, "2024-02-01"),
                            _fact("2023-01-01", "2023-12-31", 120.0, "2024-02-01"),
                            # Kwartał zgłoszony w 10-K nie jest wartością roczną.
                            _fact("2023-10-01", "2023-12-31", 35.0, "2024-02-01"),
                            _fact("2024-01-01", "2024-03-31", 30.0, "2024-05-01", "10-Q", "Q1"),
                        ]
                    }
                },
                "Assets": {
                    "units": {
                        "USD": [
                            {
                                "end": "2023-12-31",
                                "val": 500.0,
                                "filed": "2024-02-01",
                                "form": "10-K",
                                "fp": "FY",
                            }
                        ]
                    }
                },
            }
        }
    }
    df = annual_facts(payload)
    assert df["revenue"].tolist() == [105.0, 120.0]
    assert df.loc[pd.Timestamp("2023-12-31"), "total_assets"] == 500.0


def test_annual_facts_prefers_first_available_concept():
    payload = {
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {"USD": [_fact("2023-01-01", "2023-12-31", 1.0, "2024-02-01")]}
                },
                "Revenues": {
                    "units": {"USD": [_fact("2023-01-01", "2023-12-31", 2.0, "2024-02-01")]}
                },
            }
        }
    }
    assert annual_facts(payload)["revenue"].tolist() == [1.0]


def test_cik_lookup_uses_cached_table(tmp_path):
    cache = DiskCache(tmp_path)
    table = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}
    cache.json("sec", "company_tickers", lambda: table)
    assert cik_for_ticker(cache, "aapl", user_agent=None) == 320193
    with pytest.raises(KeyError):
        cik_for_ticker(cache, "XXXX", user_agent=None)


def test_normalize_news_sorts_and_deduplicates():
    items = [
        {"datetime": 1700000000, "source": "A", "headline": "h1", "summary": "s", "url": "u1"},
        {"datetime": 1700003600, "source": "B", "headline": "h2", "summary": "s", "url": "u2"},
        {"datetime": 1700000000, "source": "A", "headline": "h1", "summary": "s", "url": "u1"},
    ]
    df = normalize_news(items)
    assert df["headline"].tolist() == ["h2", "h1"]
    assert normalize_news([]).empty


def test_normalize_observations_skips_missing_values():
    payload = {
        "observations": [
            {"date": "2024-01-02", "value": "4.0"},
            {"date": "2024-01-01", "value": "3.9"},
            {"date": "2024-01-03", "value": "."},
        ]
    }
    s = normalize_observations(payload)
    assert s.tolist() == [3.9, 4.0]


def test_universe_snapshot_loads_with_date():
    from marketagents.data.universe import snapshot

    copy = snapshot()
    tickers = [row["ticker"] for row in copy.companies]
    assert len(tickers) >= 100 and {"AAPL", "MSFT", "NVDA"} <= set(tickers)
    assert copy.as_of.startswith("20")
    # Bez pamięci podręcznej (np. w testach) używana jest kopia z pliku.
    assert nasdaq100() == copy.companies


def test_universe_from_nasdaq_service_and_fallback(tmp_path):
    from marketagents.data.universe import clean_name, current, parse_nasdaq

    rows = [{"symbol": f"T{i}", "companyName": f"Firma {i} Inc. Common Stock"} for i in range(95)]
    rows.append({"symbol": "asml", "companyName": "ASML Holding N.V. New York Registry Shares"})
    payload = {"data": {"date": "Oct 13, 2026", "data": {"rows": rows}}}
    index = parse_nasdaq(payload)
    assert index.as_of == "2026-10-13" and len(index.companies) == 96
    assert {"ticker": "ASML", "name": "ASML Holding N.V."} in index.companies
    assert clean_name("Alphabet Inc. Class A Common Stock") == "Alphabet Inc."
    # Niepełna lista (np. zmieniony format odpowiedzi) jest odrzucana na rzecz kopii z pliku.
    with pytest.raises(ValueError):
        parse_nasdaq({"data": {"data": {"rows": rows[:10]}}})
    cache = DiskCache(tmp_path)
    cache.json("universe", "nasdaq100", lambda: payload)
    assert current(cache).source == "api.nasdaq.com"
    broken = DiskCache(tmp_path / "pusty", offline=True)
    assert current(broken).source == "plik nasdaq100.csv"


def test_cache_refetches_after_max_age_and_falls_back_on_error(tmp_path):
    import os
    from datetime import timedelta

    cache = DiskCache(tmp_path)
    cache.json("src", "k", lambda: {"v": 1}, timedelta(hours=1))
    path = tmp_path / "src" / "k.json"
    old = path.stat().st_mtime - 2 * 3600
    os.utime(path, (old, old))
    assert cache.json("src", "k", lambda: {"v": 2}, timedelta(hours=1)) == {"v": 2}
    os.utime(path, (old, old))

    def broken():
        raise ConnectionError("brak sieci")

    # Nieudane odświeżenie: lepiej starsze dane niż żadne.
    assert cache.json("src", "k", broken, timedelta(hours=1)) == {"v": 2}
    # Bez max_age plik jest używany bez względu na wiek.
    assert cache.json("src", "k", broken) == {"v": 2}


def test_annual_facts_fills_missing_years_from_next_concept():
    # Spółka zmieniła pojęcie przychodów: nowsze lata są tylko w drugim pojęciu.
    payload = {
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {"USD": [_fact("2021-01-01", "2021-12-31", 90.0, "2022-02-01")]}
                },
                "Revenues": {
                    "units": {"USD": [_fact("2023-01-01", "2023-12-31", 120.0, "2024-02-01")]}
                },
                "CostOfRevenue": {
                    "units": {"USD": [_fact("2023-01-01", "2023-12-31", 70.0, "2024-02-01")]}
                },
                "LiabilitiesAndStockholdersEquity": {
                    "units": {"USD": [_instant("2023-12-31", 500.0)]}
                },
                "StockholdersEquity": {"units": {"USD": [_instant("2023-12-31", 200.0)]}},
            }
        }
    }
    df = annual_facts(payload)
    assert df["revenue"].tolist() == [90.0, 120.0]
    last = df.loc[pd.Timestamp("2023-12-31")]
    # Brak pojęć GrossProfit i Liabilities: wartości wyliczone z pozostałych pozycji.
    assert last["gross_profit"] == 50.0
    assert last["total_liabilities"] == 300.0


def _instant(end: str, val: float) -> dict:
    return {"end": end, "val": val, "filed": "2024-02-01", "form": "10-K", "fp": "FY"}


def test_news_filter_keeps_only_items_about_the_company():
    from marketagents.data.news import company_aliases, relevant_news

    items = [
        {"headline": "Starbucks denies takeover plan", "summary": ""},
        {"headline": "Chipotle falls", "summary": "after SBUX statement"},
        {"headline": "Consumer stocks mixed", "summary": "Sector update"},
        {"headline": "Dutch Bros could rise", "summary": "coffee chains"},
    ]
    assert company_aliases("SBUX", "STARBUCKS CORP") == ["SBUX", "STARBUCKS"]
    kept = relevant_news(items, "SBUX", "STARBUCKS CORP")
    assert [i["headline"] for i in kept] == ["Starbucks denies takeover plan", "Chipotle falls"]
    # Symbol rozpoznawany tylko jako osobne słowo pisane wielkimi literami.
    assert not relevant_news([{"headline": "Pep talk", "summary": ""}], "PEP", None)
