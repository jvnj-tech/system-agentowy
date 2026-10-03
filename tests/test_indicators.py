import numpy as np
import pandas as pd
import pytest
from ta.momentum import RSIIndicator
from ta.trend import MACD as TaMACD
from ta.trend import EMAIndicator
from ta.volatility import BollingerBands

from marketagents.indicators import bollinger_bands, compute_all, ema, macd, rsi, sma


def test_sma_hand_computed():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    assert sma(s, 3).tolist()[2:] == [2.0, 3.0, 4.0]
    assert sma(s, 3).iloc[:2].isna().all()


def test_ema_hand_computed():
    s = pd.Series([10.0, 11.0, 12.0])
    alpha = 2 / (3 + 1)
    e1 = 10.0 + alpha * (11.0 - 10.0)
    e2 = e1 + alpha * (12.0 - e1)
    assert ema(s, 3).iloc[-1] == pytest.approx(e2)


def test_ema_matches_ta_library(close):
    expected = EMAIndicator(close, window=20, fillna=False).ema_indicator()
    pd.testing.assert_series_equal(ema(close, 20), expected, check_names=False)


def _rsi_reference(values: list[float], n: int) -> float:
    """Wilder krok po kroku, niezależnie od implementacji wektorowej."""
    changes = [b - a for a, b in zip(values, values[1:], strict=False)]
    gain = sum(max(c, 0) for c in changes[:n]) / n
    loss = sum(max(-c, 0) for c in changes[:n]) / n
    for c in changes[n:]:
        gain = (gain * (n - 1) + max(c, 0)) / n
        loss = (loss * (n - 1) + max(-c, 0)) / n
    return 100 - 100 / (1 + gain / loss)


def test_rsi_matches_step_by_step_reference(close):
    result = rsi(close, 14)
    assert result.iloc[:14].isna().all()
    for end in (15, 100, 300):
        assert result.iloc[end - 1] == pytest.approx(_rsi_reference(close.iloc[:end].tolist(), 14))


def test_rsi_converges_to_ta_library(close):
    # Biblioteka ta inicjuje średnie inaczej, więc porównujemy dopiero po okresie rozgrzewki.
    expected = RSIIndicator(close, window=14, fillna=False).rsi()
    np.testing.assert_allclose(rsi(close, 14).iloc[200:], expected.iloc[200:], atol=1e-3)


def test_rsi_bounds_and_extremes(close):
    values = rsi(close, 14).dropna()
    assert ((values >= 0) & (values <= 100)).all()
    rising = pd.Series(np.arange(1.0, 31.0))
    assert rsi(rising, 14).iloc[-1] == 100.0
    flat = pd.Series([5.0] * 30)
    assert rsi(flat, 14).iloc[-1] == 50.0


def test_macd_matches_ta_library(close):
    ours = macd(close)
    ref = TaMACD(close, window_slow=26, window_fast=12, window_sign=9, fillna=False)
    assert ours["macd"].first_valid_index() == close.index[25]
    pd.testing.assert_series_equal(ours["macd"], ref.macd(), check_names=False)
    pd.testing.assert_series_equal(ours["macd_signal"], ref.macd_signal(), check_names=False)
    pd.testing.assert_series_equal(ours["macd_hist"], ref.macd_diff(), check_names=False)


def test_macd_rejects_bad_periods(close):
    with pytest.raises(ValueError):
        macd(close, fast=26, slow=12)


def test_bollinger_matches_ta_library(close):
    ours = bollinger_bands(close, 20, 2.0)
    ref = BollingerBands(close, window=20, window_dev=2, fillna=False)
    np.testing.assert_allclose(ours["bb_upper"].iloc[19:], ref.bollinger_hband().iloc[19:])
    np.testing.assert_allclose(ours["bb_lower"].iloc[19:], ref.bollinger_lband().iloc[19:])


def test_compute_all_columns(close):
    prices = pd.DataFrame({"close": close})
    out = compute_all(prices)
    assert list(out.columns) == [
        "sma_20",
        "sma_50",
        "sma_200",
        "ema_20",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_mid",
        "bb_upper",
        "bb_lower",
    ]
    assert out.iloc[-1].notna().all()


def test_custom_moving_average_windows(close):
    from marketagents.indicators import MovingAverages

    out = compute_all(pd.DataFrame({"close": close}), MovingAverages((10, 30, 150), 12))
    assert {"sma_10", "sma_30", "sma_150", "ema_12"} <= set(out.columns)
    assert "sma_200" not in out.columns
    np.testing.assert_allclose(out["sma_30"].iloc[-1], close.iloc[-30:].mean())


def test_moving_averages_validation():
    from marketagents.indicators import MovingAverages

    assert MovingAverages.parse("10, 30,150", "12") == MovingAverages((10, 30, 150), 12)
    for windows in [(50, 20, 200), (20, 20, 200), (1, 20, 200)]:
        with pytest.raises(ValueError):
            MovingAverages(windows)
    with pytest.raises(ValueError):
        MovingAverages.parse("20,50", "20")
