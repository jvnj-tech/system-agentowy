import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def close() -> pd.Series:
    """Syntetyczny szereg cen (błądzenie losowe ze stałym ziarnem)."""
    rng = np.random.default_rng(42)
    values = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))
    return pd.Series(values, index=pd.bdate_range("2024-01-01", periods=300), name="close")
