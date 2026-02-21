"""Common test fixtures for option-meme tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """10-row OHLCV DataFrame with DatetimeIndex, single year (2025)."""
    dates = pd.date_range("2025-01-02 09:00", periods=10, freq="min")
    rng = np.random.RandomState(42)
    base = 50000.0
    df = pd.DataFrame(
        {
            "open": base + rng.randn(10) * 100,
            "high": base + 200 + rng.rand(10) * 100,
            "low": base - 200 + rng.rand(10) * 100,
            "close": base + rng.randn(10) * 100,
            "volume": rng.randint(1000, 50000, size=10),
        },
        index=dates,
    )
    df.index.name = "datetime"
    return df


@pytest.fixture
def sample_ohlcv_df_multi_year() -> pd.DataFrame:
    """6-row OHLCV: 3 rows in 2024-12-31, 3 rows in 2025-01-02 (year split test)."""
    dates = pd.DatetimeIndex(
        [
            "2024-12-31 09:00",
            "2024-12-31 09:01",
            "2024-12-31 09:02",
            "2025-01-02 09:00",
            "2025-01-02 09:01",
            "2025-01-02 09:02",
        ]
    )
    rng = np.random.RandomState(99)
    base = 50000.0
    df = pd.DataFrame(
        {
            "open": base + rng.randn(6) * 100,
            "high": base + 200 + rng.rand(6) * 100,
            "low": base - 200 + rng.rand(6) * 100,
            "close": base + rng.randn(6) * 100,
            "volume": rng.randint(1000, 50000, size=6),
        },
        index=dates,
    )
    df.index.name = "datetime"
    return df


@pytest.fixture
def empty_ohlcv_df() -> pd.DataFrame:
    """Empty DataFrame with OHLCV columns only."""
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
