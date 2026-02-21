"""Fixtures for features module tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def feature_df() -> pd.DataFrame:
    """60-row single-day DataFrame (2025-01-02 09:00~09:59).

    OHLCV with high >= max(open,close), low <= min(open,close), base=50000.
    """
    rng = np.random.RandomState(42)
    n = 60
    base = 50000.0

    dates = pd.date_range("2025-01-02 09:00", periods=n, freq="min")
    open_p = base + rng.randn(n) * 100
    close_p = base + rng.randn(n) * 100
    high_p = np.maximum(open_p, close_p) + rng.rand(n) * 50
    low_p = np.minimum(open_p, close_p) - rng.rand(n) * 50
    volume = rng.randint(1000, 50000, size=n)

    return pd.DataFrame(
        {
            "datetime": dates,
            "open": open_p,
            "high": high_p,
            "low": low_p,
            "close": close_p,
            "volume": volume,
            "date": dates.date,
            "minutes_from_open": np.arange(n),
        }
    )


@pytest.fixture
def flat_price_df() -> pd.DataFrame:
    """60-row DataFrame with flat prices (OHLC=50000, volume=10000)."""
    n = 60
    dates = pd.date_range("2025-01-02 09:00", periods=n, freq="min")

    return pd.DataFrame(
        {
            "datetime": dates,
            "open": 50000.0,
            "high": 50000.0,
            "low": 50000.0,
            "close": 50000.0,
            "volume": 10000,
            "date": dates.date,
            "minutes_from_open": np.arange(n),
        }
    )


@pytest.fixture
def empty_feature_df() -> pd.DataFrame:
    """0-row DataFrame with all required columns and proper dtypes."""
    return pd.DataFrame(
        {
            "datetime": pd.Series(dtype="datetime64[ns]"),
            "open": pd.Series(dtype="float64"),
            "high": pd.Series(dtype="float64"),
            "low": pd.Series(dtype="float64"),
            "close": pd.Series(dtype="float64"),
            "volume": pd.Series(dtype="float64"),
            "date": pd.Series(dtype="object"),
            "minutes_from_open": pd.Series(dtype="float64"),
        }
    )


@pytest.fixture
def two_day_df() -> pd.DataFrame:
    """120-row DataFrame spanning 2 days (2025-01-02, 2025-01-03)."""
    rng = np.random.RandomState(123)
    n = 120
    base = 50000.0

    dates = pd.DatetimeIndex(
        list(pd.date_range("2025-01-02 09:00", periods=60, freq="min"))
        + list(pd.date_range("2025-01-03 09:00", periods=60, freq="min"))
    )
    open_p = base + rng.randn(n) * 100
    close_p = base + rng.randn(n) * 100
    high_p = np.maximum(open_p, close_p) + rng.rand(n) * 50
    low_p = np.minimum(open_p, close_p) - rng.rand(n) * 50
    volume = rng.randint(1000, 50000, size=n)

    return pd.DataFrame(
        {
            "datetime": dates,
            "open": open_p,
            "high": high_p,
            "low": low_p,
            "close": close_p,
            "volume": volume,
            "date": dates.date,
            "minutes_from_open": list(range(60)) * 2,
        }
    )
