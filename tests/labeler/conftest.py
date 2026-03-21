"""Labeler-specific test fixtures."""

import datetime

import numpy as np
import pandas as pd
import pytest


def _make_raw_ohlcv(start: str, periods: int, base: float, seed: int = 42) -> pd.DataFrame:
    """Helper to build a raw OHLCV DataFrame with 'datetime' column."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range(start, periods=periods, freq="min")
    df = pd.DataFrame(
        {
            "datetime": dates,
            "open": base + rng.randn(periods) * base * 0.005,
            "high": base + abs(rng.randn(periods)) * base * 0.008,
            "low": base - abs(rng.randn(periods)) * base * 0.008,
            "close": base + rng.randn(periods) * base * 0.005,
            "volume": rng.randint(1000, 50000, size=periods),
        }
    )
    return df


@pytest.fixture
def raw_ohlcv_kr() -> pd.DataFrame:
    """450-row tz-naive OHLCV: 08:30~16:00 KST, base=50000.

    Pre-market 30 min (08:30-08:59) + full session 390 min (09:00-15:29) + after 30 min (15:30-15:59).
    """
    return _make_raw_ohlcv("2025-01-02 08:30", periods=450, base=50000.0, seed=42)


@pytest.fixture
def raw_ohlcv_us() -> pd.DataFrame:
    """450-row tz-naive OHLCV: 09:00~16:30 ET, base=100.

    Pre-market 30 min (09:00-09:29) + full session 390 min (09:30-15:59) + after 30 min (16:00-16:29).
    """
    return _make_raw_ohlcv("2025-01-02 09:00", periods=450, base=100.0, seed=99)


@pytest.fixture
def session_kr(raw_ohlcv_kr) -> pd.DataFrame:
    """390-row full session DataFrame for KR market (result of extract_session)."""
    from src.labeler.session_extractor import extract_session

    return extract_session(raw_ohlcv_kr, "kr")


@pytest.fixture
def early_session_kr(raw_ohlcv_kr) -> pd.DataFrame:
    """390-row session DataFrame for KR market (backward compat alias)."""
    from src.labeler.session_extractor import extract_session

    return extract_session(raw_ohlcv_kr, "kr")


@pytest.fixture
def day_df_with_peaks() -> pd.DataFrame:
    """60-row DataFrame with sine-wave close prices producing clear peaks/troughs.

    The signal has amplitude large enough relative to base price to produce
    2-3 peaks and 2-3 troughs with default detection parameters.
    """
    n = 60
    base = 50000.0
    t = np.linspace(0, 4 * np.pi, n)
    # Sine wave: amplitude = 0.5% of base (250) -> well above 0.3% prominence threshold
    amplitude = base * 0.005
    close = base + amplitude * np.sin(t)

    dates = pd.date_range("2025-01-02 09:00", periods=n, freq="min")
    rng = np.random.RandomState(42)
    df = pd.DataFrame(
        {
            "datetime": dates,
            "open": np.full(n, base),
            "high": close + 10,
            "low": close - 10,
            "close": close,
            "volume": rng.randint(1000, 50000, size=n),
            "date": datetime.date(2025, 1, 2),
            "minutes_from_open": np.arange(n),
        }
    )
    return df
