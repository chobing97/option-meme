"""Inference test fixtures."""

import numpy as np
import pandas as pd
import pytest

FEATURE_COLS = [
    "pf_return",
    "pf_momentum",
    "vf_vol_ratio",
    "tf_rsi",
    "tmf_elapsed",
]


def _make_featured_df(
    n_bars_per_symbol: int = 30,
    symbols: list[str] | None = None,
    market: str = "kr",
    seed: int = 42,
) -> pd.DataFrame:
    """Build a minimal featured DataFrame for testing.

    Mimics the schema of data/processed/featured/{market}_featured.parquet:
    OHLCV + metadata columns + feature columns with correct prefixes.
    """
    if symbols is None:
        symbols = ["005930", "000660"]

    rng = np.random.RandomState(seed)
    rows = []
    base_date = pd.Timestamp("2025-01-02")

    for sym in symbols:
        dts = pd.date_range(
            base_date + pd.Timedelta("09:00:00"),
            periods=n_bars_per_symbol,
            freq="min",
        )
        base_price = 50000.0
        n = n_bars_per_symbol
        block = pd.DataFrame(
            {
                "datetime": dts,
                "open": base_price + rng.randn(n) * 100,
                "high": base_price + 200 + rng.rand(n) * 100,
                "low": base_price - 200 + rng.rand(n) * 100,
                "close": base_price + rng.randn(n) * 100,
                "volume": rng.randint(1000, 50000, size=n),
                "date": dts.date,
                "minutes_from_open": np.arange(n, dtype=np.int32),
                "label": np.zeros(n, dtype=np.int64),
                "symbol": sym,
                "market": market,
            }
        )
        # Add feature columns
        for col in FEATURE_COLS:
            block[col] = rng.randn(n).astype(np.float32)
        rows.append(block)

    return pd.concat(rows, ignore_index=True)


@pytest.fixture
def featured_df() -> pd.DataFrame:
    """60-row featured DataFrame: 2 symbols x 30 bars."""
    return _make_featured_df()


@pytest.fixture
def featured_df_with_nans(featured_df) -> pd.DataFrame:
    """featured_df with NaN/inf injected into feature columns."""
    df = featured_df.copy()
    rng = np.random.RandomState(99)
    for col in FEATURE_COLS:
        idx = rng.choice(len(df), 6, replace=False)
        df.loc[df.index[idx[:2]], col] = np.nan
        df.loc[df.index[idx[2:4]], col] = np.inf
        df.loc[df.index[idx[4:]], col] = -np.inf
    return df


@pytest.fixture
def no_feature_df() -> pd.DataFrame:
    """DataFrame without any feature-prefixed columns."""
    dts = pd.date_range("2025-01-02 09:00", periods=5, freq="min")
    return pd.DataFrame(
        {
            "datetime": dts,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1000,
            "date": dts.date,
            "minutes_from_open": range(5),
            "label": 0,
            "symbol": "TEST",
            "market": "kr",
        }
    )
