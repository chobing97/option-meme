"""Model test fixtures and helpers."""

import numpy as np
import pandas as pd
import pytest

from src.model.dataset import time_based_split

FEATURE_COLS = ["pf_return", "pf_momentum", "vf_vol_ratio", "tf_rsi", "tmf_elapsed"]


@pytest.fixture
def feature_cols():
    return FEATURE_COLS.copy()


@pytest.fixture
def model_df():
    """600 rows, 20 days x 30 rows/day, spread across 2022-01 ~ 2024-12."""
    rng = np.random.RandomState(42)

    # Pick 20 trading days evenly spread across 36 months
    all_days = pd.bdate_range("2022-01-03", "2024-12-31")
    selected = all_days[np.linspace(0, len(all_days) - 1, 20, dtype=int)]

    datetimes = []
    for day in selected:
        times = pd.date_range(day + pd.Timedelta("09:00:00"), periods=30, freq="min")
        datetimes.extend(times)

    datetimes = pd.DatetimeIndex(datetimes)
    n = len(datetimes)  # 600

    base_price = 100.0
    df = pd.DataFrame(
        {
            "datetime": datetimes,
            "date": datetimes.date,
            "open": base_price + rng.randn(n) * 2,
            "high": base_price + 3 + rng.rand(n) * 2,
            "low": base_price - 3 + rng.rand(n) * 2,
            "close": base_price + rng.randn(n) * 2,
            "volume": rng.randint(1000, 50000, size=n),
        }
    )

    # Feature columns with required prefixes
    for col in FEATURE_COLS:
        df[col] = rng.randn(n).astype(np.float32)

    # Labels: 0=neither, 1=peak (~5%), 2=trough (~5%)
    labels = np.zeros(n, dtype=int)
    n_positive = int(n * 0.05)  # 30 each
    indices = rng.choice(n, n_positive * 2, replace=False)
    labels[indices[:n_positive]] = 1
    labels[indices[n_positive:]] = 2
    df["label"] = labels

    return df


@pytest.fixture
def split_result(model_df):
    return time_based_split(model_df, train_years=1, val_months=6, test_months=6)


@pytest.fixture
def model_df_with_nans(model_df):
    """model_df with NaN/inf/-inf injected into feature columns."""
    df = model_df.copy()
    rng = np.random.RandomState(99)
    for col in FEATURE_COLS:
        mask = rng.choice(len(df), 10, replace=False)
        df.loc[df.index[mask[:4]], col] = np.nan
        df.loc[df.index[mask[4:7]], col] = np.inf
        df.loc[df.index[mask[7:]], col] = -np.inf
    return df


def _make_eval_df(n, peak_idx=None, trough_idx=None):
    """Helper: minimal DataFrame for evaluate tests."""
    if peak_idx is None:
        peak_idx = []
    if trough_idx is None:
        trough_idx = []
    datetimes = pd.date_range("2024-01-02 09:00", periods=n, freq="min")
    labels = np.zeros(n, dtype=int)
    for i in peak_idx:
        labels[i] = 1
    for i in trough_idx:
        labels[i] = 2
    return pd.DataFrame(
        {
            "datetime": datetimes,
            "date": datetimes.date,
            "close": 100.0 + np.arange(n) * 0.1,
            "label": labels,
        }
    )
