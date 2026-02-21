"""Time-based features: elapsed time, cyclical day/month encoding."""

import numpy as np
import pandas as pd

from config.settings import EARLY_SESSION_MINUTES


def compute_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute time-based features.

    Args:
        df: DataFrame with 'datetime', 'date', 'minutes_from_open' columns.

    Returns:
        DataFrame with new time feature columns added.
    """
    result = df.copy()
    dt = pd.to_datetime(result["datetime"])

    # ── Minutes from market open (normalized) ──────────
    if "minutes_from_open" in result.columns:
        result["tmf_elapsed_norm"] = result["minutes_from_open"] / EARLY_SESSION_MINUTES
    else:
        # Fallback: compute from first bar per day
        first_minute = result.groupby("date")["datetime"].transform("min")
        elapsed = (dt - pd.to_datetime(first_minute)).dt.total_seconds() / 60
        result["tmf_elapsed_norm"] = elapsed / EARLY_SESSION_MINUTES

    # ── Cyclical day-of-week encoding ──────────────────
    dow = dt.dt.dayofweek  # 0=Mon, 4=Fri
    result["tmf_dow_sin"] = np.sin(2 * np.pi * dow / 5)
    result["tmf_dow_cos"] = np.cos(2 * np.pi * dow / 5)

    # ── Cyclical month encoding ────────────────────────
    month = dt.dt.month
    result["tmf_month_sin"] = np.sin(2 * np.pi * month / 12)
    result["tmf_month_cos"] = np.cos(2 * np.pi * month / 12)

    # ── Session progress (quadratic for non-linearity) ─
    if "minutes_from_open" in result.columns:
        progress = result["minutes_from_open"] / EARLY_SESSION_MINUTES
        result["tmf_progress_sq"] = progress ** 2

    return result
