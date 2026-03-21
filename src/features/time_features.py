"""Time-based features: elapsed time, cyclical day/month encoding."""

import numpy as np
import pandas as pd

from config.settings import KR_SESSION_MINUTES, US_SESSION_MINUTES


def compute_time_features(
    df: pd.DataFrame,
    session_minutes: int = 390,
) -> pd.DataFrame:
    """Compute time-based features.

    Args:
        df: DataFrame with 'datetime', 'date', 'minutes_from_open' columns.
        session_minutes: Total session duration in minutes (default 390).

    Returns:
        DataFrame with new time feature columns added.
    """
    result = df.copy()
    dt = pd.to_datetime(result["datetime"])

    # ── Minutes from market open (normalized) ──────────
    if "minutes_from_open" in result.columns:
        mfo = result["minutes_from_open"]
        result["tmf_elapsed_norm"] = mfo / session_minutes
    else:
        # Fallback: compute from first bar per day
        first_minute = result.groupby("date")["datetime"].transform("min")
        elapsed = (dt - pd.to_datetime(first_minute)).dt.total_seconds() / 60
        mfo = elapsed
        result["tmf_elapsed_norm"] = elapsed / session_minutes

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
        progress = result["minutes_from_open"] / session_minutes
        result["tmf_progress_sq"] = progress ** 2

    # ── New features ───────────────────────────────────
    if "minutes_from_open" in result.columns:
        mfo = result["minutes_from_open"]

        # Raw minutes from open (not normalized)
        result["tmf_minutes_from_open"] = mfo

        # Cyclical encoding of session progress
        result["tmf_elapsed_sin"] = np.sin(2 * np.pi * mfo / session_minutes)
        result["tmf_elapsed_cos"] = np.cos(2 * np.pi * mfo / session_minutes)

        # Session phase: divide session into 6 phases (0~5)
        result["tmf_session_phase"] = np.floor(mfo / (session_minutes / 6)).astype(float)

        # First/last hour indicators
        result["tmf_is_first_hour"] = (mfo < 60).astype(int)
        result["tmf_is_last_hour"] = (mfo >= (session_minutes - 60)).astype(int)

    return result
