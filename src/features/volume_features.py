"""Volume-based features for early session analysis."""

import numpy as np
import pandas as pd

from config.settings import MA_PERIODS


def compute_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute volume features.

    Args:
        df: DataFrame with 'volume', 'close', 'date' columns.

    Returns:
        DataFrame with new volume feature columns added.
    """
    result = df.copy()
    vol = result["volume"].astype(float)

    # ── Volume MA ratios ───────────────────────────────
    for period in MA_PERIODS:
        vol_ma = vol.rolling(period, min_periods=1).mean()
        result[f"vf_vol_ma{period}_ratio"] = np.where(
            vol_ma > 0, vol / vol_ma, 1.0,
        )

    # ── Cumulative volume share within the day ─────────
    cum_vol = result.groupby("date")["volume"].cumsum()
    day_total_vol = result.groupby("date")["volume"].transform("sum")
    result["vf_cum_vol_share"] = np.where(
        day_total_vol > 0, cum_vol / day_total_vol, 0,
    )

    # ── Volume change rate ─────────────────────────────
    result["vf_vol_change"] = vol.pct_change().fillna(0).clip(-10, 10)

    # ── Volume spike (vs 20-bar average) ───────────────
    vol_ma20 = vol.rolling(20, min_periods=1).mean()
    result["vf_vol_spike"] = np.where(vol_ma20 > 0, vol / vol_ma20, 1.0)

    # ── OBV (On-Balance Volume, session-relative) ──────
    price_dir = np.sign(result["close"].diff().fillna(0))
    signed_vol = vol * price_dir
    result["vf_obv"] = result.groupby("date").apply(
        lambda g: signed_vol.loc[g.index].cumsum()
    ).reset_index(level=0, drop=True)

    # Normalize OBV by total day volume
    result["vf_obv_norm"] = np.where(
        day_total_vol > 0, result["vf_obv"] / day_total_vol, 0,
    )

    # ── VWAP deviation ─────────────────────────────────
    cum_pv = result.groupby("date").apply(
        lambda g: (g["close"] * g["volume"]).cumsum()
    ).reset_index(level=0, drop=True)
    vwap = np.where(cum_vol > 0, cum_pv / cum_vol, result["close"])
    session_open = result.groupby("date")["open"].transform("first")
    result["vf_vwap_dev"] = (result["close"] - vwap) / session_open

    return result
