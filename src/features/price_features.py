"""Price-based features normalized relative to session open price."""

import numpy as np
import pandas as pd

from config.settings import RETURN_WINDOWS


def compute_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute price features for early session data.

    All features are normalized relative to the session open price
    to ensure cross-symbol comparability.

    Args:
        df: DataFrame with OHLCV columns and 'date' column.

    Returns:
        DataFrame with new price feature columns added.
    """
    result = df.copy()

    # Session open price (first bar's open for each day)
    session_open = result.groupby("date")["open"].transform("first")

    # ── Return from session open ───────────────────────
    result["pf_open_return"] = (result["close"] - session_open) / session_open

    # ── Multi-period returns ───────────────────────────
    for w in RETURN_WINDOWS:
        result[f"pf_return_{w}m"] = result["close"].pct_change(w)

    # ── Intraday range position (0=low, 1=high) ───────
    day_high = result.groupby("date")["high"].transform("cummax")
    day_low = result.groupby("date")["low"].transform("cummin")
    range_size = day_high - day_low
    result["pf_range_position"] = np.where(
        range_size > 0,
        (result["close"] - day_low) / range_size,
        0.5,
    )

    # ── Cumulative intraday range ──────────────────────
    result["pf_cum_range"] = (day_high - day_low) / session_open

    # ── Bar range (high-low relative to open) ─────────
    result["pf_bar_range"] = (result["high"] - result["low"]) / session_open

    # ── Upper/lower shadow ratios ──────────────────────
    bar_range = result["high"] - result["low"]
    body = (result["close"] - result["open"]).abs()
    result["pf_body_ratio"] = np.where(bar_range > 0, body / bar_range, 0)

    result["pf_upper_shadow"] = np.where(
        bar_range > 0,
        (result["high"] - result[["open", "close"]].max(axis=1)) / bar_range,
        0,
    )
    result["pf_lower_shadow"] = np.where(
        bar_range > 0,
        (result[["open", "close"]].min(axis=1) - result["low"]) / bar_range,
        0,
    )

    # ── Bar direction (1=bullish, -1=bearish, 0=doji) ─
    result["pf_bar_direction"] = np.sign(result["close"] - result["open"])

    # ── Consecutive direction count ────────────────────
    direction = result["pf_bar_direction"]
    # Count consecutive same-direction bars
    groups = (direction != direction.shift()).cumsum()
    result["pf_consec_direction"] = direction.groupby(groups).cumcount() + 1
    result["pf_consec_direction"] *= direction  # negative for bearish streaks

    # ── Price momentum (close vs recent close average) ─
    result["pf_momentum_5"] = result["close"] / result["close"].rolling(5).mean() - 1
    result["pf_momentum_10"] = result["close"] / result["close"].rolling(10).mean() - 1

    # ── Price acceleration (change of change) ──────────
    result["pf_acceleration"] = result["pf_return_1m"].diff()

    return result
