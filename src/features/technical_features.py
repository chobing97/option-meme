"""Technical indicator features: MA, RSI, MACD, Bollinger Bands."""

import numpy as np
import pandas as pd

from config.settings import (
    BB_PERIOD,
    BB_STD,
    MA_PERIODS,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    RSI_PERIODS,
)


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicator features.

    Args:
        df: DataFrame with OHLCV columns.

    Returns:
        DataFrame with new technical feature columns added.
    """
    result = df.copy()
    close = result["close"].astype(float)
    session_open = result.groupby("date")["open"].transform("first")

    # ── Moving Average ratios ──────────────────────────
    for period in MA_PERIODS:
        ma = close.rolling(period, min_periods=1).mean()
        result[f"tf_ma{period}_ratio"] = close / ma - 1

    # ── MA crossover signals ───────────────────────────
    ma5 = close.rolling(5, min_periods=1).mean()
    ma10 = close.rolling(10, min_periods=1).mean()
    ma20 = close.rolling(20, min_periods=1).mean()
    result["tf_ma5_10_cross"] = (ma5 - ma10) / session_open
    result["tf_ma5_20_cross"] = (ma5 - ma20) / session_open

    # ── RSI ────────────────────────────────────────────
    for period in RSI_PERIODS:
        result[f"tf_rsi{period}"] = _compute_rsi(close, period)

    # ── MACD ───────────────────────────────────────────
    ema_fast = close.ewm(span=MACD_FAST, adjust=False, min_periods=1).mean()
    ema_slow = close.ewm(span=MACD_SLOW, adjust=False, min_periods=1).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False, min_periods=1).mean()
    macd_hist = macd_line - signal_line

    result["tf_macd"] = macd_line / session_open
    result["tf_macd_signal"] = signal_line / session_open
    result["tf_macd_hist"] = macd_hist / session_open

    # ── Bollinger Bands ────────────────────────────────
    bb_ma = close.rolling(BB_PERIOD, min_periods=1).mean()
    bb_std = close.rolling(BB_PERIOD, min_periods=1).std().fillna(0)
    bb_upper = bb_ma + BB_STD * bb_std
    bb_lower = bb_ma - BB_STD * bb_std
    bb_width = bb_upper - bb_lower

    result["tf_bb_position"] = np.where(
        bb_width > 0,
        (close - bb_lower) / bb_width,
        0.5,
    )
    result["tf_bb_width"] = bb_width / session_open

    # ── Stochastic-like oscillator ─────────────────────
    roll_high = result["high"].rolling(14, min_periods=1).max()
    roll_low = result["low"].rolling(14, min_periods=1).min()
    hl_range = roll_high - roll_low
    result["tf_stoch"] = np.where(
        hl_range > 0,
        (close - roll_low) / hl_range,
        0.5,
    )

    return result


def _compute_rsi(series: pd.Series, period: int) -> pd.Series:
    """Compute RSI (Relative Strength Index) normalized to [0, 1]."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(span=period, adjust=False, min_periods=1).mean()
    avg_loss = loss.ewm(span=period, adjust=False, min_periods=1).mean()

    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100)
    rsi = 1 - 1 / (1 + rs)
    return pd.Series(rsi, index=series.index)
