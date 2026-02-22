"""Unified feature engineering pipeline combining all feature modules."""

from typing import Optional

import pandas as pd
from loguru import logger

from config.settings import LOOKBACK_WINDOW
from src.features.price_features import compute_price_features
from src.features.volume_features import compute_volume_features
from src.features.technical_features import compute_technical_features
from src.features.time_features import compute_time_features


# Prefix → module mapping for feature identification
FEATURE_PREFIXES = {
    "pf_": "price",
    "vf_": "volume",
    "tf_": "technical",
    "tmf_": "time",
    "mf_": "market",
}


def build_features(
    df: pd.DataFrame,
    include_market: bool = False,
    market_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Run full feature pipeline on labeled early session data.

    Pipeline order:
    1. Price features
    2. Volume features
    3. Technical features
    4. Time features
    5. Market features (optional, requires index data)

    Args:
        df: Labeled DataFrame with OHLCV, 'date', 'minutes_from_open' columns.
        include_market: Whether to add market-level features.
        market_df: Index/ETF DataFrame for market features (required if include_market).

    Returns:
        DataFrame with all feature columns added.
    """
    logger.info(f"Building features for {len(df)} bars...")

    result = compute_price_features(df)
    result = compute_volume_features(result)
    result = compute_technical_features(result)
    result = compute_time_features(result)

    if include_market and market_df is not None:
        result = _add_market_features(result, market_df)

    # Report feature count
    feature_cols = get_feature_columns(result)
    logger.info(f"Generated {len(feature_cols)} features")

    return result


def build_lookback_features(
    df: pd.DataFrame,
    lookback: int = LOOKBACK_WINDOW,
    fill_method: str = "0fill",
) -> pd.DataFrame:
    """Create lagged feature matrix using lookback window.

    For each bar at time t, creates features from t-lookback to t-1.
    This ensures no future data leakage.

    Args:
        df: DataFrame with feature columns (output from build_features).
        lookback: Number of past bars to include. If 0, skip lag creation.
        fill_method: How to handle early bars with insufficient history.
            "drop" — drop rows with NaN lags (M1 style).
            "0fill" — fill NaN lags with 0 (M2/M3/M4 style).

    Returns:
        DataFrame with lookback features (flattened).
    """
    if lookback == 0:
        logger.debug("lookback=0: skipping lag feature creation")
        return df

    feature_cols = get_feature_columns(df)

    if not feature_cols:
        logger.warning("No feature columns found")
        return df

    result = df.copy()

    # Create lagged features (concat at once to avoid fragmentation)
    lag_cols = {
        f"{col}_lag{lag}": df[col].shift(lag)
        for lag in range(1, lookback + 1)
        for col in feature_cols
    }
    result = pd.concat([result, pd.DataFrame(lag_cols, index=df.index)], axis=1)

    if fill_method == "drop":
        # Drop rows where the furthest lag is NaN (first `lookback` rows)
        sentinel_col = f"{feature_cols[0]}_lag{lookback}"
        result = result.dropna(subset=[sentinel_col])
        logger.debug(f"Dropped early bars with incomplete lookback (fill_method=drop)")
    else:
        # Fill NaN in lag features with 0 (early bars without full lookback history)
        lag_feature_cols = [c for c in result.columns if "_lag" in c]
        result[lag_feature_cols] = result[lag_feature_cols].fillna(0)
        logger.debug(f"Filled {len(lag_feature_cols)} lag feature columns with 0 for early bars")

    return result


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get list of feature column names from DataFrame."""
    feature_cols = []
    for col in df.columns:
        for prefix in FEATURE_PREFIXES:
            if col.startswith(prefix):
                feature_cols.append(col)
                break
    return sorted(feature_cols)


def get_all_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get all feature columns including lagged versions."""
    feature_cols = []
    for col in df.columns:
        for prefix in FEATURE_PREFIXES:
            if col.startswith(prefix):
                feature_cols.append(col)
                break
    return sorted(feature_cols)


def feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics for feature columns.

    Returns DataFrame with feature name, count, mean, std, nulls, etc.
    """
    feature_cols = get_feature_columns(df)
    if not feature_cols:
        return pd.DataFrame()

    stats = df[feature_cols].describe().T
    stats["null_count"] = df[feature_cols].isnull().sum()
    stats["null_pct"] = stats["null_count"] / len(df)
    stats["inf_count"] = df[feature_cols].apply(lambda x: x.isin([float("inf"), float("-inf")]).sum())

    return stats


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clean feature values: replace inf with NaN, forward-fill within days."""
    import numpy as np

    result = df.copy()
    feature_cols = get_all_feature_columns(result)

    for col in feature_cols:
        # Replace inf/-inf with NaN
        result[col] = result[col].replace([np.inf, -np.inf], np.nan)

    # Forward-fill within each day
    if "date" in result.columns:
        for col in feature_cols:
            result[col] = result.groupby("date")[col].ffill()

    # Fill remaining NaN with 0
    result[feature_cols] = result[feature_cols].fillna(0)

    return result


def _add_market_features(
    df: pd.DataFrame,
    market_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add market-level features (index returns at same timestamp).

    Args:
        df: Symbol DataFrame with 'datetime' column.
        market_df: Index/ETF DataFrame with 'datetime', 'close' columns.

    Returns:
        DataFrame with market features added.
    """
    result = df.copy()

    if market_df.empty:
        return result

    market = market_df.copy()
    market["datetime"] = pd.to_datetime(market["datetime"])
    market = market.sort_values("datetime")

    # Market returns at various windows
    for w in [1, 3, 5, 10]:
        market[f"mf_idx_return_{w}m"] = market["close"].pct_change(w)

    # Market volatility
    market["mf_idx_volatility"] = market["close"].pct_change().rolling(10).std()

    # Merge on nearest datetime
    market_features = market[
        ["datetime"] + [c for c in market.columns if c.startswith("mf_")]
    ].copy()

    result = pd.merge_asof(
        result.sort_values("datetime"),
        market_features.sort_values("datetime"),
        on="datetime",
        direction="backward",
        tolerance=pd.Timedelta("2min"),
    )

    return result
