"""Unified feature engineering pipeline combining all feature modules."""

import re
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from config.settings import LOOKBACK_WINDOW, PROCESSED_DIR
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
    session_minutes: int = 390,
) -> pd.DataFrame:
    """Run full feature pipeline on labeled session data.

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
        session_minutes: Total session duration in minutes (default 390).

    Returns:
        DataFrame with all feature columns added.
    """
    logger.info(f"Building features for {len(df)} bars...")

    result = compute_price_features(df)
    result = compute_volume_features(result)
    result = compute_technical_features(result)
    result = compute_time_features(result, session_minutes=session_minutes)

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
    """Get base feature columns only (excluding lag features)."""
    return get_base_feature_columns(df)


def get_all_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get all feature columns including lagged versions."""
    feature_cols = []
    for col in df.columns:
        for prefix in FEATURE_PREFIXES:
            if col.startswith(prefix):
                feature_cols.append(col)
                break
    return sorted(feature_cols)


def get_base_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get base feature columns only (excluding lag features)."""
    return [c for c in get_all_feature_columns(df) if not re.search(r"_lag\d+$", c)]


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


# ── Partitioned I/O helpers ──────────────────────────────────────


def _featured_dir() -> Path:
    """Return the featured directory root, respecting any runtime patches on PROCESSED_DIR."""
    return PROCESSED_DIR / "featured"


def save_featured_partitioned(
    df: pd.DataFrame,
    market: str,
    label_config: str,
    model_config: str,
    timeframe: str,
    symbol: str,
    year: int,
) -> Path:
    """Save featured DataFrame for a single symbol/year partition.

    Target path::

        FEATURED_DIR / timeframe / label_config / model_config / market / symbol / {year}.parquet

    Returns:
        The written file path.
    """
    out_dir = _featured_dir() / timeframe / label_config / model_config / market / symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{year}.parquet"
    df.to_parquet(out_path, index=False, compression="snappy")
    logger.debug(f"Saved featured partition {out_path} ({len(df)} rows)")
    return out_path


def load_all_featured(
    market: str,
    label_config: str,
    model_config: str,
    timeframe: str = "1m",
) -> pd.DataFrame:
    """Load all featured partitions (symbol/year) into a single DataFrame.

    Falls back to legacy single-file layout
    (``{market}_featured.parquet``) if the partitioned directory is absent.
    """
    feat_root = _featured_dir()
    base_dir = feat_root / timeframe / label_config / model_config / market
    if base_dir.exists() and base_dir.is_dir():
        files = sorted(base_dir.rglob("*.parquet"))
        if files:
            dfs = [pd.read_parquet(f) for f in files]
            result = pd.concat(dfs, ignore_index=True)
            logger.info(
                f"Loaded {len(result)} featured rows from {len(files)} partitions "
                f"({market} {label_config}/{model_config})"
            )
            return result

    # Legacy single-file fallback
    legacy = feat_root / timeframe / label_config / model_config / f"{market}_featured.parquet"
    if legacy.exists():
        df = pd.read_parquet(legacy)
        logger.info(f"Loaded {len(df)} featured rows from legacy file {legacy}")
        return df

    logger.warning(f"No featured data found for {market} [{label_config}/{model_config}]")
    return pd.DataFrame()


def get_featured_partition_info(
    market: str,
    label_config: str,
    model_config: str,
    timeframe: str = "1m",
) -> list[dict]:
    """Get partition file info (path, symbol, num_rows) without loading data.

    Uses Parquet metadata to read row counts, consuming negligible memory.

    Returns:
        List of dicts: [{"path": Path, "symbol": str, "num_rows": int}, ...]
    """
    import pyarrow.parquet as pq

    base_dir = _featured_dir() / timeframe / label_config / model_config / market
    if not base_dir.exists():
        return []

    infos = []
    for sym_dir in sorted(base_dir.iterdir()):
        if not sym_dir.is_dir():
            continue
        for f in sorted(sym_dir.glob("*.parquet")):
            try:
                num_rows = pq.read_metadata(f).num_rows
                infos.append({"path": f, "symbol": sym_dir.name, "num_rows": num_rows})
            except Exception as e:
                logger.warning(f"Failed to read metadata for {f}: {e}")

    return infos


def build_incremental_chunks(
    market: str,
    label_config: str,
    model_config: str,
    timeframe: str = "1m",
    memory_budget_ratio: float = 0.4,
) -> list[list[dict]]:
    """Build memory-safe chunks for incremental model training.

    Uses greedy bin-packing: sort partitions by size (descending),
    assign each to the smallest chunk.

    Args:
        market: Market key.
        label_config: Label variant key.
        model_config: Model variant key.
        timeframe: Timeframe key.
        memory_budget_ratio: Fraction of available memory to use per chunk.

    Returns:
        List of chunks, each chunk is a list of partition info dicts.
    """
    import psutil

    infos = get_featured_partition_info(market, label_config, model_config, timeframe)
    if not infos:
        return []

    total_rows = sum(p["num_rows"] for p in infos)

    # Estimate bytes per row from first partition's column count
    import pyarrow.parquet as pq
    n_cols = pq.read_schema(infos[0]["path"]).names.__len__()
    bytes_per_row = n_cols * 8  # float64

    available_bytes = psutil.virtual_memory().available
    budget_bytes = int(available_bytes * memory_budget_ratio)
    max_rows_per_chunk = max(1, budget_bytes // bytes_per_row)

    n_chunks = max(1, (total_rows + max_rows_per_chunk - 1) // max_rows_per_chunk)

    logger.info(
        f"Incremental chunking: {total_rows:,} rows, {n_cols} cols, "
        f"~{bytes_per_row * total_rows / 1024**3:.1f}GB total, "
        f"budget {budget_bytes / 1024**3:.1f}GB/chunk → {n_chunks} chunks"
    )

    if n_chunks == 1:
        return [infos]

    # Greedy bin-packing: sort by size desc, assign to smallest chunk
    sorted_infos = sorted(infos, key=lambda x: x["num_rows"], reverse=True)
    chunks: list[list[dict]] = [[] for _ in range(n_chunks)]
    chunk_sizes = [0] * n_chunks

    for info in sorted_infos:
        min_idx = chunk_sizes.index(min(chunk_sizes))
        chunks[min_idx].append(info)
        chunk_sizes[min_idx] += info["num_rows"]

    for i, (chunk, size) in enumerate(zip(chunks, chunk_sizes)):
        n_symbols = len(set(p["symbol"] for p in chunk))
        logger.debug(f"  Chunk {i}: {size:,} rows, {n_symbols} symbols")

    return chunks


def load_chunk(chunk: list[dict]) -> pd.DataFrame:
    """Load a single chunk of partitions into a DataFrame.

    Args:
        chunk: List of partition info dicts from build_incremental_chunks.

    Returns:
        Combined DataFrame for this chunk.
    """
    dfs = [pd.read_parquet(p["path"]) for p in chunk]
    if not dfs:
        return pd.DataFrame()
    result = pd.concat(dfs, ignore_index=True)
    return result


def list_featured_symbols(
    market: str,
    label_config: str,
    model_config: str,
    timeframe: str = "1m",
) -> list[str]:
    """Return sorted list of symbols that have featured data."""
    base_dir = _featured_dir() / timeframe / label_config / model_config / market
    if not base_dir.exists():
        return []
    return sorted(d.name for d in base_dir.iterdir() if d.is_dir())
