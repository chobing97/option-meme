"""End-to-end labeling pipeline: raw bars → labeled early session data."""

from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger
from tqdm import tqdm

from config.settings import (
    LABELED_DIR,
    PEAK_DISTANCE,
    PEAK_PROMINENCE_PCT,
    PEAK_WIDTH,
)
from src.collector.storage import load_bars
from src.labeler.peak_trough_detector import DetectionResult, label_day
from src.labeler.session_extractor import extract_early_session, split_by_day


def label_symbol(
    market: str,
    symbol: str,
    prominence_pct: float = PEAK_PROMINENCE_PCT,
    distance: int = PEAK_DISTANCE,
    width: int = PEAK_WIDTH,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Generate labeled data for a single symbol.

    Pipeline:
    1. Load raw bars from Parquet
    2. Extract early session (first 60 min)
    3. Split by day
    4. Detect peaks/troughs per day
    5. Combine and return labeled DataFrame

    Args:
        market: 'kr' or 'us'
        symbol: Ticker symbol
        prominence_pct: Minimum peak prominence (fraction of open price)
        distance: Min distance between peaks
        width: Min peak width
        start_date: Optional start date filter
        end_date: Optional end date filter

    Returns:
        DataFrame with OHLCV + label + metadata columns, or None if no data.
    """
    # 1. Load raw data
    raw_df = load_bars(market, symbol, start_date, end_date)
    if raw_df.empty:
        logger.warning(f"No raw data for {market}/{symbol}")
        return None

    # 2. Extract early session
    early_df = extract_early_session(raw_df, market)
    if early_df.empty:
        logger.warning(f"No early session data for {market}/{symbol}")
        return None

    # 3. Split by day and label
    days = split_by_day(early_df)
    labeled_dfs = []

    for date_str, day_df in days.items():
        if len(day_df) < 10:  # Skip days with too few bars
            continue

        result = label_day(day_df, prominence_pct, distance, width)

        day_labeled = day_df.copy()
        day_labeled["label"] = result.labels
        day_labeled["symbol"] = symbol
        day_labeled["market"] = market
        labeled_dfs.append(day_labeled)

    if not labeled_dfs:
        logger.warning(f"No labeled days for {market}/{symbol}")
        return None

    # 4. Combine
    labeled = pd.concat(labeled_dfs, ignore_index=True)
    labeled = labeled.sort_values("datetime").reset_index(drop=True)

    logger.info(
        f"Labeled {market}/{symbol}: {len(labeled)} bars, "
        f"{(labeled['label'] == 1).sum()} peaks, "
        f"{(labeled['label'] == 2).sum()} troughs"
    )

    return labeled


def label_all_symbols(
    market: str,
    symbols: Optional[list[str]] = None,
    prominence_pct: float = PEAK_PROMINENCE_PCT,
    distance: int = PEAK_DISTANCE,
    width: int = PEAK_WIDTH,
    save: bool = True,
) -> pd.DataFrame:
    """Label all symbols for a market and optionally save to Parquet.

    Args:
        market: 'kr' or 'us'
        symbols: Optional list of specific symbols. If None, discover from raw dir.
        prominence_pct: Peak detection prominence threshold
        distance: Peak detection minimum distance
        width: Peak detection minimum width
        save: Whether to save labeled data to Parquet

    Returns:
        Combined labeled DataFrame.
    """
    from config.settings import RAW_DIR

    if symbols is None:
        market_dir = RAW_DIR / market
        if not market_dir.exists():
            logger.error(f"No raw data directory for market {market}")
            return pd.DataFrame()
        symbols = [d.name for d in market_dir.iterdir() if d.is_dir()]

    if not symbols:
        logger.warning(f"No symbols found for {market}")
        return pd.DataFrame()

    all_labeled = []
    for symbol in tqdm(symbols, desc=f"Labeling {market}"):
        labeled = label_symbol(market, symbol, prominence_pct, distance, width)
        if labeled is not None:
            all_labeled.append(labeled)

    if not all_labeled:
        return pd.DataFrame()

    combined = pd.concat(all_labeled, ignore_index=True)

    if save:
        save_labeled(combined, market)

    return combined


def save_labeled(df: pd.DataFrame, market: str) -> Path:
    """Save labeled DataFrame to Parquet."""
    output_path = LABELED_DIR / f"{market}_labeled.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False, compression="snappy")
    logger.info(f"Saved labeled data to {output_path} ({len(df)} rows)")
    return output_path


def load_labeled(market: str) -> pd.DataFrame:
    """Load labeled data from Parquet."""
    path = LABELED_DIR / f"{market}_labeled.parquet"
    if not path.exists():
        logger.warning(f"No labeled data at {path}")
        return pd.DataFrame()
    return pd.read_parquet(path)


def label_statistics(df: pd.DataFrame) -> dict:
    """Compute labeling statistics for quality assessment.

    Args:
        df: Labeled DataFrame with 'label', 'date', 'symbol' columns.

    Returns:
        Dict with label distribution and per-day statistics.
    """
    if df.empty:
        return {}

    label_counts = df["label"].value_counts().to_dict()
    total = len(df)

    stats = {
        "total_bars": total,
        "label_counts": label_counts,
        "label_ratios": {k: v / total for k, v in label_counts.items()},
        "n_symbols": df["symbol"].nunique() if "symbol" in df.columns else 0,
        "n_days": df["date"].nunique() if "date" in df.columns else 0,
    }

    if "date" in df.columns and "symbol" in df.columns:
        per_day = df.groupby(["date", "symbol"])["label"].agg(
            n_peaks=lambda x: (x == 1).sum(),
            n_troughs=lambda x: (x == 2).sum(),
        )
        stats["avg_peaks_per_day"] = float(per_day["n_peaks"].mean())
        stats["avg_troughs_per_day"] = float(per_day["n_troughs"].mean())
        stats["std_peaks_per_day"] = float(per_day["n_peaks"].std())
        stats["std_troughs_per_day"] = float(per_day["n_troughs"].std())

    return stats
