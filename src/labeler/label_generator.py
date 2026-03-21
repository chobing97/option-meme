"""End-to-end labeling pipeline: raw bars → labeled early session data."""

from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger
from tqdm import tqdm

from config.settings import (
    LABELED_DIR,
    LABELED_MANUAL_DIR,
    PEAK_DISTANCE,
    PEAK_PROMINENCE_PCT,
    PEAK_WIDTH,
)
from src.collector.storage import load_bars
from src.labeler.peak_trough_detector import DetectionResult, label_day
from src.labeler.session_extractor import extract_session, split_by_day


# ── Partitioned I/O helpers ──────────────────────────────────────


def save_labeled_partitioned(
    df: pd.DataFrame,
    market: str,
    label_config: str,
    timeframe: str = "1m",
) -> list[Path]:
    """Save labeled DataFrame partitioned by symbol and year.

    Groups *df* by ``symbol`` and the year component of ``datetime``,
    then writes each partition to::

        LABELED_DIR / timeframe / label_config / market / symbol / {year}.parquet

    Returns:
        List of saved file paths.
    """
    if df.empty:
        return []

    saved: list[Path] = []
    df = df.copy()
    df["_year"] = pd.to_datetime(df["datetime"]).dt.year

    for (symbol, year), part in df.groupby(["symbol", "_year"]):
        out_dir = LABELED_DIR / timeframe / label_config / market / str(symbol)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{year}.parquet"
        part.drop(columns=["_year"]).to_parquet(out_path, index=False, compression="snappy")
        saved.append(out_path)

    logger.info(
        f"Saved {len(saved)} labeled partitions for {market} [{label_config}] "
        f"({len(df)} rows total)"
    )
    return saved


def list_labeled_symbols(
    market: str,
    label_config: str,
    timeframe: str = "1m",
) -> list[str]:
    """Return sorted list of symbols that have labeled data."""
    market_dir = LABELED_DIR / timeframe / label_config / market
    if not market_dir.exists():
        return []
    return sorted(d.name for d in market_dir.iterdir() if d.is_dir())


def list_labeled_years(
    market: str,
    label_config: str,
    timeframe: str,
    symbol: str,
) -> list[int]:
    """Return sorted list of years available for a labeled symbol."""
    sym_dir = LABELED_DIR / timeframe / label_config / market / symbol
    if not sym_dir.exists():
        return []
    return sorted(
        int(p.stem) for p in sym_dir.glob("*.parquet") if p.stem.isdigit()
    )


def load_labeled(
    market: str,
    label_config: str | None = None,
    timeframe: str = "1m",
    symbol: str | None = None,
    year: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Load labeled data from the partitioned directory layout.

    Supports several access patterns:

    * ``symbol`` + ``year``: load a single partition file.
    * ``symbol`` only: load all years for that symbol.
    * Neither: load *all* symbols / years (e.g. for model training).

    ``start_date`` / ``end_date`` are applied as post-load filters on the
    ``datetime`` column when provided.

    Falls back to the legacy single-file layout
    (``LABELED_DIR / timeframe / label_config / {market}_labeled.parquet``)
    if the partitioned directory does not exist.
    """
    # Legacy single-file fallback (no label_config or directory missing)
    if label_config is None:
        legacy = LABELED_DIR / f"{market}_labeled.parquet"
        if legacy.exists():
            return pd.read_parquet(legacy)
        return pd.DataFrame()

    base_dir = LABELED_DIR / timeframe / label_config / market

    # Legacy single-file fallback when partitioned dir is absent
    if not base_dir.exists():
        legacy = LABELED_DIR / timeframe / label_config / f"{market}_labeled.parquet"
        if legacy.exists():
            return pd.read_parquet(legacy)
        return pd.DataFrame()

    # Collect parquet files to read
    files: list[Path] = []
    if symbol is not None:
        sym_dir = base_dir / symbol
        if not sym_dir.exists():
            return pd.DataFrame()
        if year is not None:
            f = sym_dir / f"{year}.parquet"
            if f.exists():
                files.append(f)
        else:
            files.extend(sorted(sym_dir.glob("*.parquet")))
    else:
        for sym_dir in sorted(base_dir.iterdir()):
            if sym_dir.is_dir():
                files.extend(sorted(sym_dir.glob("*.parquet")))

    if not files:
        return pd.DataFrame()

    dfs = [pd.read_parquet(f) for f in files]
    result = pd.concat(dfs, ignore_index=True)

    # Optional date filtering
    if start_date is not None or end_date is not None:
        result["datetime"] = pd.to_datetime(result["datetime"])
        if start_date is not None:
            result = result[result["datetime"] >= pd.Timestamp(start_date)]
        if end_date is not None:
            result = result[result["datetime"] <= pd.Timestamp(end_date)]

    return result


def label_symbol(
    market: str,
    symbol: str,
    prominence_pct: float = PEAK_PROMINENCE_PCT,
    distance: int = PEAK_DISTANCE,
    width: int = PEAK_WIDTH,
    shift: int = 1,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Generate labeled data for a single symbol.

    Pipeline:
    1. Load raw bars from Parquet
    2. Extract regular session
    3. Split by day
    4. Detect peaks/troughs per day
    5. Combine and return labeled DataFrame

    Args:
        market: 'kr' or 'us'
        symbol: Ticker symbol
        prominence_pct: Minimum peak prominence (fraction of open price)
        distance: Min distance between peaks
        width: Min peak width
        shift: Label shift (0=on peak bar, 1=next bar)
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

    # 2. Extract regular session
    session_df = extract_session(raw_df, market)
    if session_df.empty:
        logger.warning(f"No session data for {market}/{symbol}")
        return None

    # 3. Split by day and label
    days = split_by_day(session_df)
    labeled_dfs = []

    for date_str, day_df in days.items():
        if len(day_df) < 10:  # Skip days with too few bars
            continue

        result = label_day(day_df, prominence_pct, distance, width, shift=shift)

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
    shift: int = 1,
    save: bool = True,
    label_config: str | None = None,
    timeframe: str = "1m",
) -> pd.DataFrame:
    """Label all symbols for a market and optionally save to Parquet.

    When *label_config* is provided and *save* is True, data is saved in
    partitioned layout (symbol/year).  Otherwise falls back to the legacy
    single-file ``save_labeled()`` helper.

    Args:
        market: 'kr' or 'us'
        symbols: Optional list of specific symbols. If None, discover from raw dir.
        prominence_pct: Peak detection prominence threshold
        distance: Peak detection minimum distance
        width: Peak detection minimum width
        shift: Label shift (0=on peak bar, 1=next bar)
        save: Whether to save labeled data to Parquet
        label_config: Label variant key (e.g. 'L1', 'L2'). Enables partitioned save.
        timeframe: Timeframe key ('1m' or '5m').

    Returns:
        Combined labeled DataFrame.
    """
    from config.settings import RAW_STOCK_DIR

    if symbols is None:
        market_dir = RAW_STOCK_DIR / market
        if not market_dir.exists():
            logger.error(f"No raw data directory for market {market}")
            return pd.DataFrame()
        symbols = [d.name for d in market_dir.iterdir() if d.is_dir()]

    if not symbols:
        logger.warning(f"No symbols found for {market}")
        return pd.DataFrame()

    all_labeled = []
    for symbol in tqdm(symbols, desc=f"Labeling {market}"):
        labeled = label_symbol(market, symbol, prominence_pct, distance, width, shift=shift)
        if labeled is not None:
            all_labeled.append(labeled)

    if not all_labeled:
        return pd.DataFrame()

    combined = pd.concat(all_labeled, ignore_index=True)

    if save:
        if label_config is not None:
            save_labeled_partitioned(combined, market, label_config, timeframe)
        else:
            save_labeled(combined, market)

    return combined


def apply_manual_overrides(
    df: pd.DataFrame, market: str,
    label_config: str | None = None, timeframe: str = "1m",
) -> pd.DataFrame:
    """수작업 레이블로 자동 레이블을 오버라이드."""
    manual_path = None
    # Timeframe-aware path first
    if label_config:
        tf_path = LABELED_MANUAL_DIR / timeframe / label_config / f"{market}_manual.parquet"
        if tf_path.exists():
            manual_path = tf_path
        else:
            # Legacy fallback (no timeframe)
            legacy = LABELED_MANUAL_DIR / label_config / f"{market}_manual.parquet"
            if legacy.exists():
                manual_path = legacy
    else:
        legacy = LABELED_MANUAL_DIR / f"{market}_manual.parquet"
        if legacy.exists():
            manual_path = legacy

    if manual_path is None or not manual_path.exists():
        return df

    manual = pd.read_parquet(manual_path)
    if manual.empty:
        return df

    merged = df.merge(
        manual[["symbol", "datetime", "label"]],
        on=["symbol", "datetime"],
        how="left",
        suffixes=("", "_manual"),
    )
    has_override = merged["label_manual"].notna()
    merged.loc[has_override, "label"] = merged.loc[has_override, "label_manual"].astype(int)
    merged.drop(columns=["label_manual"], inplace=True)

    n_overrides = has_override.sum()
    if n_overrides > 0:
        logger.info(f"수작업 레이블 {n_overrides}건 오버라이드 적용 ({market})")

    return merged


def save_labeled(df: pd.DataFrame, market: str) -> Path:
    """Save labeled DataFrame to Parquet."""
    output_path = LABELED_DIR / f"{market}_labeled.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False, compression="snappy")
    logger.info(f"Saved labeled data to {output_path} ({len(df)} rows)")
    return output_path


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
