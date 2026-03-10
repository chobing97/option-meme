"""Parquet-based storage for 1-minute OHLCV bar data with incremental merge."""

from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

from config.settings import RAW_STOCK_DIR


# Schema for consistent Parquet files
OHLCV_SCHEMA = pa.schema([
    ("datetime", pa.timestamp("us")),
    ("open", pa.float64()),
    ("high", pa.float64()),
    ("low", pa.float64()),
    ("close", pa.float64()),
    ("volume", pa.int64()),
    ("source", pa.string()),
])


def get_parquet_path(market: str, symbol: str, year: int) -> Path:
    """Get path for a symbol's yearly Parquet file.

    Example: data/raw/stock/kr/005930/2024.parquet
    """
    path = RAW_STOCK_DIR / market / symbol / f"{year}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_bars(
    df: pd.DataFrame,
    market: str,
    symbol: str,
    source: str = "",
) -> dict[int, int]:
    """Save bars to yearly Parquet files with incremental merge.

    Splits data by year, merges with existing files, removes duplicates.

    Args:
        df: DataFrame with datetime index and OHLCV columns
        market: 'kr' or 'us'
        symbol: Ticker symbol
        source: Data source identifier (e.g. 'databento', 'yfinance', 'tvdatafeed')

    Returns:
        Dict of {year: bar_count} for saved files.
    """
    if df is None or df.empty:
        return {}

    df = _normalize_df(df, source=source)
    results = {}

    for year, year_df in df.groupby(df["datetime"].dt.year):
        path = get_parquet_path(market, symbol, year)
        merged = _merge_with_existing(year_df, path)
        _write_parquet(merged, path)
        results[year] = len(merged)
        logger.debug(f"Saved {len(merged)} bars to {path}")

    return results


def load_bars(
    market: str,
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Load bars from Parquet files for a symbol.

    Args:
        market: 'kr' or 'us'
        symbol: Ticker symbol
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)

    Returns:
        DataFrame with datetime index and OHLCV columns.
    """
    symbol_dir = RAW_STOCK_DIR / market / symbol
    if not symbol_dir.exists() and market == "kr":
        stripped = symbol.lstrip("0")
        if stripped != symbol:
            symbol_dir = RAW_STOCK_DIR / market / stripped
    if not symbol_dir.exists():
        return pd.DataFrame()

    parquet_files = sorted(symbol_dir.glob("*.parquet"))
    if not parquet_files:
        return pd.DataFrame()

    dfs = []
    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to read {pf}: {e}")

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values("datetime").reset_index(drop=True)

    if start_date:
        combined = combined[combined["datetime"] >= pd.Timestamp(start_date)]
    if end_date:
        combined = combined[combined["datetime"] <= pd.Timestamp(end_date)]

    return combined


def _normalize_df(df: pd.DataFrame, source: str = "") -> pd.DataFrame:
    """Normalize DataFrame to consistent format."""
    result = df.copy()

    # If datetime is in index, move to column
    if "datetime" not in result.columns:
        if result.index.name == "datetime" or isinstance(result.index, pd.DatetimeIndex):
            result = result.reset_index()
            if result.columns[0] != "datetime":
                result = result.rename(columns={result.columns[0]: "datetime"})

    result["datetime"] = pd.to_datetime(result["datetime"])

    # Ensure required columns exist
    required = ["datetime", "open", "high", "low", "close", "volume"]
    for col in required:
        if col not in result.columns:
            raise ValueError(f"Missing required column: {col}")

    # source 컬럼: 파라미터 > 기존 컬럼 > 빈 문자열
    if "source" not in result.columns:
        result["source"] = source
    elif source:
        result["source"] = source

    result = result[required + ["source"]].copy()
    result["volume"] = result["volume"].fillna(0).astype(int)

    return result


def _merge_with_existing(new_df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Merge new data with existing Parquet file, removing duplicates."""
    if path.exists():
        try:
            existing = pd.read_parquet(path)
            # 기존 파일에 source 컬럼이 없으면 추가 (하위 호환)
            if "source" not in existing.columns:
                existing["source"] = ""
            merged = pd.concat([existing, new_df], ignore_index=True)
        except Exception as e:
            logger.warning(f"Failed to read existing {path}, overwriting: {e}")
            merged = new_df
    else:
        merged = new_df

    # Deduplicate on datetime, keep last (newer data)
    merged = merged.drop_duplicates(subset=["datetime"], keep="last")
    merged = merged.sort_values("datetime").reset_index(drop=True)

    return merged


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write DataFrame to Parquet with consistent schema."""
    table = pa.Table.from_pandas(df, schema=OHLCV_SCHEMA, preserve_index=False)
    pq.write_table(table, path, compression="snappy")


def get_symbol_date_range(market: str, symbol: str) -> Optional[tuple[str, str]]:
    """Get the date range of collected data for a symbol."""
    df = load_bars(market, symbol)
    if df.empty:
        return None
    return (
        df["datetime"].min().strftime("%Y-%m-%d"),
        df["datetime"].max().strftime("%Y-%m-%d"),
    )


def validate_bars(df: pd.DataFrame) -> dict:
    """Run basic data quality checks on a DataFrame of bars.

    Returns dict with validation results.
    """
    if df.empty:
        return {"valid": False, "error": "empty dataframe"}

    results = {
        "valid": True,
        "bar_count": len(df),
        "date_range": (str(df["datetime"].min()), str(df["datetime"].max())),
        "null_counts": df.isnull().sum().to_dict(),
        "negative_prices": int((df[["open", "high", "low", "close"]] < 0).any(axis=1).sum()),
        "zero_volume_pct": float((df["volume"] == 0).mean()),
        "duplicate_datetimes": int(df["datetime"].duplicated().sum()),
    }

    if results["negative_prices"] > 0:
        results["valid"] = False
    if results["duplicate_datetimes"] > 0:
        results["valid"] = False

    return results
