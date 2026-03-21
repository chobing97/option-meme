"""1분봉 → 5분봉 리샘플러.

1분봉 raw data를 5분봉으로 집계하여 raw-generated 디렉토리에 저장.
세션 경계(일별)를 존중하여 일 단위로 리샘플링.
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

from config.settings import RAW_GENERATED_DIR, RAW_STOCK_DIR
from collector.storage import OHLCV_SCHEMA, load_bars


# 5분봉 저장 경로: data/raw-generated/stock/5m/{market}/{symbol}/{year}.parquet
RAW_5M_DIR = RAW_GENERATED_DIR / "stock" / "5m"


def resample_1m_to_5m(df: pd.DataFrame) -> pd.DataFrame:
    """1분봉 DataFrame을 5분봉으로 리샘플링.

    일별로 그룹핑하여 세션 경계를 넘지 않도록 처리.
    OHLCV 집계: open=first, high=max, low=min, close=last, volume=sum.

    Args:
        df: 1분봉 DataFrame (datetime 컬럼 필수, OHLCV 컬럼 필수)

    Returns:
        5분봉 DataFrame (동일 스키마)
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume", "source"])

    work = df.copy()

    # datetime 컬럼 보장
    if "datetime" not in work.columns:
        if work.index.name == "datetime" or isinstance(work.index, pd.DatetimeIndex):
            work = work.reset_index()
            if work.columns[0] != "datetime":
                work = work.rename(columns={work.columns[0]: "datetime"})
        else:
            raise ValueError("datetime 컬럼 또는 DatetimeIndex가 필요합니다")

    work["datetime"] = pd.to_datetime(work["datetime"])
    work = work.sort_values("datetime").reset_index(drop=True)

    # source 컬럼 보존
    has_source = "source" in work.columns
    source_value = work["source"].iloc[0] if has_source and len(work) > 0 else ""

    # 일별 그룹핑을 위한 date 컬럼
    if "date" in work.columns:
        work["_date"] = pd.to_datetime(work["date"]).dt.date
    else:
        work["_date"] = work["datetime"].dt.date

    resampled_parts = []
    for date, day_df in work.groupby("_date"):
        day_df = day_df.sort_values("datetime").reset_index(drop=True)
        bars_5m = _resample_day(day_df)
        if not bars_5m.empty:
            resampled_parts.append(bars_5m)

    if not resampled_parts:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume", "source"])

    result = pd.concat(resampled_parts, ignore_index=True)
    result["source"] = source_value if has_source else ""
    result = result[["datetime", "open", "high", "low", "close", "volume", "source"]]

    return result


def _resample_day(day_df: pd.DataFrame) -> pd.DataFrame:
    """하루치 1분봉을 5분봉으로 집계.

    pd.resample("5min")을 사용하여 시계열 정렬 기반 리샘플링.
    gap이 있는 데이터도 올바른 5분 윈도우로 집계됨.
    5분봉의 datetime은 윈도우의 시작 시각.
    """
    if day_df.empty:
        return pd.DataFrame()

    indexed = day_df.set_index("datetime")
    resampled = indexed.resample("5min").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    # 데이터가 없는 5분 윈도우 제거
    resampled = resampled.dropna(subset=["open"])
    resampled["volume"] = resampled["volume"].fillna(0).astype(int)
    resampled = resampled.reset_index()

    return resampled


def resample_symbol(market: str, symbol: str) -> None:
    """심볼의 1분봉 데이터를 5분봉으로 변환하여 저장.

    Args:
        market: 'kr' or 'us'
        symbol: 종목 코드
    """
    df_1m = load_bars(market, symbol)
    if df_1m.empty:
        logger.warning(f"[{market}/{symbol}] 1분봉 데이터 없음, 건너뜀")
        return

    df_5m = resample_1m_to_5m(df_1m)
    if df_5m.empty:
        logger.warning(f"[{market}/{symbol}] 리샘플링 결과 없음")
        return

    # 연도별로 분할 저장
    df_5m["datetime"] = pd.to_datetime(df_5m["datetime"])
    saved_count = 0
    for year, year_df in df_5m.groupby(df_5m["datetime"].dt.year):
        path = _get_5m_parquet_path(market, symbol, year)
        _write_parquet(year_df.reset_index(drop=True), path)
        saved_count += len(year_df)
        logger.debug(f"Saved {len(year_df)} 5m bars to {path}")

    logger.info(f"[{market}/{symbol}] 1m→5m 리샘플링 완료: {len(df_1m)} → {saved_count} bars")


def resample_all(market: str) -> None:
    """마켓 내 모든 심볼의 1분봉을 5분봉으로 리샘플링.

    Args:
        market: 'kr' or 'us'
    """
    market_dir = RAW_STOCK_DIR / market
    if not market_dir.exists():
        logger.warning(f"마켓 디렉토리 없음: {market_dir}")
        return

    symbols = sorted([d.name for d in market_dir.iterdir() if d.is_dir()])
    logger.info(f"[{market}] {len(symbols)}개 심볼 리샘플링 시작")

    for symbol in symbols:
        try:
            resample_symbol(market, symbol)
        except Exception as e:
            logger.error(f"[{market}/{symbol}] 리샘플링 실패: {e}")


def load_resampled_bars(
    market: str,
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """5분봉 리샘플링 데이터 로드.

    Args:
        market: 'kr' or 'us'
        symbol: 종목 코드
        start_date: 시작일 필터 (YYYY-MM-DD)
        end_date: 종료일 필터 (YYYY-MM-DD)

    Returns:
        5분봉 DataFrame
    """
    symbol_dir = RAW_5M_DIR / market / symbol
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


def _get_5m_parquet_path(market: str, symbol: str, year: int) -> Path:
    """5분봉 Parquet 파일 경로."""
    path = RAW_5M_DIR / market / symbol / f"{year}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Parquet 저장 (OHLCV_SCHEMA 사용)."""
    # volume을 int64로 변환
    write_df = df.copy()
    write_df["volume"] = write_df["volume"].fillna(0).astype(int)
    table = pa.Table.from_pandas(write_df, schema=OHLCV_SCHEMA, preserve_index=False)
    pq.write_table(table, path, compression="snappy")
