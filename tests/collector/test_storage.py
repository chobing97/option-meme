"""Tests for src.collector.storage — Parquet I/O with tmp_path."""

from unittest.mock import patch

import pandas as pd
import pytest

from src.collector.storage import (
    get_parquet_path,
    load_bars,
    save_bars,
    validate_bars,
)


@pytest.fixture(autouse=True)
def _patch_raw_dir(tmp_path):
    """Redirect RAW_STOCK_DIR to tmp_path for every test in this module."""
    with patch("src.collector.storage.RAW_STOCK_DIR", tmp_path):
        yield


# ── get_parquet_path ────────────────────────────────────────


def test_get_parquet_path_structure(tmp_path):
    path = get_parquet_path("kr", "005930", 2025)
    assert path.name == "2025.parquet"
    assert "kr" in path.parts
    assert "005930" in path.parts
    assert path.parent.exists()


# ── save_bars ───────────────────────────────────────────────


def test_save_bars_creates_file(tmp_path, sample_ohlcv_df):
    result = save_bars(sample_ohlcv_df, market="kr", symbol="005930")
    assert 2025 in result
    assert result[2025] == 10

    path = get_parquet_path("kr", "005930", 2025)
    assert path.exists()
    loaded = pd.read_parquet(path)
    assert len(loaded) == 10


def test_save_bars_splits_by_year(tmp_path, sample_ohlcv_df_multi_year):
    result = save_bars(sample_ohlcv_df_multi_year, market="kr", symbol="005930")
    assert result == {2024: 3, 2025: 3}

    assert get_parquet_path("kr", "005930", 2024).exists()
    assert get_parquet_path("kr", "005930", 2025).exists()


def test_save_bars_incremental_merge(tmp_path, sample_ohlcv_df):
    # First save: 10 rows
    save_bars(sample_ohlcv_df, market="kr", symbol="005930")

    # Second save: 5 new rows with different datetimes
    dates2 = pd.date_range("2025-01-02 09:10", periods=5, freq="min")
    df2 = sample_ohlcv_df.iloc[:5].copy()
    df2.index = dates2
    df2.index.name = "datetime"

    result = save_bars(df2, market="kr", symbol="005930")
    assert result[2025] == 15  # 10 original + 5 new


def test_save_bars_dedup_keep_last(tmp_path, sample_ohlcv_df):
    # Save original data
    save_bars(sample_ohlcv_df, market="kr", symbol="005930")

    # Save overlapping data with different prices
    df_overlay = sample_ohlcv_df.copy()
    df_overlay["close"] = 99999.0
    save_bars(df_overlay, market="kr", symbol="005930")

    path = get_parquet_path("kr", "005930", 2025)
    loaded = pd.read_parquet(path)
    assert len(loaded) == 10  # no duplicates
    assert (loaded["close"] == 99999.0).all()  # overlay wins (keep="last")


def test_save_bars_empty_returns_empty(tmp_path, empty_ohlcv_df):
    result = save_bars(empty_ohlcv_df, market="kr", symbol="005930")
    assert result == {}

    symbol_dir = tmp_path / "kr" / "005930"
    assert not symbol_dir.exists()


# ── load_bars ───────────────────────────────────────────────


def test_load_bars_with_date_filter(tmp_path, sample_ohlcv_df_multi_year):
    save_bars(sample_ohlcv_df_multi_year, market="kr", symbol="005930")

    loaded = load_bars("kr", "005930", start_date="2025-01-01", end_date="2025-12-31")
    assert len(loaded) == 3
    assert (loaded["datetime"] >= pd.Timestamp("2025-01-01")).all()


def test_load_bars_nonexistent_symbol(tmp_path):
    loaded = load_bars("kr", "NONEXISTENT")
    assert loaded.empty


# ── validate_bars ───────────────────────────────────────────


def test_validate_bars_negative_price(tmp_path, sample_ohlcv_df):
    df = sample_ohlcv_df.reset_index()
    df.loc[0, "close"] = -100.0
    result = validate_bars(df)
    assert result["valid"] is False
    assert result["negative_prices"] >= 1


def test_validate_bars_duplicate_datetime(tmp_path, sample_ohlcv_df):
    df = sample_ohlcv_df.reset_index()
    # Duplicate first row's datetime
    df.loc[1, "datetime"] = df.loc[0, "datetime"]
    result = validate_bars(df)
    assert result["valid"] is False
    assert result["duplicate_datetimes"] >= 1
