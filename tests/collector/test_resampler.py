"""Tests for collector/resampler.py — 1분봉→5분봉 리샘플링 검증."""

from unittest.mock import patch, MagicMock
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from collector.resampler import (
    resample_1m_to_5m,
    resample_symbol,
    load_resampled_bars,
    RAW_5M_DIR,
)


def _make_1m_df(n_bars: int, start: str = "2025-01-02 09:00", source: str = "") -> pd.DataFrame:
    """테스트용 1분봉 DataFrame 생성."""
    dates = pd.date_range(start, periods=n_bars, freq="min")
    rng = np.random.RandomState(42)
    base = 100.0
    df = pd.DataFrame({
        "datetime": dates,
        "open": base + rng.randn(n_bars),
        "high": base + 2 + rng.rand(n_bars),
        "low": base - 2 + rng.rand(n_bars),
        "close": base + rng.randn(n_bars),
        "volume": rng.randint(100, 1000, size=n_bars),
    })
    if source:
        df["source"] = source
    return df


class TestResample1mTo5m:
    """resample_1m_to_5m 단위 테스트."""

    def test_basic_aggregation(self):
        """5개 1분봉 → 1개 5분봉: OHLCV 집계 규칙 검증."""
        df = pd.DataFrame({
            "datetime": pd.date_range("2025-01-02 09:00", periods=5, freq="min"),
            "open": [100, 101, 102, 103, 104],
            "high": [105, 110, 108, 107, 106],
            "low": [95, 96, 97, 98, 99],
            "close": [101, 102, 103, 104, 105],
            "volume": [10, 20, 30, 40, 50],
        })
        result = resample_1m_to_5m(df)

        assert len(result) == 1
        row = result.iloc[0]
        assert row["open"] == 100         # first
        assert row["high"] == 110         # max
        assert row["low"] == 95           # min
        assert row["close"] == 105        # last
        assert row["volume"] == 150       # sum

    def test_datetime_alignment(self):
        """5분봉 datetime은 첫 번째 1분봉의 datetime."""
        df = _make_1m_df(10)
        result = resample_1m_to_5m(df)

        assert result["datetime"].iloc[0] == pd.Timestamp("2025-01-02 09:00")
        assert result["datetime"].iloc[1] == pd.Timestamp("2025-01-02 09:05")

    def test_10_bars_makes_2_5m_bars(self):
        """10개 1분봉 → 2개 5분봉."""
        df = _make_1m_df(10)
        result = resample_1m_to_5m(df)
        assert len(result) == 2

    def test_empty_dataframe(self):
        """빈 DataFrame → 빈 DataFrame 반환."""
        df = pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
        result = resample_1m_to_5m(df)
        assert result.empty

    def test_none_input(self):
        """None 입력 → 빈 DataFrame 반환."""
        result = resample_1m_to_5m(None)
        assert result.empty

    def test_partial_bars_at_session_end(self):
        """세션 끝에 3개 남은 경우에도 1개 5분봉 생성."""
        df = _make_1m_df(8)  # 5 + 3
        result = resample_1m_to_5m(df)
        assert len(result) == 2  # 5개 그룹 + 3개 그룹

    def test_per_day_resampling(self):
        """일별 경계를 넘지 않고 리샘플링."""
        # 2일치 데이터: 각 5분봉씩
        day1 = _make_1m_df(5, start="2025-01-02 09:00")
        day2 = _make_1m_df(5, start="2025-01-03 09:00")
        df = pd.concat([day1, day2], ignore_index=True)
        result = resample_1m_to_5m(df)

        assert len(result) == 2
        # 각 5분봉이 해당 일자에 맞는지 확인
        assert result["datetime"].iloc[0].date() == pd.Timestamp("2025-01-02").date()
        assert result["datetime"].iloc[1].date() == pd.Timestamp("2025-01-03").date()

    def test_source_column_preserved(self):
        """source 컬럼이 있으면 보존."""
        df = _make_1m_df(10, source="databento")
        result = resample_1m_to_5m(df)

        assert "source" in result.columns
        assert result["source"].iloc[0] == "databento"

    def test_source_column_absent(self):
        """source 컬럼이 없어도 빈 문자열로 생성."""
        df = _make_1m_df(5)
        assert "source" not in df.columns
        result = resample_1m_to_5m(df)
        assert "source" in result.columns
        assert result["source"].iloc[0] == ""

    def test_single_bar(self):
        """1개 1분봉 → 1개 5분봉 (불완전 바)."""
        df = _make_1m_df(1)
        result = resample_1m_to_5m(df)
        assert len(result) == 1

    def test_gap_in_data(self):
        """1분봉에 gap이 있어도 올바른 5분 윈도우로 집계."""
        # 09:00~09:03 (4봉) + gap + 09:08~09:09 (2봉) = 6봉
        times = pd.to_datetime([
            "2025-01-02 09:00", "2025-01-02 09:01",
            "2025-01-02 09:02", "2025-01-02 09:03",
            "2025-01-02 09:08", "2025-01-02 09:09",
        ])
        df = pd.DataFrame({
            "datetime": times,
            "open": [100, 101, 102, 103, 108, 109],
            "high": [105, 106, 107, 108, 113, 114],
            "low": [95, 96, 97, 98, 103, 104],
            "close": [101, 102, 103, 104, 109, 110],
            "volume": [10, 20, 30, 40, 80, 90],
        })
        result = resample_1m_to_5m(df)

        # 09:00~09:04 윈도우: 4봉 (09:00~09:03)
        # 09:05~09:09 윈도우: 2봉 (09:08~09:09)
        assert len(result) == 2
        assert result["datetime"].iloc[0] == pd.Timestamp("2025-01-02 09:00")
        assert result["datetime"].iloc[1] == pd.Timestamp("2025-01-02 09:05")
        # 첫 윈도우: open=first(100), close=last(104), volume=sum(100)
        assert result["open"].iloc[0] == 100
        assert result["close"].iloc[0] == 104
        assert result["volume"].iloc[0] == 100
        # 둘째 윈도우: open=first(108), close=last(110)
        assert result["open"].iloc[1] == 108
        assert result["close"].iloc[1] == 110


class TestResampleSymbol:
    """resample_symbol 통합 테스트 (I/O mock)."""

    @patch("collector.resampler._write_parquet")
    @patch("collector.resampler._get_5m_parquet_path")
    @patch("collector.resampler.load_bars")
    def test_calls_load_and_save(self, mock_load, mock_path, mock_write):
        """load_bars → resample → save 흐름 확인."""
        mock_load.return_value = _make_1m_df(10)
        mock_path.return_value = Path("/tmp/test/2025.parquet")

        resample_symbol("us", "AAPL")

        mock_load.assert_called_once_with("us", "AAPL")
        assert mock_write.called

    @patch("collector.resampler.load_bars")
    def test_empty_data_skips(self, mock_load):
        """빈 데이터면 저장하지 않음."""
        mock_load.return_value = pd.DataFrame()
        resample_symbol("us", "AAPL")
        # No error raised


class TestLoadResampledBars:
    """load_resampled_bars 경로 구성 테스트."""

    def test_nonexistent_dir_returns_empty(self):
        """존재하지 않는 디렉토리 → 빈 DataFrame."""
        result = load_resampled_bars("us", "NONEXISTENT_SYMBOL_XYZ")
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_path_construction(self):
        """경로가 RAW_5M_DIR / market / symbol 형태인지 확인."""
        expected_dir = RAW_5M_DIR / "kr" / "005930"
        assert str(expected_dir).endswith("raw-generated/stock/5m/kr/005930")
