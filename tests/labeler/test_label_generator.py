"""Tests for src.labeler.label_generator."""

import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.labeler.label_generator import (
    label_all_symbols,
    label_statistics,
    label_symbol,
    load_labeled,
    save_labeled,
    save_labeled_partitioned,
    list_labeled_symbols,
    list_labeled_years,
)
from src.labeler.peak_trough_detector import DetectionResult


def _make_session_df(n_bars=60, date_val=datetime.date(2025, 1, 2)):
    """Helper: minimal session DF with required columns."""
    base = 50000.0
    rng = np.random.RandomState(42)
    return pd.DataFrame(
        {
            "datetime": pd.date_range(f"{date_val} 09:00", periods=n_bars, freq="min"),
            "open": np.full(n_bars, base),
            "high": base + rng.rand(n_bars) * 100,
            "low": base - rng.rand(n_bars) * 100,
            "close": base + rng.randn(n_bars) * 50,
            "volume": rng.randint(1000, 50000, size=n_bars),
            "date": date_val,
            "minutes_from_open": np.arange(n_bars),
        }
    )


def _make_detection_result(n_bars=60, date_str="2025-01-02"):
    """Helper: a DetectionResult with all-zero labels."""
    return DetectionResult(
        date=date_str,
        peak_indices=np.array([10]),
        trough_indices=np.array([20]),
        peak_prominences=np.array([100.0]),
        trough_prominences=np.array([80.0]),
        labels=np.zeros(n_bars, dtype=int),
        prices=np.ones(n_bars),
        n_bars=n_bars,
    )


# -- label_symbol -----------------------------------------------------------


_MOD = "src.labeler.label_generator"


class TestLabelSymbol:
    @patch(f"{_MOD}.label_day")
    @patch(f"{_MOD}.split_by_day")
    @patch(f"{_MOD}.extract_session")
    @patch(f"{_MOD}.load_bars")
    def test_success(self, mock_load, mock_extract, mock_split, mock_label):
        day_df = _make_session_df()
        mock_load.return_value = pd.DataFrame({"datetime": [1], "close": [1]})  # non-empty
        mock_extract.return_value = day_df
        mock_split.return_value = {"2025-01-02": day_df}
        mock_label.return_value = _make_detection_result()

        result = label_symbol("kr", "005930")

        assert result is not None
        assert "label" in result.columns
        assert "symbol" in result.columns
        assert "market" in result.columns

    @patch(f"{_MOD}.load_bars")
    def test_no_raw_data(self, mock_load):
        mock_load.return_value = pd.DataFrame()

        assert label_symbol("kr", "005930") is None

    @patch(f"{_MOD}.extract_session")
    @patch(f"{_MOD}.load_bars")
    def test_no_session_data(self, mock_load, mock_extract):
        mock_load.return_value = pd.DataFrame({"datetime": [1]})
        mock_extract.return_value = pd.DataFrame()

        assert label_symbol("kr", "005930") is None

    @patch(f"{_MOD}.label_day")
    @patch(f"{_MOD}.split_by_day")
    @patch(f"{_MOD}.extract_session")
    @patch(f"{_MOD}.load_bars")
    def test_skips_short_days(self, mock_load, mock_extract, mock_split, mock_label):
        short_df = _make_session_df(n_bars=5)
        mock_load.return_value = pd.DataFrame({"datetime": [1]})
        mock_extract.return_value = short_df
        mock_split.return_value = {"2025-01-02": short_df}

        result = label_symbol("kr", "005930")

        mock_label.assert_not_called()
        assert result is None

    @patch(f"{_MOD}.label_day")
    @patch(f"{_MOD}.split_by_day")
    @patch(f"{_MOD}.extract_session")
    @patch(f"{_MOD}.load_bars")
    def test_shift_passed_to_label_day(self, mock_load, mock_extract, mock_split, mock_label):
        """Verify that the shift parameter is forwarded to label_day."""
        day_df = _make_session_df()
        mock_load.return_value = pd.DataFrame({"datetime": [1], "close": [1]})
        mock_extract.return_value = day_df
        mock_split.return_value = {"2025-01-02": day_df}
        mock_label.return_value = _make_detection_result()

        label_symbol("kr", "005930", shift=0)

        mock_label.assert_called_once()
        _, kwargs = mock_label.call_args
        assert kwargs.get("shift") == 0

    @patch(f"{_MOD}.label_day")
    @patch(f"{_MOD}.split_by_day")
    @patch(f"{_MOD}.extract_session")
    @patch(f"{_MOD}.load_bars")
    def test_shift_default_is_1(self, mock_load, mock_extract, mock_split, mock_label):
        """Default shift should be 1."""
        day_df = _make_session_df()
        mock_load.return_value = pd.DataFrame({"datetime": [1], "close": [1]})
        mock_extract.return_value = day_df
        mock_split.return_value = {"2025-01-02": day_df}
        mock_label.return_value = _make_detection_result()

        label_symbol("kr", "005930")

        mock_label.assert_called_once()
        _, kwargs = mock_label.call_args
        assert kwargs.get("shift") == 1


# -- label_all_symbols ------------------------------------------------------


class TestLabelAllSymbols:
    @patch(f"{_MOD}.save_labeled")
    @patch(f"{_MOD}.label_symbol")
    def test_calls_per_symbol(self, mock_label_sym, mock_save):
        labeled_df = _make_session_df()
        labeled_df["label"] = 0
        labeled_df["symbol"] = "A"
        labeled_df["market"] = "us"
        mock_label_sym.return_value = labeled_df

        result = label_all_symbols("us", symbols=["AAPL", "MSFT", "GOOG"], save=False)

        assert mock_label_sym.call_count == 3

    @patch(f"{_MOD}.save_labeled")
    @patch(f"{_MOD}.label_symbol")
    def test_shift_passed_through(self, mock_label_sym, mock_save):
        """Verify that shift parameter is forwarded to label_symbol."""
        labeled_df = _make_session_df()
        labeled_df["label"] = 0
        labeled_df["symbol"] = "A"
        labeled_df["market"] = "us"
        mock_label_sym.return_value = labeled_df

        label_all_symbols("us", symbols=["AAPL"], shift=0, save=False)

        mock_label_sym.assert_called_once()
        _, kwargs = mock_label_sym.call_args
        assert kwargs.get("shift") == 0


# -- save_labeled / load_labeled --------------------------------------------


class TestSaveLoadLabeled:
    @patch(f"{_MOD}.LABELED_DIR")
    def test_save_creates_file(self, mock_dir, tmp_path):
        mock_dir.__truediv__ = lambda self, x: tmp_path / x
        mock_dir.parent = tmp_path

        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        path = save_labeled(df, "kr")

        assert (tmp_path / "kr_labeled.parquet").exists()
        loaded = pd.read_parquet(tmp_path / "kr_labeled.parquet")
        pd.testing.assert_frame_equal(loaded, df)

    @patch(f"{_MOD}.LABELED_DIR")
    def test_load_legacy_existing(self, mock_dir, tmp_path):
        mock_dir.__truediv__ = lambda self, x: tmp_path / x

        df = pd.DataFrame({"x": [10, 20]})
        df.to_parquet(tmp_path / "us_labeled.parquet", index=False)

        result = load_labeled("us")
        pd.testing.assert_frame_equal(result, df)

    @patch(f"{_MOD}.LABELED_DIR")
    def test_load_not_found(self, mock_dir, tmp_path):
        mock_dir.__truediv__ = lambda self, x: tmp_path / x

        result = load_labeled("kr")
        assert result.empty


class TestPartitionedSaveLoad:
    """Tests for partitioned labeled data storage (symbol/year layout)."""

    @patch(f"{_MOD}.LABELED_DIR")
    def test_save_partitioned_creates_files(self, mock_dir, tmp_path):
        mock_dir.__truediv__ = lambda self, x: tmp_path / x

        df = pd.DataFrame({
            "datetime": pd.to_datetime(["2024-06-01 09:00", "2025-01-02 09:00"]),
            "close": [100.0, 200.0],
            "symbol": ["AAPL", "AAPL"],
            "label": [0, 1],
        })

        saved = save_labeled_partitioned(df, "us", "L1", "1m")
        assert len(saved) == 2
        assert (tmp_path / "1m" / "L1" / "us" / "AAPL" / "2024.parquet").exists()
        assert (tmp_path / "1m" / "L1" / "us" / "AAPL" / "2025.parquet").exists()

    @patch(f"{_MOD}.LABELED_DIR")
    def test_list_labeled_symbols(self, mock_dir, tmp_path):
        mock_dir.__truediv__ = lambda self, x: tmp_path / x

        df = pd.DataFrame({
            "datetime": pd.to_datetime(["2025-01-02 09:00", "2025-01-02 09:00"]),
            "close": [100.0, 200.0],
            "symbol": ["AAPL", "MSFT"],
            "label": [0, 1],
        })
        save_labeled_partitioned(df, "us", "L1", "1m")

        symbols = list_labeled_symbols("us", "L1", "1m")
        assert symbols == ["AAPL", "MSFT"]

    @patch(f"{_MOD}.LABELED_DIR")
    def test_list_labeled_years(self, mock_dir, tmp_path):
        mock_dir.__truediv__ = lambda self, x: tmp_path / x

        df = pd.DataFrame({
            "datetime": pd.to_datetime(["2024-06-01 09:00", "2025-01-02 09:00"]),
            "close": [100.0, 200.0],
            "symbol": ["AAPL", "AAPL"],
            "label": [0, 1],
        })
        save_labeled_partitioned(df, "us", "L1", "1m")

        years = list_labeled_years("us", "L1", "1m", "AAPL")
        assert years == [2024, 2025]

    @patch(f"{_MOD}.LABELED_DIR")
    def test_load_labeled_partitioned_single_symbol(self, mock_dir, tmp_path):
        mock_dir.__truediv__ = lambda self, x: tmp_path / x

        df = pd.DataFrame({
            "datetime": pd.to_datetime(["2025-01-02 09:00", "2025-01-02 09:01"]),
            "close": [100.0, 200.0],
            "symbol": ["AAPL", "AAPL"],
            "label": [0, 1],
        })
        save_labeled_partitioned(df, "us", "L1", "1m")

        loaded = load_labeled("us", label_config="L1", timeframe="1m", symbol="AAPL")
        assert len(loaded) == 2
        assert set(loaded["symbol"]) == {"AAPL"}

    @patch(f"{_MOD}.LABELED_DIR")
    def test_load_labeled_partitioned_all(self, mock_dir, tmp_path):
        mock_dir.__truediv__ = lambda self, x: tmp_path / x

        df = pd.DataFrame({
            "datetime": pd.to_datetime(["2025-01-02 09:00", "2025-01-02 09:00"]),
            "close": [100.0, 200.0],
            "symbol": ["AAPL", "MSFT"],
            "label": [0, 1],
        })
        save_labeled_partitioned(df, "us", "L1", "1m")

        loaded = load_labeled("us", label_config="L1", timeframe="1m")
        assert len(loaded) == 2
        assert set(loaded["symbol"]) == {"AAPL", "MSFT"}

    @patch(f"{_MOD}.LABELED_DIR")
    def test_load_labeled_partitioned_by_year(self, mock_dir, tmp_path):
        mock_dir.__truediv__ = lambda self, x: tmp_path / x

        df = pd.DataFrame({
            "datetime": pd.to_datetime(["2024-06-01 09:00", "2025-01-02 09:00"]),
            "close": [100.0, 200.0],
            "symbol": ["AAPL", "AAPL"],
            "label": [0, 1],
        })
        save_labeled_partitioned(df, "us", "L1", "1m")

        loaded = load_labeled("us", label_config="L1", timeframe="1m", symbol="AAPL", year=2025)
        assert len(loaded) == 1
        assert loaded["close"].iloc[0] == 200.0


# -- label_statistics -------------------------------------------------------


class TestLabelStatistics:
    def test_basic(self):
        df = pd.DataFrame(
            {
                "label": [0, 0, 0, 1, 1, 2, 0, 0, 1, 2],
                "date": [datetime.date(2025, 1, 2)] * 5 + [datetime.date(2025, 1, 3)] * 5,
                "symbol": ["A"] * 10,
            }
        )

        stats = label_statistics(df)

        assert stats["total_bars"] == 10
        assert stats["label_counts"][0] == 5
        assert stats["label_counts"][1] == 3
        assert stats["label_counts"][2] == 2
        assert stats["n_symbols"] == 1
        assert stats["n_days"] == 2

    def test_empty_df(self):
        assert label_statistics(pd.DataFrame()) == {}
