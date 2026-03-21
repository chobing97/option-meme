"""Tests for src.features.feature_pipeline module."""

import numpy as np
import pandas as pd
import pytest

from src.features.feature_pipeline import (
    build_features,
    build_incremental_chunks,
    build_lookback_features,
    clean_features,
    feature_summary,
    get_all_feature_columns,
    get_base_feature_columns,
    get_feature_columns,
    get_featured_partition_info,
    load_chunk,
    save_featured_partitioned,
)


class TestFeaturePipeline:
    """Tests for feature pipeline functions."""

    def test_all_prefixes_present(self, feature_df):
        result = build_features(feature_df)
        feature_cols = get_feature_columns(result)
        # 45 base + 6 new time features = 51
        assert len(feature_cols) == 51
        # Each prefix must be present
        for prefix in ["pf_", "vf_", "tf_", "tmf_"]:
            count = sum(1 for c in feature_cols if c.startswith(prefix))
            assert count > 0, f"No columns with prefix {prefix}"

    def test_output_length_matches_input(self, feature_df):
        result = build_features(feature_df)
        assert len(result) == 60

    def test_original_columns_preserved(self, feature_df):
        result = build_features(feature_df)
        for col in ["open", "high", "low", "close", "volume", "datetime", "date"]:
            assert col in result.columns, f"Original column '{col}' missing"

    def test_lookback_lag_columns(self, feature_df):
        featured = build_features(feature_df)
        base_cols = get_feature_columns(featured)
        result = build_lookback_features(featured, lookback=10)
        # Verify lag1~lag10 exist for at least the first base feature
        for lag in range(1, 11):
            col_name = f"{base_cols[0]}_lag{lag}"
            assert col_name in result.columns, f"Missing {col_name}"

    def test_lookback_0fill_preserves_all_rows(self, feature_df):
        featured = build_features(feature_df)
        result = build_lookback_features(featured, lookback=10, fill_method="0fill")
        # 0-fill: all rows preserved
        assert len(result) == 60

    def test_lookback_drop_removes_early_rows(self, feature_df):
        featured = build_features(feature_df)
        result = build_lookback_features(featured, lookback=10, fill_method="drop")
        # drop: first 10 rows removed (+ 2 from base NaN in pf_acceleration)
        assert len(result) == 48

    def test_lookback_zero_skips_lag_creation(self, feature_df):
        featured = build_features(feature_df)
        result = build_lookback_features(featured, lookback=0)
        # No lag columns should be created
        lag_cols = [c for c in result.columns if "_lag" in c]
        assert len(lag_cols) == 0
        assert len(result) == 60

    def test_clean_features_removes_inf_nan(self, feature_df):
        result = build_features(feature_df)
        feature_cols = get_feature_columns(result)
        # Inject inf and NaN
        result.loc[result.index[0], feature_cols[0]] = float("inf")
        result.loc[result.index[1], feature_cols[1]] = float("nan")
        result = clean_features(result)
        # No inf or NaN should remain
        all_feat = get_all_feature_columns(result)
        for col in all_feat:
            assert not result[col].isin([float("inf"), float("-inf")]).any(), (
                f"inf found in {col}"
            )
            assert not result[col].isnull().any(), f"NaN found in {col}"

    def test_feature_summary_shape(self, feature_df):
        result = build_features(feature_df)
        summary = feature_summary(result)
        feature_cols = get_feature_columns(result)
        assert len(summary) == len(feature_cols)
        assert "null_count" in summary.columns
        assert "inf_count" in summary.columns


class TestSessionMinutesParameter:
    """Tests for session_minutes parameter in build_features."""

    def test_accepts_session_minutes(self, feature_df):
        """build_features should accept session_minutes without error."""
        result = build_features(feature_df, session_minutes=390)
        assert len(result) == 60

    def test_session_minutes_affects_normalization(self, feature_df):
        """Different session_minutes should produce different normalized values."""
        result_60 = build_features(feature_df, session_minutes=60)
        result_390 = build_features(feature_df, session_minutes=390)
        # tmf_elapsed_norm should differ because normalization base differs
        assert not np.allclose(
            result_60["tmf_elapsed_norm"].values,
            result_390["tmf_elapsed_norm"].values,
        )

    def test_session_minutes_default_is_390(self, feature_df):
        """Default session_minutes should be 390."""
        result_default = build_features(feature_df)
        result_390 = build_features(feature_df, session_minutes=390)
        np.testing.assert_array_equal(
            result_default["tmf_elapsed_norm"].values,
            result_390["tmf_elapsed_norm"].values,
        )


class TestLookbackFeatures:
    """Tests for build_lookback_features fill methods."""

    def test_0fill_has_zeros_in_early_lags(self, feature_df):
        """0fill method should fill NaN lags with 0 for early bars."""
        featured = build_features(feature_df)
        result = build_lookback_features(featured, lookback=5, fill_method="0fill")
        lag_cols = [c for c in result.columns if "_lag5" in c]
        # First row should have 0 for lag5 (no history available)
        for col in lag_cols:
            assert result.iloc[0][col] == 0.0, f"{col} not 0-filled"

    def test_drop_removes_early_rows(self, feature_df):
        """drop method should remove rows with incomplete lookback history."""
        featured = build_features(feature_df)
        base_count = len(featured)
        result = build_lookback_features(featured, lookback=3, fill_method="drop")
        # With lookback=3, first 3 rows (at minimum) have incomplete history
        # Plus any rows already NaN from base features (e.g., pf_acceleration)
        assert len(result) < base_count


class TestCleanFeatures:
    """Tests for clean_features inf/nan handling."""

    def test_replaces_inf_with_zero(self, feature_df):
        result = build_features(feature_df)
        feature_cols = get_feature_columns(result)
        result.loc[result.index[0], feature_cols[0]] = float("inf")
        result.loc[result.index[1], feature_cols[0]] = float("-inf")
        cleaned = clean_features(result)
        assert not np.isinf(cleaned[feature_cols[0]]).any()

    def test_forward_fills_within_day(self, feature_df):
        """NaN values should be forward-filled within each day."""
        result = build_features(feature_df)
        feature_cols = get_feature_columns(result)
        col = feature_cols[0]
        # Set a value and then NaN the next row
        val = result.loc[result.index[5], col]
        result.loc[result.index[6], col] = np.nan
        cleaned = clean_features(result)
        # After ffill, row 6 should have row 5's value
        assert cleaned.loc[cleaned.index[6], col] == val

    def test_remaining_nan_filled_with_zero(self, feature_df):
        """After ffill, remaining NaN should be filled with 0."""
        result = build_features(feature_df)
        feature_cols = get_feature_columns(result)
        # Set first row to NaN (can't be forward-filled)
        result.loc[result.index[0], feature_cols[0]] = np.nan
        cleaned = clean_features(result)
        assert cleaned.loc[cleaned.index[0], feature_cols[0]] == 0.0


class TestColumnHelpers:
    """Tests for get_base_feature_columns vs get_all_feature_columns."""

    def test_base_excludes_lag_columns(self, feature_df):
        featured = build_features(feature_df)
        featured = build_lookback_features(featured, lookback=3, fill_method="0fill")
        base_cols = get_base_feature_columns(featured)
        all_cols = get_all_feature_columns(featured)
        # base should have fewer columns
        assert len(base_cols) < len(all_cols)
        # No lag columns in base
        for col in base_cols:
            assert "_lag" not in col, f"Lag column {col} in base columns"

    def test_all_includes_lag_columns(self, feature_df):
        featured = build_features(feature_df)
        featured = build_lookback_features(featured, lookback=3, fill_method="0fill")
        all_cols = get_all_feature_columns(featured)
        lag_cols = [c for c in all_cols if "_lag" in c]
        assert len(lag_cols) > 0

    def test_no_lookback_base_equals_all(self, feature_df):
        """Without lookback, base and all feature columns should be equal."""
        featured = build_features(feature_df)
        base_cols = get_base_feature_columns(featured)
        all_cols = get_all_feature_columns(featured)
        assert base_cols == all_cols

    def test_get_feature_columns_returns_base_only(self, feature_df):
        """get_feature_columns should return only base features, no lag columns."""
        featured = build_features(feature_df)
        featured = build_lookback_features(featured, lookback=5, fill_method="0fill")
        cols = get_feature_columns(featured)
        # Must not contain any lag columns
        for col in cols:
            assert "_lag" not in col, f"get_feature_columns returned lag column: {col}"
        # Must equal get_base_feature_columns
        assert cols == get_base_feature_columns(featured)

    def test_get_all_feature_columns_includes_lags(self, feature_df):
        """get_all_feature_columns must include both base and lag columns."""
        featured = build_features(feature_df)
        featured = build_lookback_features(featured, lookback=5, fill_method="0fill")
        all_cols = get_all_feature_columns(featured)
        base_cols = get_feature_columns(featured)
        lag_cols = [c for c in all_cols if "_lag" in c]
        # All columns = base + lag
        assert len(all_cols) == len(base_cols) + len(lag_cols)
        assert len(lag_cols) == len(base_cols) * 5  # 5 lags per base feature


class TestGetFeaturedPartitionInfo:
    """Tests for get_featured_partition_info."""

    def _make_partitions(self, tmp_path, market="us", label_config="L2", model_config="M3"):
        """Create test partitions and return base_dir."""
        from unittest.mock import patch

        rng = np.random.RandomState(42)
        for symbol, n_rows in [("AAPL", 100), ("MSFT", 200)]:
            sym_dir = tmp_path / "featured" / "1m" / label_config / model_config / market / symbol
            sym_dir.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame({
                "datetime": pd.date_range("2025-01-02 09:00", periods=n_rows, freq="min"),
                "close": rng.randn(n_rows),
                "pf_return": rng.randn(n_rows),
            })
            df.to_parquet(sym_dir / "2025.parquet", index=False)
        return tmp_path

    def test_reads_row_counts_without_loading_data(self, tmp_path):
        base = self._make_partitions(tmp_path)
        from unittest.mock import patch
        with patch("src.features.feature_pipeline.PROCESSED_DIR", base):
            infos = get_featured_partition_info("us", "L2", "M3", "1m")

        assert len(infos) == 2
        symbols = {p["symbol"] for p in infos}
        assert symbols == {"AAPL", "MSFT"}
        rows = {p["symbol"]: p["num_rows"] for p in infos}
        assert rows["AAPL"] == 100
        assert rows["MSFT"] == 200

    def test_empty_dir_returns_empty_list(self, tmp_path):
        from unittest.mock import patch
        with patch("src.features.feature_pipeline.PROCESSED_DIR", tmp_path):
            infos = get_featured_partition_info("us", "L2", "M3", "1m")
        assert infos == []

    def test_all_infos_have_required_keys(self, tmp_path):
        base = self._make_partitions(tmp_path)
        from unittest.mock import patch
        with patch("src.features.feature_pipeline.PROCESSED_DIR", base):
            infos = get_featured_partition_info("us", "L2", "M3", "1m")
        for info in infos:
            assert "path" in info
            assert "symbol" in info
            assert "num_rows" in info


class TestBuildIncrementalChunks:
    """Tests for build_incremental_chunks."""

    def _make_partitions(self, tmp_path, symbols_rows):
        from unittest.mock import patch

        rng = np.random.RandomState(42)
        for symbol, n_rows in symbols_rows.items():
            sym_dir = tmp_path / "featured" / "1m" / "L2" / "M3" / "us" / symbol
            sym_dir.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame({
                "datetime": pd.date_range("2025-01-02 09:00", periods=n_rows, freq="min"),
                "close": rng.randn(n_rows),
                "pf_return": rng.randn(n_rows),
            })
            df.to_parquet(sym_dir / "2025.parquet", index=False)
        return tmp_path

    def test_small_data_returns_single_chunk(self, tmp_path):
        """Data that fits in memory → 1 chunk."""
        base = self._make_partitions(tmp_path, {"AAPL": 100, "MSFT": 100})
        from unittest.mock import patch
        with patch("src.features.feature_pipeline.PROCESSED_DIR", base):
            chunks = build_incremental_chunks("us", "L2", "M3", "1m")
        assert len(chunks) == 1
        # Single chunk contains all partitions
        total = sum(p["num_rows"] for p in chunks[0])
        assert total == 200

    def test_chunks_cover_all_rows(self, tmp_path):
        """All partitions must appear in exactly one chunk."""
        base = self._make_partitions(tmp_path, {"A": 500, "B": 300, "C": 200})
        from unittest.mock import patch
        with patch("src.features.feature_pipeline.PROCESSED_DIR", base):
            # Force small budget to get multiple chunks
            chunks = build_incremental_chunks(
                "us", "L2", "M3", "1m", memory_budget_ratio=0.00001,
            )
        all_symbols = set()
        total_rows = 0
        for chunk in chunks:
            for p in chunk:
                all_symbols.add(p["symbol"])
                total_rows += p["num_rows"]
        assert all_symbols == {"A", "B", "C"}
        assert total_rows == 1000

    def test_empty_data_returns_empty(self, tmp_path):
        from unittest.mock import patch
        with patch("src.features.feature_pipeline.PROCESSED_DIR", tmp_path):
            chunks = build_incremental_chunks("us", "L2", "M3", "1m")
        assert chunks == []


class TestLoadChunk:
    """Tests for load_chunk."""

    def test_loads_and_concatenates(self, tmp_path):
        rng = np.random.RandomState(42)
        paths = []
        for i, n in enumerate([50, 30]):
            p = tmp_path / f"part{i}.parquet"
            df = pd.DataFrame({"val": rng.randn(n)})
            df.to_parquet(p, index=False)
            paths.append({"path": p, "symbol": f"SYM{i}", "num_rows": n})

        result = load_chunk(paths)
        assert len(result) == 80

    def test_empty_chunk_returns_empty_df(self):
        result = load_chunk([])
        assert result.empty
