"""Tests for src.features.feature_pipeline module."""

import numpy as np
import pandas as pd
import pytest

from src.features.feature_pipeline import (
    build_features,
    build_lookback_features,
    clean_features,
    feature_summary,
    get_all_feature_columns,
    get_feature_columns,
)


class TestFeaturePipeline:
    """Tests for feature pipeline functions."""

    def test_all_prefixes_present(self, feature_df):
        result = build_features(feature_df)
        feature_cols = get_feature_columns(result)
        assert len(feature_cols) == 45
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

    def test_lookback_rows_dropped(self, feature_df):
        featured = build_features(feature_df)
        result = build_lookback_features(featured, lookback=10)
        # 10 rows from shift + 2 from base NaN in pf_acceleration (first sorted feature)
        assert len(result) == 48

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
