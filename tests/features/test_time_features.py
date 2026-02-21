"""Tests for src.features.time_features module."""

import numpy as np
import pandas as pd
import pytest

from src.features.time_features import compute_time_features


class TestComputeTimeFeatures:
    """Tests for compute_time_features."""

    def test_expected_columns_created(self, feature_df):
        result = compute_time_features(feature_df)
        tmf_cols = [c for c in result.columns if c.startswith("tmf_")]
        assert len(tmf_cols) == 6

    def test_output_length_matches_input(self, feature_df):
        result = compute_time_features(feature_df)
        assert len(result) == 60

    def test_elapsed_norm_range(self, feature_df):
        result = compute_time_features(feature_df)
        col = result["tmf_elapsed_norm"]
        assert col.iloc[0] == 0.0
        assert abs(col.iloc[-1] - 59 / 60) < 1e-10
        # Monotonically increasing
        assert (col.diff().dropna() > 0).all()

    def test_cyclical_bounded(self, feature_df):
        result = compute_time_features(feature_df)
        for name in ["tmf_dow_sin", "tmf_dow_cos", "tmf_month_sin", "tmf_month_cos"]:
            col = result[name]
            assert col.min() >= -1, f"{name} min < -1"
            assert col.max() <= 1, f"{name} max > 1"

    def test_fallback_without_minutes_from_open(self, feature_df):
        df = feature_df.drop(columns=["minutes_from_open"])
        result = compute_time_features(df)
        assert "tmf_elapsed_norm" in result.columns
        assert "tmf_progress_sq" not in result.columns
