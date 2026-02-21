"""Tests for src.features.volume_features module."""

import numpy as np
import pandas as pd
import pytest

from src.features.volume_features import compute_volume_features


class TestComputeVolumeFeatures:
    """Tests for compute_volume_features."""

    def test_expected_columns_created(self, feature_df):
        result = compute_volume_features(feature_df)
        vf_cols = [c for c in result.columns if c.startswith("vf_")]
        assert len(vf_cols) == 9

    def test_output_length_matches_input(self, feature_df):
        result = compute_volume_features(feature_df)
        assert len(result) == 60

    def test_cum_vol_share_bounded(self, feature_df):
        result = compute_volume_features(feature_df)
        col = result["vf_cum_vol_share"]
        assert col.min() >= 0
        assert col.max() <= 1
        # Monotonically non-decreasing (single day)
        assert (col.diff().dropna() >= -1e-10).all()
        # Last value approximately 1.0
        assert abs(col.iloc[-1] - 1.0) < 1e-10

    def test_vol_change_clipped(self, feature_df):
        result = compute_volume_features(feature_df)
        col = result["vf_vol_change"]
        assert col.min() >= -10
        assert col.max() <= 10

    def test_constant_volume(self, flat_price_df):
        result = compute_volume_features(flat_price_df)
        # Constant volume → MA ratios all 1.0
        for period in [5, 10, 20]:
            col = result[f"vf_vol_ma{period}_ratio"]
            assert (col == 1.0).all(), f"vf_vol_ma{period}_ratio not all 1.0"
        # vol_change = 0 (pct_change of constant is 0, first NaN → fillna(0))
        assert (result["vf_vol_change"] == 0).all()

    def test_empty_df(self, empty_feature_df):
        result = compute_volume_features(empty_feature_df)
        assert result.empty
