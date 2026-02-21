"""Tests for src.features.technical_features module."""

import numpy as np
import pandas as pd
import pytest

from src.features.technical_features import compute_technical_features


class TestComputeTechnicalFeatures:
    """Tests for compute_technical_features."""

    def test_expected_columns_created(self, feature_df):
        result = compute_technical_features(feature_df)
        tf_cols = [c for c in result.columns if c.startswith("tf_")]
        assert len(tf_cols) == 13

    def test_output_length_matches_input(self, feature_df):
        result = compute_technical_features(feature_df)
        assert len(result) == 60

    def test_rsi_bounded(self, feature_df):
        result = compute_technical_features(feature_df)
        for col_name in ["tf_rsi7", "tf_rsi14"]:
            col = result[col_name].dropna()
            assert col.min() >= 0, f"{col_name} min < 0"
            assert col.max() <= 1, f"{col_name} max > 1"

    def test_bb_position_finite(self, feature_df):
        result = compute_technical_features(feature_df)
        col = result["tf_bb_position"]
        assert np.isfinite(col).all()

    def test_stochastic_bounded(self, feature_df):
        result = compute_technical_features(feature_df)
        col = result["tf_stoch"].dropna()
        assert col.min() >= 0
        assert col.max() <= 1

    def test_flat_prices_defaults(self, flat_price_df):
        result = compute_technical_features(flat_price_df)
        assert (result["tf_bb_position"] == 0.5).all()
        assert (result["tf_stoch"] == 0.5).all()

    def test_empty_df(self, empty_feature_df):
        result = compute_technical_features(empty_feature_df)
        assert result.empty
