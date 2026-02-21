"""Tests for src.features.price_features module."""

import numpy as np
import pandas as pd
import pytest

from src.features.price_features import compute_price_features


class TestComputePriceFeatures:
    """Tests for compute_price_features."""

    def test_expected_columns_created(self, feature_df):
        result = compute_price_features(feature_df)
        pf_cols = [c for c in result.columns if c.startswith("pf_")]
        assert len(pf_cols) == 17

    def test_output_length_matches_input(self, feature_df):
        result = compute_price_features(feature_df)
        assert len(result) == 60

    def test_range_position_bounded(self, feature_df):
        result = compute_price_features(feature_df)
        col = result["pf_range_position"].dropna()
        assert col.min() >= 0
        assert col.max() <= 1

    def test_body_ratio_bounded(self, feature_df):
        result = compute_price_features(feature_df)
        col = result["pf_body_ratio"].dropna()
        assert col.min() >= 0
        assert col.max() <= 1

    def test_shadow_ratios_bounded(self, feature_df):
        result = compute_price_features(feature_df)
        for name in ["pf_upper_shadow", "pf_lower_shadow"]:
            col = result[name].dropna()
            assert col.min() >= 0, f"{name} min < 0"
            assert col.max() <= 1, f"{name} max > 1"

    def test_flat_prices_defaults(self, flat_price_df):
        result = compute_price_features(flat_price_df)
        assert (result["pf_range_position"] == 0.5).all()
        assert (result["pf_body_ratio"] == 0).all()
        assert (result["pf_bar_direction"] == 0).all()

    def test_empty_df(self, empty_feature_df):
        result = compute_price_features(empty_feature_df)
        assert result.empty
