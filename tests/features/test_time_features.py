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
        expected = {
            "tmf_elapsed_norm", "tmf_dow_sin", "tmf_dow_cos",
            "tmf_month_sin", "tmf_month_cos", "tmf_progress_sq",
            "tmf_minutes_from_open", "tmf_elapsed_sin", "tmf_elapsed_cos",
            "tmf_session_phase", "tmf_is_first_hour", "tmf_is_last_hour",
        }
        assert expected == set(tmf_cols)
        assert len(tmf_cols) == 12

    def test_output_length_matches_input(self, feature_df):
        result = compute_time_features(feature_df)
        assert len(result) == 60

    def test_elapsed_norm_range(self, feature_df):
        result = compute_time_features(feature_df)
        col = result["tmf_elapsed_norm"]
        assert col.iloc[0] == 0.0
        assert abs(col.iloc[-1] - 59 / 390) < 1e-10
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
        # New features requiring minutes_from_open should also be absent
        assert "tmf_minutes_from_open" not in result.columns
        assert "tmf_elapsed_sin" not in result.columns
        assert "tmf_elapsed_cos" not in result.columns
        assert "tmf_session_phase" not in result.columns
        assert "tmf_is_first_hour" not in result.columns
        assert "tmf_is_last_hour" not in result.columns


class TestNewTimeFeatures:
    """Tests for new time features added for full session support."""

    def test_minutes_from_open_passthrough(self, feature_df):
        result = compute_time_features(feature_df)
        np.testing.assert_array_equal(
            result["tmf_minutes_from_open"].values,
            feature_df["minutes_from_open"].values,
        )

    def test_elapsed_sin_cos_bounded(self, feature_df):
        result = compute_time_features(feature_df)
        assert result["tmf_elapsed_sin"].min() >= -1.0
        assert result["tmf_elapsed_sin"].max() <= 1.0
        assert result["tmf_elapsed_cos"].min() >= -1.0
        assert result["tmf_elapsed_cos"].max() <= 1.0

    def test_elapsed_sin_cos_at_zero(self, feature_df):
        """At minutes_from_open=0, sin=0 and cos=1."""
        result = compute_time_features(feature_df)
        row0 = result.iloc[0]
        assert abs(row0["tmf_elapsed_sin"] - 0.0) < 1e-10
        assert abs(row0["tmf_elapsed_cos"] - 1.0) < 1e-10

    def test_elapsed_sin_cos_at_known_point(self):
        """At minutes_from_open = session_minutes/4, sin=1 and cos=0."""
        session = 60
        df = pd.DataFrame({
            "datetime": pd.date_range("2025-01-02 09:00", periods=session, freq="min"),
            "date": [pd.Timestamp("2025-01-02").date()] * session,
            "minutes_from_open": np.arange(session),
        })
        result = compute_time_features(df, session_minutes=session)
        # At quarter point (15 min), sin(2*pi*15/60) = sin(pi/2) = 1
        row_15 = result[result["minutes_from_open"] == 15].iloc[0]
        assert abs(row_15["tmf_elapsed_sin"] - 1.0) < 1e-10
        assert abs(row_15["tmf_elapsed_cos"] - 0.0) < 1e-10

    def test_session_phase_values(self, feature_df):
        """With 60 bars and session_minutes=390, all should be phase 0."""
        result = compute_time_features(feature_df)
        # minutes 0..59 / (390/6) = 0..59/65 → all floor to 0
        assert (result["tmf_session_phase"] == 0).all()

    def test_session_phase_with_short_session(self):
        """Test session_phase divides session into 6 phases correctly."""
        session = 60
        df = pd.DataFrame({
            "datetime": pd.date_range("2025-01-02 09:00", periods=session, freq="min"),
            "date": [pd.Timestamp("2025-01-02").date()] * session,
            "minutes_from_open": np.arange(session),
        })
        result = compute_time_features(df, session_minutes=session)
        phases = result["tmf_session_phase"]
        # phase_size = 60/6 = 10 min per phase
        # minutes 0-9 → phase 0, 10-19 → phase 1, ..., 50-59 → phase 5
        assert phases.iloc[0] == 0
        assert phases.iloc[9] == 0
        assert phases.iloc[10] == 1
        assert phases.iloc[50] == 5
        assert phases.iloc[59] == 5
        assert set(phases.unique()) == {0, 1, 2, 3, 4, 5}

    def test_is_first_hour(self, feature_df):
        result = compute_time_features(feature_df)
        # All 60 bars have minutes_from_open 0..59 → all < 60 → all 1
        assert (result["tmf_is_first_hour"] == 1).all()

    def test_is_first_hour_boundary(self):
        """Minute 59 is in first hour, minute 60 is not."""
        df = pd.DataFrame({
            "datetime": pd.date_range("2025-01-02 09:00", periods=61, freq="min"),
            "date": [pd.Timestamp("2025-01-02").date()] * 61,
            "minutes_from_open": np.arange(61),
        })
        result = compute_time_features(df)
        assert result.iloc[59]["tmf_is_first_hour"] == 1
        assert result.iloc[60]["tmf_is_first_hour"] == 0

    def test_is_last_hour(self, feature_df):
        """With 60 bars (0-59) and session_minutes=390, none are in last hour."""
        result = compute_time_features(feature_df)
        # Last hour starts at 390-60=330; max minutes_from_open is 59
        assert (result["tmf_is_last_hour"] == 0).all()

    def test_is_last_hour_boundary(self):
        """Test last hour boundary at session_minutes - 60."""
        session = 390
        n = 391
        df = pd.DataFrame({
            "datetime": pd.date_range("2025-01-02 09:00", periods=n, freq="min"),
            "date": [pd.Timestamp("2025-01-02").date()] * n,
            "minutes_from_open": np.arange(n),
        })
        result = compute_time_features(df, session_minutes=session)
        # minute 329 (< 330) → not last hour
        assert result.iloc[329]["tmf_is_last_hour"] == 0
        # minute 330 (>= 330) → last hour
        assert result.iloc[330]["tmf_is_last_hour"] == 1
        # minute 389 → last hour
        assert result.iloc[389]["tmf_is_last_hour"] == 1

    def test_session_minutes_parameter(self):
        """Test that session_minutes parameter changes normalization."""
        n = 60
        df = pd.DataFrame({
            "datetime": pd.date_range("2025-01-02 09:00", periods=n, freq="min"),
            "date": [pd.Timestamp("2025-01-02").date()] * n,
            "minutes_from_open": np.arange(n),
        })
        result_60 = compute_time_features(df, session_minutes=60)
        result_390 = compute_time_features(df, session_minutes=390)

        # With session_minutes=60, norm should be 0..59/60
        assert abs(result_60.iloc[-1]["tmf_elapsed_norm"] - 59 / 60) < 1e-10
        # With session_minutes=390, norm should be 0..59/390
        assert abs(result_390.iloc[-1]["tmf_elapsed_norm"] - 59 / 390) < 1e-10

    def test_nan_in_minutes_from_open(self):
        """NaN in minutes_from_open should produce NaN in derived features."""
        df = pd.DataFrame({
            "datetime": pd.date_range("2025-01-02 09:00", periods=5, freq="min"),
            "date": [pd.Timestamp("2025-01-02").date()] * 5,
            "minutes_from_open": [0.0, 1.0, np.nan, 3.0, 4.0],
        })
        result = compute_time_features(df)
        assert pd.isna(result.iloc[2]["tmf_elapsed_norm"])
        assert pd.isna(result.iloc[2]["tmf_elapsed_sin"])
        assert pd.isna(result.iloc[2]["tmf_elapsed_cos"])
        assert pd.isna(result.iloc[2]["tmf_session_phase"])

    def test_cyclical_sin_cos_identity(self, feature_df):
        """sin^2 + cos^2 should equal 1 for all bars."""
        result = compute_time_features(feature_df)
        sin_sq = result["tmf_elapsed_sin"] ** 2
        cos_sq = result["tmf_elapsed_cos"] ** 2
        np.testing.assert_allclose(sin_sq + cos_sq, 1.0, atol=1e-10)
