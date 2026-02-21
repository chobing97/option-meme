"""Tests for src.model.dataset."""

import numpy as np
import pandas as pd

from src.model.dataset import (
    TimeSeriesDataset,
    prepare_xy,
    time_based_split,
    walk_forward_splits,
)


class TestTimeBasedSplit:
    def test_splits_are_non_empty(self, split_result):
        assert len(split_result.train) > 0
        assert len(split_result.val) > 0
        assert len(split_result.test) > 0

    def test_no_temporal_overlap(self, split_result):
        assert split_result.train["datetime"].max() < split_result.val["datetime"].min()
        assert split_result.val["datetime"].max() < split_result.test["datetime"].min()

    def test_total_rows_preserved(self, model_df, split_result):
        total = len(split_result.train) + len(split_result.val) + len(split_result.test)
        assert total == len(model_df)

    def test_date_tuples_are_strings(self, split_result):
        for dates in [split_result.train_dates, split_result.val_dates, split_result.test_dates]:
            assert isinstance(dates, tuple)
            assert len(dates) == 2
            assert isinstance(dates[0], str)
            assert isinstance(dates[1], str)

    def test_sorted_by_datetime(self, split_result):
        for part in [split_result.train, split_result.val, split_result.test]:
            assert part["datetime"].is_monotonic_increasing


class TestWalkForwardSplits:
    def test_returns_list_of_tuples(self, model_df):
        result = walk_forward_splits(model_df, train_months=6, test_months=3)
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], pd.DataFrame)
            assert isinstance(item[1], pd.DataFrame)

    def test_no_temporal_overlap_per_split(self, model_df):
        splits = walk_forward_splits(model_df, train_months=6, test_months=3)
        for train, test in splits:
            assert train["datetime"].max() < test["datetime"].min()

    def test_at_least_one_split(self, model_df):
        splits = walk_forward_splits(model_df, train_months=6, test_months=3)
        assert len(splits) >= 1


class TestPrepareXY:
    def test_output_shapes(self, model_df, feature_cols):
        X, y = prepare_xy(model_df, target_label=1, feature_cols=feature_cols)
        assert X.shape == (len(model_df), 5)
        assert y.shape == (len(model_df),)

    def test_binary_target(self, model_df, feature_cols):
        X, y = prepare_xy(model_df, target_label=1, feature_cols=feature_cols)
        assert set(np.unique(y)).issubset({0, 1})

    def test_nan_inf_replaced(self, model_df_with_nans, feature_cols):
        X, y = prepare_xy(model_df_with_nans, target_label=1, feature_cols=feature_cols)
        assert np.isfinite(X).all()


class TestTimeSeriesDataset:
    def test_sequence_shape(self, model_df, feature_cols):
        ds = TimeSeriesDataset(model_df, target_label=1, lookback=3, feature_cols=feature_cols)
        assert len(ds) > 0
        seq, label = ds[0]
        assert seq.shape == (3, 5)

    def test_properties(self, model_df, feature_cols):
        ds = TimeSeriesDataset(model_df, target_label=1, lookback=3, feature_cols=feature_cols)
        assert ds.n_features == 5
        assert 0 <= ds.positive_rate <= 1
