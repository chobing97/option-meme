"""Tests for BarAccumulator (SignalDetector needs trained models, tested via engine)."""

from datetime import datetime, timedelta

import pandas as pd

from tests.trading.conftest import _make_history_df, make_bar
from src.trading.signal_detector import BarAccumulator


class TestBarAccumulator:
    def test_initial_bar_count_zero(self):
        history = _make_history_df(n_days=2)
        acc = BarAccumulator(history)
        assert acc.bar_count == 0

    def test_add_bar_increments_count(self):
        acc = BarAccumulator(_make_history_df(1))
        bar = make_bar(datetime(2026, 1, 5, 9, 0))
        acc.add_bar(bar)
        assert acc.bar_count == 1

    def test_get_feature_df_without_bars(self):
        history = _make_history_df(2, bars_per_day=5)
        acc = BarAccumulator(history)
        df = acc.get_feature_df()
        assert len(df) == 10  # 2 days * 5 bars

    def test_get_feature_df_with_bars(self):
        history = _make_history_df(1, bars_per_day=5)
        acc = BarAccumulator(history)
        acc.add_bar(make_bar(datetime(2026, 1, 5, 9, 0)))
        acc.add_bar(make_bar(datetime(2026, 1, 5, 9, 1)))
        df = acc.get_feature_df()
        assert len(df) == 7  # 5 + 2

    def test_history_not_mutated(self):
        history = _make_history_df(1, bars_per_day=5)
        original_len = len(history)
        acc = BarAccumulator(history)
        acc.add_bar(make_bar(datetime(2026, 1, 5, 9, 0)))
        assert len(history) == original_len  # original unchanged

    def test_feature_df_has_all_columns(self):
        history = _make_history_df(1, bars_per_day=3)
        acc = BarAccumulator(history)
        bar = make_bar(datetime(2026, 1, 5, 9, 0))
        acc.add_bar(bar)
        df = acc.get_feature_df()
        assert "close" in df.columns
        assert "datetime" in df.columns
