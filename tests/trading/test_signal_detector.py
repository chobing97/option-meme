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


class TestBarAccumulatorMultipleBars:
    """Tests for accumulating multiple bars throughout a full session."""

    def test_accumulate_many_bars(self):
        """Accumulate bars across a full session worth of data."""
        history = _make_history_df(3, bars_per_day=10)
        acc = BarAccumulator(history)
        start = datetime(2026, 1, 5, 9, 0)
        for i in range(60):
            acc.add_bar(make_bar(start + timedelta(minutes=i)))
        assert acc.bar_count == 60
        df = acc.get_feature_df()
        assert len(df) == 30 + 60  # 3*10 + 60

    def test_feature_df_preserves_order(self):
        """Bars should appear in order: history first, then today's bars."""
        history = _make_history_df(1, bars_per_day=3)
        acc = BarAccumulator(history)
        bar1 = make_bar(datetime(2026, 1, 5, 9, 0), close=100.0)
        bar2 = make_bar(datetime(2026, 1, 5, 9, 1), close=200.0)
        acc.add_bar(bar1)
        acc.add_bar(bar2)
        df = acc.get_feature_df()
        # Last two rows should be today's bars
        assert df.iloc[-2]["close"] == 100.0
        assert df.iloc[-1]["close"] == 200.0

    def test_minutes_from_open_in_accumulated_df(self):
        """minutes_from_open should be present in combined output."""
        history = _make_history_df(1, bars_per_day=5)
        acc = BarAccumulator(history)
        bar = make_bar(datetime(2026, 1, 5, 9, 30))
        acc.add_bar(bar)
        df = acc.get_feature_df()
        assert "minutes_from_open" in df.columns

    def test_full_session_accumulation(self):
        """Simulate full 390-bar session accumulation."""
        history = _make_history_df(2, bars_per_day=390)
        acc = BarAccumulator(history)
        start = datetime(2026, 1, 6, 9, 0)
        for i in range(390):
            acc.add_bar(make_bar(start + timedelta(minutes=i), close=50000 + i))
        assert acc.bar_count == 390
        df = acc.get_feature_df()
        assert len(df) == 2 * 390 + 390
