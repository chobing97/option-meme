"""Tests for TradeTracker: bar snapshots, equity curve, drawdown."""

from datetime import datetime, timedelta

import pytest

from src.trading.trade_tracker import TradeTracker


@pytest.fixture
def tracker():
    return TradeTracker()


class TestRecordBar:
    def test_single_bar_recorded(self, tracker):
        tracker.record_bar(
            timestamp=datetime(2026, 1, 5, 9, 31),
            symbol="SPY",
            bar_num=1,
            underlying_close=592.0,
            signal="NONE",
            peak_prob=0.3,
            trough_prob=0.1,
            action="",
            reason="",
            strike=0.0,
            fill_price=0.0,
            position_qty=0,
            position_avg_entry=0.0,
            position_mark_price=0.0,
            cash=100_000.0,
        )
        df = tracker.to_dataframe()
        assert len(df) == 1
        assert df.iloc[0]["symbol"] == "SPY"
        assert df.iloc[0]["equity"] == 100_000.0

    def test_equity_includes_position_value(self, tracker):
        tracker.record_bar(
            timestamp=datetime(2026, 1, 5, 9, 31),
            symbol="SPY",
            bar_num=1,
            underlying_close=592.0,
            signal="PEAK",
            peak_prob=0.8,
            trough_prob=0.1,
            action="BUY_PUT",
            reason="PEAK_SIGNAL",
            strike=590.0,
            fill_price=3.5,
            position_qty=2,
            position_avg_entry=3.5,
            position_mark_price=3.5,
            cash=99_300.0,
        )
        df = tracker.to_dataframe()
        # equity = cash + position_qty * mark_price * 100
        expected_equity = 99_300.0 + 2 * 3.5 * 100
        assert df.iloc[0]["equity"] == pytest.approx(expected_equity)
        assert df.iloc[0]["position_value"] == pytest.approx(2 * 3.5 * 100)


class TestDrawdown:
    def test_drawdown_from_equity_high(self, tracker):
        # Bar 1: equity = 100k (high = 100k)
        tracker.record_bar(
            timestamp=datetime(2026, 1, 5, 9, 31),
            symbol="SPY", bar_num=1, underlying_close=592.0,
            signal="NONE", peak_prob=0.0, trough_prob=0.0,
            action="", reason="", strike=0.0, fill_price=0.0,
            position_qty=0, position_avg_entry=0.0, position_mark_price=0.0,
            cash=100_000.0,
        )
        # Bar 2: equity drops to 99k
        tracker.record_bar(
            timestamp=datetime(2026, 1, 5, 9, 32),
            symbol="SPY", bar_num=2, underlying_close=590.0,
            signal="NONE", peak_prob=0.0, trough_prob=0.0,
            action="", reason="", strike=0.0, fill_price=0.0,
            position_qty=0, position_avg_entry=0.0, position_mark_price=0.0,
            cash=99_000.0,
        )
        df = tracker.to_dataframe()
        assert df.iloc[0]["drawdown_pct"] == 0.0
        assert df.iloc[1]["drawdown_pct"] == pytest.approx(-0.01)
        assert df.iloc[1]["equity_high"] == 100_000.0


class TestSummary:
    def test_summary_stats(self, tracker):
        base = datetime(2026, 1, 5, 9, 30)
        # Buy bar
        tracker.record_bar(
            timestamp=base, symbol="SPY", bar_num=1,
            underlying_close=592.0, signal="PEAK",
            peak_prob=0.8, trough_prob=0.1,
            action="BUY_PUT", reason="PEAK_SIGNAL",
            strike=590.0, fill_price=3.5,
            position_qty=1, position_avg_entry=3.5,
            position_mark_price=3.5, cash=99_650.0,
        )
        # Hold bar
        tracker.record_bar(
            timestamp=base + timedelta(minutes=1), symbol="SPY", bar_num=2,
            underlying_close=590.0, signal="NONE",
            peak_prob=0.3, trough_prob=0.2,
            action="", reason="",
            strike=0.0, fill_price=0.0,
            position_qty=1, position_avg_entry=3.5,
            position_mark_price=4.0, cash=99_650.0,
        )
        # Sell bar
        tracker.record_bar(
            timestamp=base + timedelta(minutes=2), symbol="SPY", bar_num=3,
            underlying_close=588.0, signal="TROUGH",
            peak_prob=0.1, trough_prob=0.8,
            action="SELL_PUT", reason="TROUGH_SIGNAL",
            strike=590.0, fill_price=4.2,
            position_qty=0, position_avg_entry=0.0,
            position_mark_price=0.0, cash=100_070.0,
        )

        summary = tracker.summary()
        assert summary["total_bars"] == 3
        assert summary["buys"] == 1
        assert summary["sells"] == 1
        assert summary["final_equity"] == 100_070.0

    def test_empty_summary(self, tracker):
        assert tracker.summary() == {}


class TestSave:
    def test_save_to_parquet(self, tracker, tmp_path, monkeypatch):
        import config.settings as settings
        monkeypatch.setattr(settings, "TRADE_DB_DIR", tmp_path)

        tracker.record_bar(
            timestamp=datetime(2026, 1, 5, 9, 31),
            symbol="SPY", bar_num=1, underlying_close=592.0,
            signal="NONE", peak_prob=0.0, trough_prob=0.0,
            action="", reason="", strike=0.0, fill_price=0.0,
            position_qty=0, position_avg_entry=0.0,
            position_mark_price=0.0, cash=100_000.0,
        )

        out_path = tracker.save("2026-01-05", "SPY")
        assert out_path.exists()

        import pandas as pd
        df = pd.read_parquet(out_path)
        assert len(df) == 1
        assert "equity" in df.columns
        assert "drawdown_pct" in df.columns


class TestEngineIntegration:
    """Test TradeTracker works with TradingEngine."""

    def test_tracker_records_during_engine_run(self):
        from src.trading.broker.mock_broker import MockBroker
        from src.trading.broker.base import Signal, SignalType
        from src.trading.engine import TradingEngine
        from tests.trading.conftest import StubDetector, StubFeed, make_bars

        bars = make_bars(4)
        signals = [
            Signal(signal_type=SignalType.PEAK, timestamp=datetime(2026, 1, 5, 9, 0), close_price=50000.0),
            Signal(signal_type=SignalType.NONE, timestamp=datetime(2026, 1, 5, 9, 1), close_price=50010.0),
            Signal(signal_type=SignalType.TROUGH, timestamp=datetime(2026, 1, 5, 9, 2), close_price=50020.0),
            Signal(signal_type=SignalType.NONE, timestamp=datetime(2026, 1, 5, 9, 3), close_price=50030.0),
        ]

        tracker = TradeTracker()
        engine = TradingEngine(
            feeds={"A": StubFeed(bars, market="kr", symbol="A")},
            broker=MockBroker(capital=10_000_000),
            detector=StubDetector(signals),
            symbols=["A"],
            quantity=1,
            tracker=tracker,
        )
        engine.run()

        df = tracker.to_dataframe()
        assert len(df) == 4  # one snapshot per bar

        # Check that BUY and SELL actions are recorded
        actions = df["action"].tolist()
        assert "BUY_PUT" in actions
        assert "SELL_PUT" in actions

        # Check equity is tracked
        assert all(df["equity"] > 0)
        assert all(df["cash"] > 0)
