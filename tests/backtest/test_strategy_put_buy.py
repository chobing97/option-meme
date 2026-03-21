"""Tests for PutBuyStrategy — 13 tests (mirrors existing test_strategy.py)."""

import pytest

from src.backtest.strategy import PutBuyStrategy, PutBuyConfig, Action, ActionResult


class TestPutBuyStrategy:

    # 1. peak_prob >= threshold, no position -> BUY PEAK_SIGNAL
    def test_peak_buy(self, default_config, peak_bar):
        s = PutBuyStrategy(default_config)
        result = s.on_bar(peak_bar, position=None)
        assert result.action == Action.BUY
        assert result.reason == "PEAK_SIGNAL"

    # 2. peak signal but position exists -> HOLD
    def test_peak_with_position_hold(self, default_config, peak_bar, open_position):
        s = PutBuyStrategy(default_config)
        result = s.on_bar(peak_bar, position=open_position)
        assert result.action == Action.HOLD

    # 3. trough_prob >= threshold, position exists -> SELL TROUGH_SIGNAL
    def test_trough_sell(self, default_config, trough_bar, open_position):
        s = PutBuyStrategy(default_config)
        result = s.on_bar(trough_bar, position=open_position)
        assert result.action == Action.SELL
        assert result.reason == "TROUGH_SIGNAL"

    # 4. trough signal but no position -> HOLD
    def test_trough_no_position_hold(self, default_config, trough_bar):
        s = PutBuyStrategy(default_config)
        result = s.on_bar(trough_bar, position=None)
        assert result.action == Action.HOLD

    # 5. both probs below threshold -> HOLD
    def test_neutral_hold(self, default_config, neutral_bar):
        s = PutBuyStrategy(default_config)
        result = s.on_bar(neutral_bar, position=None)
        assert result.action == Action.HOLD

    # 6. TP sell
    def test_tp_sell(self, default_config, sample_bar, profitable_position):
        s = PutBuyStrategy(default_config)
        result = s.on_bar(sample_bar, position=profitable_position)
        assert result.action == Action.SELL
        assert result.reason == "TP"

    # 7. SL sell
    def test_sl_sell(self, default_config, sample_bar, losing_position):
        s = PutBuyStrategy(default_config)
        result = s.on_bar(sample_bar, position=losing_position)
        assert result.action == Action.SELL
        assert result.reason == "SL"

    # 8. Force close with position
    def test_force_close_with_position(self, default_config, open_position):
        s = PutBuyStrategy(default_config)
        bar = {"close": 150.0, "peak_prob": 0.1, "trough_prob": 0.1, "minutes_from_open": 280}
        result = s.on_bar(bar, position=open_position, session_minutes=390)
        assert result.action == Action.SELL
        assert result.reason == "FORCE_CLOSE"

    # 9. Near close, no position -> HOLD
    def test_force_close_no_position_hold(self, default_config):
        s = PutBuyStrategy(default_config)
        bar = {"close": 150.0, "peak_prob": 0.5, "trough_prob": 0.1, "minutes_from_open": 280}
        result = s.on_bar(bar, position=None, session_minutes=390)
        assert result.action == Action.HOLD

    # 10. SL priority over trough
    def test_sl_priority_over_trough(self, default_config, losing_position):
        s = PutBuyStrategy(default_config)
        bar = {"close": 150.0, "peak_prob": 0.1, "trough_prob": 0.5, "minutes_from_open": 60}
        result = s.on_bar(bar, position=losing_position)
        assert result.action == Action.SELL
        assert result.reason == "SL"

    # 11. Force close priority over TP
    def test_force_close_priority_over_tp(self, default_config, profitable_position):
        s = PutBuyStrategy(default_config)
        bar = {"close": 150.0, "peak_prob": 0.1, "trough_prob": 0.1, "minutes_from_open": 280}
        result = s.on_bar(bar, position=profitable_position, session_minutes=390)
        assert result.action == Action.SELL
        assert result.reason == "FORCE_CLOSE"

    # 12. Threshold boundary (inclusive)
    def test_threshold_boundary(self, default_config):
        s = PutBuyStrategy(default_config)
        bar = {"close": 150.0, "peak_prob": 0.3, "trough_prob": 0.1, "minutes_from_open": 60}
        result = s.on_bar(bar, position=None)
        assert result.action == Action.BUY

    # 13. name() and config_dict()
    def test_name_and_config_dict(self):
        config = PutBuyConfig(threshold=0.4, tp_pct=0.15)
        s = PutBuyStrategy(config)
        assert s.name() == "put_buy"
        d = s.config_dict()
        assert d["threshold"] == 0.4
        assert d["tp_pct"] == 0.15
        assert d["option_type"] == "put"
