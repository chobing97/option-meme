import pytest

from src.backtest.strategy import Strategy, StrategyConfig, Action, ActionResult


class TestStrategy:
    """Strategy.on_bar() 매매 판단 로직 테스트."""

    def test_peak_buy(self, default_config, peak_bar):
        """#1: peak_prob >= threshold, no position -> BUY PEAK_SIGNAL."""
        s = Strategy(default_config)
        result = s.on_bar(peak_bar, position=None)
        assert result.action == Action.BUY
        assert result.reason == "PEAK_SIGNAL"

    def test_peak_with_position_hold(self, default_config, peak_bar, open_position):
        """#2: peak signal but position exists -> no BUY (check exit rules)."""
        s = Strategy(default_config)
        result = s.on_bar(peak_bar, position=open_position)
        # open_position has 0% pnl, not near close -> HOLD
        assert result.action == Action.HOLD

    def test_trough_sell(self, default_config, trough_bar, open_position):
        """#3: trough_prob >= threshold, position exists -> SELL TROUGH_SIGNAL."""
        s = Strategy(default_config)
        result = s.on_bar(trough_bar, position=open_position)
        assert result.action == Action.SELL
        assert result.reason == "TROUGH_SIGNAL"

    def test_trough_no_position_hold(self, default_config, trough_bar):
        """#4: trough signal but no position -> HOLD."""
        s = Strategy(default_config)
        result = s.on_bar(trough_bar, position=None)
        assert result.action == Action.HOLD

    def test_neutral_hold(self, default_config, neutral_bar):
        """#5: both probs below threshold -> HOLD."""
        s = Strategy(default_config)
        result = s.on_bar(neutral_bar, position=None)
        assert result.action == Action.HOLD

    def test_tp_sell(self, default_config, sample_bar, profitable_position):
        """#6: unrealized_pnl_pct >= tp_pct -> SELL TP."""
        s = Strategy(default_config)
        result = s.on_bar(sample_bar, position=profitable_position)
        assert result.action == Action.SELL
        assert result.reason == "TP"

    def test_sl_sell(self, default_config, sample_bar, losing_position):
        """#7: unrealized_pnl_pct <= sl_pct -> SELL SL."""
        s = Strategy(default_config)
        result = s.on_bar(sample_bar, position=losing_position)
        assert result.action == Action.SELL
        assert result.reason == "SL"

    def test_force_close_with_position(self, default_config, open_position):
        """#8: minutes_from_open >= session - force_close_minutes, position -> SELL FORCE_CLOSE."""
        s = Strategy(default_config)
        bar = {"close": 150.0, "peak_prob": 0.1, "trough_prob": 0.1, "minutes_from_open": 280}
        # session_minutes=390, force_close_minutes=120 -> force_close_at=270
        result = s.on_bar(bar, position=open_position, session_minutes=390)
        assert result.action == Action.SELL
        assert result.reason == "FORCE_CLOSE"

    def test_force_close_no_position_hold(self, default_config):
        """#9: near close but no position -> HOLD (don't buy)."""
        s = Strategy(default_config)
        bar = {"close": 150.0, "peak_prob": 0.5, "trough_prob": 0.1, "minutes_from_open": 280}
        result = s.on_bar(bar, position=None, session_minutes=390)
        assert result.action == Action.HOLD

    def test_sl_priority_over_trough(self, default_config, losing_position):
        """#10: SL overrides TROUGH_SIGNAL (SL checked before trough)."""
        s = Strategy(default_config)
        bar = {"close": 150.0, "peak_prob": 0.1, "trough_prob": 0.5, "minutes_from_open": 60}
        result = s.on_bar(bar, position=losing_position)
        assert result.action == Action.SELL
        assert result.reason == "SL"

    def test_force_close_priority_over_tp(self, default_config, profitable_position):
        """#11: FORCE_CLOSE overrides TP (checked first)."""
        s = Strategy(default_config)
        bar = {"close": 150.0, "peak_prob": 0.1, "trough_prob": 0.1, "minutes_from_open": 280}
        result = s.on_bar(bar, position=profitable_position, session_minutes=390)
        assert result.action == Action.SELL
        assert result.reason == "FORCE_CLOSE"

    def test_threshold_boundary(self, default_config):
        """#12: peak_prob == threshold exactly -> BUY (boundary inclusive)."""
        s = Strategy(default_config)
        bar = {"close": 150.0, "peak_prob": 0.3, "trough_prob": 0.1, "minutes_from_open": 60}
        result = s.on_bar(bar, position=None)
        assert result.action == Action.BUY
        assert result.reason == "PEAK_SIGNAL"

    def test_custom_config(self):
        """#13: different config values respected."""
        config = StrategyConfig(threshold=0.5, tp_pct=0.20, sl_pct=-0.10)
        s = Strategy(config)
        # peak_prob=0.4 < threshold 0.5 -> HOLD
        bar = {"close": 150.0, "peak_prob": 0.4, "trough_prob": 0.1, "minutes_from_open": 60}
        result = s.on_bar(bar, position=None)
        assert result.action == Action.HOLD

        # peak_prob=0.6 >= threshold 0.5 -> BUY
        bar2 = {"close": 150.0, "peak_prob": 0.6, "trough_prob": 0.1, "minutes_from_open": 60}
        result2 = s.on_bar(bar2, position=None)
        assert result2.action == Action.BUY
