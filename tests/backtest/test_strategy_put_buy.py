"""Tests for PutBuyStrategy — 15 tests (migrated to Order-based interface)."""

import pytest
import pandas as pd

from src.backtest.strategy import PutBuyStrategy, PutBuyConfig
from src.backtest.types import Order, Side, PortfolioState


class TestPutBuyStrategy:

    # 1. peak_prob >= threshold, no position -> BUY PEAK_SIGNAL
    def test_peak_buy(self, default_config, peak_row, empty_portfolio, default_context):
        s = PutBuyStrategy(default_config)
        orders = s.on_bar(peak_row, empty_portfolio, default_context)
        assert len(orders) == 1
        assert orders[0].side == Side.BUY
        assert orders[0].reason == "PEAK_SIGNAL"
        assert orders[0].instrument_type == "option"
        assert orders[0].option_type == "put"

    # 2. peak signal but position exists -> HOLD (empty orders)
    def test_peak_with_position_hold(self, default_config, peak_row, portfolio_with_position, default_context):
        s = PutBuyStrategy(default_config)
        orders = s.on_bar(peak_row, portfolio_with_position, default_context)
        assert len(orders) == 0

    # 3. trough_prob >= threshold, position exists -> SELL TROUGH_SIGNAL
    def test_trough_sell(self, default_config, trough_row, portfolio_with_position, default_context):
        s = PutBuyStrategy(default_config)
        orders = s.on_bar(trough_row, portfolio_with_position, default_context)
        assert len(orders) == 1
        assert orders[0].side == Side.SELL
        assert orders[0].reason == "TROUGH_SIGNAL"

    # 4. trough signal but no position -> HOLD
    def test_trough_no_position_hold(self, default_config, trough_row, empty_portfolio, default_context):
        s = PutBuyStrategy(default_config)
        orders = s.on_bar(trough_row, empty_portfolio, default_context)
        assert len(orders) == 0

    # 5. both probs below threshold -> HOLD
    def test_neutral_hold(self, default_config, neutral_row, empty_portfolio, default_context):
        s = PutBuyStrategy(default_config)
        orders = s.on_bar(neutral_row, empty_portfolio, default_context)
        assert len(orders) == 0

    # 6. TP sell
    def test_tp_sell(self, default_config, sample_row, portfolio_with_profitable, default_context):
        s = PutBuyStrategy(default_config)
        orders = s.on_bar(sample_row, portfolio_with_profitable, default_context)
        assert len(orders) == 1
        assert orders[0].side == Side.SELL
        assert orders[0].reason == "TP"

    # 7. SL sell
    def test_sl_sell(self, default_config, sample_row, portfolio_with_losing, default_context):
        s = PutBuyStrategy(default_config)
        orders = s.on_bar(sample_row, portfolio_with_losing, default_context)
        assert len(orders) == 1
        assert orders[0].side == Side.SELL
        assert orders[0].reason == "SL"

    # 8. Force close with position
    def test_force_close_with_position(self, default_config, portfolio_with_position):
        s = PutBuyStrategy(default_config)
        row = pd.Series({"symbol": "AAPL", "close": 150.0, "peak_prob": 0.1, "trough_prob": 0.1, "minutes_from_open": 280})
        context = {"session_minutes": 390}
        orders = s.on_bar(row, portfolio_with_position, context)
        assert len(orders) == 1
        assert orders[0].side == Side.SELL
        assert orders[0].reason == "FORCE_CLOSE"

    # 9. Near close, no position -> HOLD
    def test_force_close_no_position_hold(self, default_config, empty_portfolio):
        s = PutBuyStrategy(default_config)
        row = pd.Series({"symbol": "AAPL", "close": 150.0, "peak_prob": 0.5, "trough_prob": 0.1, "minutes_from_open": 280})
        context = {"session_minutes": 390}
        orders = s.on_bar(row, empty_portfolio, context)
        assert len(orders) == 0

    # 10. SL priority over trough
    def test_sl_priority_over_trough(self, default_config, portfolio_with_losing):
        s = PutBuyStrategy(default_config)
        row = pd.Series({"symbol": "AAPL", "close": 150.0, "peak_prob": 0.1, "trough_prob": 0.5, "minutes_from_open": 60})
        context = {"session_minutes": 390}
        orders = s.on_bar(row, portfolio_with_losing, context)
        assert len(orders) == 1
        assert orders[0].side == Side.SELL
        assert orders[0].reason == "SL"

    # 11. Force close priority over TP
    def test_force_close_priority_over_tp(self, default_config, portfolio_with_profitable):
        s = PutBuyStrategy(default_config)
        row = pd.Series({"symbol": "AAPL", "close": 150.0, "peak_prob": 0.1, "trough_prob": 0.1, "minutes_from_open": 280})
        context = {"session_minutes": 390}
        orders = s.on_bar(row, portfolio_with_profitable, context)
        assert len(orders) == 1
        assert orders[0].side == Side.SELL
        assert orders[0].reason == "FORCE_CLOSE"

    # 12. Threshold boundary (inclusive)
    def test_threshold_boundary(self, default_config, empty_portfolio, default_context):
        s = PutBuyStrategy(default_config)
        row = pd.Series({"symbol": "AAPL", "close": 150.0, "peak_prob": 0.3, "trough_prob": 0.1, "minutes_from_open": 60})
        orders = s.on_bar(row, empty_portfolio, default_context)
        assert len(orders) == 1
        assert orders[0].side == Side.BUY

    # 13. name() and config_dict()
    def test_name_and_config_dict(self):
        config = PutBuyConfig(threshold=0.4, tp_pct=0.15)
        s = PutBuyStrategy(config)
        assert s.name() == "put_buy"
        d = s.config_dict()
        assert d["threshold"] == 0.4
        assert d["tp_pct"] == 0.15
        assert d["option_type"] == "put"

    # 14. on_day_end closes all positions
    def test_on_day_end(self, default_config, portfolio_with_position, default_context):
        s = PutBuyStrategy(default_config)
        orders = s.on_day_end(portfolio_with_position, default_context)
        assert len(orders) == 1
        assert orders[0].side == Side.SELL
        assert orders[0].reason == "FORCE_CLOSE"
        assert orders[0].symbol == "AAPL"

    # 15. on_data_end delegates to on_day_end
    def test_on_data_end(self, default_config, portfolio_with_position, default_context):
        s = PutBuyStrategy(default_config)
        day_end_orders = s.on_day_end(portfolio_with_position, default_context)
        data_end_orders = s.on_data_end(portfolio_with_position, default_context)
        assert len(day_end_orders) == len(data_end_orders)
        for a, b in zip(day_end_orders, data_end_orders):
            assert a.symbol == b.symbol
            assert a.side == b.side
            assert a.quantity == b.quantity
