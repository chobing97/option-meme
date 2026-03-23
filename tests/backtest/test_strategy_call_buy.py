"""Tests for CallBuyStrategy — 12 tests (migrated to Order-based interface)."""

import pytest
from datetime import datetime

import pandas as pd

from src.backtest.strategy.call_buy import CallBuyStrategy, CallBuyConfig
from src.backtest.types import Order, Side, PortfolioState
from src.backtest.executor.base import OptionContract, Position


@pytest.fixture
def cb_config():
    return CallBuyConfig(
        threshold=0.5,
        tp_pct=0.05,
        sl_pct=-0.10,
        force_close_minutes=120,
        min_holding_minutes=30,
        cooldown_minutes=30,
        max_trades_per_day=3,
    )


@pytest.fixture
def cb_strategy(cb_config):
    return CallBuyStrategy(cb_config)


@pytest.fixture
def cb_contract():
    return OptionContract(symbol="AAPL", strike=150.0, expiry=datetime(2026, 3, 28), option_type="call")


@pytest.fixture
def cb_position(cb_contract):
    return Position(contract=cb_contract, quantity=1, avg_entry_price=3.0, current_price=3.0, unrealized_pnl_pct=0.0)


@pytest.fixture
def cb_profitable(cb_contract):
    return Position(contract=cb_contract, quantity=1, avg_entry_price=3.0, current_price=3.18, unrealized_pnl_pct=0.06)


@pytest.fixture
def cb_losing(cb_contract):
    return Position(contract=cb_contract, quantity=1, avg_entry_price=3.0, current_price=2.67, unrealized_pnl_pct=-0.11)


@pytest.fixture
def cb_portfolio(cb_position):
    return PortfolioState(cash=100_000, positions=[cb_position], equity=100_300)


@pytest.fixture
def cb_portfolio_profitable(cb_profitable):
    return PortfolioState(cash=100_000, positions=[cb_profitable], equity=100_318)


@pytest.fixture
def cb_portfolio_losing(cb_losing):
    return PortfolioState(cash=100_000, positions=[cb_losing], equity=100_267)


@pytest.fixture
def cb_empty_portfolio():
    return PortfolioState(cash=100_000, positions=[], equity=100_000)


@pytest.fixture
def cb_context():
    return {"session_minutes": 390}


class TestCallBuyStrategy:

    # 1. Trough signal, no position -> BUY (opposite of put)
    def test_trough_buy(self, cb_strategy, cb_empty_portfolio, cb_context):
        row = pd.Series({"symbol": "AAPL", "close": 150.0, "peak_prob": 0.2, "trough_prob": 0.6, "minutes_from_open": 60})
        orders = cb_strategy.on_bar(row, cb_empty_portfolio, cb_context)
        assert len(orders) == 1
        assert orders[0].side == Side.BUY
        assert orders[0].reason == "TROUGH_SIGNAL"
        assert orders[0].option_type == "call"

    # 2. Peak signal, no position -> HOLD (call strategy doesn't buy on peak)
    def test_peak_no_buy(self, cb_strategy, cb_empty_portfolio, cb_context):
        row = pd.Series({"symbol": "AAPL", "close": 150.0, "peak_prob": 0.6, "trough_prob": 0.2, "minutes_from_open": 60})
        orders = cb_strategy.on_bar(row, cb_empty_portfolio, cb_context)
        assert len(orders) == 0

    # 3. Peak signal with position, held long enough -> SELL PEAK_SIGNAL
    def test_peak_sell_after_min_holding(self, cb_strategy, cb_portfolio, cb_context):
        cb_strategy._entry_minute = 10
        row = pd.Series({"symbol": "AAPL", "close": 150.0, "peak_prob": 0.6, "trough_prob": 0.2, "minutes_from_open": 50})
        orders = cb_strategy.on_bar(row, cb_portfolio, cb_context)
        assert len(orders) == 1
        assert orders[0].side == Side.SELL
        assert orders[0].reason == "PEAK_SIGNAL"

    # 4. Peak signal but held too short -> HOLD
    def test_peak_hold_min_holding(self, cb_strategy, cb_portfolio, cb_context):
        cb_strategy._entry_minute = 40
        row = pd.Series({"symbol": "AAPL", "close": 150.0, "peak_prob": 0.6, "trough_prob": 0.2, "minutes_from_open": 50})
        orders = cb_strategy.on_bar(row, cb_portfolio, cb_context)
        assert len(orders) == 0

    # 5. Force close with position
    def test_force_close(self, cb_strategy, cb_portfolio, cb_context):
        row = pd.Series({"symbol": "AAPL", "close": 150.0, "peak_prob": 0.1, "trough_prob": 0.1, "minutes_from_open": 280})
        orders = cb_strategy.on_bar(row, cb_portfolio, cb_context)
        assert len(orders) == 1
        assert orders[0].side == Side.SELL
        assert orders[0].reason == "FORCE_CLOSE"

    # 6. Stop loss
    def test_sl(self, cb_strategy, cb_portfolio_losing, cb_context):
        row = pd.Series({"symbol": "AAPL", "close": 150.0, "peak_prob": 0.1, "trough_prob": 0.1, "minutes_from_open": 60})
        orders = cb_strategy.on_bar(row, cb_portfolio_losing, cb_context)
        assert len(orders) == 1
        assert orders[0].side == Side.SELL
        assert orders[0].reason == "SL"

    # 7. Take profit
    def test_tp(self, cb_strategy, cb_portfolio_profitable, cb_context):
        row = pd.Series({"symbol": "AAPL", "close": 150.0, "peak_prob": 0.1, "trough_prob": 0.1, "minutes_from_open": 60})
        orders = cb_strategy.on_bar(row, cb_portfolio_profitable, cb_context)
        assert len(orders) == 1
        assert orders[0].side == Side.SELL
        assert orders[0].reason == "TP"

    # 8. Max trades per day
    def test_max_trades_per_day(self, cb_strategy, cb_empty_portfolio, cb_context):
        cb_strategy._trades_today = 3
        row = pd.Series({"symbol": "AAPL", "close": 150.0, "peak_prob": 0.2, "trough_prob": 0.6, "minutes_from_open": 60})
        orders = cb_strategy.on_bar(row, cb_empty_portfolio, cb_context)
        assert len(orders) == 0

    # 9. Cooldown after sell
    def test_cooldown_after_sell(self, cb_strategy, cb_empty_portfolio, cb_context):
        cb_strategy._last_sell_minute = 50
        row = pd.Series({"symbol": "AAPL", "close": 150.0, "peak_prob": 0.2, "trough_prob": 0.6, "minutes_from_open": 60})
        orders = cb_strategy.on_bar(row, cb_empty_portfolio, cb_context)
        assert len(orders) == 0

    # 10. on_day_start resets state
    def test_on_day_start(self, cb_strategy):
        cb_strategy._trades_today = 3
        cb_strategy._entry_minute = 10
        cb_strategy._last_sell_minute = 50
        cb_strategy.on_day_start("2026-03-20")
        assert cb_strategy._trades_today == 0
        assert cb_strategy._entry_minute is None
        assert cb_strategy._last_sell_minute is None

    # 11. name and config_dict
    def test_name_and_config_dict(self, cb_strategy):
        assert cb_strategy.name() == "call_buy"
        d = cb_strategy.config_dict()
        assert d["option_type"] == "call"
        assert d["threshold"] == 0.5

    # 12. BUY increments trades_today
    def test_buy_increments_state(self, cb_strategy, cb_empty_portfolio, cb_context):
        row = pd.Series({"symbol": "AAPL", "close": 150.0, "peak_prob": 0.2, "trough_prob": 0.6, "minutes_from_open": 60})
        orders = cb_strategy.on_bar(row, cb_empty_portfolio, cb_context)
        assert len(orders) == 1
        assert orders[0].side == Side.BUY
        assert cb_strategy._trades_today == 1
        assert cb_strategy._entry_minute == 60
