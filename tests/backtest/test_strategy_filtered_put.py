"""Tests for FilteredPutStrategy — 18 tests (migrated to Order-based interface)."""

import pytest
from datetime import datetime

import pandas as pd

from src.backtest.strategy.filtered_put import FilteredPutStrategy, FilteredPutConfig
from src.backtest.types import Order, Side, PortfolioState
from src.backtest.executor.base import OptionContract, Position


@pytest.fixture
def fp_config():
    return FilteredPutConfig(
        threshold=0.5,
        tp_pct=0.05,
        sl_pct=-0.10,
        force_close_minutes=120,
        min_holding_minutes=30,
        cooldown_minutes=30,
        max_trades_per_day=3,
        min_prob_gap=0.2,
    )


@pytest.fixture
def fp_strategy(fp_config):
    return FilteredPutStrategy(fp_config)


@pytest.fixture
def fp_contract():
    return OptionContract(symbol="AAPL", strike=150.0, expiry=datetime(2026, 3, 28), option_type="put")


@pytest.fixture
def fp_position(fp_contract):
    return Position(contract=fp_contract, quantity=1, avg_entry_price=3.0, current_price=3.0, unrealized_pnl_pct=0.0)


@pytest.fixture
def fp_profitable(fp_contract):
    return Position(contract=fp_contract, quantity=1, avg_entry_price=3.0, current_price=3.18, unrealized_pnl_pct=0.06)


@pytest.fixture
def fp_losing(fp_contract):
    return Position(contract=fp_contract, quantity=1, avg_entry_price=3.0, current_price=2.67, unrealized_pnl_pct=-0.11)


@pytest.fixture
def fp_portfolio(fp_position):
    return PortfolioState(cash=100_000, positions=[fp_position], equity=100_300)


@pytest.fixture
def fp_portfolio_profitable(fp_profitable):
    return PortfolioState(cash=100_000, positions=[fp_profitable], equity=100_318)


@pytest.fixture
def fp_portfolio_losing(fp_losing):
    return PortfolioState(cash=100_000, positions=[fp_losing], equity=100_267)


@pytest.fixture
def fp_empty_portfolio():
    return PortfolioState(cash=100_000, positions=[], equity=100_000)


@pytest.fixture
def fp_context():
    return {"session_minutes": 390}


class TestFilteredPutStrategy:

    # 1. Peak signal with sufficient gap -> BUY
    def test_peak_buy_with_gap(self, fp_strategy, fp_empty_portfolio, fp_context):
        row = pd.Series({"symbol": "AAPL", "close": 150.0, "peak_prob": 0.6, "trough_prob": 0.2, "minutes_from_open": 60})
        orders = fp_strategy.on_bar(row, fp_empty_portfolio, fp_context)
        assert len(orders) == 1
        assert orders[0].side == Side.BUY
        assert orders[0].reason == "PEAK_SIGNAL"
        assert orders[0].option_type == "put"

    # 2. Peak signal but gap too small -> HOLD
    def test_peak_insufficient_gap(self, fp_strategy, fp_empty_portfolio, fp_context):
        row = pd.Series({"symbol": "AAPL", "close": 150.0, "peak_prob": 0.5, "trough_prob": 0.4, "minutes_from_open": 60})
        orders = fp_strategy.on_bar(row, fp_empty_portfolio, fp_context)
        assert len(orders) == 0

    # 3. Peak below threshold -> HOLD
    def test_peak_below_threshold(self, fp_strategy, fp_empty_portfolio, fp_context):
        row = pd.Series({"symbol": "AAPL", "close": 150.0, "peak_prob": 0.4, "trough_prob": 0.1, "minutes_from_open": 60})
        orders = fp_strategy.on_bar(row, fp_empty_portfolio, fp_context)
        assert len(orders) == 0

    # 4. Trough signal with position, held long enough -> SELL
    def test_trough_sell_after_min_holding(self, fp_strategy, fp_portfolio, fp_context):
        fp_strategy._entry_minute = 10
        row = pd.Series({"symbol": "AAPL", "close": 150.0, "peak_prob": 0.1, "trough_prob": 0.6, "minutes_from_open": 50})
        orders = fp_strategy.on_bar(row, fp_portfolio, fp_context)
        assert len(orders) == 1
        assert orders[0].side == Side.SELL
        assert orders[0].reason == "TROUGH_SIGNAL"

    # 5. Trough signal but held too short -> HOLD
    def test_trough_hold_min_holding(self, fp_strategy, fp_portfolio, fp_context):
        fp_strategy._entry_minute = 40
        row = pd.Series({"symbol": "AAPL", "close": 150.0, "peak_prob": 0.1, "trough_prob": 0.6, "minutes_from_open": 50})
        orders = fp_strategy.on_bar(row, fp_portfolio, fp_context)
        assert len(orders) == 0

    # 6. Force close with position
    def test_force_close(self, fp_strategy, fp_portfolio, fp_context):
        row = pd.Series({"symbol": "AAPL", "close": 150.0, "peak_prob": 0.1, "trough_prob": 0.1, "minutes_from_open": 280})
        orders = fp_strategy.on_bar(row, fp_portfolio, fp_context)
        assert len(orders) == 1
        assert orders[0].side == Side.SELL
        assert orders[0].reason == "FORCE_CLOSE"

    # 7. Stop loss
    def test_sl(self, fp_strategy, fp_portfolio_losing, fp_context):
        row = pd.Series({"symbol": "AAPL", "close": 150.0, "peak_prob": 0.1, "trough_prob": 0.1, "minutes_from_open": 60})
        orders = fp_strategy.on_bar(row, fp_portfolio_losing, fp_context)
        assert len(orders) == 1
        assert orders[0].side == Side.SELL
        assert orders[0].reason == "SL"

    # 8. Take profit
    def test_tp(self, fp_strategy, fp_portfolio_profitable, fp_context):
        row = pd.Series({"symbol": "AAPL", "close": 150.0, "peak_prob": 0.1, "trough_prob": 0.1, "minutes_from_open": 60})
        orders = fp_strategy.on_bar(row, fp_portfolio_profitable, fp_context)
        assert len(orders) == 1
        assert orders[0].side == Side.SELL
        assert orders[0].reason == "TP"

    # 9. Force close priority over TP
    def test_force_close_priority_over_tp(self, fp_strategy, fp_portfolio_profitable, fp_context):
        row = pd.Series({"symbol": "AAPL", "close": 150.0, "peak_prob": 0.1, "trough_prob": 0.1, "minutes_from_open": 280})
        orders = fp_strategy.on_bar(row, fp_portfolio_profitable, fp_context)
        assert len(orders) == 1
        assert orders[0].side == Side.SELL
        assert orders[0].reason == "FORCE_CLOSE"

    # 10. Max trades per day
    def test_max_trades_per_day(self, fp_strategy, fp_empty_portfolio, fp_context):
        fp_strategy._trades_today = 3
        row = pd.Series({"symbol": "AAPL", "close": 150.0, "peak_prob": 0.6, "trough_prob": 0.2, "minutes_from_open": 60})
        orders = fp_strategy.on_bar(row, fp_empty_portfolio, fp_context)
        assert len(orders) == 0

    # 11. Cooldown after sell
    def test_cooldown_after_sell(self, fp_strategy, fp_empty_portfolio, fp_context):
        fp_strategy._last_sell_minute = 50
        row = pd.Series({"symbol": "AAPL", "close": 150.0, "peak_prob": 0.6, "trough_prob": 0.2, "minutes_from_open": 60})
        orders = fp_strategy.on_bar(row, fp_empty_portfolio, fp_context)
        assert len(orders) == 0

    # 12. Cooldown expired -> can buy
    def test_cooldown_expired(self, fp_strategy, fp_empty_portfolio, fp_context):
        fp_strategy._last_sell_minute = 20
        row = pd.Series({"symbol": "AAPL", "close": 150.0, "peak_prob": 0.6, "trough_prob": 0.2, "minutes_from_open": 60})
        orders = fp_strategy.on_bar(row, fp_empty_portfolio, fp_context)
        assert len(orders) == 1
        assert orders[0].side == Side.BUY

    # 13. Don't buy near close
    def test_no_buy_near_close(self, fp_strategy, fp_empty_portfolio, fp_context):
        row = pd.Series({"symbol": "AAPL", "close": 150.0, "peak_prob": 0.6, "trough_prob": 0.2, "minutes_from_open": 280})
        orders = fp_strategy.on_bar(row, fp_empty_portfolio, fp_context)
        assert len(orders) == 0

    # 14. on_day_start resets counters
    def test_on_day_start_resets(self, fp_strategy):
        fp_strategy._trades_today = 3
        fp_strategy._entry_minute = 10
        fp_strategy._last_sell_minute = 50
        fp_strategy.on_day_start("2026-03-20")
        assert fp_strategy._trades_today == 0
        assert fp_strategy._entry_minute is None
        assert fp_strategy._last_sell_minute is None

    # 15. reset resets internal state
    def test_reset(self, fp_strategy):
        fp_strategy._entry_minute = 10
        fp_strategy._last_sell_minute = 50
        fp_strategy.reset()
        assert fp_strategy._entry_minute is None
        assert fp_strategy._last_sell_minute is None

    # 16. name returns correct string
    def test_name(self, fp_strategy):
        assert fp_strategy.name() == "filtered_put"

    # 17. config_dict returns all fields
    def test_config_dict(self, fp_config):
        s = FilteredPutStrategy(fp_config)
        d = s.config_dict()
        assert d["threshold"] == 0.5
        assert d["min_holding_minutes"] == 30
        assert d["cooldown_minutes"] == 30
        assert d["max_trades_per_day"] == 3
        assert d["min_prob_gap"] == 0.2
        assert d["option_type"] == "put"

    # 18. BUY increments trades_today and sets entry_minute
    def test_buy_increments_state(self, fp_strategy, fp_empty_portfolio, fp_context):
        assert fp_strategy._trades_today == 0
        assert fp_strategy._entry_minute is None
        row = pd.Series({"symbol": "AAPL", "close": 150.0, "peak_prob": 0.6, "trough_prob": 0.2, "minutes_from_open": 60})
        orders = fp_strategy.on_bar(row, fp_empty_portfolio, fp_context)
        assert len(orders) == 1
        assert orders[0].side == Side.BUY
        assert fp_strategy._trades_today == 1
        assert fp_strategy._entry_minute == 60
