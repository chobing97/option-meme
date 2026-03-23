"""Tests for BacktestExecutor — order execution, portfolio, and marks."""

import pytest
from datetime import datetime

import pandas as pd

from src.backtest.executor.backtest import BacktestExecutor
from src.backtest.executor.base import OptionContract
from src.backtest.market_data import BacktestMarketData
from src.backtest.types import Order, OrderResult, Side


# ── Fixtures ──────────────────────────────────────────


@pytest.fixture
def mock_options_dir(tmp_path):
    """Create mock options data structure for AAPL."""
    sym_dir = tmp_path / "us" / "AAPL"
    sym_dir.mkdir(parents=True)

    # contracts.parquet
    contracts = pd.DataFrame(
        {
            "symbol": ["AAPL  260109P00270000"],
            "underlying": ["AAPL"],
            "expiry": [pd.Timestamp("2026-01-09")],
            "cp": ["P"],
            "strike": [270.0],
            "period_start": [pd.Timestamp("2026-01-02")],
            "stock_close": [272.0],
        }
    )
    contracts.to_parquet(sym_dir / "contracts.parquet", index=False)

    # OHLCV parquet with 5 bars
    bars = pd.DataFrame(
        {
            "datetime": pd.date_range("2026-01-02 09:30", periods=5, freq="min"),
            "symbol": ["AAPL  260109P00270000"] * 5,
            "open": [2.50, 2.55, 2.60, 2.45, 2.40],
            "high": [2.60, 2.65, 2.70, 2.55, 2.50],
            "low": [2.40, 2.45, 2.50, 2.35, 2.30],
            "close": [2.55, 2.60, 2.65, 2.50, 2.45],
            "volume": [100, 200, 150, 0, 100],  # bar at 09:33 has volume=0
            "source": ["databento"] * 5,
        }
    )
    bars.to_parquet(sym_dir / "2026.parquet", index=False)

    return tmp_path


@pytest.fixture
def market_data(mock_options_dir):
    """BacktestMarketData with AAPL data loaded."""
    md = BacktestMarketData(
        symbols=["AAPL"],
        market="us",
        data_dir=mock_options_dir,
    )
    md.load_data()
    return md


@pytest.fixture
def executor(market_data):
    """BacktestExecutor with market_data injected, capital=100_000."""
    return BacktestExecutor(
        market_data=market_data,
        capital=100_000,
        slippage_pct=0.005,
    )


@pytest.fixture
def buy_order():
    """Standard put buy order for AAPL with reference price."""
    return Order(
        symbol="AAPL",
        side=Side.BUY,
        quantity=1,
        instrument_type="option",
        option_type="put",
        strike_selection="atm",
        reference_price=272.0,  # underlying price for ATM selection
        reason="test_buy",
    )


@pytest.fixture
def sell_order():
    """Standard sell order for AAPL."""
    return Order(
        symbol="AAPL",
        side=Side.SELL,
        quantity=1,
        instrument_type="option",
        option_type="put",
        reason="test_sell",
    )


# ── execute BUY ──────────────────────────────────────


class TestExecuteBuy:
    def test_buy_filled(self, executor, buy_order):
        """Normal buy -> FILLED, cash deducted, position added."""
        ts = datetime(2026, 1, 2, 9, 30)
        result = executor.execute(buy_order, ts)

        assert result.status == "FILLED"
        assert result.fill_price > 0
        assert result.fill_time == ts
        assert result.contract is not None
        assert result.contract.strike == 270.0

        state = executor.get_portfolio_state()
        assert state.cash < 100_000
        assert len(state.positions) == 1

    def test_buy_slippage(self, executor, buy_order):
        """fill_price = ask * (1 + slippage)."""
        ts = datetime(2026, 1, 2, 9, 30)
        # Bar at 09:30: close=2.55, high=2.60, low=2.40
        # half_spread = max((2.60 - 2.40)/2, 0.01) = 0.10
        # ask = 2.55 + 0.10 = 2.65
        # fill_price = 2.65 * 1.005 = 2.66325
        result = executor.execute(buy_order, ts)

        expected_half_spread = (2.60 - 2.40) / 2  # 0.10
        expected_ask = 2.55 + expected_half_spread  # 2.65
        expected_fill = expected_ask * (1 + 0.005)  # 2.66325
        assert result.fill_price == pytest.approx(expected_fill)

    def test_buy_insufficient_cash(self, market_data, buy_order):
        """Insufficient cash -> REJECTED."""
        ex = BacktestExecutor(
            market_data=market_data,
            capital=1.0,  # very low capital
            slippage_pct=0.005,
        )
        ts = datetime(2026, 1, 2, 9, 30)
        result = ex.execute(buy_order, ts)

        assert result.status == "REJECTED"
        assert result.reject_reason == "INSUFFICIENT_CASH"

    def test_buy_no_liquidity(self, executor, buy_order):
        """volume=0 at 09:33 -> REJECTED NO_LIQUIDITY."""
        ts = datetime(2026, 1, 2, 9, 33)  # bar with volume=0
        result = executor.execute(buy_order, ts)

        assert result.status == "REJECTED"
        assert result.reject_reason == "NO_LIQUIDITY"

    def test_buy_no_chain(self, executor):
        """Expired contract -> NO_CHAIN."""
        order = Order(
            symbol="AAPL",
            side=Side.BUY,
            quantity=1,
            instrument_type="option",
            option_type="put",
            reference_price=272.0,
        )
        ts = datetime(2026, 1, 10, 10, 0)  # after expiry
        result = executor.execute(order, ts)

        assert result.status == "REJECTED"
        assert result.reject_reason == "NO_CHAIN"

    def test_buy_no_reference_price(self, executor):
        """No reference_price and no stock quote -> REJECTED."""
        order = Order(
            symbol="AAPL",
            side=Side.BUY,
            quantity=1,
            instrument_type="option",
            option_type="put",
            reference_price=0.0,  # no reference price
        )
        ts = datetime(2026, 1, 2, 9, 30)
        result = executor.execute(order, ts)

        assert result.status == "REJECTED"
        assert result.reject_reason == "NO_UNDERLYING_PRICE"


# ── execute SELL ─────────────────────────────────────


class TestExecuteSell:
    def test_sell_filled(self, executor, buy_order, sell_order):
        """Sell -> FILLED, cash increases, position removed."""
        ts_buy = datetime(2026, 1, 2, 9, 30)
        executor.execute(buy_order, ts_buy)
        cash_after_buy = executor.get_portfolio_state().cash

        ts_sell = datetime(2026, 1, 2, 9, 31)
        result = executor.execute(sell_order, ts_sell)

        assert result.status == "FILLED"
        state = executor.get_portfolio_state()
        assert state.cash > cash_after_buy
        assert len(state.positions) == 0

    def test_sell_slippage(self, executor, buy_order, sell_order):
        """fill_price = bid * (1 - slippage)."""
        ts_buy = datetime(2026, 1, 2, 9, 30)
        executor.execute(buy_order, ts_buy)

        ts_sell = datetime(2026, 1, 2, 9, 31)
        # Bar at 09:31: close=2.60, high=2.65, low=2.45
        # half_spread = (2.65 - 2.45)/2 = 0.10
        # bid = 2.60 - 0.10 = 2.50
        # fill_price = 2.50 * (1 - 0.005) = 2.4875
        result = executor.execute(sell_order, ts_sell)

        expected_half_spread = (2.65 - 2.45) / 2
        expected_bid = 2.60 - expected_half_spread
        expected_fill = expected_bid * (1 - 0.005)
        assert result.fill_price == pytest.approx(expected_fill)

    def test_sell_no_position(self, executor, sell_order):
        """Sell without position -> REJECTED NO_POSITION."""
        ts = datetime(2026, 1, 2, 9, 31)
        result = executor.execute(sell_order, ts)

        assert result.status == "REJECTED"
        assert result.reject_reason == "NO_POSITION"


# ── get_portfolio_state ──────────────────────────────


class TestPortfolioState:
    def test_initial_state(self, executor):
        """Initial state: full capital, no positions."""
        state = executor.get_portfolio_state()
        assert state.cash == 100_000
        assert len(state.positions) == 0
        assert state.equity == 100_000

    def test_state_after_buy(self, executor, buy_order):
        """After buy: cash reduced, position present, equity includes position."""
        ts = datetime(2026, 1, 2, 9, 30)
        result = executor.execute(buy_order, ts)

        state = executor.get_portfolio_state()
        assert state.cash < 100_000
        assert len(state.positions) == 1
        assert state.equity == pytest.approx(
            state.cash + state.positions[0].quantity * state.positions[0].current_price * 100
        )

    def test_get_position_by_symbol(self, executor, buy_order):
        """PortfolioState.get_position finds position by symbol."""
        ts = datetime(2026, 1, 2, 9, 30)
        executor.execute(buy_order, ts)

        state = executor.get_portfolio_state()
        pos = state.get_position("AAPL")
        assert pos is not None
        assert pos.contract.symbol == "AAPL"

        assert state.get_position("MSFT") is None


# ── update_marks ─────────────────────────────────────


class TestUpdateMarks:
    def test_update_marks(self, executor, buy_order):
        """After update_marks, position price and PnL updated."""
        ts_buy = datetime(2026, 1, 2, 9, 30)
        executor.execute(buy_order, ts_buy)

        entry_price = executor.get_portfolio_state().positions[0].avg_entry_price

        # Mark at later timestamp
        mark_ts = datetime(2026, 1, 2, 9, 32)
        executor.update_marks(mark_ts)

        pos = executor.get_portfolio_state().positions[0]
        # Bar at 09:32: close=2.65
        assert pos.current_price == pytest.approx(2.65)
        expected_pnl_pct = (2.65 - entry_price) / entry_price
        assert pos.unrealized_pnl_pct == pytest.approx(expected_pnl_pct)


# ── Cash integrity ───────────────────────────────────


class TestCashIntegrity:
    def test_initial_capital(self, executor):
        """get_portfolio_state().cash == initial capital before any trades."""
        assert executor.get_portfolio_state().cash == 100_000

    def test_buy_sell_cash_consistency(self, executor, buy_order, sell_order):
        """initial - buy_cost + sell_proceeds == final cash."""
        initial = executor.get_portfolio_state().cash

        ts_buy = datetime(2026, 1, 2, 9, 30)
        buy_result = executor.execute(buy_order, ts_buy)
        buy_cost = buy_result.fill_price * 1 * 100

        ts_sell = datetime(2026, 1, 2, 9, 31)
        sell_result = executor.execute(sell_order, ts_sell)
        sell_proceeds = sell_result.fill_price * 1 * 100

        expected_cash = initial - buy_cost + sell_proceeds
        assert executor.get_portfolio_state().cash == pytest.approx(expected_cash)


# ── Reset ────────────────────────────────────────────


class TestReset:
    def test_reset_restores_state(self, executor, buy_order):
        """Reset restores cash, clears positions."""
        ts = datetime(2026, 1, 2, 9, 30)
        executor.execute(buy_order, ts)

        state = executor.get_portfolio_state()
        assert state.cash < 100_000
        assert len(state.positions) > 0

        executor.reset()

        state = executor.get_portfolio_state()
        assert state.cash == 100_000
        assert len(state.positions) == 0


# ── Instrument type handling ─────────────────────────


class TestInstrumentType:
    def test_stock_rejected(self, executor):
        """Stock orders rejected (not supported yet)."""
        order = Order(
            symbol="AAPL",
            side=Side.BUY,
            quantity=100,
            instrument_type="stock",
        )
        ts = datetime(2026, 1, 2, 9, 30)
        result = executor.execute(order, ts)

        assert result.status == "REJECTED"
        assert result.reject_reason == "STOCK_NOT_SUPPORTED"

    def test_unknown_instrument_rejected(self, executor):
        """Unknown instrument type rejected."""
        order = Order(
            symbol="AAPL",
            side=Side.BUY,
            quantity=1,
            instrument_type="futures",
        )
        ts = datetime(2026, 1, 2, 9, 30)
        result = executor.execute(order, ts)

        assert result.status == "REJECTED"
        assert result.reject_reason == "UNKNOWN_INSTRUMENT"
