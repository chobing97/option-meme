"""Tests for BacktestExecutor — 16 test cases per plan."""

import pytest
from datetime import datetime

import pandas as pd

from src.backtest.executor.backtest import BacktestExecutor
from src.backtest.executor.base import OptionContract


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
def executor(mock_options_dir):
    """BacktestExecutor with AAPL data loaded, capital=100_000."""
    ex = BacktestExecutor(
        symbols=["AAPL"],
        market="us",
        capital=100_000,
        slippage_pct=0.005,
        data_dir=mock_options_dir,
    )
    ex.load_data()
    return ex


@pytest.fixture
def sample_contract():
    """The AAPL put contract matching mock data."""
    return OptionContract(
        symbol="AAPL",
        strike=270.0,
        expiry=datetime(2026, 1, 9),
        option_type="put",
        contract_id="AAPL  260109P00270000",
    )


# ── #1: load_data success ─────────────────────────────


class TestLoadData:
    def test_load_data_success(self, executor):
        """#1: contracts + OHLCV loaded, symbol present."""
        assert "AAPL" in executor._contracts
        assert "AAPL" in executor._ohlcv
        assert len(executor._contracts["AAPL"]) == 1
        assert len(executor._ohlcv["AAPL"]) == 5

    def test_load_data_missing_symbol(self, mock_options_dir):
        """#2: Missing symbol logs warning, no error."""
        ex = BacktestExecutor(
            symbols=["NONEXIST"],
            market="us",
            capital=100_000,
            data_dir=mock_options_dir,
        )
        ex.load_data()  # should not raise
        assert "NONEXIST" not in ex._contracts
        assert "NONEXIST" not in ex._ohlcv


# ── #3-5: get_option_chain ────────────────────────────


class TestGetOptionChain:
    def test_chain_normal(self, executor):
        """#3: Active contract returned for valid timestamp."""
        ts = datetime(2026, 1, 5, 10, 0)
        chain = executor.get_option_chain("AAPL", "put", ts)
        assert len(chain) == 1
        assert chain[0].strike == 270.0
        assert chain[0].option_type == "put"

    def test_chain_expired(self, executor):
        """#4: Expired contract not returned (timestamp.date >= expiry)."""
        ts = datetime(2026, 1, 10, 10, 0)  # after expiry 2026-01-09
        chain = executor.get_option_chain("AAPL", "put", ts)
        assert len(chain) == 0

    def test_chain_empty_before_period(self, executor):
        """#5: No active contracts before period_start."""
        ts = datetime(2025, 12, 31, 10, 0)
        chain = executor.get_option_chain("AAPL", "put", ts)
        assert len(chain) == 0


# ── #6-9: execute_buy ─────────────────────────────────


class TestExecuteBuy:
    def test_buy_filled(self, executor, sample_contract):
        """#6: Normal buy -> FILLED, cash deducted, position added."""
        ts = datetime(2026, 1, 2, 9, 30)
        result = executor.execute_buy(sample_contract, 1, ts)

        assert result.status == "FILLED"
        assert result.fill_price > 0
        assert result.fill_time == ts
        assert result.contract == sample_contract
        assert executor.get_cash() < 100_000
        assert len(executor.get_positions()) == 1

    def test_buy_slippage(self, executor, sample_contract):
        """#7: fill_price = ask * (1 + slippage)."""
        ts = datetime(2026, 1, 2, 9, 30)
        # Bar at 09:30: close=2.55, high=2.60, low=2.40
        # half_spread = max((2.60 - 2.40)/2, 0.01) = 0.10
        # ask = 2.55 + 0.10 = 2.65
        # fill_price = 2.65 * 1.005 = 2.66325
        result = executor.execute_buy(sample_contract, 1, ts)

        expected_half_spread = (2.60 - 2.40) / 2  # 0.10
        expected_ask = 2.55 + expected_half_spread  # 2.65
        expected_fill = expected_ask * (1 + 0.005)  # 2.66325
        assert result.fill_price == pytest.approx(expected_fill)

    def test_buy_insufficient_cash(self, mock_options_dir, sample_contract):
        """#8: Insufficient cash -> REJECTED."""
        ex = BacktestExecutor(
            symbols=["AAPL"],
            market="us",
            capital=1.0,  # very low capital
            slippage_pct=0.005,
            data_dir=mock_options_dir,
        )
        ex.load_data()

        ts = datetime(2026, 1, 2, 9, 30)
        result = ex.execute_buy(sample_contract, 1, ts)

        assert result.status == "REJECTED"
        assert result.reject_reason == "INSUFFICIENT_CASH"

    def test_buy_no_liquidity(self, executor, sample_contract):
        """#9: volume=0 at 09:33 -> REJECTED NO_LIQUIDITY."""
        ts = datetime(2026, 1, 2, 9, 33)  # bar with volume=0
        result = executor.execute_buy(sample_contract, 1, ts)

        assert result.status == "REJECTED"
        assert result.reject_reason == "NO_LIQUIDITY"


# ── #10-11: execute_sell ──────────────────────────────


class TestExecuteSell:
    def test_sell_filled(self, executor, sample_contract):
        """#10: Sell -> FILLED, cash increases, position removed."""
        ts_buy = datetime(2026, 1, 2, 9, 30)
        executor.execute_buy(sample_contract, 1, ts_buy)
        cash_after_buy = executor.get_cash()

        ts_sell = datetime(2026, 1, 2, 9, 31)
        result = executor.execute_sell(sample_contract, 1, ts_sell)

        assert result.status == "FILLED"
        assert executor.get_cash() > cash_after_buy
        assert len(executor.get_positions()) == 0

    def test_sell_slippage(self, executor, sample_contract):
        """#11: fill_price = bid * (1 - slippage)."""
        ts_buy = datetime(2026, 1, 2, 9, 30)
        executor.execute_buy(sample_contract, 1, ts_buy)

        ts_sell = datetime(2026, 1, 2, 9, 31)
        # Bar at 09:31: close=2.60, high=2.65, low=2.45
        # half_spread = (2.65 - 2.45)/2 = 0.10
        # bid = 2.60 - 0.10 = 2.50
        # fill_price = 2.50 * (1 - 0.005) = 2.4875
        result = executor.execute_sell(sample_contract, 1, ts_sell)

        expected_half_spread = (2.65 - 2.45) / 2
        expected_bid = 2.60 - expected_half_spread
        expected_fill = expected_bid * (1 - 0.005)
        assert result.fill_price == pytest.approx(expected_fill)


# ── #12-13: get_mark_price ────────────────────────────


class TestGetMarkPrice:
    def test_mark_price_asof(self, executor, sample_contract):
        """#12: Returns close at matching timestamp."""
        ts = datetime(2026, 1, 2, 9, 31)
        price = executor.get_mark_price(sample_contract, ts)
        assert price == pytest.approx(2.60)  # close at 09:31

    def test_mark_price_no_data(self, executor, sample_contract):
        """#13: Returns 0.0 when no data before timestamp."""
        ts = datetime(2025, 1, 1, 9, 30)  # way before any data
        price = executor.get_mark_price(sample_contract, ts)
        assert price == 0.0


# ── #14: mark-to-market ──────────────────────────────


class TestMarkToMarket:
    def test_position_unrealized_pnl(self, executor, sample_contract):
        """#14: After buy, mark position with later price -> pnl updates."""
        ts_buy = datetime(2026, 1, 2, 9, 30)
        executor.execute_buy(sample_contract, 1, ts_buy)

        pos = executor.get_positions()[0]
        entry = pos.avg_entry_price

        # Mark at later timestamp
        mark_ts = datetime(2026, 1, 2, 9, 32)
        mark_price = executor.get_mark_price(sample_contract, mark_ts)
        pos.update_mark(mark_price)

        expected_pnl_pct = (mark_price - entry) / entry
        assert pos.current_price == pytest.approx(mark_price)
        assert pos.unrealized_pnl_pct == pytest.approx(expected_pnl_pct)


# ── #15-16: cash integrity ───────────────────────────


class TestCashIntegrity:
    def test_initial_capital(self, executor):
        """#15: get_cash() == initial capital before any trades."""
        assert executor.get_cash() == 100_000

    def test_buy_sell_cash_consistency(self, executor, sample_contract):
        """#16: initial - buy_cost + sell_proceeds == final cash."""
        initial = executor.get_cash()

        ts_buy = datetime(2026, 1, 2, 9, 30)
        buy_result = executor.execute_buy(sample_contract, 1, ts_buy)
        buy_cost = buy_result.fill_price * 1 * 100  # ×100 options multiplier

        ts_sell = datetime(2026, 1, 2, 9, 31)
        sell_result = executor.execute_sell(sample_contract, 1, ts_sell)
        sell_proceeds = sell_result.fill_price * 1 * 100  # ×100

        expected_cash = initial - buy_cost + sell_proceeds
        assert executor.get_cash() == pytest.approx(expected_cash)


# ── Reset ─────────────────────────────────────────────


class TestReset:
    def test_reset_restores_state(self, executor, sample_contract):
        """Reset restores cash, clears positions, keeps data."""
        ts = datetime(2026, 1, 2, 9, 30)
        executor.execute_buy(sample_contract, 1, ts)
        assert executor.get_cash() < 100_000
        assert len(executor.get_positions()) > 0

        executor.reset()

        assert executor.get_cash() == 100_000
        assert len(executor.get_positions()) == 0
        # Data still loaded
        assert "AAPL" in executor._contracts
        assert "AAPL" in executor._ohlcv
