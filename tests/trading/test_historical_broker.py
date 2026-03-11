"""Tests for HistoricalBroker: real options data loading, quoting, execution."""

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from src.trading.broker.base import (
    OptionContract,
    Order,
    OrderSide,
    OrderStatus,
)
from src.trading.broker.historical_broker import HistoricalBroker


@pytest.fixture
def sample_options_dir(tmp_path):
    """Create minimal options data files for testing."""
    sym_dir = tmp_path / "us" / "SPY"
    sym_dir.mkdir(parents=True)

    # contracts.parquet
    contracts = pd.DataFrame({
        "symbol": ["SPY   260112P00590000", "SPY   260119P00595000"],
        "underlying": ["SPY", "SPY"],
        "expiry": ["2026-01-12", "2026-01-19"],
        "cp": ["P", "P"],
        "strike": [590.0, 595.0],
        "period_start": ["2026-01-05", "2026-01-12"],
        "stock_close": [592.0, 594.0],
    })
    contracts.to_parquet(sym_dir / "contracts.parquet", index=False)

    # ohlcv parquet: minute bars for two contracts
    bars = []
    for i in range(10):
        dt = datetime(2026, 1, 5, 9, 30 + i)
        bars.append({
            "datetime": dt,
            "symbol": "SPY   260112P00590000",
            "open": 3.0 + i * 0.1,
            "high": 3.5 + i * 0.1,
            "low": 2.8 + i * 0.1,
            "close": 3.2 + i * 0.1,
            "volume": 100 + i * 50,
            "source": "test",
        })
    # Zero-volume bar for second contract
    bars.append({
        "datetime": datetime(2026, 1, 5, 9, 40),
        "symbol": "SPY   260119P00595000",
        "open": 2.0,
        "high": 2.0,
        "low": 2.0,
        "close": 2.0,
        "volume": 0,
        "source": "test",
    })
    # Bar with volume for second contract
    bars.append({
        "datetime": datetime(2026, 1, 5, 9, 35),
        "symbol": "SPY   260119P00595000",
        "open": 2.5,
        "high": 2.8,
        "low": 2.2,
        "close": 2.5,
        "volume": 200,
        "source": "test",
    })

    ohlcv = pd.DataFrame(bars)
    ohlcv.to_parquet(sym_dir / "2026.parquet", index=False)

    return tmp_path


@pytest.fixture
def broker(sample_options_dir):
    """HistoricalBroker loaded with test data."""
    b = HistoricalBroker(market="us", capital=100_000, data_dir=sample_options_dir)
    b.load_symbols(["SPY"])
    b.connect()
    return b


class TestDataLoading:
    def test_contracts_loaded(self, broker):
        assert "SPY" in broker._contracts
        assert len(broker._contracts["SPY"]) == 2

    def test_ohlcv_loaded(self, broker):
        assert "SPY" in broker._ohlcv
        assert len(broker._ohlcv["SPY"]) == 12  # 10 + 2 bars


class TestOptionChain:
    def test_get_active_contracts(self, broker):
        broker.update_underlying_price("SPY", 592.0, datetime(2026, 1, 5, 9, 35))
        chain = broker.get_option_chain("SPY", "put")
        assert len(chain) >= 1
        assert all(c.option_type == "put" for c in chain)

    def test_no_contracts_for_unknown_symbol(self, broker):
        chain = broker.get_option_chain("AAPL", "put")
        assert chain == []

    def test_contract_has_real_strike(self, broker):
        broker.update_underlying_price("SPY", 592.0, datetime(2026, 1, 5, 9, 35))
        chain = broker.get_option_chain("SPY", "put")
        strikes = [c.strike for c in chain]
        assert 590.0 in strikes


class TestOptionQuote:
    def test_quote_from_real_data(self, broker):
        broker.update_underlying_price("SPY", 592.0, datetime(2026, 1, 5, 9, 35))
        contract = OptionContract(
            symbol="SPY",
            expiry=datetime(2026, 1, 12),
            strike=590.0,
            option_type="put",
            contract_id="SPY   260112P00590000",
        )
        quote = broker.get_option_quote(contract)
        # Bar at 09:35 (i=5): close=3.7, high=4.0, low=3.3
        assert quote.last == pytest.approx(3.7, abs=0.01)
        assert quote.bid < quote.ask
        assert quote.volume > 0

    def test_quote_zero_volume(self, broker):
        broker.update_underlying_price("SPY", 594.0, datetime(2026, 1, 5, 9, 40))
        contract = OptionContract(
            symbol="SPY",
            expiry=datetime(2026, 1, 19),
            strike=595.0,
            option_type="put",
            contract_id="SPY   260119P00595000",
        )
        quote = broker.get_option_quote(contract)
        assert quote.volume == 0

    def test_empty_quote_for_missing_contract(self, broker):
        broker.update_underlying_price("SPY", 592.0, datetime(2026, 1, 5, 9, 35))
        contract = OptionContract(
            symbol="SPY",
            expiry=datetime(2026, 1, 12),
            strike=999.0,
            option_type="put",
            contract_id="SPY   FAKE_CONTRACT",
        )
        quote = broker.get_option_quote(contract)
        assert quote.volume == 0
        assert quote.last == 0.0


class TestOrderExecution:
    def test_buy_with_liquidity(self, broker):
        broker.update_underlying_price("SPY", 592.0, datetime(2026, 1, 5, 9, 35))
        contract = OptionContract(
            symbol="SPY",
            expiry=datetime(2026, 1, 12),
            strike=590.0,
            option_type="put",
            contract_id="SPY   260112P00590000",
        )
        order = Order(side=OrderSide.BUY, contract=contract, quantity=1)
        filled = broker.submit_order(order)
        assert filled.status == OrderStatus.FILLED
        assert filled.fill_price > 0
        assert broker.get_cash_balance() < 100_000

    def test_buy_rejected_no_liquidity(self, broker):
        """Order should be rejected when volume == 0."""
        broker.update_underlying_price("SPY", 594.0, datetime(2026, 1, 5, 9, 40))
        contract = OptionContract(
            symbol="SPY",
            expiry=datetime(2026, 1, 19),
            strike=595.0,
            option_type="put",
            contract_id="SPY   260119P00595000",
        )
        order = Order(side=OrderSide.BUY, contract=contract, quantity=1)
        filled = broker.submit_order(order)
        assert filled.status == OrderStatus.REJECTED

    def test_buy_rejected_insufficient_cash(self, sample_options_dir):
        b = HistoricalBroker(market="us", capital=0.01, data_dir=sample_options_dir)
        b.load_symbols(["SPY"])
        b.connect()
        b.update_underlying_price("SPY", 592.0, datetime(2026, 1, 5, 9, 35))

        contract = OptionContract(
            symbol="SPY",
            expiry=datetime(2026, 1, 12),
            strike=590.0,
            option_type="put",
            contract_id="SPY   260112P00590000",
        )
        order = Order(side=OrderSide.BUY, contract=contract, quantity=1)
        filled = b.submit_order(order)
        assert filled.status == OrderStatus.REJECTED

    def test_sell_returns_cash(self, broker):
        broker.update_underlying_price("SPY", 592.0, datetime(2026, 1, 5, 9, 35))
        contract = OptionContract(
            symbol="SPY",
            expiry=datetime(2026, 1, 12),
            strike=590.0,
            option_type="put",
            contract_id="SPY   260112P00590000",
        )
        # Buy first
        buy_order = Order(side=OrderSide.BUY, contract=contract, quantity=1)
        broker.submit_order(buy_order)
        cash_after_buy = broker.get_cash_balance()

        # Sell
        sell_order = Order(side=OrderSide.SELL, contract=contract, quantity=1)
        filled = broker.submit_order(sell_order)
        assert filled.status == OrderStatus.FILLED
        assert broker.get_cash_balance() > cash_after_buy


class TestMarkToMarket:
    def test_position_marked_with_real_price(self, broker):
        broker.update_underlying_price("SPY", 592.0, datetime(2026, 1, 5, 9, 33))
        contract = OptionContract(
            symbol="SPY",
            expiry=datetime(2026, 1, 12),
            strike=590.0,
            option_type="put",
            contract_id="SPY   260112P00590000",
        )
        order = Order(side=OrderSide.BUY, contract=contract, quantity=1)
        broker.submit_order(order)

        # Move time forward -> price changes
        broker.update_underlying_price("SPY", 588.0, datetime(2026, 1, 5, 9, 37))
        positions = broker.get_positions()
        assert len(positions) == 1
        # Current price should be from the 09:37 bar (i=7, close=3.9)
        assert positions[0].current_price == pytest.approx(3.9, abs=0.01)
        assert positions[0].unrealized_pnl_pct != 0.0


class TestSpreadEstimation:
    def test_bid_ask_from_high_low(self, broker):
        """bid/ask should be estimated from high-low range."""
        broker.update_underlying_price("SPY", 592.0, datetime(2026, 1, 5, 9, 35))
        contract = OptionContract(
            symbol="SPY",
            expiry=datetime(2026, 1, 12),
            strike=590.0,
            option_type="put",
            contract_id="SPY   260112P00590000",
        )
        quote = broker.get_option_quote(contract)
        spread = quote.ask - quote.bid
        # i=5: high=4.0, low=3.3 -> spread = 0.7
        assert spread > 0
        assert spread == pytest.approx(0.7, abs=0.01)
