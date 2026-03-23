"""Tests for src.backtest.types — Side, Order, OrderResult, PortfolioState, Quote."""

from datetime import datetime

from src.backtest.executor.base import OptionContract, Position
from src.backtest.types import Side, Order, OrderResult, PortfolioState, Quote


class TestSide:
    def test_values(self):
        assert Side.BUY.value == "BUY"
        assert Side.SELL.value == "SELL"


class TestOrder:
    def test_required_fields(self):
        o = Order(symbol="AAPL", side=Side.BUY, quantity=1)
        assert o.symbol == "AAPL"
        assert o.side == Side.BUY
        assert o.quantity == 1

    def test_defaults(self):
        o = Order(symbol="AAPL", side=Side.BUY, quantity=1)
        assert o.reason == ""
        assert o.instrument_type == "option"
        assert o.option_type is None
        assert o.strike_selection == "atm"
        assert o.reference_price == 0.0

    def test_full_option_order(self):
        o = Order(symbol="AAPL", side=Side.BUY, quantity=2, reason="PEAK_SIGNAL",
                  instrument_type="option", option_type="put", strike_selection="atm",
                  reference_price=150.0)
        assert o.option_type == "put"
        assert o.reference_price == 150.0

    def test_stock_order(self):
        o = Order(symbol="AAPL", side=Side.BUY, quantity=100, instrument_type="stock")
        assert o.instrument_type == "stock"
        assert o.option_type is None


class TestOrderResult:
    def test_filled(self):
        order = Order(symbol="AAPL", side=Side.BUY, quantity=1)
        contract = OptionContract(symbol="AAPL", strike=150.0,
                                  expiry=datetime(2026, 3, 28), option_type="put")
        r = OrderResult(order=order, status="FILLED", fill_price=3.50,
                        fill_time=datetime(2026, 3, 21, 10, 30), contract=contract)
        assert r.status == "FILLED"
        assert r.fill_price == 3.50
        assert r.contract.strike == 150.0

    def test_rejected(self):
        order = Order(symbol="AAPL", side=Side.BUY, quantity=1)
        r = OrderResult(order=order, status="REJECTED", reject_reason="NO_CHAIN")
        assert r.status == "REJECTED"
        assert r.reject_reason == "NO_CHAIN"
        assert r.contract is None


class TestPortfolioState:
    def test_empty(self):
        ps = PortfolioState(cash=100_000)
        assert ps.cash == 100_000
        assert ps.positions == []
        assert ps.equity == 0.0

    def test_get_position_found(self):
        contract = OptionContract(symbol="AAPL", strike=150.0,
                                  expiry=datetime(2026, 3, 28), option_type="put")
        pos = Position(contract=contract, quantity=1, avg_entry_price=3.0)
        ps = PortfolioState(cash=99_700, positions=[pos], equity=100_000)
        found = ps.get_position("AAPL")
        assert found is pos

    def test_get_position_not_found(self):
        ps = PortfolioState(cash=100_000)
        assert ps.get_position("AAPL") is None

    def test_get_position_multiple(self):
        c1 = OptionContract(symbol="AAPL", strike=150.0,
                            expiry=datetime(2026, 3, 28), option_type="put")
        c2 = OptionContract(symbol="MSFT", strike=400.0,
                            expiry=datetime(2026, 3, 28), option_type="call")
        p1 = Position(contract=c1, quantity=1, avg_entry_price=3.0)
        p2 = Position(contract=c2, quantity=2, avg_entry_price=5.0)
        ps = PortfolioState(cash=90_000, positions=[p1, p2], equity=91_300)
        assert ps.get_position("MSFT") is p2
        assert ps.get_position("GOOG") is None


class TestQuote:
    def test_basic(self):
        q = Quote(open=100.0, high=105.0, low=99.0, close=103.0, volume=1000)
        assert q.close == 103.0
        assert q.timestamp is None

    def test_with_timestamp(self):
        ts = datetime(2026, 3, 21, 10, 30)
        q = Quote(open=100.0, high=105.0, low=99.0, close=103.0, volume=1000, timestamp=ts)
        assert q.timestamp == ts
