"""Tests for MockBroker: cash balance, multi-symbol, order handling."""

from datetime import datetime, timedelta

import pytest

from src.trading.broker.base import (
    OptionContract,
    Order,
    OrderSide,
    OrderStatus,
)
from src.trading.broker.mock_broker import MockBroker


class TestConnect:
    def test_connect_sets_connected(self, broker):
        # broker fixture already connected
        assert broker.get_cash_balance() == 10_000_000

    def test_custom_capital(self):
        b = MockBroker(capital=5_000_000)
        b.connect()
        assert b.get_cash_balance() == 5_000_000


class TestUpdateUnderlyingPrice:
    def test_stores_per_symbol(self, broker):
        now = datetime(2026, 1, 5, 9, 0)
        broker.update_underlying_price("AAA", 50000.0, now)
        broker.update_underlying_price("BBB", 70000.0, now)
        # Both symbols should have option chains
        chain_a = broker.get_option_chain("AAA", "put")
        chain_b = broker.get_option_chain("BBB", "put")
        assert len(chain_a) > 0
        assert len(chain_b) > 0
        # Strikes should differ
        assert chain_a[0].strike != chain_b[0].strike

    def test_no_price_returns_empty_chain(self, broker):
        chain = broker.get_option_chain("UNKNOWN", "put")
        assert chain == []


class TestGetOptionChain:
    def test_chain_around_atm(self, broker):
        broker.update_underlying_price("X", 50000.0, datetime.now())
        chain = broker.get_option_chain("X", "put")
        strikes = [c.strike for c in chain]
        # ATM strike should be close to 50000
        assert min(abs(s - 50000) for s in strikes) <= 1000

    def test_chain_expiry_is_friday(self, broker):
        broker.update_underlying_price("X", 50000.0, datetime(2026, 1, 5, 9, 0))
        chain = broker.get_option_chain("X", "put")
        for c in chain:
            assert c.expiry.weekday() == 4  # Friday

    def test_chain_symbols_match(self, broker):
        broker.update_underlying_price("MY", 50000.0, datetime.now())
        chain = broker.get_option_chain("MY", "put")
        for c in chain:
            assert c.symbol == "MY"


class TestGetOptionQuote:
    def test_quote_has_spread(self, broker):
        broker.update_underlying_price("X", 50000.0, datetime.now())
        chain = broker.get_option_chain("X", "put")
        quote = broker.get_option_quote(chain[0])
        assert quote.ask > quote.bid
        assert quote.last > 0

    def test_quote_bid_non_negative(self, broker):
        broker.update_underlying_price("X", 50000.0, datetime.now())
        chain = broker.get_option_chain("X", "put")
        for c in chain:
            q = broker.get_option_quote(c)
            assert q.bid >= 0


class TestSubmitOrder:
    def test_buy_deducts_cash(self, broker):
        broker.update_underlying_price("X", 50000.0, datetime.now())
        chain = broker.get_option_chain("X", "put")
        atm = min(chain, key=lambda c: abs(c.strike - 50000))

        initial_cash = broker.get_cash_balance()
        order = Order(side=OrderSide.BUY, contract=atm, quantity=1)
        filled = broker.submit_order(order)

        assert filled.status == OrderStatus.FILLED
        assert broker.get_cash_balance() < initial_cash
        expected = initial_cash - filled.fill_price * filled.quantity
        assert abs(broker.get_cash_balance() - expected) < 0.01

    def test_sell_recovers_cash(self, broker):
        broker.update_underlying_price("X", 50000.0, datetime.now())
        chain = broker.get_option_chain("X", "put")
        atm = min(chain, key=lambda c: abs(c.strike - 50000))

        # Buy
        buy = Order(side=OrderSide.BUY, contract=atm, quantity=2)
        broker.submit_order(buy)
        cash_after_buy = broker.get_cash_balance()

        # Sell
        sell = Order(side=OrderSide.SELL, contract=atm, quantity=2)
        filled_sell = broker.submit_order(sell)

        assert filled_sell.status == OrderStatus.FILLED
        assert broker.get_cash_balance() > cash_after_buy

    def test_buy_rejected_insufficient_cash(self, broker_small):
        broker_small.update_underlying_price("X", 50000.0, datetime.now())
        chain = broker_small.get_option_chain("X", "put")
        atm = min(chain, key=lambda c: abs(c.strike - 50000))

        order = Order(side=OrderSide.BUY, contract=atm, quantity=1)
        result = broker_small.submit_order(order)

        assert result.status == OrderStatus.REJECTED
        assert broker_small.get_cash_balance() == 100  # unchanged

    def test_buy_quantity_affects_cost(self, broker):
        broker.update_underlying_price("X", 50000.0, datetime.now())
        chain = broker.get_option_chain("X", "put")
        atm = min(chain, key=lambda c: abs(c.strike - 50000))

        initial = broker.get_cash_balance()
        order = Order(side=OrderSide.BUY, contract=atm, quantity=3)
        filled = broker.submit_order(order)

        cost = filled.fill_price * 3
        assert abs(broker.get_cash_balance() - (initial - cost)) < 0.01

    def test_order_gets_id_and_fill_time(self, broker):
        broker.update_underlying_price("X", 50000.0, datetime.now())
        chain = broker.get_option_chain("X", "put")
        order = Order(side=OrderSide.BUY, contract=chain[0], quantity=1)
        filled = broker.submit_order(order)

        assert filled.order_id != ""
        assert filled.fill_time is not None
        assert filled.fill_price > 0


class TestPositions:
    def test_buy_creates_position(self, broker):
        broker.update_underlying_price("X", 50000.0, datetime.now())
        chain = broker.get_option_chain("X", "put")
        order = Order(side=OrderSide.BUY, contract=chain[0], quantity=1)
        broker.submit_order(order)

        positions = broker.get_positions()
        assert len(positions) == 1
        assert positions[0].quantity == 1

    def test_sell_removes_position(self, broker):
        broker.update_underlying_price("X", 50000.0, datetime.now())
        chain = broker.get_option_chain("X", "put")
        c = chain[0]

        broker.submit_order(Order(side=OrderSide.BUY, contract=c, quantity=1))
        assert len(broker.get_positions()) == 1

        broker.submit_order(Order(side=OrderSide.SELL, contract=c, quantity=1))
        assert len(broker.get_positions()) == 0

    def test_averaging_up(self, broker):
        broker.update_underlying_price("X", 50000.0, datetime.now())
        chain = broker.get_option_chain("X", "put")
        c = chain[0]

        broker.submit_order(Order(side=OrderSide.BUY, contract=c, quantity=1))
        broker.submit_order(Order(side=OrderSide.BUY, contract=c, quantity=2))

        positions = broker.get_positions()
        assert len(positions) == 1
        assert positions[0].quantity == 3


class TestMarkPositions:
    def test_pnl_updates_on_price_change(self, broker):
        now = datetime(2026, 1, 5, 9, 0)
        broker.update_underlying_price("X", 50000.0, now)
        chain = broker.get_option_chain("X", "put")
        atm = min(chain, key=lambda c: abs(c.strike - 50000))

        broker.submit_order(Order(side=OrderSide.BUY, contract=atm, quantity=1))

        # Price drops -> put value rises -> positive PnL
        broker.update_underlying_price("X", 48000.0, now + timedelta(minutes=10))
        pos = broker.get_positions()[0]
        assert pos.unrealized_pnl_pct > 0

    def test_multi_symbol_mark_independent(self, broker):
        now = datetime(2026, 1, 5, 9, 0)
        broker.update_underlying_price("A", 50000.0, now)
        broker.update_underlying_price("B", 70000.0, now)

        chain_a = broker.get_option_chain("A", "put")
        chain_b = broker.get_option_chain("B", "put")

        broker.submit_order(Order(side=OrderSide.BUY, contract=chain_a[0], quantity=1))
        broker.submit_order(Order(side=OrderSide.BUY, contract=chain_b[0], quantity=1))

        # Only drop A's price
        broker.update_underlying_price("A", 45000.0, now + timedelta(minutes=5))

        positions = broker.get_positions()
        pos_a = next(p for p in positions if p.contract.symbol == "A")
        pos_b = next(p for p in positions if p.contract.symbol == "B")

        # A's put should gain value, B's should be unchanged
        assert pos_a.unrealized_pnl_pct != pos_b.unrealized_pnl_pct


class TestStrikeTick:
    def test_tick_levels(self):
        assert MockBroker._strike_tick(150000) == 2500
        assert MockBroker._strike_tick(70000) == 1000
        assert MockBroker._strike_tick(30000) == 500
        assert MockBroker._strike_tick(5000) == 100
        assert MockBroker._strike_tick(200) == 5
        assert MockBroker._strike_tick(50) == 1
