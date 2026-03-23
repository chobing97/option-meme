import pytest
from datetime import datetime

from src.backtest.executor.base import Executor, OptionContract, Position


class TestExecutorABC:
    def test_cannot_instantiate(self):
        """ABC cannot be instantiated directly -> TypeError."""
        with pytest.raises(TypeError):
            Executor()

    def test_abstract_methods_exist(self):
        """All 4 abstract methods present."""
        expected = {
            "execute", "get_portfolio_state", "update_marks", "reset",
        }
        assert expected == Executor.__abstractmethods__


class TestOptionContract:
    def test_auto_contract_id(self):
        """contract_id auto-generated."""
        c = OptionContract("AAPL", 150.0, datetime(2026, 3, 28), "put")
        assert c.contract_id == "AAPL_P_150_20260328"

    def test_custom_contract_id(self):
        """Explicit contract_id preserved."""
        c = OptionContract("AAPL", 150.0, datetime(2026, 3, 28), "put", contract_id="CUSTOM_ID")
        assert c.contract_id == "CUSTOM_ID"

    def test_call_option_contract_id(self):
        """Call option contract_id."""
        c = OptionContract("AAPL", 155.0, datetime(2026, 4, 4), "call")
        assert c.contract_id == "AAPL_C_155_20260404"


class TestPosition:
    def test_update_mark(self):
        """Position unrealized_pnl_pct = (current - entry) / entry."""
        contract = OptionContract("AAPL", 150.0, datetime(2026, 3, 28), "put")
        pos = Position(contract=contract, quantity=1, avg_entry_price=3.0)
        pos.update_mark(3.3)
        assert pos.current_price == 3.3
        assert pos.unrealized_pnl_pct == pytest.approx(0.1)

    def test_update_mark_loss(self):
        """Loss case: unrealized_pnl_pct < 0."""
        contract = OptionContract("AAPL", 150.0, datetime(2026, 3, 28), "put")
        pos = Position(contract=contract, quantity=1, avg_entry_price=3.0)
        pos.update_mark(2.7)
        assert pos.unrealized_pnl_pct == pytest.approx(-0.1)

    def test_update_mark_zero_entry(self):
        """entry_price=0 -> unrealized_pnl_pct stays 0."""
        contract = OptionContract("AAPL", 150.0, datetime(2026, 3, 28), "put")
        pos = Position(contract=contract, quantity=1, avg_entry_price=0.0)
        pos.update_mark(3.0)
        assert pos.unrealized_pnl_pct == 0.0
