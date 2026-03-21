import pytest
from datetime import datetime

from src.backtest.executor.base import Executor, OptionContract, Position, FillResult


class TestExecutorABC:
    def test_cannot_instantiate(self):
        """#1: ABC 직접 인스턴스화 불가 -> TypeError."""
        with pytest.raises(TypeError):
            Executor()

    def test_abstract_methods_exist(self):
        """#2: 모든 abstract method 8개 존재."""
        expected = {
            "get_option_chain", "execute_buy", "execute_sell",
            "get_mark_price", "get_cash", "get_positions",
            "get_position", "reset",
        }
        assert expected == Executor.__abstractmethods__


class TestFillResult:
    def test_filled_result(self):
        """#3: FillResult FILLED 상태."""
        contract = OptionContract("AAPL", 150.0, datetime(2026, 3, 28), "put")
        fr = FillResult(
            status="FILLED",
            fill_price=3.05,
            fill_time=datetime(2026, 3, 1, 9, 45),
            contract=contract,
        )
        assert fr.status == "FILLED"
        assert fr.fill_price == 3.05
        assert fr.reject_reason == ""

    def test_rejected_result(self):
        """FillResult REJECTED 상태."""
        fr = FillResult(status="REJECTED", reject_reason="NO_LIQUIDITY")
        assert fr.status == "REJECTED"
        assert fr.reject_reason == "NO_LIQUIDITY"
        assert fr.contract is None


class TestOptionContract:
    def test_auto_contract_id(self):
        """#4: contract_id 자동 생성."""
        c = OptionContract("AAPL", 150.0, datetime(2026, 3, 28), "put")
        assert c.contract_id == "AAPL_P_150_20260328"

    def test_custom_contract_id(self):
        """명시적 contract_id 지정 시 유지."""
        c = OptionContract("AAPL", 150.0, datetime(2026, 3, 28), "put", contract_id="CUSTOM_ID")
        assert c.contract_id == "CUSTOM_ID"

    def test_call_option_contract_id(self):
        """call 옵션 contract_id."""
        c = OptionContract("AAPL", 155.0, datetime(2026, 4, 4), "call")
        assert c.contract_id == "AAPL_C_155_20260404"


class TestPosition:
    def test_update_mark(self):
        """#5: Position unrealized_pnl_pct = (current - entry) / entry."""
        contract = OptionContract("AAPL", 150.0, datetime(2026, 3, 28), "put")
        pos = Position(contract=contract, quantity=1, avg_entry_price=3.0)
        pos.update_mark(3.3)
        assert pos.current_price == 3.3
        assert pos.unrealized_pnl_pct == pytest.approx(0.1)  # (3.3 - 3.0) / 3.0

    def test_update_mark_loss(self):
        """손실 시 unrealized_pnl_pct < 0."""
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
