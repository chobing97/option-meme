"""Tests for Notifier ABC and ConsoleNotifier."""

from datetime import datetime

from src.trading.notifier.base import Notifier, TradeEvent
from src.trading.notifier.console import ConsoleNotifier


class TestTradeEvent:
    def test_default_details(self):
        e = TradeEvent(
            event_type="BUY", market="kr", symbol="5930",
            timestamp=datetime.now(),
        )
        assert e.details == {}

    def test_with_details(self):
        e = TradeEvent(
            event_type="SELL", market="kr", symbol="5930",
            timestamp=datetime.now(),
            details={"strike": 70000, "pnl_pct": 0.05},
        )
        assert e.details["strike"] == 70000


class TestConsoleNotifier:
    def test_is_notifier(self):
        assert isinstance(ConsoleNotifier(), Notifier)

    def test_buy_event(self, capsys):
        n = ConsoleNotifier()
        n.notify(TradeEvent(
            event_type="BUY", market="kr", symbol="5930",
            timestamp=datetime.now(),
            details={"strike": 70000.0, "expiry": "2026-02-28",
                     "fill_price": 1200.0, "quantity": 2},
        ))
        out = capsys.readouterr().out
        assert "BUY PUT" in out
        assert "70,000" in out
        assert "qty=2" in out

    def test_sell_event(self, capsys):
        n = ConsoleNotifier()
        n.notify(TradeEvent(
            event_type="SELL", market="kr", symbol="5930",
            timestamp=datetime.now(),
            details={"strike": 70000.0, "fill_price": 1500.0,
                     "pnl_pct": 0.08, "reason": "STOP_LOSS", "quantity": 1},
        ))
        out = capsys.readouterr().out
        assert "SELL PUT" in out
        assert "STOP_LOSS" in out
        assert "+8.0%" in out

    def test_session_end_event(self, capsys):
        n = ConsoleNotifier()
        n.notify(TradeEvent(
            event_type="SESSION_END", market="kr", symbol="5930",
            timestamp=datetime.now(),
            details={"buys": 3, "sells": 2, "net_pnl": 500.0,
                     "total_cost": 5000.0, "peak_signals": 4,
                     "trough_signals": 3, "cash_balance": 9_500_000},
        ))
        out = capsys.readouterr().out
        assert "Session Summary" in out
        assert "3 buy" in out
        assert "9,500,000" in out

    def test_session_end_no_trades(self, capsys):
        n = ConsoleNotifier()
        n.notify(TradeEvent(
            event_type="SESSION_END", market="us", symbol="AAPL",
            timestamp=datetime.now(),
            details={"buys": 0, "sells": 0, "net_pnl": 0,
                     "total_cost": 0, "peak_signals": 1,
                     "trough_signals": 0, "cash_balance": 10_000_000},
        ))
        out = capsys.readouterr().out
        assert "no trades" in out
