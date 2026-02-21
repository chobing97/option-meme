"""Tests for TradingEngine: multi-symbol, rules, notifier/DB integration."""

from datetime import datetime, timedelta, time

import pytest

from src.trading.broker.base import (
    Order,
    OrderSide,
    Signal,
    SignalType,
)
from src.trading.broker.mock_broker import MockBroker
from src.trading.engine import TradingEngine
from src.trading.notifier.console import ConsoleNotifier
from src.trading.trade_db import TradeDB
from tests.trading.conftest import (
    StubDetector,
    StubFeed,
    make_bars,
    make_bar,
)


def _make_engine(
    bars_per_symbol: dict[str, list],
    signals_per_bar: list[Signal],
    capital: float = 10_000_000,
    quantity: int = 1,
    notifiers=None,
    trade_db=None,
    market: str = "kr",
):
    """Helper: build a TradingEngine with stub feeds and detector."""
    symbols = list(bars_per_symbol.keys())
    feeds = {
        sym: StubFeed(bars, market=market, symbol=sym)
        for sym, bars in bars_per_symbol.items()
    }
    broker = MockBroker(capital=capital)
    detector = StubDetector(signals_per_bar)
    return TradingEngine(
        feeds=feeds,
        broker=broker,
        detector=detector,
        symbols=symbols,
        quantity=quantity,
        notifiers=notifiers or [],
        trade_db=trade_db,
    )


def _signal(stype: SignalType, close: float = 50000.0, dt=None):
    if dt is None:
        dt = datetime(2026, 1, 5, 9, 5)
    return Signal(signal_type=stype, timestamp=dt, close_price=close)


class TestSingleSymbolBasic:
    def test_no_signals_no_trades(self):
        bars = make_bars(5)
        signals = [_signal(SignalType.NONE)] * 5
        engine = _make_engine({"A": bars}, signals)
        result = engine.run()
        assert result["A"]["buys"] == 0
        assert result["A"]["sells"] == 0

    def test_peak_triggers_buy(self):
        bars = make_bars(3)
        signals = [
            _signal(SignalType.PEAK, close=50000.0),
            _signal(SignalType.NONE),
            _signal(SignalType.NONE),
        ]
        engine = _make_engine({"A": bars}, signals)
        result = engine.run()
        assert result["A"]["buys"] == 1
        assert result["A"]["sells"] == 0

    def test_peak_then_trough_buy_sell(self):
        bars = make_bars(4)
        signals = [
            _signal(SignalType.PEAK, close=50000.0),
            _signal(SignalType.NONE),
            _signal(SignalType.TROUGH, close=50020.0),
            _signal(SignalType.NONE),
        ]
        engine = _make_engine({"A": bars}, signals)
        result = engine.run()
        assert result["A"]["buys"] == 1
        assert result["A"]["sells"] == 1

    def test_no_double_buy(self):
        """Second peak while holding should not buy again."""
        bars = make_bars(4)
        signals = [
            _signal(SignalType.PEAK, close=50000.0),
            _signal(SignalType.PEAK, close=50010.0),
            _signal(SignalType.NONE),
            _signal(SignalType.NONE),
        ]
        engine = _make_engine({"A": bars}, signals)
        result = engine.run()
        assert result["A"]["buys"] == 1


class TestQuantity:
    def test_quantity_passed_to_order(self):
        bars = make_bars(3)
        signals = [
            _signal(SignalType.PEAK, close=50000.0),
            _signal(SignalType.TROUGH, close=50020.0),
            _signal(SignalType.NONE),
        ]
        engine = _make_engine({"A": bars}, signals, quantity=5)
        result = engine.run()
        assert result["A"]["buys"] == 1
        assert result["A"]["sells"] == 1


class TestStopLoss:
    def test_stop_loss_triggers_sell(self):
        """Price rises after buying put -> put value drops -> stop loss."""
        start = datetime(2026, 1, 5, 9, 0)
        bars = [
            make_bar(start, close=50000.0),
            # Price jumps up significantly -> put loses value
            make_bar(start + timedelta(minutes=1), close=55000.0),
            make_bar(start + timedelta(minutes=2), close=55000.0),
        ]
        signals = [
            _signal(SignalType.PEAK, close=50000.0),
            _signal(SignalType.NONE, close=55000.0),
            _signal(SignalType.NONE, close=55000.0),
        ]
        engine = _make_engine({"A": bars}, signals)
        result = engine.run()
        # Should have bought and then stop-lossed
        assert result["A"]["buys"] == 1
        assert result["A"]["sells"] == 1


class TestProfitTarget:
    def test_profit_target_triggers_sell(self):
        """Price drops after buying put -> put gains value -> profit target."""
        start = datetime(2026, 1, 5, 9, 0)
        bars = [
            make_bar(start, close=50000.0),
            # Price drops sharply -> put value rises
            make_bar(start + timedelta(minutes=1), close=44000.0),
            make_bar(start + timedelta(minutes=2), close=44000.0),
        ]
        signals = [
            _signal(SignalType.PEAK, close=50000.0),
            _signal(SignalType.NONE, close=44000.0),
            _signal(SignalType.NONE, close=44000.0),
        ]
        engine = _make_engine({"A": bars}, signals)
        result = engine.run()
        assert result["A"]["buys"] == 1
        assert result["A"]["sells"] == 1


class TestForceClose:
    def test_compute_force_close_time(self):
        assert TradingEngine._compute_force_close_time("kr") == time(13, 30)
        assert TradingEngine._compute_force_close_time("us") == time(14, 0)
        assert TradingEngine._compute_force_close_time("other") is None

    def test_is_force_close_time(self):
        fc = time(13, 30)
        early = datetime(2026, 1, 5, 9, 0)
        late = datetime(2026, 1, 5, 13, 35)
        assert not TradingEngine._is_force_close_time(early, fc)
        assert TradingEngine._is_force_close_time(late, fc)

    def test_force_close_sells_position(self):
        """Bars after force close time should trigger close."""
        late = datetime(2026, 1, 5, 13, 35)  # after 13:30
        early = datetime(2026, 1, 5, 9, 0)

        bars = [
            make_bar(early, close=50000.0),
            make_bar(late, close=50000.0),
        ]
        signals = [
            _signal(SignalType.PEAK, close=50000.0),
            _signal(SignalType.NONE, close=50000.0),
        ]
        engine = _make_engine({"A": bars}, signals, market="kr")
        result = engine.run()
        assert result["A"]["buys"] == 1
        assert result["A"]["sells"] == 1


class TestCashBalance:
    def test_rejected_buy_on_insufficient_cash(self, capsys):
        bars = make_bars(2)
        signals = [
            _signal(SignalType.PEAK, close=50000.0),
            _signal(SignalType.NONE),
        ]
        engine = _make_engine({"A": bars}, signals, capital=1)
        result = engine.run()
        assert result["A"]["buys"] == 0  # rejected

    def test_cash_decreases_on_buy(self):
        bars = make_bars(2)
        signals = [
            _signal(SignalType.PEAK, close=50000.0),
            _signal(SignalType.NONE),
        ]
        engine = _make_engine({"A": bars}, signals, capital=10_000_000)
        result = engine.run()
        assert engine.broker.get_cash_balance() < 10_000_000


class TestMultiSymbol:
    def test_two_symbols_independent_trades(self):
        start = datetime(2026, 1, 5, 9, 0)
        bars_a = [make_bar(start, 50000.0), make_bar(start + timedelta(minutes=1), 50010.0)]
        bars_b = [make_bar(start, 70000.0), make_bar(start + timedelta(minutes=1), 70010.0)]

        # Total signals = bars_a + bars_b = 4 signals in interleaved order
        # Engine processes: A bar1, B bar1, A bar2, B bar2
        signals = [
            _signal(SignalType.PEAK, close=50000.0),   # A bar 1 -> buy A
            _signal(SignalType.NONE, close=70000.0),    # B bar 1
            _signal(SignalType.NONE, close=50010.0),    # A bar 2
            _signal(SignalType.PEAK, close=70010.0),    # B bar 2 -> buy B
        ]
        engine = _make_engine({"A": bars_a, "B": bars_b}, signals)
        result = engine.run()
        assert result["A"]["buys"] == 1
        assert result["B"]["buys"] == 1

    def test_shared_cash_across_symbols(self):
        start = datetime(2026, 1, 5, 9, 0)
        bars_a = [make_bar(start, 50000.0)]
        bars_b = [make_bar(start, 70000.0)]

        signals = [
            _signal(SignalType.PEAK, close=50000.0),
            _signal(SignalType.PEAK, close=70000.0),
        ]
        engine = _make_engine({"A": bars_a, "B": bars_b}, signals, capital=10_000_000)
        result = engine.run()
        # Both should have bought using shared cash
        total_buys = result["A"]["buys"] + result["B"]["buys"]
        assert total_buys == 2
        assert engine.broker.get_cash_balance() < 10_000_000


class TestNotifierIntegration:
    def test_buy_event_sent(self, capsys):
        bars = make_bars(2)
        signals = [
            _signal(SignalType.PEAK, close=50000.0),
            _signal(SignalType.NONE),
        ]
        notifier = ConsoleNotifier()
        engine = _make_engine({"A": bars}, signals, notifiers=[notifier])
        engine.run()
        out = capsys.readouterr().out
        assert "BUY PUT" in out

    def test_session_end_event_sent(self, capsys):
        bars = make_bars(1)
        signals = [_signal(SignalType.NONE)]
        notifier = ConsoleNotifier()
        engine = _make_engine({"A": bars}, signals, notifiers=[notifier])
        engine.run()
        out = capsys.readouterr().out
        assert "Session Summary" in out


class TestTradeDBIntegration:
    def test_trade_recorded(self, tmp_path):
        bars = make_bars(3)
        signals = [
            _signal(SignalType.PEAK, close=50000.0),
            _signal(SignalType.TROUGH, close=50020.0),
            _signal(SignalType.NONE),
        ]
        db = TradeDB(tmp_path / "test.db")
        engine = _make_engine({"A": bars}, signals, trade_db=db)
        engine.run()

        import sqlite3
        conn = sqlite3.connect(str(tmp_path / "test.db"))
        trade_count = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        summary_count = conn.execute("SELECT COUNT(*) FROM daily_summary").fetchone()[0]
        conn.close()
        db.close()

        assert trade_count == 2  # 1 buy + 1 sell
        assert summary_count == 1

    def test_summary_recorded_even_no_trades(self, tmp_path):
        bars = make_bars(1)
        signals = [_signal(SignalType.NONE)]
        db = TradeDB(tmp_path / "test.db")
        engine = _make_engine({"A": bars}, signals, trade_db=db)
        engine.run()

        import sqlite3
        conn = sqlite3.connect(str(tmp_path / "test.db"))
        summary_count = conn.execute("SELECT COUNT(*) FROM daily_summary").fetchone()[0]
        conn.close()
        db.close()

        assert summary_count == 1


class TestSignalCounting:
    def test_peak_and_trough_counted(self):
        bars = make_bars(4)
        signals = [
            _signal(SignalType.PEAK, close=50000.0),
            _signal(SignalType.TROUGH, close=50010.0),
            _signal(SignalType.PEAK, close=50020.0),
            _signal(SignalType.NONE),
        ]
        engine = _make_engine({"A": bars}, signals)
        result = engine.run()
        assert result["A"]["peak_signals"] == 2
        assert result["A"]["trough_signals"] == 1
