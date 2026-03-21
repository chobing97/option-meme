import pytest
from datetime import datetime

from src.backtest.result import Trade, BarSnapshot, SimulationResult
from src.backtest.strategy import StrategyConfig


class TestTrade:
    def test_trade_creation(self):
        """Trade 생성 시 필수 필드 존재 및 기본값 확인."""
        t = Trade(
            trade_id=1,
            symbol="AAPL",
            entry_time=datetime(2026, 3, 1, 9, 45),
            entry_price=3.0,
            entry_strike=150.0,
            entry_expiry=datetime(2026, 3, 7),
            entry_underlying=150.0,
        )
        assert t.trade_id == 1
        assert t.symbol == "AAPL"
        assert t.is_open is True
        assert t.pnl == 0.0
        assert t.pnl_pct == 0.0
        assert t.quantity == 1

    def test_trade_pnl_calculation(self):
        """close() 호출 시 PnL 계산 정확성."""
        t = Trade(
            trade_id=1,
            symbol="AAPL",
            entry_time=datetime(2026, 3, 1, 9, 45),
            entry_price=3.0,
            entry_strike=150.0,
            entry_expiry=datetime(2026, 3, 7),
            entry_underlying=150.0,
            quantity=2,
        )
        t.close(
            exit_time=datetime(2026, 3, 1, 11, 30),
            exit_price=3.6,
            exit_underlying=148.0,
            exit_reason="TROUGH_SIGNAL",
            holding_bars=105,
            holding_minutes=105,
        )
        assert t.pnl == pytest.approx(1.2)  # (3.6 - 3.0) * 2
        assert t.pnl_pct == pytest.approx(0.2)  # (3.6 - 3.0) / 3.0
        assert t.is_open is False

    def test_trade_win_loss(self):
        """pnl > 0 -> win, pnl <= 0 -> loss."""
        t = Trade(
            trade_id=1, symbol="AAPL",
            entry_time=datetime(2026, 3, 1, 9, 45),
            entry_price=3.0, entry_strike=150.0,
            entry_expiry=datetime(2026, 3, 7), entry_underlying=150.0,
        )
        # Winning trade
        t.close(datetime(2026, 3, 1, 10, 0), 3.5, 149.0, "TP", 15, 15)
        assert t.is_win is True

        # Losing trade
        t2 = Trade(
            trade_id=2, symbol="AAPL",
            entry_time=datetime(2026, 3, 1, 9, 45),
            entry_price=3.0, entry_strike=150.0,
            entry_expiry=datetime(2026, 3, 7), entry_underlying=150.0,
        )
        t2.close(datetime(2026, 3, 1, 10, 0), 2.5, 151.0, "SL", 15, 15)
        assert t2.is_win is False

    def test_trade_zero_entry_price(self):
        """entry_price=0 일 때 pnl_pct 0.0 (ZeroDivisionError 방지)."""
        t = Trade(
            trade_id=1, symbol="AAPL",
            entry_time=datetime(2026, 3, 1, 9, 45),
            entry_price=0.0, entry_strike=150.0,
            entry_expiry=datetime(2026, 3, 7), entry_underlying=150.0,
        )
        t.close(datetime(2026, 3, 1, 10, 0), 1.0, 149.0, "TP", 15, 15)
        assert t.pnl_pct == 0.0


class TestBarSnapshot:
    def test_bar_snapshot_creation(self):
        """BarSnapshot 생성 및 equity = cash + position_value 검증."""
        snap = BarSnapshot(
            timestamp=datetime(2026, 3, 1, 9, 45),
            symbol="AAPL",
            underlying_close=150.0,
            peak_prob=0.5,
            trough_prob=0.1,
            action="BUY",
            reason="PEAK_SIGNAL",
            position_qty=1,
            option_mark_price=3.0,
            cash=9_700.0,
            position_value=300.0,
            equity=10_000.0,
            drawdown_pct=0.0,
        )
        assert snap.equity == snap.cash + snap.position_value

    def test_bar_snapshot_drawdown(self):
        """equity < equity_high -> drawdown_pct < 0."""
        snap = BarSnapshot(
            timestamp=datetime(2026, 3, 1, 10, 0),
            symbol="AAPL",
            underlying_close=148.0,
            peak_prob=0.1,
            trough_prob=0.3,
            action="",
            reason="",
            position_qty=1,
            option_mark_price=2.5,
            cash=9_700.0,
            position_value=250.0,
            equity=9_950.0,
            drawdown_pct=-0.005,  # (9950 - 10000) / 10000
        )
        assert snap.drawdown_pct < 0


class TestSimulationResult:
    def _make_trades(self, n=3):
        trades = []
        for i in range(n):
            t = Trade(
                trade_id=i, symbol="AAPL",
                entry_time=datetime(2026, 3, 1, 9, 45 + i),
                entry_price=3.0, entry_strike=150.0,
                entry_expiry=datetime(2026, 3, 7), entry_underlying=150.0,
            )
            t.close(datetime(2026, 3, 1, 10, 45 + i), 3.3, 149.0, "TP", 60, 60)
            trades.append(t)
        return trades

    def _make_snapshots(self, n=5):
        snaps = []
        for i in range(n):
            snaps.append(BarSnapshot(
                timestamp=datetime(2026, 3, 1, 9, 30 + i),
                symbol="AAPL",
                underlying_close=150.0 + i * 0.1,
                peak_prob=0.2, trough_prob=0.1,
                action="", reason="",
                position_qty=0, option_mark_price=0.0,
                cash=10_000.0, position_value=0.0,
                equity=10_000.0, drawdown_pct=0.0,
            ))
        return snaps

    def test_trades_df(self):
        """trades_df() -> DataFrame with correct columns and rows."""
        trades = self._make_trades(3)
        config = StrategyConfig()
        result = SimulationResult(trades, [], config, {})
        df = result.trades_df()
        assert len(df) == 3
        assert "trade_id" in df.columns
        assert "pnl" in df.columns
        assert "exit_reason" in df.columns

    def test_equity_df(self):
        """equity_df() -> DataFrame with correct columns and time-sorted."""
        snaps = self._make_snapshots(5)
        config = StrategyConfig()
        result = SimulationResult([], snaps, config, {})
        df = result.equity_df()
        assert len(df) == 5
        assert "equity" in df.columns
        assert "timestamp" in df.columns
        # timestamps should be sorted
        assert list(df["timestamp"]) == sorted(df["timestamp"])

    def test_empty_result(self):
        """빈 trades/snapshots -> empty DataFrames, no error."""
        config = StrategyConfig()
        result = SimulationResult([], [], config, {})
        assert result.trades_df().empty
        assert result.equity_df().empty

    def test_save(self, tmp_path):
        """save() -> parquet files created."""
        trades = self._make_trades(2)
        snaps = self._make_snapshots(3)
        config = StrategyConfig()
        result = SimulationResult(trades, snaps, config, {"market": "us"})
        result.save(tmp_path / "output")
        assert (tmp_path / "output" / "trades.parquet").exists()
        assert (tmp_path / "output" / "equity.parquet").exists()
