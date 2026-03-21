"""Tests for Analyzer — 13 test cases."""

import pytest
import math
from datetime import datetime, timedelta

from src.backtest.analyzer import Analyzer
from src.backtest.result import Trade, BarSnapshot, SimulationResult
from src.backtest.strategy import PutBuyConfig as StrategyConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_trade(
    trade_id: int = 1,
    symbol: str = "AAPL",
    entry_price: float = 3.0,
    exit_price: float = 3.3,
    exit_reason: str = "TROUGH_SIGNAL",
    entry_time: datetime = None,
    exit_time: datetime = None,
    holding_minutes: int = 30,
) -> Trade:
    if entry_time is None:
        entry_time = datetime(2026, 3, 20, 9, 45)
    if exit_time is None:
        exit_time = entry_time + timedelta(minutes=holding_minutes)
    t = Trade(
        trade_id=trade_id,
        symbol=symbol,
        entry_time=entry_time,
        entry_price=entry_price,
        entry_strike=150.0,
        entry_expiry=datetime(2026, 3, 28),
        entry_underlying=150.0,
        quantity=1,
    )
    t.close(
        exit_time=exit_time,
        exit_price=exit_price,
        exit_underlying=149.0,
        exit_reason=exit_reason,
        holding_bars=holding_minutes,
        holding_minutes=holding_minutes,
    )
    return t


def _make_snapshots(
    equities: list[float],
    base_time: datetime = datetime(2026, 3, 20, 9, 30),
    day_size: int = 0,
) -> list[BarSnapshot]:
    """Build snapshots from equity values. If day_size > 0, advance date every day_size bars."""
    snaps = []
    equity_high = equities[0] if equities else 0
    for i, eq in enumerate(equities):
        if day_size > 0:
            day_offset = i // day_size
            minute_offset = i % day_size
            ts = base_time + timedelta(days=day_offset, minutes=minute_offset)
        else:
            ts = base_time + timedelta(minutes=i)
        equity_high = max(equity_high, eq)
        dd = (eq - equity_high) / equity_high if equity_high > 0 else 0.0
        snaps.append(BarSnapshot(
            timestamp=ts,
            symbol="AAPL",
            underlying_close=150.0,
            peak_prob=0.1,
            trough_prob=0.1,
            action="",
            reason="",
            position_qty=0,
            option_mark_price=0.0,
            cash=eq,
            position_value=0.0,
            equity=eq,
            drawdown_pct=dd,
        ))
    return snaps


def _make_result(trades=None, snapshots=None, config=None) -> SimulationResult:
    if trades is None:
        trades = []
    if snapshots is None:
        snapshots = []
    if config is None:
        config = StrategyConfig()
    return SimulationResult(trades=trades, snapshots=snapshots, config=config, metadata={"market": "us"})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestAnalyzer:

    # 1. Win rate: 3 wins 2 losses -> 60%
    def test_win_rate(self):
        trades = [
            _make_trade(1, exit_price=3.5, exit_reason="TP"),       # win
            _make_trade(2, exit_price=3.2, exit_reason="TROUGH"),   # win
            _make_trade(3, exit_price=3.1, exit_reason="TROUGH"),   # win
            _make_trade(4, exit_price=2.8, exit_reason="SL"),       # loss
            _make_trade(5, exit_price=2.9, exit_reason="SL"),       # loss
        ]
        snaps = _make_snapshots([10_000_000] * 5)
        result = _make_result(trades, snaps)
        m = Analyzer().compute_metrics(result)
        assert m["win_rate"] == pytest.approx(0.6)
        assert m["total_trades"] == 5

    # 2. Profit factor: total_wins / total_losses
    def test_profit_factor(self):
        trades = [
            _make_trade(1, exit_price=4.0),   # pnl = 1.0
            _make_trade(2, exit_price=2.0),   # pnl = -1.0
        ]
        snaps = _make_snapshots([10_000_000] * 2)
        result = _make_result(trades, snaps)
        m = Analyzer().compute_metrics(result)
        assert m["profit_factor"] == pytest.approx(1.0)

    # 3. MDD from equity curve
    def test_mdd(self):
        # Equity: 100, 110, 90, 95 -> peak=110, trough=90, MDD = (90-110)/110 = -18.18%
        snaps = _make_snapshots([100, 110, 90, 95])
        result = _make_result(snapshots=snaps)
        m = Analyzer().compute_metrics(result)
        assert m["max_drawdown_pct"] == pytest.approx((90 - 110) / 110, abs=1e-6)

    # 4. Sharpe ratio
    def test_sharpe_ratio(self):
        # Multi-day equity for meaningful Sharpe; need at least 1 trade to avoid early return
        snaps = _make_snapshots([100, 102, 104, 103, 106, 108], day_size=2)
        trades = [_make_trade(1, exit_price=3.5)]
        result = _make_result(trades=trades, snapshots=snaps)
        m = Analyzer().compute_metrics(result)
        # Sharpe should be non-zero with positive returns
        assert isinstance(m["sharpe_ratio"], float)
        assert m["sharpe_ratio"] > 0

    # 5. Average holding time
    def test_avg_holding_minutes(self):
        trades = [
            _make_trade(1, holding_minutes=20),
            _make_trade(2, holding_minutes=40),
            _make_trade(3, holding_minutes=60),
        ]
        snaps = _make_snapshots([10_000_000] * 3)
        result = _make_result(trades, snaps)
        m = Analyzer().compute_metrics(result)
        assert m["avg_holding_minutes"] == pytest.approx(40.0)

    # 6. Exit reason distribution
    def test_exit_reasons(self):
        trades = [
            _make_trade(1, exit_reason="TROUGH_SIGNAL"),
            _make_trade(2, exit_reason="TROUGH_SIGNAL"),
            _make_trade(3, exit_reason="TP"),
            _make_trade(4, exit_reason="SL"),
        ]
        snaps = _make_snapshots([10_000_000] * 4)
        result = _make_result(trades, snaps)
        m = Analyzer().compute_metrics(result)
        assert m["exit_reasons"] == {"TROUGH_SIGNAL": 2, "TP": 1, "SL": 1}

    # 7. Monthly returns
    def test_monthly_returns(self):
        t1 = _make_trade(
            1, exit_price=3.5,
            entry_time=datetime(2026, 1, 15, 10, 0),
            exit_time=datetime(2026, 1, 15, 11, 0),
        )
        t2 = _make_trade(
            2, exit_price=3.2,
            entry_time=datetime(2026, 2, 10, 10, 0),
            exit_time=datetime(2026, 2, 10, 11, 0),
        )
        snaps = _make_snapshots([10_000_000] * 2)
        result = _make_result([t1, t2], snaps)
        m = Analyzer().compute_metrics(result)
        assert "2026-01" in m["monthly_returns"]
        assert "2026-02" in m["monthly_returns"]

    # 8. Weekday returns
    def test_weekday_returns(self):
        # 2026-03-20 is a Friday
        t1 = _make_trade(
            1, exit_price=3.5,
            entry_time=datetime(2026, 3, 20, 10, 0),
            exit_time=datetime(2026, 3, 20, 11, 0),
        )
        snaps = _make_snapshots([10_000_000] * 2)
        result = _make_result([t1], snaps)
        m = Analyzer().compute_metrics(result)
        assert "Friday" in m["weekday_returns"]

    # 9. Compare 2 results -> 2-row table
    def test_compare_two_results(self):
        r1 = _make_result(
            [_make_trade(1, exit_price=3.5)],
            _make_snapshots([10000, 10100]),
            StrategyConfig(threshold=0.3, tp_pct=0.10, sl_pct=-0.05),
        )
        r2 = _make_result(
            [_make_trade(1, exit_price=2.5)],
            _make_snapshots([10000, 9900]),
            StrategyConfig(threshold=0.4, tp_pct=0.15, sl_pct=-0.03),
        )
        df = Analyzer().compare([r1, r2])
        assert len(df) == 2

    # 10. Compare columns exist
    def test_compare_columns(self):
        r1 = _make_result(
            [_make_trade(1)],
            _make_snapshots([10000, 10050]),
            StrategyConfig(threshold=0.3),
        )
        df = Analyzer().compare([r1])
        expected_cols = {
            "threshold", "tp_pct", "sl_pct", "total_return", "win_rate",
            "max_drawdown_pct", "sharpe_ratio", "total_trades", "profit_factor",
            "avg_holding_minutes",
        }
        assert expected_cols.issubset(set(df.columns))

    # 11. Empty result -> zeros
    def test_empty_result(self):
        result = _make_result()
        m = Analyzer().compute_metrics(result)
        assert m["total_trades"] == 0
        assert m["win_rate"] == 0.0
        assert m["profit_factor"] == 0.0
        assert m["sharpe_ratio"] == 0.0
        assert m["total_pnl"] == 0.0

    # 12. Single trade
    def test_single_trade(self):
        t = _make_trade(1, exit_price=3.5)  # win
        snaps = _make_snapshots([10000, 10050])
        result = _make_result([t], snaps)
        m = Analyzer().compute_metrics(result)
        assert m["total_trades"] == 1
        assert m["win_rate"] == 1.0
        assert m["profit_factor"] == float("inf")

    # 13. to_dataframes format
    def test_to_dataframes_format(self):
        t = _make_trade(1, exit_price=3.5)
        snaps = _make_snapshots([10000, 10050])
        result = _make_result([t], snaps)
        dfs = Analyzer().to_dataframes(result)
        assert "trades" in dfs
        assert "equity" in dfs
        assert "metrics" in dfs
        assert hasattr(dfs["trades"], "columns")  # is DataFrame
        assert hasattr(dfs["equity"], "columns")
        assert isinstance(dfs["metrics"], dict)
