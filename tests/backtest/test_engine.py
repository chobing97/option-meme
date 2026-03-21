"""Tests for BacktestEngine — 13 test cases using MockExecutor."""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

from src.backtest.engine import BacktestEngine
from src.backtest.strategy import Strategy, StrategyConfig
from src.backtest.executor.base import (
    Executor, OptionContract, Position, FillResult,
)


# ---------------------------------------------------------------------------
# MockExecutor
# ---------------------------------------------------------------------------
class MockExecutor(Executor):
    """Controllable mock executor for engine tests."""

    def __init__(self, capital: float = 10_000_000):
        self._initial_capital = capital
        self._cash = capital
        self._positions: list[Position] = []
        self._mark_prices: dict[str, float] = {}   # contract_id -> price
        self._chains: dict[str, list[OptionContract]] = {}  # symbol -> contracts
        self._fill_price: float = 3.0

    # --- Executor ABC ---
    def get_option_chain(self, symbol: str, option_type: str, timestamp: datetime) -> list[OptionContract]:
        return self._chains.get(symbol, [])

    def execute_buy(self, contract: OptionContract, quantity: int, timestamp: datetime) -> FillResult:
        cost = self._fill_price * quantity * 100
        if cost > self._cash:
            return FillResult(status="REJECTED", reject_reason="INSUFFICIENT_CASH")
        self._cash -= cost
        pos = Position(
            contract=contract,
            quantity=quantity,
            avg_entry_price=self._fill_price,
            current_price=self._fill_price,
            unrealized_pnl_pct=0.0,
        )
        self._positions.append(pos)
        return FillResult(
            status="FILLED",
            fill_price=self._fill_price,
            fill_time=timestamp,
            contract=contract,
        )

    def execute_sell(self, contract: OptionContract, quantity: int, timestamp: datetime) -> FillResult:
        mark = self._mark_prices.get(contract.contract_id, self._fill_price)
        self._cash += mark * quantity * 100
        self._positions = [p for p in self._positions if p.contract.contract_id != contract.contract_id]
        return FillResult(
            status="FILLED",
            fill_price=mark,
            fill_time=timestamp,
            contract=contract,
        )

    def get_mark_price(self, contract: OptionContract, timestamp: datetime) -> float:
        return self._mark_prices.get(contract.contract_id, self._fill_price)

    def get_cash(self) -> float:
        return self._cash

    def get_positions(self) -> list[Position]:
        return list(self._positions)

    def get_position(self, symbol: str) -> Optional[Position]:
        for p in self._positions:
            if p.contract.symbol == symbol:
                return p
        return None

    def reset(self) -> None:
        self._cash = self._initial_capital
        self._positions = []
        self._mark_prices = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_chain(symbol: str, strike: float = 150.0) -> list[OptionContract]:
    return [
        OptionContract(symbol=symbol, strike=strike, expiry=datetime(2026, 3, 28), option_type="put"),
    ]


def _make_pred_df(
    bars: list[dict],
    symbol: str = "AAPL",
    base_time: datetime = datetime(2026, 3, 20, 9, 30),
) -> pd.DataFrame:
    """Build pred_df from a list of bar dicts (peak_prob, trough_prob, close, minutes_from_open)."""
    rows = []
    for i, b in enumerate(bars):
        rows.append({
            "datetime": base_time + timedelta(minutes=i),
            "symbol": symbol,
            "close": b.get("close", 150.0),
            "peak_prob": b.get("peak_prob", 0.1),
            "trough_prob": b.get("trough_prob", 0.1),
            "minutes_from_open": b.get("minutes_from_open", i),
        })
    return pd.DataFrame(rows)


def _neutral_bar(**overrides) -> dict:
    d = {"peak_prob": 0.1, "trough_prob": 0.1, "close": 150.0, "minutes_from_open": 60}
    d.update(overrides)
    return d


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestBacktestEngine:
    """13 engine test cases."""

    # 1. No signals -> 0 trades
    def test_no_signals_zero_trades(self):
        executor = MockExecutor()
        executor._chains["AAPL"] = _make_chain("AAPL")
        strategy = Strategy(StrategyConfig(threshold=0.3))
        engine = BacktestEngine(strategy, executor)

        bars = [_neutral_bar(minutes_from_open=i) for i in range(10)]
        df = _make_pred_df(bars)

        result = engine.run(df, market="us")
        assert len(result.trades) == 0

    # 2. One PEAK -> TROUGH cycle -> 1 trade
    def test_one_cycle_one_trade(self):
        executor = MockExecutor()
        executor._chains["AAPL"] = _make_chain("AAPL")
        strategy = Strategy(StrategyConfig(threshold=0.3))
        engine = BacktestEngine(strategy, executor)

        bars = [
            _neutral_bar(minutes_from_open=0),
            {"peak_prob": 0.5, "trough_prob": 0.1, "close": 150.0, "minutes_from_open": 1},  # BUY
            _neutral_bar(minutes_from_open=2),
            {"peak_prob": 0.1, "trough_prob": 0.5, "close": 148.0, "minutes_from_open": 3},  # SELL
            _neutral_bar(minutes_from_open=4),
        ]
        df = _make_pred_df(bars)
        result = engine.run(df, market="us")

        assert len(result.trades) == 1
        assert result.trades[0].exit_reason == "TROUGH_SIGNAL"
        assert result.trades[0].symbol == "AAPL"

    # 3. Multiple cycles -> N trades
    def test_multiple_cycles(self):
        executor = MockExecutor()
        executor._chains["AAPL"] = _make_chain("AAPL")
        strategy = Strategy(StrategyConfig(threshold=0.3))
        engine = BacktestEngine(strategy, executor)

        bars = []
        for cycle in range(3):
            offset = cycle * 4
            bars.append({"peak_prob": 0.5, "trough_prob": 0.1, "close": 150.0, "minutes_from_open": offset})
            bars.append(_neutral_bar(minutes_from_open=offset + 1))
            bars.append({"peak_prob": 0.1, "trough_prob": 0.5, "close": 148.0, "minutes_from_open": offset + 2})
            bars.append(_neutral_bar(minutes_from_open=offset + 3))

        df = _make_pred_df(bars)
        result = engine.run(df, market="us")
        assert len(result.trades) == 3

    # 4. TP exit
    def test_tp_exit(self):
        executor = MockExecutor()
        executor._fill_price = 3.0
        executor._chains["AAPL"] = _make_chain("AAPL")
        strategy = Strategy(StrategyConfig(threshold=0.3, tp_pct=0.10))
        engine = BacktestEngine(strategy, executor)

        # After BUY, set mark price to trigger TP (3.0 * 1.12 = 3.36)
        bars = [
            {"peak_prob": 0.5, "trough_prob": 0.1, "close": 150.0, "minutes_from_open": 0},  # BUY
        ]
        # Add bars where mark price is high
        for i in range(1, 5):
            bars.append(_neutral_bar(minutes_from_open=i))

        df = _make_pred_df(bars)

        # After first bar processes buy, set mark price high for TP
        contract_id = _make_chain("AAPL")[0].contract_id
        executor._mark_prices[contract_id] = 3.36  # +12% > tp_pct 10%

        result = engine.run(df, market="us")
        assert len(result.trades) == 1
        assert result.trades[0].exit_reason == "TP"

    # 5. SL exit
    def test_sl_exit(self):
        executor = MockExecutor()
        executor._fill_price = 3.0
        executor._chains["AAPL"] = _make_chain("AAPL")
        strategy = Strategy(StrategyConfig(threshold=0.3, sl_pct=-0.05))
        engine = BacktestEngine(strategy, executor)

        bars = [
            {"peak_prob": 0.5, "trough_prob": 0.1, "close": 150.0, "minutes_from_open": 0},  # BUY
        ]
        for i in range(1, 5):
            bars.append(_neutral_bar(minutes_from_open=i))

        df = _make_pred_df(bars)

        contract_id = _make_chain("AAPL")[0].contract_id
        executor._mark_prices[contract_id] = 2.82  # -6% < sl_pct -5%

        result = engine.run(df, market="us")
        assert len(result.trades) == 1
        assert result.trades[0].exit_reason == "SL"

    # 6. Force close at session end
    def test_force_close_session_end(self):
        executor = MockExecutor()
        executor._chains["AAPL"] = _make_chain("AAPL")
        strategy = Strategy(StrategyConfig(threshold=0.3, force_close_minutes=120))
        engine = BacktestEngine(strategy, executor)

        bars = [
            {"peak_prob": 0.5, "trough_prob": 0.1, "close": 150.0, "minutes_from_open": 10},  # BUY
            _neutral_bar(minutes_from_open=100),
            _neutral_bar(minutes_from_open=270),  # 390-120=270 -> force close
        ]
        df = _make_pred_df(bars)
        result = engine.run(df, market="us", session_minutes=390)

        assert len(result.trades) == 1
        assert result.trades[0].exit_reason == "FORCE_CLOSE"

    # 7. Day boundary -> force close open position
    def test_day_boundary_force_close(self):
        executor = MockExecutor()
        executor._chains["AAPL"] = _make_chain("AAPL")
        strategy = Strategy(StrategyConfig(threshold=0.3))
        engine = BacktestEngine(strategy, executor)

        day1 = datetime(2026, 3, 20, 9, 30)
        day2 = datetime(2026, 3, 21, 9, 30)

        rows = [
            {"datetime": day1, "symbol": "AAPL", "close": 150.0, "peak_prob": 0.5, "trough_prob": 0.1, "minutes_from_open": 10},
            {"datetime": day1 + timedelta(minutes=30), "symbol": "AAPL", "close": 150.0, "peak_prob": 0.1, "trough_prob": 0.1, "minutes_from_open": 40},
            # Day boundary: position should be force-closed
            {"datetime": day2, "symbol": "AAPL", "close": 149.0, "peak_prob": 0.1, "trough_prob": 0.1, "minutes_from_open": 0},
        ]
        df = pd.DataFrame(rows)

        result = engine.run(df, market="us")
        assert len(result.trades) == 1
        assert result.trades[0].exit_reason == "FORCE_CLOSE"

    # 8. Equity curve continuity
    def test_equity_curve_continuity(self):
        executor = MockExecutor(capital=10_000_000)
        executor._chains["AAPL"] = _make_chain("AAPL")
        strategy = Strategy(StrategyConfig(threshold=0.3))
        engine = BacktestEngine(strategy, executor)

        bars = [_neutral_bar(minutes_from_open=i) for i in range(5)]
        df = _make_pred_df(bars)

        result = engine.run(df, market="us")
        assert len(result.snapshots) == 5
        # No trades, equity should stay constant
        for snap in result.snapshots:
            assert snap.equity == 10_000_000

    # 9. Drawdown accuracy
    def test_drawdown_accuracy(self):
        executor = MockExecutor()
        executor._fill_price = 3.0
        executor._chains["AAPL"] = _make_chain("AAPL")
        strategy = Strategy(StrategyConfig(threshold=0.3))
        engine = BacktestEngine(strategy, executor)

        # BUY then price drops -> negative drawdown
        bars = [
            {"peak_prob": 0.5, "trough_prob": 0.1, "close": 150.0, "minutes_from_open": 0},  # BUY
            _neutral_bar(minutes_from_open=1),
            _neutral_bar(minutes_from_open=2),
        ]
        df = _make_pred_df(bars)

        contract_id = _make_chain("AAPL")[0].contract_id
        executor._mark_prices[contract_id] = 2.5  # price drop -> drawdown

        result = engine.run(df, market="us")
        # After BUY at 3.0 with qty=1, position_value = 1*3.0*100 = 300 initially
        # Then mark drops to 2.5, position_value = 250, loss of 50
        # Check that at least one snapshot has negative drawdown
        dd_vals = [s.drawdown_pct for s in result.snapshots]
        assert min(dd_vals) < 0

    # 10. Symbol with no option chain -> 0 trades
    def test_no_chain_zero_trades(self):
        executor = MockExecutor()
        # No chain set for AAPL
        strategy = Strategy(StrategyConfig(threshold=0.3))
        engine = BacktestEngine(strategy, executor)

        bars = [
            {"peak_prob": 0.5, "trough_prob": 0.1, "close": 150.0, "minutes_from_open": 0},
        ]
        df = _make_pred_df(bars)
        result = engine.run(df, market="us")
        assert len(result.trades) == 0

    # 11. Multiple symbols
    def test_multiple_symbols(self):
        executor = MockExecutor()
        executor._chains["AAPL"] = _make_chain("AAPL", strike=150.0)
        executor._chains["MSFT"] = _make_chain("MSFT", strike=300.0)
        strategy = Strategy(StrategyConfig(threshold=0.3))
        engine = BacktestEngine(strategy, executor)

        base = datetime(2026, 3, 20, 9, 30)
        rows = [
            # AAPL buy
            {"datetime": base, "symbol": "AAPL", "close": 150.0, "peak_prob": 0.5, "trough_prob": 0.1, "minutes_from_open": 0},
            # MSFT buy
            {"datetime": base + timedelta(minutes=1), "symbol": "MSFT", "close": 300.0, "peak_prob": 0.5, "trough_prob": 0.1, "minutes_from_open": 1},
            # AAPL sell
            {"datetime": base + timedelta(minutes=2), "symbol": "AAPL", "close": 149.0, "peak_prob": 0.1, "trough_prob": 0.5, "minutes_from_open": 2},
            # MSFT sell
            {"datetime": base + timedelta(minutes=3), "symbol": "MSFT", "close": 299.0, "peak_prob": 0.1, "trough_prob": 0.5, "minutes_from_open": 3},
        ]
        df = pd.DataFrame(rows)

        result = engine.run(df, market="us")
        assert len(result.trades) == 2
        symbols = {t.symbol for t in result.trades}
        assert symbols == {"AAPL", "MSFT"}

    # 12. run_grid with multiple configs
    def test_run_grid(self):
        executor = MockExecutor()
        executor._chains["AAPL"] = _make_chain("AAPL")
        strategy = Strategy(StrategyConfig(threshold=0.3))
        engine = BacktestEngine(strategy, executor)

        bars = [
            {"peak_prob": 0.5, "trough_prob": 0.1, "close": 150.0, "minutes_from_open": 0},
            {"peak_prob": 0.1, "trough_prob": 0.5, "close": 148.0, "minutes_from_open": 1},
        ]
        df = _make_pred_df(bars)

        configs = [
            StrategyConfig(threshold=0.3),
            StrategyConfig(threshold=0.6),  # too high -> no trades
        ]
        results = engine.run_grid(df, "us", configs)

        assert len(results) == 2
        assert len(results[0].trades) == 1  # threshold=0.3 triggers
        assert len(results[1].trades) == 0  # threshold=0.6 too high

    # 13. Empty pred_df -> empty result
    def test_empty_pred_df(self):
        executor = MockExecutor()
        strategy = Strategy(StrategyConfig())
        engine = BacktestEngine(strategy, executor)

        df = pd.DataFrame(columns=["datetime", "symbol", "close", "peak_prob", "trough_prob", "minutes_from_open"])
        result = engine.run(df, market="us")

        assert len(result.trades) == 0
        assert len(result.snapshots) == 0
        assert result.metadata["total_bars"] == 0
