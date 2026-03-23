"""Tests for BacktestEngine — 18 test cases using MockExecutor + MockMarketData."""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

from src.backtest.engine import BacktestEngine
from src.backtest.strategy import PutBuyStrategy as Strategy, PutBuyConfig as StrategyConfig
from src.backtest.strategy import FilteredPutStrategy, FilteredPutConfig, CallBuyStrategy, CallBuyConfig
from src.backtest.executor.base import Executor, OptionContract, Position
from src.backtest.market_data import MarketData
from src.backtest.types import Side, Order, OrderResult, PortfolioState, Quote


# ---------------------------------------------------------------------------
# MockMarketData
# ---------------------------------------------------------------------------
class MockMarketData(MarketData):
    """Mock market data for engine tests."""

    def __init__(self):
        self._chains: dict[str, list[OptionContract]] = {}
        self._mark_prices: dict[str, float] = {}  # contract_id -> price

    def get_stock_quote(self, symbol: str, timestamp: datetime) -> Optional[Quote]:
        return None

    def get_option_chain(self, symbol: str, option_type: str, timestamp: datetime) -> list[OptionContract]:
        return self._chains.get(symbol, [])

    def get_option_quote(self, contract: OptionContract, timestamp: datetime) -> Optional[Quote]:
        price = self._mark_prices.get(contract.contract_id, 3.0)
        return Quote(open=price, high=price + 0.1, low=price - 0.1, close=price, volume=100)


# ---------------------------------------------------------------------------
# MockExecutor
# ---------------------------------------------------------------------------
class MockExecutor(Executor):
    """Controllable mock executor for engine tests."""

    def __init__(self, capital: float = 10_000_000, market_data: MockMarketData = None):
        self._initial_capital = capital
        self._cash = capital
        self._positions: list[Position] = []
        self._market_data = market_data or MockMarketData()
        self._fill_price: float = 3.0

    def execute(self, order: Order, timestamp: datetime) -> OrderResult:
        if order.side == Side.BUY:
            return self._execute_buy(order, timestamp)
        elif order.side == Side.SELL:
            return self._execute_sell(order, timestamp)
        return OrderResult(order=order, status="REJECTED", reject_reason="UNKNOWN_SIDE")

    def _execute_buy(self, order: Order, timestamp: datetime) -> OrderResult:
        # Get chain
        chain = self._market_data.get_option_chain(order.symbol, order.option_type or "put", timestamp)
        if not chain:
            return OrderResult(order=order, status="REJECTED", reject_reason="NO_CHAIN")

        # ATM selection
        ref_price = order.reference_price if order.reference_price > 0 else 150.0
        contract = min(chain, key=lambda c: abs(c.strike - ref_price))

        # Use fixed fill price for buy (simulates entry price independent of mark)
        fill_price = self._fill_price

        cost = fill_price * order.quantity * 100
        if cost > self._cash:
            return OrderResult(order=order, status="REJECTED", reject_reason="INSUFFICIENT_CASH")

        self._cash -= cost
        pos = Position(
            contract=contract,
            quantity=order.quantity,
            avg_entry_price=fill_price,
            current_price=fill_price,
            unrealized_pnl_pct=0.0,
        )
        self._positions.append(pos)
        return OrderResult(
            order=order,
            status="FILLED",
            fill_price=fill_price,
            fill_time=timestamp,
            contract=contract,
        )

    def _execute_sell(self, order: Order, timestamp: datetime) -> OrderResult:
        pos = next((p for p in self._positions if p.contract.symbol == order.symbol), None)
        if pos is None:
            return OrderResult(order=order, status="REJECTED", reject_reason="NO_POSITION")

        quote = self._market_data.get_option_quote(pos.contract, timestamp)
        fill_price = quote.close if quote else self._fill_price

        self._cash += fill_price * pos.quantity * 100
        self._positions = [p for p in self._positions if p.contract.contract_id != pos.contract.contract_id]
        return OrderResult(
            order=order,
            status="FILLED",
            fill_price=fill_price,
            fill_time=timestamp,
            contract=pos.contract,
        )

    def get_portfolio_state(self) -> PortfolioState:
        pos_value = sum(p.quantity * p.current_price * 100 for p in self._positions)
        return PortfolioState(cash=self._cash, positions=list(self._positions), equity=self._cash + pos_value)

    def update_marks(self, timestamp: datetime) -> None:
        for pos in self._positions:
            quote = self._market_data.get_option_quote(pos.contract, timestamp)
            if quote:
                pos.update_mark(quote.close)

    def reset(self) -> None:
        self._cash = self._initial_capital
        self._positions = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_chain(symbol: str, strike: float = 150.0, option_type: str = "put") -> list[OptionContract]:
    return [
        OptionContract(symbol=symbol, strike=strike, expiry=datetime(2026, 3, 28), option_type=option_type),
    ]


def _make_pred_df(
    bars: list[dict],
    symbol: str = "AAPL",
    base_time: datetime = datetime(2026, 3, 20, 9, 30),
) -> pd.DataFrame:
    """Build pred_df from a list of bar dicts."""
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


def _make_engine(capital=10_000_000, chains=None, mark_prices=None, strategy=None):
    """Helper to create engine with MockMarketData + MockExecutor."""
    md = MockMarketData()
    if chains:
        md._chains = chains
    if mark_prices:
        md._mark_prices = mark_prices

    executor = MockExecutor(capital=capital, market_data=md)
    if strategy is None:
        strategy = Strategy(StrategyConfig(threshold=0.3))

    engine = BacktestEngine(strategy, executor, md)
    return engine, executor, md


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestBacktestEngine:
    """16 engine test cases + 2 new (on_day_end, on_data_end)."""

    # 1. No signals -> 0 trades
    def test_no_signals_zero_trades(self):
        engine, _, _ = _make_engine(chains={"AAPL": _make_chain("AAPL")})
        bars = [_neutral_bar(minutes_from_open=i) for i in range(10)]
        df = _make_pred_df(bars)
        result = engine.run(df, market="us")
        assert len(result.trades) == 0

    # 2. One PEAK -> TROUGH cycle -> 1 trade
    def test_one_cycle_one_trade(self):
        engine, _, _ = _make_engine(chains={"AAPL": _make_chain("AAPL")})
        bars = [
            _neutral_bar(minutes_from_open=0),
            {"peak_prob": 0.5, "trough_prob": 0.1, "close": 150.0, "minutes_from_open": 1},
            _neutral_bar(minutes_from_open=2),
            {"peak_prob": 0.1, "trough_prob": 0.5, "close": 148.0, "minutes_from_open": 3},
            _neutral_bar(minutes_from_open=4),
        ]
        df = _make_pred_df(bars)
        result = engine.run(df, market="us")
        assert len(result.trades) == 1
        assert result.trades[0].exit_reason == "TROUGH_SIGNAL"
        assert result.trades[0].symbol == "AAPL"

    # 3. Multiple cycles -> N trades
    def test_multiple_cycles(self):
        engine, _, _ = _make_engine(chains={"AAPL": _make_chain("AAPL")})
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
        contract_id = _make_chain("AAPL")[0].contract_id
        engine, _, _ = _make_engine(
            chains={"AAPL": _make_chain("AAPL")},
            mark_prices={contract_id: 3.36},  # +12% > tp_pct 10%
        )
        engine.strategy = Strategy(StrategyConfig(threshold=0.3, tp_pct=0.10))
        engine.strategy.set_market_data(engine.market_data)

        bars = [
            {"peak_prob": 0.5, "trough_prob": 0.1, "close": 150.0, "minutes_from_open": 0},
        ]
        for i in range(1, 5):
            bars.append(_neutral_bar(minutes_from_open=i))
        df = _make_pred_df(bars)
        result = engine.run(df, market="us")
        assert len(result.trades) == 1
        assert result.trades[0].exit_reason == "TP"

    # 5. SL exit
    def test_sl_exit(self):
        contract_id = _make_chain("AAPL")[0].contract_id
        engine, _, _ = _make_engine(
            chains={"AAPL": _make_chain("AAPL")},
            mark_prices={contract_id: 2.82},  # -6% < sl_pct -5%
        )
        engine.strategy = Strategy(StrategyConfig(threshold=0.3, sl_pct=-0.05))
        engine.strategy.set_market_data(engine.market_data)

        bars = [
            {"peak_prob": 0.5, "trough_prob": 0.1, "close": 150.0, "minutes_from_open": 0},
        ]
        for i in range(1, 5):
            bars.append(_neutral_bar(minutes_from_open=i))
        df = _make_pred_df(bars)
        result = engine.run(df, market="us")
        assert len(result.trades) == 1
        assert result.trades[0].exit_reason == "SL"

    # 6. Force close at session end
    def test_force_close_session_end(self):
        engine, _, _ = _make_engine(chains={"AAPL": _make_chain("AAPL")})
        engine.strategy = Strategy(StrategyConfig(threshold=0.3, force_close_minutes=120))
        engine.strategy.set_market_data(engine.market_data)

        bars = [
            {"peak_prob": 0.5, "trough_prob": 0.1, "close": 150.0, "minutes_from_open": 10},
            _neutral_bar(minutes_from_open=100),
            _neutral_bar(minutes_from_open=270),  # 390-120=270 -> force close
        ]
        df = _make_pred_df(bars)
        result = engine.run(df, market="us", session_minutes=390)
        assert len(result.trades) == 1
        assert result.trades[0].exit_reason == "FORCE_CLOSE"

    # 7. Day boundary -> force close open position via on_day_end
    def test_day_boundary_force_close(self):
        engine, _, _ = _make_engine(chains={"AAPL": _make_chain("AAPL")})

        day1 = datetime(2026, 3, 20, 9, 30)
        day2 = datetime(2026, 3, 21, 9, 30)

        rows = [
            {"datetime": day1, "symbol": "AAPL", "close": 150.0, "peak_prob": 0.5, "trough_prob": 0.1, "minutes_from_open": 10},
            {"datetime": day1 + timedelta(minutes=30), "symbol": "AAPL", "close": 150.0, "peak_prob": 0.1, "trough_prob": 0.1, "minutes_from_open": 40},
            {"datetime": day2, "symbol": "AAPL", "close": 149.0, "peak_prob": 0.1, "trough_prob": 0.1, "minutes_from_open": 0},
        ]
        df = pd.DataFrame(rows)
        result = engine.run(df, market="us")
        assert len(result.trades) == 1
        assert result.trades[0].exit_reason == "FORCE_CLOSE"

    # 8. Equity curve continuity
    def test_equity_curve_continuity(self):
        engine, _, _ = _make_engine(capital=10_000_000, chains={"AAPL": _make_chain("AAPL")})
        bars = [_neutral_bar(minutes_from_open=i) for i in range(5)]
        df = _make_pred_df(bars)
        result = engine.run(df, market="us")
        assert len(result.snapshots) == 5
        for snap in result.snapshots:
            assert snap.equity == 10_000_000

    # 9. Drawdown accuracy
    def test_drawdown_accuracy(self):
        contract_id = _make_chain("AAPL")[0].contract_id
        engine, _, _ = _make_engine(
            chains={"AAPL": _make_chain("AAPL")},
            mark_prices={contract_id: 2.5},  # price drop -> drawdown
        )

        bars = [
            {"peak_prob": 0.5, "trough_prob": 0.1, "close": 150.0, "minutes_from_open": 0},
            _neutral_bar(minutes_from_open=1),
            _neutral_bar(minutes_from_open=2),
        ]
        df = _make_pred_df(bars)
        result = engine.run(df, market="us")
        dd_vals = [s.drawdown_pct for s in result.snapshots]
        assert min(dd_vals) < 0

    # 10. Symbol with no option chain -> 0 trades (executor rejects)
    def test_no_chain_zero_trades(self):
        engine, _, _ = _make_engine()  # no chains set
        bars = [
            {"peak_prob": 0.5, "trough_prob": 0.1, "close": 150.0, "minutes_from_open": 0},
        ]
        df = _make_pred_df(bars)
        result = engine.run(df, market="us")
        assert len(result.trades) == 0

    # 11. Multiple symbols
    def test_multiple_symbols(self):
        engine, _, _ = _make_engine(chains={
            "AAPL": _make_chain("AAPL", strike=150.0),
            "MSFT": _make_chain("MSFT", strike=300.0),
        })

        base = datetime(2026, 3, 20, 9, 30)
        rows = [
            {"datetime": base, "symbol": "AAPL", "close": 150.0, "peak_prob": 0.5, "trough_prob": 0.1, "minutes_from_open": 0},
            {"datetime": base + timedelta(minutes=1), "symbol": "MSFT", "close": 300.0, "peak_prob": 0.5, "trough_prob": 0.1, "minutes_from_open": 1},
            {"datetime": base + timedelta(minutes=2), "symbol": "AAPL", "close": 149.0, "peak_prob": 0.1, "trough_prob": 0.5, "minutes_from_open": 2},
            {"datetime": base + timedelta(minutes=3), "symbol": "MSFT", "close": 299.0, "peak_prob": 0.1, "trough_prob": 0.5, "minutes_from_open": 3},
        ]
        df = pd.DataFrame(rows)
        result = engine.run(df, market="us")
        assert len(result.trades) == 2
        symbols = {t.symbol for t in result.trades}
        assert symbols == {"AAPL", "MSFT"}

    # 12. run_grid with multiple configs
    def test_run_grid(self):
        engine, _, _ = _make_engine(chains={"AAPL": _make_chain("AAPL")})
        bars = [
            {"peak_prob": 0.5, "trough_prob": 0.1, "close": 150.0, "minutes_from_open": 0},
            {"peak_prob": 0.1, "trough_prob": 0.5, "close": 148.0, "minutes_from_open": 1},
        ]
        df = _make_pred_df(bars)
        strategies = [
            Strategy(StrategyConfig(threshold=0.3)),
            Strategy(StrategyConfig(threshold=0.6)),
        ]
        results = engine.run_grid(df, "us", strategies)
        assert len(results) == 2
        assert len(results[0].trades) == 1
        assert len(results[1].trades) == 0

    # 13. Empty pred_df -> empty result
    def test_empty_pred_df(self):
        engine, _, _ = _make_engine()
        df = pd.DataFrame(columns=["datetime", "symbol", "close", "peak_prob", "trough_prob", "minutes_from_open"])
        result = engine.run(df, market="us")
        assert len(result.trades) == 0
        assert len(result.snapshots) == 0
        assert result.metadata["total_bars"] == 0

    # 14. FilteredPut engine run
    def test_filtered_put_engine_run(self):
        config = FilteredPutConfig(threshold=0.3, min_prob_gap=0.2, min_holding_minutes=0)
        strategy = FilteredPutStrategy(config)
        engine, _, _ = _make_engine(chains={"AAPL": _make_chain("AAPL")}, strategy=strategy)

        bars = [
            {"peak_prob": 0.6, "trough_prob": 0.1, "close": 150.0, "minutes_from_open": 0},
            _neutral_bar(minutes_from_open=1),
            {"peak_prob": 0.1, "trough_prob": 0.6, "close": 148.0, "minutes_from_open": 2},
        ]
        df = _make_pred_df(bars)
        result = engine.run(df, market="us")
        assert len(result.trades) == 1
        assert result.trades[0].exit_reason == "TROUGH_SIGNAL"

    # 15. CallBuy engine run
    def test_call_buy_engine_run(self):
        config = CallBuyConfig(threshold=0.3, min_holding_minutes=0)
        strategy = CallBuyStrategy(config)
        engine, _, md = _make_engine(
            chains={"AAPL": _make_chain("AAPL", option_type="call")},
            strategy=strategy,
        )

        bars = [
            {"peak_prob": 0.1, "trough_prob": 0.6, "close": 150.0, "minutes_from_open": 0},
            _neutral_bar(minutes_from_open=1),
            {"peak_prob": 0.6, "trough_prob": 0.1, "close": 152.0, "minutes_from_open": 2},
        ]
        df = _make_pred_df(bars)
        result = engine.run(df, market="us")
        assert len(result.trades) == 1
        assert result.trades[0].exit_reason == "PEAK_SIGNAL"

    # 16. on_day_start is called on day boundary
    def test_on_day_start_called_on_day_boundary(self):
        config = FilteredPutConfig(threshold=0.3, max_trades_per_day=1, min_prob_gap=0.2, min_holding_minutes=0)
        strategy = FilteredPutStrategy(config)
        engine, _, _ = _make_engine(chains={"AAPL": _make_chain("AAPL")}, strategy=strategy)

        day1 = datetime(2026, 3, 20, 9, 30)
        day2 = datetime(2026, 3, 21, 9, 30)

        rows = [
            {"datetime": day1, "symbol": "AAPL", "close": 150.0, "peak_prob": 0.6, "trough_prob": 0.1, "minutes_from_open": 0},
            {"datetime": day1 + timedelta(minutes=5), "symbol": "AAPL", "close": 148.0, "peak_prob": 0.1, "trough_prob": 0.6, "minutes_from_open": 5},
            {"datetime": day2, "symbol": "AAPL", "close": 150.0, "peak_prob": 0.6, "trough_prob": 0.1, "minutes_from_open": 0},
            {"datetime": day2 + timedelta(minutes=5), "symbol": "AAPL", "close": 148.0, "peak_prob": 0.1, "trough_prob": 0.6, "minutes_from_open": 5},
        ]
        df = pd.DataFrame(rows)
        result = engine.run(df, market="us")
        assert len(result.trades) >= 2

    # 17. on_day_end closes positions at day boundary
    def test_on_day_end_closes_positions(self):
        engine, _, _ = _make_engine(chains={"AAPL": _make_chain("AAPL")})

        day1 = datetime(2026, 3, 20, 9, 30)
        day2 = datetime(2026, 3, 21, 9, 30)

        rows = [
            {"datetime": day1, "symbol": "AAPL", "close": 150.0, "peak_prob": 0.5, "trough_prob": 0.1, "minutes_from_open": 0},
            {"datetime": day2, "symbol": "AAPL", "close": 149.0, "peak_prob": 0.1, "trough_prob": 0.1, "minutes_from_open": 0},
            {"datetime": day2 + timedelta(minutes=1), "symbol": "AAPL", "close": 149.0, "peak_prob": 0.1, "trough_prob": 0.1, "minutes_from_open": 1},
        ]
        df = pd.DataFrame(rows)
        result = engine.run(df, market="us")
        # Position opened on day1 should be closed via on_day_end at day boundary
        assert len(result.trades) == 1
        assert result.trades[0].exit_reason == "FORCE_CLOSE"
        # After force close, no position should remain on day2
        portfolio = engine.executor.get_portfolio_state()
        assert len(portfolio.positions) == 0

    # 18. on_data_end closes remaining positions
    def test_on_data_end_closes_remaining(self):
        engine, _, _ = _make_engine(chains={"AAPL": _make_chain("AAPL")})

        base = datetime(2026, 3, 20, 9, 30)
        rows = [
            {"datetime": base, "symbol": "AAPL", "close": 150.0, "peak_prob": 0.5, "trough_prob": 0.1, "minutes_from_open": 0},
            {"datetime": base + timedelta(minutes=1), "symbol": "AAPL", "close": 150.0, "peak_prob": 0.1, "trough_prob": 0.1, "minutes_from_open": 1},
            # Data ends with position still open — on_data_end should close it
        ]
        df = pd.DataFrame(rows)
        result = engine.run(df, market="us")
        assert len(result.trades) == 1
        assert result.trades[0].exit_reason == "FORCE_CLOSE"
