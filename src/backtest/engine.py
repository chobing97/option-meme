"""BacktestEngine — pure loop connecting Strategy + Executor + MarketData."""

from __future__ import annotations

import pandas as pd
from datetime import datetime
from typing import Optional

from src.backtest.result import Trade, BarSnapshot, SimulationResult
from src.backtest.strategy.base import BaseStrategy
from src.backtest.executor.base import Executor
from src.backtest.market_data import MarketData
from src.backtest.types import Side, OrderResult


class BacktestEngine:
    """Pure loop engine. Delegates all decisions to Strategy, execution to Executor."""

    def __init__(self, strategy: BaseStrategy, executor: Executor, market_data: MarketData):
        self.strategy = strategy
        self.executor = executor
        self.market_data = market_data
        # Inject market_data into strategy
        self.strategy.set_market_data(market_data)

    # ------------------------------------------------------------------
    def run(
        self,
        df: pd.DataFrame,
        market: str = "us",
        session_minutes: int = 390,
    ) -> SimulationResult:
        """Run backtest. Engine does NOT know about DataFrame columns — Strategy does."""
        if df.empty:
            return SimulationResult(
                trades=[],
                snapshots=[],
                config=self.strategy.config_dict(),
                metadata={"market": market, "symbols": [], "date_range": "", "timeframe": "1m", "total_bars": 0},
            )

        df = df.sort_values("datetime").reset_index(drop=True)

        trades: list[Trade] = []
        snapshots: list[BarSnapshot] = []

        open_trades: dict[str, Trade] = {}
        entry_bar_counts: dict[str, int] = {}
        bar_counts: dict[str, int] = {}

        trade_id_counter = 0
        portfolio = self.executor.get_portfolio_state()
        equity_high = portfolio.equity if portfolio.equity > 0 else portfolio.cash

        prev_date: Optional[datetime] = None
        last_ts = None

        for idx, row in df.iterrows():
            ts: datetime = row["datetime"]
            if isinstance(ts, pd.Timestamp):
                ts = ts.to_pydatetime()
            symbol: str = row["symbol"]
            last_ts = ts

            current_date = ts.date()

            # --- Day boundary: delegate to strategy.on_day_end() ---
            if prev_date is not None and current_date != prev_date:
                portfolio = self.executor.get_portfolio_state()
                close_orders = self.strategy.on_day_end(portfolio, {"session_minutes": session_minutes})
                for order in close_orders:
                    result = self.executor.execute(order, ts)
                    if result.status == "FILLED":
                        self._record_sell(
                            result, order, ts, open_trades, entry_bar_counts,
                            bar_counts, trades, snapshots, equity_high, row,
                        )
                equity_high = max(equity_high, self.executor.get_portfolio_state().equity)
                self.strategy.reset()
                self.strategy.on_day_start(str(current_date))

            prev_date = current_date

            # Track bars per symbol
            bar_counts[symbol] = bar_counts.get(symbol, 0) + 1

            # Mark-to-market all positions
            self.executor.update_marks(ts)

            # Get portfolio state for strategy
            portfolio = self.executor.get_portfolio_state()

            # Strategy decides
            context = {"session_minutes": session_minutes, "bar_index": idx, "timestamp": ts}
            orders = self.strategy.on_bar(row, portfolio, context)

            # Execute orders
            action_str = ""
            reason_str = ""
            for order in orders:
                result = self.executor.execute(order, ts)
                if result.status == "FILLED":
                    if order.side == Side.BUY:
                        trade_id_counter += 1
                        t = Trade(
                            trade_id=trade_id_counter,
                            symbol=order.symbol,
                            entry_time=ts,
                            entry_price=result.fill_price,
                            entry_strike=result.contract.strike if result.contract else None,
                            entry_expiry=result.contract.expiry if result.contract else None,
                            entry_underlying=order.reference_price,
                            instrument_type=order.instrument_type,
                            quantity=order.quantity,
                        )
                        open_trades[order.symbol] = t
                        entry_bar_counts[order.symbol] = bar_counts.get(order.symbol, 0)
                        action_str = "BUY"
                        reason_str = order.reason
                    elif order.side == Side.SELL:
                        self._record_sell(
                            result, order, ts, open_trades, entry_bar_counts,
                            bar_counts, trades, snapshots, equity_high, row,
                        )
                        action_str = "SELL"
                        reason_str = order.reason

            # Compute snapshot
            portfolio = self.executor.get_portfolio_state()
            equity_high = max(equity_high, portfolio.equity)
            dd_pct = (portfolio.equity - equity_high) / equity_high if equity_high > 0 else 0.0

            pos = portfolio.get_position(symbol)
            close_price = float(row.get("close", 0.0)) if hasattr(row, "get") else float(row["close"]) if "close" in row.index else 0.0
            snapshots.append(BarSnapshot(
                timestamp=ts,
                symbol=symbol,
                underlying_close=close_price,
                peak_prob=float(row.get("peak_prob", 0.0)) if hasattr(row, "get") else 0.0,
                trough_prob=float(row.get("trough_prob", 0.0)) if hasattr(row, "get") else 0.0,
                action=action_str,
                reason=reason_str,
                position_qty=pos.quantity if pos else 0,
                option_mark_price=pos.current_price if pos else 0.0,
                cash=portfolio.cash,
                position_value=portfolio.equity - portfolio.cash,
                equity=portfolio.equity,
                drawdown_pct=dd_pct,
            ))

        # End of data: delegate to strategy.on_data_end()
        if last_ts is not None:
            portfolio = self.executor.get_portfolio_state()
            if portfolio.positions:
                end_orders = self.strategy.on_data_end(portfolio, {"session_minutes": session_minutes})
                for order in end_orders:
                    result = self.executor.execute(order, last_ts)
                    if result.status == "FILLED":
                        self._record_sell(
                            result, order, last_ts, open_trades, entry_bar_counts,
                            bar_counts, trades, snapshots, equity_high,
                        )

        # Build metadata
        symbols = sorted(df["symbol"].unique().tolist())
        date_min = df["datetime"].min()
        date_max = df["datetime"].max()
        metadata = {
            "market": market,
            "symbols": symbols,
            "date_range": f"{date_min} ~ {date_max}",
            "timeframe": "1m",
            "total_bars": len(df),
        }

        return SimulationResult(
            trades=trades,
            snapshots=snapshots,
            config=self.strategy.config_dict(),
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    def run_grid(
        self,
        df: pd.DataFrame,
        market: str,
        strategies: list[BaseStrategy],
        session_minutes: int = 390,
    ) -> list[SimulationResult]:
        """Run backtest for each strategy. Reset executor between runs."""
        results = []
        for strategy in strategies:
            self.strategy = strategy
            self.strategy.set_market_data(self.market_data)
            self.executor.reset()
            result = self.run(df, market, session_minutes)
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _record_sell(
        self,
        result: OrderResult,
        order,
        ts: datetime,
        open_trades: dict[str, Trade],
        entry_bar_counts: dict[str, int],
        bar_counts: dict[str, int],
        trades: list[Trade],
        snapshots: list[BarSnapshot],
        equity_high: float,
        row=None,
    ) -> None:
        """Record a completed sell trade."""
        symbol = order.symbol
        if symbol not in open_trades:
            return

        t = open_trades[symbol]
        holding_bars = bar_counts.get(symbol, 0) - entry_bar_counts.get(symbol, 0)
        holding_minutes = (ts - t.entry_time).total_seconds() / 60

        # Get underlying close from row if available
        exit_underlying = 0.0
        if row is not None:
            exit_underlying = float(row.get("close", 0.0)) if hasattr(row, "get") else 0.0
        elif snapshots:
            for snap in reversed(snapshots):
                if snap.symbol == symbol:
                    exit_underlying = snap.underlying_close
                    break

        t.close(
            exit_time=ts,
            exit_price=result.fill_price,
            exit_underlying=exit_underlying,
            exit_reason=order.reason,
            holding_bars=holding_bars,
            holding_minutes=int(holding_minutes),
        )
        trades.append(t)
        del open_trades[symbol]
        if symbol in entry_bar_counts:
            del entry_bar_counts[symbol]
