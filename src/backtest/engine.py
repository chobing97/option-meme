"""BacktestEngine — connects Strategy + Executor and runs the backtest loop."""

from __future__ import annotations

import pandas as pd
from datetime import datetime
from typing import Optional

from src.backtest.result import Trade, BarSnapshot, SimulationResult
from src.backtest.strategy.base import BaseStrategy, Action
from src.backtest.executor.base import Executor, Position


class BacktestEngine:
    """Connects Strategy + Executor, runs the backtest loop."""

    def __init__(self, strategy: BaseStrategy, executor: Executor):
        self.strategy = strategy
        self.executor = executor

    # ------------------------------------------------------------------
    def run(
        self,
        pred_df: pd.DataFrame,
        market: str = "us",
        session_minutes: int = 390,
    ) -> SimulationResult:
        """
        Run backtest on prediction data.

        pred_df columns required:
            datetime, symbol, close, peak_prob, trough_prob, minutes_from_open
        """
        if pred_df.empty:
            return SimulationResult(
                trades=[],
                snapshots=[],
                config=self.strategy.config_dict(),
                metadata={"market": market, "symbols": [], "date_range": "", "timeframe": "1m", "total_bars": 0},
            )

        df = pred_df.sort_values("datetime").reset_index(drop=True)

        trades: list[Trade] = []
        snapshots: list[BarSnapshot] = []

        # Per-symbol state
        open_trades: dict[str, Trade] = {}          # symbol -> open Trade
        entry_bar_counts: dict[str, int] = {}       # symbol -> bar index at entry
        bar_counts: dict[str, int] = {}             # symbol -> bars seen

        trade_id_counter = 0
        equity_high = self.executor.get_cash()

        prev_date: Optional[datetime] = None

        for idx, row in df.iterrows():
            ts: datetime = row["datetime"]
            if isinstance(ts, pd.Timestamp):
                ts = ts.to_pydatetime()
            symbol: str = row["symbol"]
            close: float = row["close"]
            peak_prob: float = row["peak_prob"]
            trough_prob: float = row["trough_prob"]
            minutes_from_open: float = row["minutes_from_open"]

            current_date = ts.date()

            # --- Day boundary: force close all open positions from previous day ---
            if prev_date is not None and current_date != prev_date:
                self._force_close_all(
                    open_trades, entry_bar_counts, bar_counts,
                    trades, prev_date, snapshots, equity_high,
                )
                equity_high = max(equity_high, self._current_equity())
                self.strategy.reset()
                self.strategy.on_day_start(str(current_date))

            prev_date = current_date

            # Track bars per symbol
            bar_counts[symbol] = bar_counts.get(symbol, 0) + 1

            # Get current position for this symbol
            position: Optional[Position] = self.executor.get_position(symbol)

            # Update mark price if position exists
            if position is not None:
                mark = self.executor.get_mark_price(position.contract, ts)
                position.update_mark(mark)

            # Build bar dict for strategy
            bar = {
                "close": close,
                "peak_prob": peak_prob,
                "trough_prob": trough_prob,
                "minutes_from_open": minutes_from_open,
            }

            action_result = self.strategy.on_bar(bar, position, session_minutes)

            action_str = ""
            reason_str = ""

            if action_result.action == Action.BUY and symbol not in open_trades:
                _cfg = self.strategy.config_dict()
                chain = self.executor.get_option_chain(symbol, _cfg.get("option_type", "put"), ts)
                if chain:
                    # ATM selection
                    atm = min(chain, key=lambda c: abs(c.strike - close))
                    fill = self.executor.execute_buy(atm, _cfg.get("quantity", 1), ts)
                    if fill.status == "FILLED":
                        trade_id_counter += 1
                        t = Trade(
                            trade_id=trade_id_counter,
                            symbol=symbol,
                            entry_time=ts,
                            entry_price=fill.fill_price,
                            entry_strike=atm.strike,
                            entry_expiry=atm.expiry,
                            entry_underlying=close,
                            quantity=_cfg.get("quantity", 1),
                        )
                        open_trades[symbol] = t
                        entry_bar_counts[symbol] = bar_counts[symbol]
                        action_str = "BUY"
                        reason_str = action_result.reason

            elif action_result.action == Action.SELL and symbol in open_trades:
                t = open_trades[symbol]
                pos = self.executor.get_position(symbol)
                if pos is not None:
                    fill = self.executor.execute_sell(pos.contract, pos.quantity, ts)
                    if fill.status == "FILLED":
                        holding_bars = bar_counts[symbol] - entry_bar_counts.get(symbol, bar_counts[symbol])
                        holding_minutes = (ts - t.entry_time).total_seconds() / 60
                        t.close(
                            exit_time=ts,
                            exit_price=fill.fill_price,
                            exit_underlying=close,
                            exit_reason=action_result.reason,
                            holding_bars=holding_bars,
                            holding_minutes=int(holding_minutes),
                        )
                        trades.append(t)
                        del open_trades[symbol]
                        if symbol in entry_bar_counts:
                            del entry_bar_counts[symbol]
                        action_str = "SELL"
                        reason_str = action_result.reason

            # Compute snapshot
            cash = self.executor.get_cash()
            pos_value = self._total_position_value()
            equity = cash + pos_value
            equity_high = max(equity_high, equity)
            dd_pct = (equity - equity_high) / equity_high if equity_high > 0 else 0.0

            pos = self.executor.get_position(symbol)
            snapshots.append(BarSnapshot(
                timestamp=ts,
                symbol=symbol,
                underlying_close=close,
                peak_prob=peak_prob,
                trough_prob=trough_prob,
                action=action_str,
                reason=reason_str,
                position_qty=pos.quantity if pos else 0,
                option_mark_price=pos.current_price if pos else 0.0,
                cash=cash,
                position_value=pos_value,
                equity=equity,
                drawdown_pct=dd_pct,
            ))

        # End of data: force close remaining positions
        if open_trades and prev_date is not None:
            self._force_close_all(
                open_trades, entry_bar_counts, bar_counts,
                trades, prev_date, snapshots, equity_high,
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
        pred_df: pd.DataFrame,
        market: str,
        strategies: list[BaseStrategy],
        session_minutes: int = 390,
    ) -> list[SimulationResult]:
        """Run backtest for each strategy. Reset executor between runs."""
        results = []
        for strategy in strategies:
            self.strategy = strategy
            self.executor.reset()
            result = self.run(pred_df, market, session_minutes)
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _total_position_value(self) -> float:
        """Sum position values: qty * current_price * 100 (options multiplier)."""
        total = 0.0
        for pos in self.executor.get_positions():
            total += pos.quantity * pos.current_price * 100
        return total

    def _current_equity(self) -> float:
        return self.executor.get_cash() + self._total_position_value()

    def _force_close_all(
        self,
        open_trades: dict[str, Trade],
        entry_bar_counts: dict[str, int],
        bar_counts: dict[str, int],
        trades: list[Trade],
        close_date,
        snapshots: list[BarSnapshot],
        equity_high: float,
    ) -> None:
        """Force close all open positions (day boundary or end of data)."""
        for symbol in list(open_trades.keys()):
            t = open_trades[symbol]
            pos = self.executor.get_position(symbol)
            if pos is None:
                del open_trades[symbol]
                continue
            # Use last known time for the close
            close_ts = t.entry_time  # fallback
            # Find the latest bar timestamp for this symbol from snapshots
            for snap in reversed(snapshots):
                if snap.symbol == symbol:
                    close_ts = snap.timestamp
                    break

            fill = self.executor.execute_sell(pos.contract, pos.quantity, close_ts)
            if fill.status == "FILLED":
                holding_bars = bar_counts.get(symbol, 0) - entry_bar_counts.get(symbol, 0)
                holding_minutes = (close_ts - t.entry_time).total_seconds() / 60
                # Use last known underlying close
                exit_underlying = 0.0
                for snap in reversed(snapshots):
                    if snap.symbol == symbol:
                        exit_underlying = snap.underlying_close
                        break
                t.close(
                    exit_time=close_ts,
                    exit_price=fill.fill_price,
                    exit_underlying=exit_underlying,
                    exit_reason="FORCE_CLOSE",
                    holding_bars=holding_bars,
                    holding_minutes=int(holding_minutes),
                )
                trades.append(t)
                # Emit snapshot for force-close event
                cash = self.executor.get_cash()
                pos_value = self._total_position_value()
                equity = cash + pos_value
                snapshots.append(BarSnapshot(
                    timestamp=close_ts,
                    symbol=symbol,
                    underlying_close=exit_underlying,
                    peak_prob=0.0,
                    trough_prob=0.0,
                    action="SELL",
                    reason="FORCE_CLOSE",
                    position_qty=0,
                    option_mark_price=0.0,
                    cash=cash,
                    position_value=pos_value,
                    equity=equity,
                    drawdown_pct=(equity - equity_high) / equity_high if equity_high > 0 else 0.0,
                ))
            del open_trades[symbol]
            if symbol in entry_bar_counts:
                del entry_bar_counts[symbol]
