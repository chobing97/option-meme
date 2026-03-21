"""Analyzer — compute performance metrics, compare results, export to DataFrames."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd

from src.backtest.result import SimulationResult


class Analyzer:
    """Compute and compare backtest performance metrics."""

    # ------------------------------------------------------------------
    def compute_metrics(self, result: SimulationResult) -> dict:
        """
        Compute performance metrics from simulation result.

        Returns dict with standard trading metrics.
        Handles edge cases: 0 trades, all wins, single day, etc.
        """
        trades = result.trades
        snapshots = result.snapshots

        total_trades = len(trades)

        # --- Equity ---
        if snapshots:
            initial_equity = snapshots[0].equity
            final_equity = snapshots[-1].equity
        else:
            initial_equity = 0.0
            final_equity = 0.0

        total_return = (final_equity - initial_equity) / initial_equity if initial_equity > 0 else 0.0

        if total_trades == 0:
            return {
                "total_return": total_return,
                "total_trades": 0,
                "win_rate": 0.0,
                "avg_win_pct": 0.0,
                "avg_loss_pct": 0.0,
                "profit_factor": 0.0,
                "max_drawdown_pct": self._compute_mdd(snapshots),
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "avg_holding_minutes": 0.0,
                "total_pnl": 0.0,
                "exit_reasons": {},
                "monthly_returns": {},
                "weekday_returns": {},
            }

        # --- Win / loss (break-even excluded from losses) ---
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl < 0]

        win_rate = len(wins) / total_trades

        avg_win_pct = float(np.mean([t.pnl_pct for t in wins])) if wins else 0.0
        avg_loss_pct = float(np.mean([t.pnl_pct for t in losses])) if losses else 0.0

        sum_wins = sum(t.pnl for t in wins)
        sum_losses = abs(sum(t.pnl for t in losses))
        if sum_losses == 0:
            profit_factor = float("inf") if sum_wins > 0 else 0.0
        else:
            profit_factor = sum_wins / sum_losses

        # --- Drawdown ---
        max_drawdown_pct = self._compute_mdd(snapshots)

        # --- Sharpe / Sortino ---
        sharpe_ratio = self._compute_sharpe(snapshots)
        sortino_ratio = self._compute_sortino(snapshots)

        # --- Holding time ---
        avg_holding_minutes = float(np.mean([t.holding_minutes for t in trades]))

        # --- Total PnL ---
        total_pnl = sum(t.pnl for t in trades)

        # --- Exit reasons ---
        exit_reasons: dict[str, int] = defaultdict(int)
        for t in trades:
            exit_reasons[t.exit_reason] += 1
        exit_reasons = dict(exit_reasons)

        # --- Monthly returns ---
        monthly_returns: dict[str, float] = defaultdict(float)
        for t in trades:
            if t.exit_time is not None:
                key = t.exit_time.strftime("%Y-%m")
                monthly_returns[key] += t.pnl

        # --- Weekday returns ---
        weekday_returns: dict[str, float] = defaultdict(float)
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        for t in trades:
            if t.exit_time is not None:
                wd = day_names[t.exit_time.weekday()]
                weekday_returns[wd] += t.pnl

        return {
            "total_return": total_return,
            "total_trades": total_trades,
            "win_rate": win_rate,
            "avg_win_pct": avg_win_pct,
            "avg_loss_pct": avg_loss_pct,
            "profit_factor": profit_factor,
            "max_drawdown_pct": max_drawdown_pct,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "avg_holding_minutes": avg_holding_minutes,
            "total_pnl": total_pnl,
            "exit_reasons": exit_reasons,
            "monthly_returns": dict(monthly_returns),
            "weekday_returns": dict(weekday_returns),
        }

    # ------------------------------------------------------------------
    def compare(self, results: list[SimulationResult]) -> pd.DataFrame:
        """
        Compare multiple backtest results in a single table.
        Sort by total_return descending.
        """
        rows = []
        for r in results:
            m = self.compute_metrics(r)
            cfg = r.config if isinstance(r.config, dict) else vars(r.config)
            rows.append({
                "threshold": cfg.get("threshold", 0),
                "tp_pct": cfg.get("tp_pct", 0),
                "sl_pct": cfg.get("sl_pct", 0),
                "total_return": m["total_return"],
                "win_rate": m["win_rate"],
                "max_drawdown_pct": m["max_drawdown_pct"],
                "sharpe_ratio": m["sharpe_ratio"],
                "total_trades": m["total_trades"],
                "profit_factor": m["profit_factor"],
                "avg_holding_minutes": m["avg_holding_minutes"],
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("total_return", ascending=False).reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    def to_dataframes(self, result: SimulationResult) -> dict:
        """
        Convert result to dashboard-friendly format.
        Returns: {"trades": pd.DataFrame, "equity": pd.DataFrame, "metrics": dict}
        """
        return {
            "trades": result.trades_df(),
            "equity": result.equity_df(),
            "metrics": self.compute_metrics(result),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_mdd(snapshots) -> float:
        """Maximum drawdown percentage from equity curve."""
        if not snapshots:
            return 0.0
        equities = [s.equity for s in snapshots]
        peak = equities[0]
        mdd = 0.0
        for eq in equities:
            if eq > peak:
                peak = eq
            dd = (eq - peak) / peak if peak > 0 else 0.0
            if dd < mdd:
                mdd = dd
        return mdd

    @staticmethod
    def _compute_sharpe(snapshots, risk_free: float = 0.0) -> float:
        """Annualized Sharpe ratio from daily returns."""
        daily_returns = Analyzer._daily_returns(snapshots)
        if len(daily_returns) < 2:
            return 0.0
        mean_r = np.mean(daily_returns) - risk_free
        std_r = np.std(daily_returns, ddof=1)
        if std_r == 0:
            return 0.0
        return float(mean_r / std_r * math.sqrt(252))

    @staticmethod
    def _compute_sortino(snapshots, risk_free: float = 0.0) -> float:
        """Annualized Sortino ratio (downside deviation only)."""
        daily_returns = Analyzer._daily_returns(snapshots)
        if len(daily_returns) < 2:
            return 0.0
        mean_r = np.mean(daily_returns) - risk_free
        downside = [r for r in daily_returns if r < 0]
        if len(downside) < 2:
            return 0.0
        down_std = float(np.std(downside, ddof=1))
        if down_std == 0:
            return 0.0
        return float(mean_r / down_std * math.sqrt(252))

    @staticmethod
    def _daily_returns(snapshots) -> list[float]:
        """Extract daily returns from snapshots (last equity per day)."""
        if not snapshots:
            return []
        # Group by date, take last equity of each day
        daily_equity: dict[str, float] = {}
        for s in snapshots:
            day_key = s.timestamp.date() if hasattr(s.timestamp, "date") else str(s.timestamp)[:10]
            daily_equity[day_key] = s.equity
        equities = list(daily_equity.values())
        if len(equities) < 2:
            return []
        returns = []
        for i in range(1, len(equities)):
            prev = equities[i - 1]
            if prev > 0:
                returns.append((equities[i] - prev) / prev)
        return returns
