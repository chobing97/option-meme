"""Trade tracker: record per-bar snapshots for equity curve & trade analysis."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from config.settings import TRADE_DB_DIR


@dataclass
class BarSnapshot:
    """Single bar state snapshot."""

    timestamp: datetime
    symbol: str
    bar_num: int
    underlying_close: float
    signal: str  # "PEAK", "TROUGH", "NONE"
    peak_prob: float
    trough_prob: float
    action: str  # "BUY_PUT", "SELL_PUT", "" (empty = no action)
    reason: str  # "PEAK_SIGNAL", "STOP_LOSS", "PROFIT_TARGET", "FORCE_CLOSE", "TROUGH_SIGNAL", ""
    strike: float  # option strike (0 if no action)
    fill_price: float  # option fill price (0 if no action)
    position_qty: int  # current put position quantity after this bar
    position_avg_entry: float  # avg entry price of current position
    position_mark_price: float  # current mark-to-market price of option
    position_value: float  # position_qty * position_mark_price * 100
    cash: float
    equity: float  # cash + position_value
    equity_high: float  # running max equity (for drawdown calc)
    drawdown_pct: float  # (equity - equity_high) / equity_high


class TradeTracker:
    """Records per-bar snapshots during a trading session.

    Produces a DataFrame / parquet with the full equity curve,
    trade entry/exit markers, and drawdown tracking.
    """

    def __init__(self):
        self._snapshots: list[BarSnapshot] = []
        self._equity_high: dict[str, float] = {}  # per-symbol running max

    def record_bar(
        self,
        timestamp: datetime,
        symbol: str,
        bar_num: int,
        underlying_close: float,
        signal: str,
        peak_prob: float,
        trough_prob: float,
        action: str,
        reason: str,
        strike: float,
        fill_price: float,
        position_qty: int,
        position_avg_entry: float,
        position_mark_price: float,
        cash: float,
    ) -> None:
        """Record a single bar snapshot."""
        position_value = position_qty * position_mark_price * 100
        equity = cash + position_value

        # Update equity high watermark
        prev_high = self._equity_high.get(symbol, equity)
        equity_high = max(prev_high, equity)
        self._equity_high[symbol] = equity_high

        drawdown_pct = (
            (equity - equity_high) / equity_high if equity_high > 0 else 0.0
        )

        self._snapshots.append(
            BarSnapshot(
                timestamp=timestamp,
                symbol=symbol,
                bar_num=bar_num,
                underlying_close=underlying_close,
                signal=signal,
                peak_prob=peak_prob,
                trough_prob=trough_prob,
                action=action,
                reason=reason,
                strike=strike,
                fill_price=fill_price,
                position_qty=position_qty,
                position_avg_entry=position_avg_entry,
                position_mark_price=position_mark_price,
                position_value=position_value,
                cash=cash,
                equity=equity,
                equity_high=equity_high,
                drawdown_pct=drawdown_pct,
            )
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Convert snapshots to DataFrame."""
        if not self._snapshots:
            return pd.DataFrame()

        records = [
            {
                "timestamp": s.timestamp,
                "symbol": s.symbol,
                "bar_num": s.bar_num,
                "underlying_close": s.underlying_close,
                "signal": s.signal,
                "peak_prob": s.peak_prob,
                "trough_prob": s.trough_prob,
                "action": s.action,
                "reason": s.reason,
                "strike": s.strike,
                "fill_price": s.fill_price,
                "position_qty": s.position_qty,
                "position_avg_entry": s.position_avg_entry,
                "position_mark_price": s.position_mark_price,
                "position_value": s.position_value,
                "cash": s.cash,
                "equity": s.equity,
                "equity_high": s.equity_high,
                "drawdown_pct": s.drawdown_pct,
            }
            for s in self._snapshots
        ]
        return pd.DataFrame(records)

    def save(self, session_date: str, symbol: str | None = None) -> Path:
        """Save snapshots to parquet. Returns the output path."""
        out_dir = TRADE_DB_DIR / "backtests"
        out_dir.mkdir(parents=True, exist_ok=True)

        suffix = f"_{symbol}" if symbol else ""
        out_path = out_dir / f"{session_date}{suffix}.parquet"

        df = self.to_dataframe()
        if df.empty:
            logger.warning("No snapshots to save")
            return out_path

        df.to_parquet(out_path, index=False)
        logger.info(f"Saved {len(df)} bar snapshots to {out_path}")
        return out_path

    def summary(self) -> dict:
        """Compute summary statistics from snapshots."""
        df = self.to_dataframe()
        if df.empty:
            return {}

        trades = df[df["action"] != ""]
        buys = trades[trades["action"] == "BUY_PUT"]
        sells = trades[trades["action"] == "SELL_PUT"]

        return {
            "total_bars": len(df),
            "total_trades": len(trades),
            "buys": len(buys),
            "sells": len(sells),
            "final_equity": float(df["equity"].iloc[-1]),
            "initial_equity": float(df["equity"].iloc[0]),
            "return_pct": float(
                (df["equity"].iloc[-1] - df["equity"].iloc[0]) / df["equity"].iloc[0]
            ),
            "max_drawdown_pct": float(df["drawdown_pct"].min()),
            "equity_high": float(df["equity_high"].max()),
        }
