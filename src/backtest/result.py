from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import pandas as pd


@dataclass
class Trade:
    trade_id: int
    symbol: str
    entry_time: datetime
    entry_price: float          # option buy price
    entry_strike: float
    entry_expiry: datetime
    entry_underlying: float     # stock price at entry
    exit_time: Optional[datetime] = None
    exit_price: float = 0.0
    exit_underlying: float = 0.0
    exit_reason: str = ""       # TROUGH_SIGNAL / TP / SL / FORCE_CLOSE
    quantity: int = 1
    pnl: float = 0.0
    pnl_pct: float = 0.0
    holding_bars: int = 0
    holding_minutes: int = 0

    def close(self, exit_time, exit_price, exit_underlying, exit_reason, holding_bars, holding_minutes):
        """Close the trade and compute PnL."""
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.exit_underlying = exit_underlying
        self.exit_reason = exit_reason
        self.holding_bars = holding_bars
        self.holding_minutes = holding_minutes
        self.pnl = (exit_price - self.entry_price) * self.quantity
        self.pnl_pct = (exit_price - self.entry_price) / self.entry_price if self.entry_price > 0 else 0.0

    @property
    def is_open(self) -> bool:
        return self.exit_time is None

    @property
    def is_win(self) -> bool:
        return self.pnl > 0


@dataclass
class BarSnapshot:
    timestamp: datetime
    symbol: str
    underlying_close: float
    peak_prob: float
    trough_prob: float
    action: str                 # "BUY" / "SELL" / ""
    reason: str                 # "PEAK_SIGNAL" / "TP" / "SL" / "FORCE_CLOSE" / "TROUGH_SIGNAL" / ""
    position_qty: int
    option_mark_price: float
    cash: float
    position_value: float       # position_qty * option_mark_price * 100
    equity: float               # cash + position_value
    drawdown_pct: float         # (equity - equity_high) / equity_high


# Forward reference for StrategyConfig
class SimulationResult:
    """Container for backtest results."""

    def __init__(self, trades: list[Trade], snapshots: list[BarSnapshot], config, metadata: dict):
        self.trades = trades
        self.snapshots = snapshots
        self.config = config
        self.metadata = metadata

    def trades_df(self) -> pd.DataFrame:
        """Convert trades to DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([vars(t) for t in self.trades])

    def equity_df(self) -> pd.DataFrame:
        """Convert snapshots to DataFrame."""
        if not self.snapshots:
            return pd.DataFrame()
        return pd.DataFrame([vars(s) for s in self.snapshots])

    def save(self, path) -> None:
        """Save trades and equity curve to parquet files."""
        from pathlib import Path
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        trades = self.trades_df()
        if not trades.empty:
            trades.to_parquet(path / "trades.parquet", index=False)
        equity = self.equity_df()
        if not equity.empty:
            equity.to_parquet(path / "equity.parquet", index=False)
