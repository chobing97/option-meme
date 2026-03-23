from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.backtest.types import Order, OrderResult, PortfolioState


@dataclass
class OptionContract:
    symbol: str           # underlying symbol (e.g. "AAPL")
    strike: float
    expiry: datetime
    option_type: str      # "put" or "call"
    contract_id: str = ""

    def __post_init__(self):
        if not self.contract_id:
            exp_str = self.expiry.strftime("%Y%m%d")
            self.contract_id = f"{self.symbol}_{self.option_type[0].upper()}_{self.strike:.0f}_{exp_str}"


@dataclass
class Position:
    contract: OptionContract
    quantity: int
    avg_entry_price: float
    current_price: float = 0.0
    unrealized_pnl_pct: float = 0.0

    def update_mark(self, price: float) -> None:
        self.current_price = price
        if self.avg_entry_price > 0:
            self.unrealized_pnl_pct = (price - self.avg_entry_price) / self.avg_entry_price


# Kept for backward compatibility — new code should use OrderResult from types.py
@dataclass
class FillResult:
    status: str               # "FILLED" or "REJECTED"
    fill_price: float = 0.0
    fill_time: Optional[datetime] = None
    contract: Optional[OptionContract] = None
    reject_reason: str = ""   # "NO_LIQUIDITY" / "INSUFFICIENT_CASH" / "NO_CHAIN" / ""


class Executor(ABC):
    """Abstract broker interface. Executes orders, manages cash/positions."""

    @abstractmethod
    def execute(self, order: Order, timestamp: datetime) -> OrderResult:
        """Execute a trading order. Returns fill result."""
        ...

    @abstractmethod
    def get_portfolio_state(self) -> PortfolioState:
        """Return snapshot of cash + positions + equity."""
        ...

    @abstractmethod
    def update_marks(self, timestamp: datetime) -> None:
        """Mark-to-market all positions at given timestamp."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset state for grid search."""
        ...
