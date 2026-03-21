from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


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


@dataclass
class FillResult:
    status: str               # "FILLED" or "REJECTED"
    fill_price: float = 0.0
    fill_time: Optional[datetime] = None
    contract: Optional[OptionContract] = None
    reject_reason: str = ""   # "NO_LIQUIDITY" / "INSUFFICIENT_CASH" / "NO_CHAIN" / ""


class Executor(ABC):
    """Abstract interface for trade execution. Implemented by BacktestExecutor and LiveExecutor."""

    @abstractmethod
    def get_option_chain(self, symbol: str, option_type: str, timestamp: datetime) -> list[OptionContract]:
        """Get available option contracts at given timestamp."""
        ...

    @abstractmethod
    def execute_buy(self, contract: OptionContract, quantity: int, timestamp: datetime) -> FillResult:
        """Execute a buy order. Returns fill result."""
        ...

    @abstractmethod
    def execute_sell(self, contract: OptionContract, quantity: int, timestamp: datetime) -> FillResult:
        """Execute a sell order. Returns fill result."""
        ...

    @abstractmethod
    def get_mark_price(self, contract: OptionContract, timestamp: datetime) -> float:
        """Get current mark-to-market price for a contract."""
        ...

    @abstractmethod
    def get_cash(self) -> float:
        """Get current cash balance."""
        ...

    @abstractmethod
    def get_positions(self) -> list[Position]:
        """Get all open positions."""
        ...

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific underlying symbol, or None."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset executor state (for grid search - run multiple backtests)."""
        ...
