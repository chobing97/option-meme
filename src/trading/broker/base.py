"""Broker abstract base class and trading data models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    REJECTED = "REJECTED"


class SignalType(Enum):
    PEAK = "PEAK"
    TROUGH = "TROUGH"
    NONE = "NONE"


@dataclass
class OptionContract:
    symbol: str
    expiry: datetime
    strike: float
    option_type: str  # "put" or "call"
    contract_id: str = ""

    def __post_init__(self):
        if not self.contract_id:
            exp_str = self.expiry.strftime("%Y%m%d")
            self.contract_id = (
                f"{self.symbol}_{self.option_type[0].upper()}"
                f"_{self.strike:.0f}_{exp_str}"
            )


@dataclass
class OptionQuote:
    contract: OptionContract
    bid: float
    ask: float
    last: float
    volume: int
    timestamp: datetime


@dataclass
class Order:
    side: OrderSide
    contract: OptionContract
    quantity: int
    status: OrderStatus = OrderStatus.PENDING
    fill_price: float = 0.0
    fill_time: Optional[datetime] = None
    order_id: str = ""


@dataclass
class Position:
    contract: OptionContract
    quantity: int
    avg_entry_price: float
    current_price: float = 0.0
    unrealized_pnl_pct: float = 0.0

    def update_mark(self, current_price: float) -> None:
        self.current_price = current_price
        if self.avg_entry_price > 0:
            self.unrealized_pnl_pct = (
                (current_price - self.avg_entry_price) / self.avg_entry_price
            )


@dataclass
class Signal:
    signal_type: SignalType
    timestamp: datetime
    close_price: float
    peak_prob: float = 0.0
    trough_prob: float = 0.0


class Broker(ABC):
    """Abstract broker interface for option trading."""

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def get_option_chain(
        self,
        symbol: str,
        option_type: str = "put",
        min_expiry_days: int = 7,
    ) -> list[OptionContract]: ...

    @abstractmethod
    def get_option_quote(self, contract: OptionContract) -> OptionQuote: ...

    @abstractmethod
    def submit_order(self, order: Order) -> Order: ...

    @abstractmethod
    def get_positions(self) -> list[Position]: ...

    @abstractmethod
    def get_cash_balance(self) -> float: ...

    @abstractmethod
    def update_underlying_price(
        self, symbol: str, price: float, timestamp: datetime
    ) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...
