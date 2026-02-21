"""Notifier abstract base class and trade event model."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TradeEvent:
    """Represents a trading event for notification."""

    event_type: str  # "BUY", "SELL", "SESSION_END"
    market: str
    symbol: str
    timestamp: datetime
    details: dict = field(default_factory=dict)


class Notifier(ABC):
    """Abstract notifier interface."""

    @abstractmethod
    def notify(self, event: TradeEvent) -> None: ...
