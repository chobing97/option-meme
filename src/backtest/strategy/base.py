from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class Action(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class ActionResult:
    action: Action
    reason: str = ""   # PEAK_SIGNAL / TROUGH_SIGNAL / TP / SL / FORCE_CLOSE / ""


class BaseStrategy(ABC):
    @abstractmethod
    def on_bar(self, bar: dict, position, session_minutes: int = 390) -> ActionResult:
        ...

    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def config_dict(self) -> dict:
        """Return strategy config as dict for result storage."""
        ...

    def reset(self) -> None:
        """Reset internal state (entry time, cooldown, etc). Called on day boundary."""
        pass

    def on_day_start(self, date: str) -> None:
        """Called at start of each trading day. Reset daily counters."""
        pass
