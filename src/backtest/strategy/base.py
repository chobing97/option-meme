from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pandas as pd

from src.backtest.types import Order, Side, PortfolioState

if TYPE_CHECKING:
    from src.backtest.market_data import MarketData


class BaseStrategy(ABC):
    """Abstract strategy interface. Returns list[Order] for each bar."""

    _market_data: "MarketData | None" = None

    def set_market_data(self, market_data: "MarketData") -> None:
        """Inject MarketData reference. Called by Engine before run()."""
        self._market_data = market_data

    @abstractmethod
    def on_bar(self, row: pd.Series, portfolio: PortfolioState, context: dict) -> list[Order]:
        """Evaluate bar and return orders. Empty list = HOLD."""
        ...

    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def config_dict(self) -> dict:
        """Return strategy config as dict for result storage."""
        ...

    def on_day_end(self, portfolio: PortfolioState, context: dict) -> list[Order]:
        """Called on day boundary. Default: close all positions."""
        return [Order(symbol=p.contract.symbol, side=Side.SELL, quantity=p.quantity,
                      reason="FORCE_CLOSE", instrument_type="option", option_type=p.contract.option_type)
                for p in portfolio.positions]

    def on_data_end(self, portfolio: PortfolioState, context: dict) -> list[Order]:
        """Called when data ends. Default: close all positions."""
        return self.on_day_end(portfolio, context)

    def reset(self) -> None:
        """Reset internal state. Called on day boundary."""
        pass

    def on_day_start(self, date: str) -> None:
        """Called at start of each trading day."""
        pass
