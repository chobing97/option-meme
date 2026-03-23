from dataclasses import dataclass, asdict

import pandas as pd

from src.backtest.strategy.base import BaseStrategy
from src.backtest.types import Order, Side, PortfolioState


@dataclass
class PutBuyConfig:
    threshold: float = 0.3
    tp_pct: float = 0.10
    sl_pct: float = -0.05
    force_close_minutes: int = 120
    quantity: int = 1
    option_type: str = "put"
    strike_selection: str = "atm"


class PutBuyStrategy(BaseStrategy):
    """Original put-buy strategy: PEAK -> buy put, TROUGH -> sell put."""

    def __init__(self, config: PutBuyConfig):
        self.config = config

    def name(self) -> str:
        return "put_buy"

    def config_dict(self) -> dict:
        return asdict(self.config)

    def on_bar(self, row: pd.Series, portfolio: PortfolioState, context: dict) -> list[Order]:
        c = self.config
        symbol = row.get("symbol", "")
        close = row.get("close", 0.0)
        peak_prob = row.get("peak_prob", 0.0)
        trough_prob = row.get("trough_prob", 0.0)
        minutes_from_open = row.get("minutes_from_open", 0)
        session_minutes = context.get("session_minutes", 390)

        position = portfolio.get_position(symbol)

        if position is not None:
            # 1. Force close
            force_close_at = session_minutes - c.force_close_minutes
            if minutes_from_open >= force_close_at:
                return [Order(symbol=symbol, side=Side.SELL, quantity=position.quantity,
                             reason="FORCE_CLOSE", instrument_type="option", option_type=c.option_type)]

            # 2. Stop loss
            if position.unrealized_pnl_pct <= c.sl_pct:
                return [Order(symbol=symbol, side=Side.SELL, quantity=position.quantity,
                             reason="SL", instrument_type="option", option_type=c.option_type)]

            # 3. Take profit
            if position.unrealized_pnl_pct >= c.tp_pct:
                return [Order(symbol=symbol, side=Side.SELL, quantity=position.quantity,
                             reason="TP", instrument_type="option", option_type=c.option_type)]

            # 4. Trough signal
            if trough_prob >= c.threshold and trough_prob > peak_prob:
                return [Order(symbol=symbol, side=Side.SELL, quantity=position.quantity,
                             reason="TROUGH_SIGNAL", instrument_type="option", option_type=c.option_type)]

            return []

        else:
            # Don't buy too close to close
            force_close_at = session_minutes - c.force_close_minutes
            if minutes_from_open >= force_close_at:
                return []

            # Peak signal -> BUY
            if peak_prob >= c.threshold and peak_prob > trough_prob:
                return [Order(symbol=symbol, side=Side.BUY, quantity=c.quantity,
                             reason="PEAK_SIGNAL", instrument_type="option", option_type=c.option_type,
                             strike_selection=c.strike_selection, reference_price=close)]

            return []
