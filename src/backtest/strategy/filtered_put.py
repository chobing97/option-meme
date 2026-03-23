from dataclasses import dataclass, asdict

import pandas as pd

from src.backtest.strategy.base import BaseStrategy
from src.backtest.types import Order, Side, PortfolioState


@dataclass
class FilteredPutConfig:
    threshold: float = 0.5
    tp_pct: float = 0.05
    sl_pct: float = -0.10
    force_close_minutes: int = 120
    quantity: int = 1
    option_type: str = "put"
    strike_selection: str = "atm"
    min_holding_minutes: int = 30
    cooldown_minutes: int = 30
    max_trades_per_day: int = 3
    min_prob_gap: float = 0.2


class FilteredPutStrategy(BaseStrategy):
    """Put-buy strategy with additional filters: min holding, cooldown, max trades, prob gap."""

    def __init__(self, config: FilteredPutConfig):
        self.config = config
        self._entry_minute: int | None = None
        self._last_sell_minute: int | None = None
        self._trades_today: int = 0

    def name(self) -> str:
        return "filtered_put"

    def config_dict(self) -> dict:
        return asdict(self.config)

    def reset(self) -> None:
        self._entry_minute = None
        self._last_sell_minute = None

    def on_day_start(self, date: str) -> None:
        self._trades_today = 0
        self._entry_minute = None
        self._last_sell_minute = None

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
            # 1. Force close (always takes priority)
            force_close_at = session_minutes - c.force_close_minutes
            if minutes_from_open >= force_close_at:
                self._last_sell_minute = minutes_from_open
                return [Order(symbol=symbol, side=Side.SELL, quantity=position.quantity,
                             reason="FORCE_CLOSE", instrument_type="option", option_type=c.option_type)]

            # 2. Stop loss (always takes priority)
            if position.unrealized_pnl_pct <= c.sl_pct:
                self._last_sell_minute = minutes_from_open
                return [Order(symbol=symbol, side=Side.SELL, quantity=position.quantity,
                             reason="SL", instrument_type="option", option_type=c.option_type)]

            # 3. Take profit (always takes priority)
            if position.unrealized_pnl_pct >= c.tp_pct:
                self._last_sell_minute = minutes_from_open
                return [Order(symbol=symbol, side=Side.SELL, quantity=position.quantity,
                             reason="TP", instrument_type="option", option_type=c.option_type)]

            # 4. Trough signal -- but check min_holding first
            if trough_prob >= c.threshold and trough_prob > peak_prob:
                if self._entry_minute is not None:
                    held = minutes_from_open - self._entry_minute
                    if held < c.min_holding_minutes:
                        return []
                self._last_sell_minute = minutes_from_open
                return [Order(symbol=symbol, side=Side.SELL, quantity=position.quantity,
                             reason="TROUGH_SIGNAL", instrument_type="option", option_type=c.option_type)]

            return []

        else:
            # 5. Don't buy near close
            force_close_at = session_minutes - c.force_close_minutes
            if minutes_from_open >= force_close_at:
                return []

            # 6. Max trades per day
            if self._trades_today >= c.max_trades_per_day:
                return []

            # 7. Cooldown after last sell
            if self._last_sell_minute is not None:
                elapsed = minutes_from_open - self._last_sell_minute
                if elapsed < c.cooldown_minutes:
                    return []

            # 8. Prob gap filter
            if peak_prob - trough_prob < c.min_prob_gap:
                return []

            # 9. Peak signal -> BUY
            if peak_prob >= c.threshold:
                self._entry_minute = minutes_from_open
                self._trades_today += 1
                return [Order(symbol=symbol, side=Side.BUY, quantity=c.quantity,
                             reason="PEAK_SIGNAL", instrument_type="option", option_type=c.option_type,
                             strike_selection=c.strike_selection, reference_price=close)]

            return []
