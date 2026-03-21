from dataclasses import dataclass, asdict

from src.backtest.strategy.base import BaseStrategy, Action, ActionResult


@dataclass
class CallBuyConfig:
    threshold: float = 0.5
    tp_pct: float = 0.05
    sl_pct: float = -0.10
    force_close_minutes: int = 120
    quantity: int = 1
    option_type: str = "call"
    strike_selection: str = "atm"
    min_holding_minutes: int = 30
    cooldown_minutes: int = 30
    max_trades_per_day: int = 3


class CallBuyStrategy(BaseStrategy):
    """TROUGH -> call buy, PEAK -> call sell (opposite of put strategies)."""

    def __init__(self, config: CallBuyConfig):
        self.config = config
        self._entry_minute: int | None = None
        self._last_sell_minute: int | None = None
        self._trades_today: int = 0

    def name(self) -> str:
        return "call_buy"

    def config_dict(self) -> dict:
        return asdict(self.config)

    def reset(self) -> None:
        self._entry_minute = None
        self._last_sell_minute = None

    def on_day_start(self, date: str) -> None:
        self._trades_today = 0
        self._entry_minute = None
        self._last_sell_minute = None

    def on_bar(self, bar: dict, position, session_minutes: int = 390) -> ActionResult:
        c = self.config
        peak_prob = bar.get("peak_prob", 0.0)
        trough_prob = bar.get("trough_prob", 0.0)
        minutes_from_open = bar.get("minutes_from_open", 0)

        if position is not None:
            # 1. Force close (always takes priority)
            force_close_at = session_minutes - c.force_close_minutes
            if minutes_from_open >= force_close_at:
                self._last_sell_minute = minutes_from_open
                return ActionResult(Action.SELL, "FORCE_CLOSE")

            # 2. Stop loss (always takes priority)
            if position.unrealized_pnl_pct <= c.sl_pct:
                self._last_sell_minute = minutes_from_open
                return ActionResult(Action.SELL, "SL")

            # 3. Take profit (always takes priority)
            if position.unrealized_pnl_pct >= c.tp_pct:
                self._last_sell_minute = minutes_from_open
                return ActionResult(Action.SELL, "TP")

            # 4. Peak signal -> SELL call (opposite of put: peak = exit for call)
            if peak_prob >= c.threshold and peak_prob > trough_prob:
                if self._entry_minute is not None:
                    held = minutes_from_open - self._entry_minute
                    if held < c.min_holding_minutes:
                        return ActionResult(Action.HOLD)
                self._last_sell_minute = minutes_from_open
                return ActionResult(Action.SELL, "PEAK_SIGNAL")

            return ActionResult(Action.HOLD)

        else:
            # 5. Don't buy near close
            force_close_at = session_minutes - c.force_close_minutes
            if minutes_from_open >= force_close_at:
                return ActionResult(Action.HOLD)

            # 6. Max trades per day
            if self._trades_today >= c.max_trades_per_day:
                return ActionResult(Action.HOLD)

            # 7. Cooldown after last sell
            if self._last_sell_minute is not None:
                elapsed = minutes_from_open - self._last_sell_minute
                if elapsed < c.cooldown_minutes:
                    return ActionResult(Action.HOLD)

            # 8. Trough signal -> BUY call (opposite of put: trough = entry for call)
            if trough_prob >= c.threshold and trough_prob > peak_prob:
                self._entry_minute = minutes_from_open
                self._trades_today += 1
                return ActionResult(Action.BUY, "TROUGH_SIGNAL")

            return ActionResult(Action.HOLD)
