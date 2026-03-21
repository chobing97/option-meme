from dataclasses import dataclass, asdict

from src.backtest.strategy.base import BaseStrategy, Action, ActionResult


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

    def on_bar(self, bar: dict, position, session_minutes: int = 390) -> ActionResult:
        """
        Determine action for current bar.

        Args:
            bar: dict with keys: close, peak_prob, trough_prob, minutes_from_open
            position: Position object or None (from executor.base)
            session_minutes: total session length in minutes (for force close calc)

        Returns:
            ActionResult with action and reason.

        Priority when position exists:
            1. Force close (minutes_from_open >= session_minutes - force_close_minutes)
            2. Stop loss (unrealized_pnl_pct <= sl_pct)
            3. Take profit (unrealized_pnl_pct >= tp_pct)
            4. Trough signal (trough_prob >= threshold and trough_prob > peak_prob)
            5. Hold

        When no position:
            6. Peak signal (peak_prob >= threshold and peak_prob > trough_prob)
            7. Hold
        """
        c = self.config
        peak_prob = bar.get("peak_prob", 0.0)
        trough_prob = bar.get("trough_prob", 0.0)
        minutes_from_open = bar.get("minutes_from_open", 0)

        if position is not None:
            # 1. Force close
            force_close_at = session_minutes - c.force_close_minutes
            if minutes_from_open >= force_close_at:
                return ActionResult(Action.SELL, "FORCE_CLOSE")

            # 2. Stop loss
            if position.unrealized_pnl_pct <= c.sl_pct:
                return ActionResult(Action.SELL, "SL")

            # 3. Take profit
            if position.unrealized_pnl_pct >= c.tp_pct:
                return ActionResult(Action.SELL, "TP")

            # 4. Trough signal
            if trough_prob >= c.threshold and trough_prob > peak_prob:
                return ActionResult(Action.SELL, "TROUGH_SIGNAL")

            return ActionResult(Action.HOLD)

        else:
            # Don't buy too close to close
            force_close_at = session_minutes - c.force_close_minutes
            if minutes_from_open >= force_close_at:
                return ActionResult(Action.HOLD)

            # 6. Peak signal
            if peak_prob >= c.threshold and peak_prob > trough_prob:
                return ActionResult(Action.BUY, "PEAK_SIGNAL")

            return ActionResult(Action.HOLD)
