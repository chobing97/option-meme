"""Trading engine: multi-symbol loop + trading rules."""

from datetime import datetime, time

from loguru import logger

from config.settings import (
    KR_MARKET_CLOSE,
    TRADE_FORCE_CLOSE_MINUTES,
    TRADE_PROFIT_TARGET_PCT,
    TRADE_STOP_LOSS_PCT,
    US_MARKET_CLOSE,
)
from src.trading.broker.base import (
    Broker,
    Order,
    OrderSide,
    OrderStatus,
    SignalType,
)
from src.trading.datafeed.base import DataFeed
from src.trading.notifier.base import Notifier, TradeEvent
from src.trading.signal_detector import BarAccumulator, SignalDetector
from src.trading.trade_tracker import TradeTracker


class TradingEngine:
    """Multi-symbol trading loop: poll bars -> detect signals -> execute trades.

    Rules (priority order):
    1. Force close: 장마감 N분 전 → close all puts
    2. Stop loss: unrealized PnL <= -5% → close all puts
    3. Profit target: unrealized PnL >= +10% → close all puts
    4. TROUGH + has put → SELL all puts
    5. PEAK + no put + sufficient cash → BUY ATM put
    """

    def __init__(
        self,
        feeds: dict[str, DataFeed],
        broker: Broker,
        detector: SignalDetector,
        symbols: list[str],
        quantity: int = 1,
        notifiers: list[Notifier] | None = None,
        tracker: TradeTracker | None = None,
    ):
        self.feeds = feeds
        self.broker = broker
        self.detector = detector
        self.symbols = symbols
        self.quantity = quantity
        self.notifiers = notifiers or []
        self.tracker = tracker

        # Per-symbol tracking
        self._buy_orders: dict[str, list[Order]] = {s: [] for s in symbols}
        self._sell_orders: dict[str, list[Order]] = {s: [] for s in symbols}
        self._signals_summary: dict[str, dict] = {
            s: {"peaks": 0, "troughs": 0} for s in symbols
        }

    def run(self) -> dict:
        """Run the trading session for all symbols. Returns summary dict."""
        # Connect all feeds + broker
        for symbol in self.symbols:
            self.feeds[symbol].connect()
        self.broker.connect()

        # Print header
        model_name = "LightGBM" if self.detector.model_type == "gbm" else "LSTM"
        market = getattr(self.feeds[self.symbols[0]], "market", "?")

        print(f"\n=== Trading: {market} / {', '.join(self.symbols)} ===")
        print(
            f"Model: {model_name} | Threshold: {self.detector.threshold} "
            f"| Qty: {self.quantity} | Mode: Mock"
        )
        print(f"Cash: {self.broker.get_cash_balance():,.0f}")
        print()

        # Initialize accumulators per symbol
        accumulators: dict[str, BarAccumulator] = {}
        for symbol in self.symbols:
            history = self.feeds[symbol].get_history(n_days=5)
            accumulators[symbol] = BarAccumulator(history)

        # Determine session date and force close time
        session_date = None
        force_close_time = self._compute_force_close_time(market)

        # Main loop: iterate while any feed is active
        bar_nums: dict[str, int] = {s: 0 for s in self.symbols}

        while any(self.feeds[s].is_session_active() for s in self.symbols):
            for symbol in self.symbols:
                feed = self.feeds[symbol]
                if not feed.is_session_active():
                    continue

                bar = feed.get_latest_bar()
                if bar is None:
                    continue

                bar_nums[symbol] += 1
                accumulators[symbol].add_bar(bar)

                close_price = float(bar["close"])
                dt = bar["datetime"]
                self.broker.update_underlying_price(symbol, close_price, dt)

                if session_date is None:
                    session_date = dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)[:10]

                # Detect signal
                signal = self.detector.detect(accumulators[symbol])

                # Print bar line
                time_str = dt.strftime("%H:%M") if hasattr(dt, "strftime") else str(dt)
                sig_label = signal.signal_type.value
                print(
                    f"[{symbol}] Bar {bar_nums[symbol]:>3}: {time_str}  "
                    f"close={close_price:,.0f}  "
                    f"peak={signal.peak_prob:.2f}  "
                    f"trough={signal.trough_prob:.2f}  "
                    f"signal={sig_label}"
                )

                if signal.signal_type == SignalType.PEAK:
                    self._signals_summary[symbol]["peaks"] += 1
                elif signal.signal_type == SignalType.TROUGH:
                    self._signals_summary[symbol]["troughs"] += 1

                # === Trading rules (priority order) ===
                action = ""
                reason = ""
                trade_strike = 0.0
                trade_fill_price = 0.0

                # 1. Force close check
                if force_close_time and self._is_force_close_time(dt, force_close_time):
                    if self._has_put(symbol):
                        self._close_all_puts(symbol, "FORCE_CLOSE", session_date, market)
                        action, reason = "SELL_PUT", "FORCE_CLOSE"
                    self._record_bar_snapshot(
                        symbol, bar_nums[symbol], close_price, dt, signal,
                        action, reason, trade_strike, trade_fill_price,
                    )
                    continue

                # 2. Stop loss check
                if self._check_stop_loss(symbol):
                    self._close_all_puts(symbol, "STOP_LOSS", session_date, market)
                    action, reason = "SELL_PUT", "STOP_LOSS"
                    self._record_bar_snapshot(
                        symbol, bar_nums[symbol], close_price, dt, signal,
                        action, reason, trade_strike, trade_fill_price,
                    )
                    continue

                # 3. Profit target check
                if self._check_profit_target(symbol):
                    self._close_all_puts(symbol, "PROFIT_TARGET", session_date, market)
                    action, reason = "SELL_PUT", "PROFIT_TARGET"
                    self._record_bar_snapshot(
                        symbol, bar_nums[symbol], close_price, dt, signal,
                        action, reason, trade_strike, trade_fill_price,
                    )
                    continue

                # 4. Trough signal -> sell
                if signal.signal_type == SignalType.TROUGH and self._has_put(symbol):
                    self._close_all_puts(symbol, "TROUGH_SIGNAL", session_date, market)
                    action, reason = "SELL_PUT", "TROUGH_SIGNAL"

                # 5. Peak signal -> buy
                elif (
                    signal.signal_type == SignalType.PEAK
                    and not self._has_put(symbol)
                ):
                    filled = self._buy_atm_put(symbol, signal.close_price, session_date, market)
                    if filled and filled.status == OrderStatus.FILLED:
                        action = "BUY_PUT"
                        reason = "PEAK_SIGNAL"
                        trade_strike = filled.contract.strike
                        trade_fill_price = filled.fill_price

                self._record_bar_snapshot(
                    symbol, bar_nums[symbol], close_price, dt, signal,
                    action, reason, trade_strike, trade_fill_price,
                )

        # End of session
        for symbol in self.symbols:
            self.feeds[symbol].disconnect()
        self.broker.disconnect()

        return self._finalize(session_date or "", market)

    # ── Position helpers ─────────────────────────────

    def _has_put(self, symbol: str) -> bool:
        return any(
            p.contract.option_type == "put" and p.contract.symbol == symbol
            for p in self.broker.get_positions()
        )

    def _get_puts(self, symbol: str) -> list:
        return [
            p for p in self.broker.get_positions()
            if p.contract.option_type == "put" and p.contract.symbol == symbol
        ]

    # ── Rule checks ──────────────────────────────────

    def _check_stop_loss(self, symbol: str) -> bool:
        for pos in self._get_puts(symbol):
            if pos.unrealized_pnl_pct <= TRADE_STOP_LOSS_PCT:
                return True
        return False

    def _check_profit_target(self, symbol: str) -> bool:
        for pos in self._get_puts(symbol):
            if pos.unrealized_pnl_pct >= TRADE_PROFIT_TARGET_PCT:
                return True
        return False

    @staticmethod
    def _compute_force_close_time(market: str) -> time | None:
        """Compute the time after which force close should trigger."""
        if market == "kr":
            close_str = KR_MARKET_CLOSE
        elif market == "us":
            close_str = US_MARKET_CLOSE
        else:
            return None
        h, m = map(int, close_str.split(":"))
        total_min = h * 60 + m - TRADE_FORCE_CLOSE_MINUTES
        return time(total_min // 60, total_min % 60)

    @staticmethod
    def _is_force_close_time(dt: datetime, force_time: time) -> bool:
        bar_time = dt.time() if hasattr(dt, "time") else None
        if bar_time is None:
            return False
        return bar_time >= force_time

    # ── Trade execution ──────────────────────────────

    def _buy_atm_put(
        self, symbol: str, close_price: float,
        session_date: str | None, market: str,
    ) -> Order | None:
        chain = self.broker.get_option_chain(symbol, "put")
        if not chain:
            logger.warning(f"[{symbol}] No option chain available")
            return None

        atm = min(chain, key=lambda c: abs(c.strike - close_price))

        order = Order(
            side=OrderSide.BUY,
            contract=atm,
            quantity=self.quantity,
        )
        filled = self.broker.submit_order(order)

        if filled.status == OrderStatus.REJECTED:
            print(f"  [{symbol}] BUY REJECTED: insufficient cash")
            return filled

        self._buy_orders[symbol].append(filled)

        # Notify
        exp_str = atm.expiry.strftime("%Y-%m-%d")
        for n in self.notifiers:
            n.notify(TradeEvent(
                event_type="BUY",
                market=market,
                symbol=symbol,
                timestamp=filled.fill_time,
                details={
                    "strike": atm.strike,
                    "expiry": exp_str,
                    "fill_price": filled.fill_price,
                    "quantity": filled.quantity,
                },
            ))

        return filled

    def _close_all_puts(
        self, symbol: str, reason: str,
        session_date: str | None, market: str,
    ) -> None:
        for pos in self._get_puts(symbol):
            order = Order(
                side=OrderSide.SELL,
                contract=pos.contract,
                quantity=pos.quantity,
            )
            filled = self.broker.submit_order(order)
            self._sell_orders[symbol].append(filled)

            pnl_pct = (
                (filled.fill_price - pos.avg_entry_price) / pos.avg_entry_price
                if pos.avg_entry_price > 0 else 0
            )

            # Notify
            for n in self.notifiers:
                n.notify(TradeEvent(
                    event_type="SELL",
                    market=market,
                    symbol=symbol,
                    timestamp=filled.fill_time,
                    details={
                        "strike": pos.contract.strike,
                        "fill_price": filled.fill_price,
                        "pnl_pct": pnl_pct,
                        "reason": reason,
                        "quantity": filled.quantity,
                    },
                ))

    # ── Tracker ──────────────────────────────────────

    def _record_bar_snapshot(
        self,
        symbol: str,
        bar_num: int,
        close_price: float,
        dt: datetime,
        signal,
        action: str,
        reason: str,
        trade_strike: float,
        trade_fill_price: float,
    ) -> None:
        """Record a bar snapshot to the tracker if available."""
        if not self.tracker:
            return

        puts = self._get_puts(symbol)
        pos_qty = sum(p.quantity for p in puts)
        pos_avg_entry = puts[0].avg_entry_price if puts else 0.0
        pos_mark_price = puts[0].current_price if puts else 0.0

        # For SELL actions, position is already closed — use last sell order info
        if action == "SELL_PUT" and not puts and self._sell_orders[symbol]:
            last_sell = self._sell_orders[symbol][-1]
            trade_strike = last_sell.contract.strike
            trade_fill_price = last_sell.fill_price

        self.tracker.record_bar(
            timestamp=dt,
            symbol=symbol,
            bar_num=bar_num,
            underlying_close=close_price,
            signal=signal.signal_type.value,
            peak_prob=signal.peak_prob,
            trough_prob=signal.trough_prob,
            action=action,
            reason=reason,
            strike=trade_strike,
            fill_price=trade_fill_price,
            position_qty=pos_qty,
            position_avg_entry=pos_avg_entry,
            position_mark_price=pos_mark_price,
            cash=self.broker.get_cash_balance(),
        )

    # ── Summary ──────────────────────────────────────

    def _finalize(self, session_date: str, market: str) -> dict:
        """Print per-symbol summaries, record to DB, return combined result."""
        combined = {}

        for symbol in self.symbols:
            buys = self._buy_orders[symbol]
            sells = self._sell_orders[symbol]

            net_pnl = 0.0
            for sell in sells:
                for buy in buys:
                    if buy.contract.contract_id == sell.contract.contract_id:
                        net_pnl += (sell.fill_price - buy.fill_price) * sell.quantity
                        break

            total_cost = sum(o.fill_price * o.quantity for o in buys) if buys else 0

            summary = {
                "buys": len(buys),
                "sells": len(sells),
                "net_pnl": net_pnl,
                "total_cost": total_cost,
                "peak_signals": self._signals_summary[symbol]["peaks"],
                "trough_signals": self._signals_summary[symbol]["troughs"],
                "cash_balance": self.broker.get_cash_balance(),
            }

            # Notify session end
            for n in self.notifiers:
                n.notify(TradeEvent(
                    event_type="SESSION_END",
                    market=market,
                    symbol=symbol,
                    timestamp=datetime.now(),
                    details=summary,
                ))

            combined[symbol] = summary

        return combined
