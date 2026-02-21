"""Mock Broker: in-memory option trading with synthetic option chains."""

import uuid
from datetime import datetime, timedelta

from loguru import logger

from config.settings import (
    TRADE_MOCK_CAPITAL,
    TRADE_MOCK_RISK_FREE,
    TRADE_MOCK_SLIPPAGE_PCT,
    TRADE_MOCK_VOLATILITY,
    TRADE_MIN_EXPIRY_DAYS,
)
from src.trading.broker.base import (
    Broker,
    OptionContract,
    OptionQuote,
    Order,
    OrderSide,
    OrderStatus,
    Position,
)
from src.trading.option_pricer import price_put


class MockBroker(Broker):
    """In-memory mock broker for backtesting option trades.

    - Generates synthetic option chains based on current underlying price
    - Executes orders immediately at ask + slippage (buy) or bid - slippage (sell)
    - Tracks positions and marks to market using Black-Scholes
    - Manages cash balance: deduct on buy, recover on sell
    """

    def __init__(self, capital: float = TRADE_MOCK_CAPITAL):
        self._positions: list[Position] = []
        self._orders: list[Order] = []
        self._cash: float = capital
        self._underlying_prices: dict[str, float] = {}
        self._current_time: datetime = datetime.now()
        self._connected: bool = False

    def connect(self) -> None:
        self._connected = True
        logger.info(f"MockBroker connected (capital={self._cash:,.0f})")

    def update_underlying_price(
        self, symbol: str, price: float, timestamp: datetime
    ) -> None:
        """Update underlying price for a symbol and re-mark its positions."""
        self._underlying_prices[symbol] = price
        self._current_time = timestamp
        self._mark_positions(symbol)

    def get_option_chain(
        self,
        symbol: str,
        option_type: str = "put",
        min_expiry_days: int = TRADE_MIN_EXPIRY_DAYS,
    ) -> list[OptionContract]:
        """Generate synthetic option contracts around current price.

        Creates contracts at ATM and +/- 1%, 2%, 3% strikes
        with the nearest Friday expiry >= min_expiry_days.
        """
        spot = self._underlying_prices.get(symbol, 0.0)
        if spot <= 0:
            return []

        expiry = self._next_friday(min_expiry_days)
        tick = self._strike_tick(spot)
        atm_strike = round(spot / tick) * tick

        strikes = []
        for pct in [-0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03]:
            s = round((atm_strike * (1 + pct)) / tick) * tick
            if s > 0 and s not in strikes:
                strikes.append(s)

        contracts = []
        for strike in sorted(strikes):
            contracts.append(
                OptionContract(
                    symbol=symbol,
                    expiry=expiry,
                    strike=strike,
                    option_type=option_type,
                )
            )

        return contracts

    def get_option_quote(self, contract: OptionContract) -> OptionQuote:
        """Price an option contract using Black-Scholes."""
        spot = self._underlying_prices.get(contract.symbol, 0.0)
        days_to_expiry = max(
            (contract.expiry - self._current_time).total_seconds() / 86400, 0
        )

        mid = price_put(
            spot=spot,
            strike=contract.strike,
            days_to_expiry=days_to_expiry,
            vol=TRADE_MOCK_VOLATILITY,
            r=TRADE_MOCK_RISK_FREE,
        )

        half_spread = max(mid * 0.01, 1.0)
        bid = max(mid - half_spread, 0.0)
        ask = mid + half_spread

        return OptionQuote(
            contract=contract,
            bid=bid,
            ask=ask,
            last=mid,
            volume=100,
            timestamp=self._current_time,
        )

    def submit_order(self, order: Order) -> Order:
        """Execute order immediately (market order semantics).

        BUY: deduct cost from cash. REJECTED if insufficient.
        SELL: recover proceeds to cash.
        """
        order.order_id = str(uuid.uuid4())[:8]
        quote = self.get_option_quote(order.contract)

        if order.side == OrderSide.BUY:
            fill_price = quote.ask * (1 + TRADE_MOCK_SLIPPAGE_PCT)
            total_cost = fill_price * order.quantity
            if total_cost > self._cash:
                order.status = OrderStatus.REJECTED
                order.fill_time = self._current_time
                logger.warning(
                    f"Order REJECTED: cost={total_cost:,.0f} > cash={self._cash:,.0f}"
                )
                self._orders.append(order)
                return order
            order.fill_price = fill_price
            self._cash -= total_cost
            self._add_position(order)
        else:
            order.fill_price = quote.bid * (1 - TRADE_MOCK_SLIPPAGE_PCT)
            proceeds = order.fill_price * order.quantity
            self._cash += proceeds
            self._remove_position(order)

        order.status = OrderStatus.FILLED
        order.fill_time = self._current_time
        self._orders.append(order)
        return order

    def get_positions(self) -> list[Position]:
        return list(self._positions)

    def get_cash_balance(self) -> float:
        return self._cash

    def get_orders(self) -> list[Order]:
        return list(self._orders)

    def disconnect(self) -> None:
        self._connected = False

    # ── Internal ─────────────────────────────────────

    def _add_position(self, order: Order) -> None:
        """Add or increase a position from a BUY order."""
        for pos in self._positions:
            if pos.contract.contract_id == order.contract.contract_id:
                total_qty = pos.quantity + order.quantity
                pos.avg_entry_price = (
                    (pos.avg_entry_price * pos.quantity + order.fill_price * order.quantity)
                    / total_qty
                )
                pos.quantity = total_qty
                return

        self._positions.append(
            Position(
                contract=order.contract,
                quantity=order.quantity,
                avg_entry_price=order.fill_price,
                current_price=order.fill_price,
            )
        )

    def _remove_position(self, order: Order) -> None:
        """Remove or reduce a position from a SELL order."""
        self._positions = [
            p for p in self._positions
            if p.contract.contract_id != order.contract.contract_id
        ]

    def _mark_positions(self, symbol: str) -> None:
        """Mark positions for a specific symbol to market."""
        spot = self._underlying_prices.get(symbol, 0.0)
        for pos in self._positions:
            if pos.contract.symbol != symbol:
                continue
            days_to_expiry = max(
                (pos.contract.expiry - self._current_time).total_seconds() / 86400, 0
            )
            current_price = price_put(
                spot=spot,
                strike=pos.contract.strike,
                days_to_expiry=days_to_expiry,
                vol=TRADE_MOCK_VOLATILITY,
                r=TRADE_MOCK_RISK_FREE,
            )
            pos.update_mark(current_price)

    def _next_friday(self, min_days: int) -> datetime:
        """Find the next Friday that is at least min_days away."""
        base = self._current_time + timedelta(days=min_days)
        days_ahead = 4 - base.weekday()  # Friday = 4
        if days_ahead < 0:
            days_ahead += 7
        friday = base + timedelta(days=days_ahead)
        return friday.replace(hour=15, minute=30, second=0, microsecond=0)

    @staticmethod
    def _strike_tick(price: float) -> float:
        """Determine strike price tick size based on price level."""
        if price >= 100000:
            return 2500
        elif price >= 50000:
            return 1000
        elif price >= 10000:
            return 500
        elif price >= 1000:
            return 100
        elif price >= 100:
            return 5
        else:
            return 1
