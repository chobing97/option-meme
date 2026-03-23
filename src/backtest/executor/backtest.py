"""BacktestExecutor: executes trades against historical options OHLCV data.

Now delegates all data lookups to BacktestMarketData.
"""

from datetime import datetime
from typing import Optional

from config.settings import TRADE_MOCK_CAPITAL, TRADE_MOCK_SLIPPAGE_PCT
from src.backtest.executor.base import Executor, OptionContract, Position
from src.backtest.market_data import BacktestMarketData
from src.backtest.types import Order, OrderResult, PortfolioState, Side


class BacktestExecutor(Executor):
    """Executor that uses BacktestMarketData for price lookups and manages cash/positions."""

    def __init__(
        self,
        market_data: BacktestMarketData,
        capital: float = TRADE_MOCK_CAPITAL,
        slippage_pct: float = TRADE_MOCK_SLIPPAGE_PCT,
    ):
        self._market_data = market_data
        self._initial_capital = capital
        self._slippage_pct = slippage_pct
        self._cash: float = capital
        self._positions: list[Position] = []

    # ── Executor interface ────────────────────────────

    def execute(self, order: Order, timestamp: datetime) -> OrderResult:
        """Execute a trading order. Routes to option or stock execution."""
        if order.instrument_type == "option":
            return self._execute_option(order, timestamp)
        elif order.instrument_type == "stock":
            return self._execute_stock(order, timestamp)
        return OrderResult(order=order, status="REJECTED", reject_reason="UNKNOWN_INSTRUMENT")

    def get_portfolio_state(self) -> PortfolioState:
        """Return snapshot of cash + positions + equity."""
        pos_value = sum(p.quantity * p.current_price * 100 for p in self._positions)
        return PortfolioState(
            cash=self._cash,
            positions=list(self._positions),
            equity=self._cash + pos_value,
        )

    def update_marks(self, timestamp: datetime) -> None:
        """Mark-to-market all positions at given timestamp."""
        for pos in self._positions:
            quote = self._market_data.get_option_quote(pos.contract, timestamp)
            if quote:
                pos.update_mark(quote.close)

    def reset(self) -> None:
        """Reset cash and positions. Market data is kept for grid search efficiency."""
        self._cash = self._initial_capital
        self._positions = []

    # ── Option execution ──────────────────────────────

    def _execute_option(self, order: Order, timestamp: datetime) -> OrderResult:
        if order.side == Side.BUY:
            return self._execute_option_buy(order, timestamp)
        elif order.side == Side.SELL:
            return self._execute_option_sell(order, timestamp)
        return OrderResult(order=order, status="REJECTED", reject_reason="UNKNOWN_SIDE")

    def _execute_option_buy(self, order: Order, timestamp: datetime) -> OrderResult:
        """Buy option: select contract, check liquidity/cash, fill."""
        # Get option chain
        chain = self._market_data.get_option_chain(
            order.symbol, order.option_type, timestamp
        )
        if not chain:
            return OrderResult(order=order, status="REJECTED", reject_reason="NO_CHAIN")

        # Determine underlying price for strike selection
        underlying_price = order.reference_price
        if underlying_price <= 0:
            stock_quote = self._market_data.get_stock_quote(order.symbol, timestamp)
            if stock_quote:
                underlying_price = stock_quote.close
            else:
                # Fallback: cannot determine underlying price
                return OrderResult(
                    order=order, status="REJECTED", reject_reason="NO_UNDERLYING_PRICE"
                )

        # Select contract
        contract = self._select_contract(chain, underlying_price, order.strike_selection)

        # Get quote for selected contract
        quote = self._market_data.get_option_quote(contract, timestamp)
        if quote is None:
            return OrderResult(order=order, status="REJECTED", reject_reason="NO_LIQUIDITY")

        # Liquidity check
        if quote.volume == 0:
            return OrderResult(order=order, status="REJECTED", reject_reason="NO_LIQUIDITY")

        # Calculate fill price
        half_spread = max((quote.high - quote.low) / 2, 0.01)
        ask = quote.close + half_spread
        fill_price = ask * (1 + self._slippage_pct)

        # Cash check (options multiplier x100)
        total_cost = fill_price * order.quantity * 100
        if total_cost > self._cash:
            return OrderResult(
                order=order, status="REJECTED", reject_reason="INSUFFICIENT_CASH"
            )

        # Execute
        self._cash -= total_cost
        self._add_position(contract, order.quantity, fill_price)

        return OrderResult(
            order=order,
            status="FILLED",
            fill_price=fill_price,
            fill_time=timestamp,
            contract=contract,
        )

    def _execute_option_sell(self, order: Order, timestamp: datetime) -> OrderResult:
        """Sell option: find position, check liquidity, fill."""
        # Find existing position for symbol
        pos = self._get_position_for_symbol(order.symbol)
        if pos is None:
            return OrderResult(order=order, status="REJECTED", reject_reason="NO_POSITION")

        # Get quote for the held contract
        quote = self._market_data.get_option_quote(pos.contract, timestamp)
        if quote is None:
            return OrderResult(order=order, status="REJECTED", reject_reason="NO_LIQUIDITY")

        # Liquidity check
        if quote.volume == 0:
            return OrderResult(order=order, status="REJECTED", reject_reason="NO_LIQUIDITY")

        # Calculate fill price
        half_spread = max((quote.high - quote.low) / 2, 0.01)
        bid = quote.close - half_spread
        fill_price = bid * (1 - self._slippage_pct)

        # Execute (options multiplier x100)
        sell_qty = min(order.quantity, pos.quantity)
        proceeds = fill_price * sell_qty * 100
        self._cash += proceeds

        contract = pos.contract
        self._remove_position(pos.contract.contract_id)

        return OrderResult(
            order=order,
            status="FILLED",
            fill_price=fill_price,
            fill_time=timestamp,
            contract=contract,
        )

    # ── Stock execution (placeholder) ─────────────────

    def _execute_stock(self, order: Order, timestamp: datetime) -> OrderResult:
        return OrderResult(order=order, status="REJECTED", reject_reason="STOCK_NOT_SUPPORTED")

    # ── Internal helpers ──────────────────────────────

    def _select_contract(
        self,
        chain: list[OptionContract],
        underlying_price: float,
        strike_selection: str,
    ) -> OptionContract:
        """Select contract from chain based on strike selection method."""
        if strike_selection == "atm":
            return min(chain, key=lambda c: abs(c.strike - underlying_price))
        # Fallback: first contract
        return chain[0]

    def _get_position_for_symbol(self, symbol: str) -> Optional[Position]:
        """Find open position for underlying symbol."""
        for pos in self._positions:
            if pos.contract.symbol == symbol:
                return pos
        return None

    def _add_position(
        self, contract: OptionContract, quantity: int, fill_price: float
    ) -> None:
        """Add or average into existing position."""
        for pos in self._positions:
            if pos.contract.contract_id == contract.contract_id:
                total_qty = pos.quantity + quantity
                pos.avg_entry_price = (
                    pos.avg_entry_price * pos.quantity + fill_price * quantity
                ) / total_qty
                pos.quantity = total_qty
                return

        self._positions.append(
            Position(
                contract=contract,
                quantity=quantity,
                avg_entry_price=fill_price,
                current_price=fill_price,
            )
        )

    def _remove_position(self, contract_id: str) -> None:
        """Remove position entirely (assumes full quantity sell)."""
        self._positions = [
            p for p in self._positions if p.contract.contract_id != contract_id
        ]
