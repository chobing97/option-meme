"""Historical Broker: backtest option trades using real OHLCV data."""

import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from config.settings import (
    RAW_OPTIONS_DIR,
    TRADE_MOCK_CAPITAL,
    TRADE_MOCK_SLIPPAGE_PCT,
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


class HistoricalBroker(Broker):
    """Broker that uses real historical options OHLCV for backtesting.

    - Loads contracts.parquet for option chain lookup
    - Loads {year}.parquet for real 1-min OHLCV prices
    - Matches option quotes by timestamp (asof join)
    - Rejects orders when volume == 0 (no liquidity)
    - Marks positions to market using real option prices
    """

    def __init__(
        self,
        market: str = "us",
        capital: float = TRADE_MOCK_CAPITAL,
        slippage_pct: float = TRADE_MOCK_SLIPPAGE_PCT,
        data_dir: Path | None = None,
    ):
        self._market = market
        self._capital = capital
        self._slippage_pct = slippage_pct
        self._data_dir = data_dir or RAW_OPTIONS_DIR

        self._positions: list[Position] = []
        self._orders: list[Order] = []
        self._cash: float = capital
        self._underlying_prices: dict[str, float] = {}
        self._current_time: datetime = datetime.now()
        self._connected: bool = False

        # Per-symbol data loaded on connect
        self._contracts: dict[str, pd.DataFrame] = {}  # symbol -> contracts df
        self._ohlcv: dict[str, pd.DataFrame] = {}  # symbol -> ohlcv df (indexed by datetime+option_symbol)

    def connect(self) -> None:
        """Mark broker as connected. Call load_symbols() first to load data."""
        if not self._connected:
            self._connected = True
            logger.info(
                f"HistoricalBroker connected (capital={self._cash:,.0f}, "
                f"symbols={list(self._contracts.keys())})"
            )

    def load_symbols(self, symbols: list[str]) -> None:
        """Load option data for given underlying symbols."""
        for symbol in symbols:
            self._load_symbol_data(symbol)

    def _load_symbol_data(self, symbol: str) -> None:
        """Load contracts + OHLCV parquets for a single underlying."""
        base_dir = self._data_dir / self._market / symbol
        if not base_dir.exists():
            logger.warning(f"No options data for {symbol} at {base_dir}")
            return

        # Load contracts
        contracts_path = base_dir / "contracts.parquet"
        if contracts_path.exists():
            df = pd.read_parquet(contracts_path)
            df["expiry"] = pd.to_datetime(df["expiry"])
            df["period_start"] = pd.to_datetime(df["period_start"])
            self._contracts[symbol] = df
            logger.debug(f"Loaded {len(df)} contracts for {symbol}")

        # Load all year OHLCV files and concat
        ohlcv_parts = []
        for p in sorted(base_dir.glob("*.parquet")):
            if p.name == "contracts.parquet":
                continue
            ohlcv_parts.append(pd.read_parquet(p))

        if ohlcv_parts:
            ohlcv = pd.concat(ohlcv_parts, ignore_index=True)
            ohlcv["datetime"] = pd.to_datetime(ohlcv["datetime"])
            ohlcv = ohlcv.sort_values(["symbol", "datetime"]).reset_index(drop=True)
            self._ohlcv[symbol] = ohlcv
            logger.debug(
                f"Loaded {len(ohlcv)} option bars for {symbol} "
                f"({ohlcv['symbol'].nunique()} contracts)"
            )

    def update_underlying_price(
        self, symbol: str, price: float, timestamp: datetime
    ) -> None:
        self._underlying_prices[symbol] = price
        # Strip timezone to match tz-naive OHLCV data
        if hasattr(timestamp, "tzinfo") and timestamp.tzinfo is not None:
            timestamp = timestamp.replace(tzinfo=None)
        self._current_time = timestamp
        self._mark_positions(symbol)

    def get_option_chain(
        self,
        symbol: str,
        option_type: str = "put",
        min_expiry_days: int = 7,
    ) -> list[OptionContract]:
        """Get real option contracts active at current timestamp."""
        if symbol not in self._contracts:
            return []

        contracts_df = self._contracts[symbol]
        current_date = self._current_time.date() if hasattr(self._current_time, "date") else self._current_time

        # Find contracts where period_start <= current_date < expiry
        mask = (
            (contracts_df["period_start"].dt.date <= current_date)
            & (contracts_df["expiry"].dt.date > current_date)
        )
        active = contracts_df[mask]

        if active.empty:
            # Fallback: find the contract whose period_start is closest to current_date
            logger.debug(f"[{symbol}] No active contracts for {current_date}, trying closest")
            return []

        result = []
        for _, row in active.iterrows():
            cp = row["cp"].upper()
            if option_type == "put" and cp != "P":
                continue
            if option_type == "call" and cp != "C":
                continue

            result.append(
                OptionContract(
                    symbol=symbol,
                    expiry=row["expiry"].to_pydatetime(),
                    strike=float(row["strike"]),
                    option_type=option_type,
                    contract_id=row["symbol"].strip(),
                )
            )

        return result

    def get_option_quote(self, contract: OptionContract) -> OptionQuote:
        """Get real option quote from OHLCV data at current timestamp."""
        underlying = contract.symbol
        option_symbol = contract.contract_id

        ohlcv = self._ohlcv.get(underlying)
        if ohlcv is None:
            return self._empty_quote(contract)

        # Filter to this option contract
        contract_bars = ohlcv[ohlcv["symbol"].str.strip() == option_symbol]
        if contract_bars.empty:
            return self._empty_quote(contract)

        # Asof match: find the latest bar at or before current_time
        mask = contract_bars["datetime"] <= self._current_time
        matched = contract_bars[mask]

        if matched.empty:
            return self._empty_quote(contract)

        bar = matched.iloc[-1]
        mid = float(bar["close"])
        half_spread = max(float(bar["high"] - bar["low"]) / 2, 0.01)

        return OptionQuote(
            contract=contract,
            bid=max(mid - half_spread, 0.01),
            ask=mid + half_spread,
            last=mid,
            volume=int(bar["volume"]),
            timestamp=bar["datetime"].to_pydatetime() if hasattr(bar["datetime"], "to_pydatetime") else bar["datetime"],
        )

    def submit_order(self, order: Order) -> Order:
        """Execute order using real option prices. Reject if no liquidity."""
        order.order_id = str(uuid.uuid4())[:8]
        quote = self.get_option_quote(order.contract)

        # Liquidity check: reject if volume == 0
        if quote.volume == 0:
            order.status = OrderStatus.REJECTED
            order.fill_time = self._current_time
            logger.warning(
                f"Order REJECTED: no liquidity for {order.contract.contract_id} "
                f"at {self._current_time}"
            )
            self._orders.append(order)
            return order

        if order.side == OrderSide.BUY:
            fill_price = quote.ask * (1 + self._slippage_pct)
            total_cost = fill_price * order.quantity
            if total_cost > self._cash:
                order.status = OrderStatus.REJECTED
                order.fill_time = self._current_time
                logger.warning(
                    f"Order REJECTED: cost={total_cost:,.2f} > cash={self._cash:,.2f}"
                )
                self._orders.append(order)
                return order
            order.fill_price = fill_price
            self._cash -= total_cost
            self._add_position(order)
        else:
            fill_price = quote.bid * (1 - self._slippage_pct)
            order.fill_price = fill_price
            proceeds = fill_price * order.quantity
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
        self._positions = [
            p for p in self._positions
            if p.contract.contract_id != order.contract.contract_id
        ]

    def _mark_positions(self, symbol: str) -> None:
        """Mark positions to market using real option OHLCV prices."""
        for pos in self._positions:
            if pos.contract.symbol != symbol:
                continue
            quote = self.get_option_quote(pos.contract)
            if quote.volume > 0:
                pos.update_mark(quote.last)
            # If no quote available, keep last known price

    def _empty_quote(self, contract: OptionContract) -> OptionQuote:
        """Return a zero-volume quote when no data is available."""
        return OptionQuote(
            contract=contract,
            bid=0.0,
            ask=0.0,
            last=0.0,
            volume=0,
            timestamp=self._current_time,
        )
