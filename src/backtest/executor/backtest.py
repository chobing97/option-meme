"""BacktestExecutor: executes trades against historical options OHLCV data."""

from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from config.settings import RAW_OPTIONS_DIR, TRADE_MOCK_CAPITAL, TRADE_MOCK_SLIPPAGE_PCT
from src.backtest.executor.base import Executor, FillResult, OptionContract, Position


class BacktestExecutor(Executor):
    """Executor that replays historical option OHLCV for backtesting.

    Data layout per symbol:
        data/raw/options/us/{SYMBOL}/contracts.parquet
        data/raw/options/us/{SYMBOL}/{year}.parquet
    """

    def __init__(
        self,
        symbols: list[str],
        market: str = "us",
        capital: float = TRADE_MOCK_CAPITAL,
        slippage_pct: float = TRADE_MOCK_SLIPPAGE_PCT,
        data_dir: Path | None = None,
    ):
        self._symbols = symbols
        self._market = market
        self._initial_capital = capital
        self._slippage_pct = slippage_pct
        self._data_dir = data_dir or RAW_OPTIONS_DIR

        # State
        self._cash: float = capital
        self._positions: list[Position] = []

        # Loaded data (keyed by underlying symbol)
        self._contracts: dict[str, pd.DataFrame] = {}
        self._ohlcv: dict[str, pd.DataFrame] = {}
        # Per-contract indexed OHLCV for fast asof lookup
        self._contract_ohlcv: dict[str, pd.DataFrame] = {}

    # ── Data loading ──────────────────────────────────

    def load_data(self) -> None:
        """Load contracts.parquet and OHLCV parquets for all symbols."""
        for symbol in self._symbols:
            self._load_symbol(symbol)

    def _load_symbol(self, symbol: str) -> None:
        base_dir = self._data_dir / self._market / symbol
        if not base_dir.exists():
            logger.warning(f"No options data for {symbol} at {base_dir}")
            return

        # Contracts
        contracts_path = base_dir / "contracts.parquet"
        if contracts_path.exists():
            df = pd.read_parquet(contracts_path)
            df["expiry"] = pd.to_datetime(df["expiry"])
            df["period_start"] = pd.to_datetime(df["period_start"])
            self._contracts[symbol] = df
            logger.debug(f"Loaded {len(df)} contracts for {symbol}")
        else:
            logger.warning(f"No contracts.parquet for {symbol}")
            return

        # OHLCV (all year parquets)
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
            # Build per-contract indexed DataFrames for fast asof lookup
            for cid, grp in ohlcv.groupby("symbol"):
                cid_stripped = cid.strip() if isinstance(cid, str) else str(cid).strip()
                self._contract_ohlcv[cid_stripped] = (
                    grp.sort_values("datetime").reset_index(drop=True)
                )
            logger.debug(
                f"Loaded {len(ohlcv)} option bars for {symbol} "
                f"({ohlcv['symbol'].nunique()} contracts)"
            )

    # ── Executor interface ────────────────────────────

    def get_option_chain(
        self, symbol: str, option_type: str, timestamp: datetime
    ) -> list[OptionContract]:
        """Return contracts where period_start <= timestamp.date() < expiry."""
        if symbol not in self._contracts:
            return []

        contracts_df = self._contracts[symbol]
        ts = timestamp
        if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
            ts = ts.replace(tzinfo=None)
        current_date = ts.date() if hasattr(ts, "date") else ts

        mask = (
            (contracts_df["period_start"].dt.date <= current_date)
            & (contracts_df["expiry"].dt.date > current_date)
        )
        active = contracts_df[mask]

        if active.empty:
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

    def execute_buy(
        self, contract: OptionContract, quantity: int, timestamp: datetime
    ) -> FillResult:
        """Buy: ask * (1 + slippage). Reject if no volume or insufficient cash."""
        quote = self._get_quote(contract.contract_id, timestamp)
        if quote is None:
            return FillResult(
                status="REJECTED",
                reject_reason="NO_LIQUIDITY",
            )

        # Liquidity check
        if quote["volume"] == 0:
            return FillResult(
                status="REJECTED",
                reject_reason="NO_LIQUIDITY",
            )

        # Calculate fill price
        half_spread = max((quote["high"] - quote["low"]) / 2, 0.01)
        ask = quote["close"] + half_spread
        fill_price = ask * (1 + self._slippage_pct)

        # Cash check (options multiplier ×100)
        total_cost = fill_price * quantity * 100
        if total_cost > self._cash:
            return FillResult(
                status="REJECTED",
                reject_reason="INSUFFICIENT_CASH",
            )

        # Execute
        self._cash -= total_cost
        self._add_position(contract, quantity, fill_price)

        return FillResult(
            status="FILLED",
            fill_price=fill_price,
            fill_time=timestamp,
            contract=contract,
        )

    def execute_sell(
        self, contract: OptionContract, quantity: int, timestamp: datetime
    ) -> FillResult:
        """Sell: bid * (1 - slippage). Reject if no volume."""
        quote = self._get_quote(contract.contract_id, timestamp)
        if quote is None:
            return FillResult(
                status="REJECTED",
                reject_reason="NO_LIQUIDITY",
            )

        # Liquidity check
        if quote["volume"] == 0:
            return FillResult(
                status="REJECTED",
                reject_reason="NO_LIQUIDITY",
            )

        # Calculate fill price
        half_spread = max((quote["high"] - quote["low"]) / 2, 0.01)
        bid = quote["close"] - half_spread
        fill_price = bid * (1 - self._slippage_pct)

        # Execute (options multiplier ×100)
        proceeds = fill_price * quantity * 100
        self._cash += proceeds
        self._remove_position(contract.contract_id)

        return FillResult(
            status="FILLED",
            fill_price=fill_price,
            fill_time=timestamp,
            contract=contract,
        )

    def get_mark_price(self, contract: OptionContract, timestamp: datetime) -> float:
        """Asof match in OHLCV -> return close price. 0.0 if no data."""
        quote = self._get_quote(contract.contract_id, timestamp)
        if quote is None:
            return 0.0
        return float(quote["close"])

    def get_cash(self) -> float:
        return self._cash

    def get_positions(self) -> list[Position]:
        return list(self._positions)

    def get_position(self, symbol: str) -> Position | None:
        for pos in self._positions:
            if pos.contract.symbol == symbol:
                return pos
        return None

    def reset(self) -> None:
        """Reset cash and positions. Keep loaded data for grid search efficiency."""
        self._cash = self._initial_capital
        self._positions = []

    # ── Internal helpers ──────────────────────────────

    def _get_quote(self, contract_id: str, timestamp: datetime) -> dict | None:
        """Asof lookup in per-contract indexed OHLCV. O(log n) via searchsorted."""
        contract_bars = self._contract_ohlcv.get(contract_id)
        if contract_bars is None or contract_bars.empty:
            return None

        # Strip timezone for comparison with tz-naive OHLCV data
        ts = timestamp
        if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
            ts = ts.replace(tzinfo=None)
        if isinstance(ts, pd.Timestamp):
            ts = ts.tz_localize(None) if ts.tzinfo else ts

        # Binary search for last bar <= timestamp
        idx = contract_bars["datetime"].searchsorted(ts, side="right") - 1
        if idx < 0:
            return None

        bar = contract_bars.iloc[idx]
        return {
            "open": float(bar["open"]),
            "high": float(bar["high"]),
            "low": float(bar["low"]),
            "close": float(bar["close"]),
            "volume": int(bar["volume"]),
        }

    def _select_atm(
        self, chain: list[OptionContract], underlying_price: float
    ) -> OptionContract:
        """Select the contract closest to underlying price."""
        return min(chain, key=lambda c: abs(c.strike - underlying_price))

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
