"""MarketData interface and BacktestMarketData implementation.

Provides read-only market data access, decoupled from execution logic.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from config.settings import RAW_OPTIONS_DIR
from src.backtest.executor.base import OptionContract
from src.backtest.types import Quote


class MarketData(ABC):
    """Read-only market data interface. Like a HTS quote screen."""

    @abstractmethod
    def get_stock_quote(self, symbol: str, timestamp: datetime) -> Optional[Quote]:
        """Get stock quote at timestamp (asof lookup)."""
        ...

    @abstractmethod
    def get_option_chain(
        self, symbol: str, option_type: str, timestamp: datetime
    ) -> list[OptionContract]:
        """Get active option contracts at timestamp."""
        ...

    @abstractmethod
    def get_option_quote(
        self, contract: OptionContract, timestamp: datetime
    ) -> Optional[Quote]:
        """Get option quote for specific contract at timestamp."""
        ...


class BacktestMarketData(MarketData):
    """MarketData backed by historical parquet files.

    Data layout per symbol:
        data/raw/options/us/{SYMBOL}/contracts.parquet
        data/raw/options/us/{SYMBOL}/{year}.parquet
    """

    def __init__(
        self,
        symbols: list[str],
        market: str = "us",
        data_dir: Path | None = None,
    ):
        self._symbols = symbols
        self._market = market
        self._data_dir = data_dir or RAW_OPTIONS_DIR

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

    # ── MarketData interface ──────────────────────────

    def get_stock_quote(self, symbol: str, timestamp: datetime) -> Optional[Quote]:
        # For backtesting, we don't have separate stock OHLCV in this data source.
        # Return None — strategies that need stock quotes should use the row data.
        return None

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

    def get_option_quote(
        self, contract: OptionContract, timestamp: datetime
    ) -> Optional[Quote]:
        """Get option quote for specific contract at timestamp."""
        return self._get_quote(contract.contract_id, timestamp)

    # ── Internal helpers ──────────────────────────────

    def _get_quote(self, contract_id: str, timestamp: datetime) -> Optional[Quote]:
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
        return Quote(
            open=float(bar["open"]),
            high=float(bar["high"]),
            low=float(bar["low"]),
            close=float(bar["close"]),
            volume=int(bar["volume"]),
            timestamp=bar["datetime"].to_pydatetime() if hasattr(bar["datetime"], "to_pydatetime") else bar["datetime"],
        )
