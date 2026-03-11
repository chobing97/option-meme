"""Historical DataFeed: replays historical parquet data as a live feed."""

from collections import deque
from typing import Optional

import pandas as pd
from loguru import logger

from src.collector.storage import load_bars
from src.labeler.session_extractor import extract_early_session
from src.trading.datafeed.base import DataFeed


class HistoricalDataFeed(DataFeed):
    """Replays parquet bar data as if it were a live feed.

    On connect():
    - The replay date becomes the live queue
    - Earlier dates become the history buffer

    Supports pre-loaded early_df to avoid reloading data on each date.
    """

    def __init__(
        self,
        market: str,
        symbol: str,
        date: Optional[str] = None,
        early_df: Optional[pd.DataFrame] = None,
    ):
        self.market = market
        self.symbol = symbol
        self._target_date = date
        self._early_df = early_df

        self._history: pd.DataFrame = pd.DataFrame()
        self._queue: deque[pd.Series] = deque()
        self.replay_date: str = ""
        self._connected: bool = False

    def connect(self) -> None:
        early_df = self._early_df
        if early_df is None:
            raw_df = load_bars(self.market, self.symbol)
            if raw_df.empty:
                raise FileNotFoundError(
                    f"No data for {self.market}/{self.symbol}. "
                    f"Run: ./optionmeme collector --market {self.market} --symbol {self.symbol}"
                )
            early_df = extract_early_session(raw_df, self.market)
            if early_df.empty:
                raise ValueError(f"No early session bars for {self.market}/{self.symbol}")

        if "date_str" not in early_df.columns:
            early_df = early_df.copy()
            early_df["date_str"] = early_df["date"].astype(str)

        available_dates = sorted(early_df["date_str"].unique())

        if self._target_date:
            if self._target_date not in available_dates:
                raise ValueError(
                    f"Date {self._target_date} not available. "
                    f"Latest: {available_dates[-5:]}"
                )
            target = self._target_date
        else:
            target = available_dates[-1]

        self.replay_date = target

        # Split: history = last 5 dates before target, queue = target date
        history_dates = [d for d in available_dates if d < target]
        history_dates = history_dates[-5:]

        self._history = early_df[early_df["date_str"].isin(history_dates)].copy()
        replay_df = early_df[early_df["date_str"] == target].copy()
        replay_df = replay_df.sort_values("datetime").reset_index(drop=True)

        for _, row in replay_df.iterrows():
            self._queue.append(row)

        self._connected = True
        logger.info(
            f"HistoricalDataFeed connected: {self.market}/{self.symbol} "
            f"replay={target} ({len(self._queue)} bars), "
            f"history={len(history_dates)} days ({len(self._history)} bars)"
        )

    def get_latest_bar(self) -> Optional[pd.Series]:
        if not self._queue:
            return None
        return self._queue.popleft()

    def get_history(self, n_days: int = 5) -> pd.DataFrame:
        return self._history.copy()

    def is_session_active(self) -> bool:
        return self._connected and len(self._queue) > 0

    def disconnect(self) -> None:
        self._connected = False
        self._queue.clear()

    @staticmethod
    def get_available_dates(
        market: str, symbol: str, date_from: str | None = None, date_to: str | None = None,
    ) -> tuple[pd.DataFrame, list[str]]:
        """Load data once and return (early_df, list of available dates).

        Optionally filter by date_from/date_to range.
        """
        raw_df = load_bars(market, symbol)
        if raw_df.empty:
            raise FileNotFoundError(f"No data for {market}/{symbol}")

        early_df = extract_early_session(raw_df, market)
        if early_df.empty:
            raise ValueError(f"No early session bars for {market}/{symbol}")

        early_df["date_str"] = early_df["date"].astype(str)
        dates = sorted(early_df["date_str"].unique())

        if date_from:
            dates = [d for d in dates if d >= date_from]
        if date_to:
            dates = [d for d in dates if d <= date_to]

        # Need at least 5 history days before first replay date
        all_dates = sorted(early_df["date_str"].unique())
        if dates:
            first_replay = dates[0]
            history_count = len([d for d in all_dates if d < first_replay])
            if history_count < 5:
                dates = dates[5 - history_count:]

        return early_df, dates
