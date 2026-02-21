"""TradingView data feed client wrapper with rate limiting and retry logic."""

import time
from typing import Optional

import pandas as pd
from loguru import logger
from tvDatafeed import Interval, TvDatafeed, TvDatafeedLive

from config.settings import (
    TV_BACKOFF_BASE,
    TV_MAX_BARS,
    TV_MAX_RETRIES,
    TV_RATE_LIMIT_SEC,
)


class TVClient:
    """Wrapper around tvDatafeed with rate limiting, retries, and session management."""

    def __init__(self, username: Optional[str] = None, password: Optional[str] = None):
        self._username = username
        self._password = password
        self._tv: Optional[TvDatafeed] = None
        self._last_request_time: float = 0.0
        self._connect()

    def _connect(self) -> None:
        """Establish or re-establish TvDatafeed session."""
        try:
            if self._username and self._password:
                self._tv = TvDatafeed(self._username, self._password)
            else:
                self._tv = TvDatafeed()
            logger.info("TvDatafeed session established")
        except Exception as e:
            logger.error(f"Failed to connect to TvDatafeed: {e}")
            raise

    def _rate_limit(self) -> None:
        """Enforce minimum interval between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < TV_RATE_LIMIT_SEC:
            sleep_time = TV_RATE_LIMIT_SEC - elapsed
            time.sleep(sleep_time)
        self._last_request_time = time.time()

    def get_hist(
        self,
        symbol: str,
        exchange: str,
        n_bars: int = TV_MAX_BARS,
    ) -> Optional[pd.DataFrame]:
        """Fetch 1-minute historical bars with retry and exponential backoff.

        Args:
            symbol: Ticker symbol (e.g., '005930', 'AAPL')
            exchange: Exchange name (e.g., 'KRX', 'NASDAQ')
            n_bars: Number of bars to fetch (max ~5000)

        Returns:
            DataFrame with columns [open, high, low, close, volume] indexed by datetime,
            or None if all retries fail.
        """
        n_bars = min(n_bars, TV_MAX_BARS)

        for attempt in range(1, TV_MAX_RETRIES + 1):
            self._rate_limit()
            try:
                df = self._tv.get_hist(
                    symbol=symbol,
                    exchange=exchange,
                    interval=Interval.in_1_minute,
                    n_bars=n_bars,
                )
                if df is not None and not df.empty:
                    df = df[["open", "high", "low", "close", "volume"]].copy()
                    df.index = pd.to_datetime(df.index)
                    df.index.name = "datetime"
                    logger.debug(
                        f"Fetched {len(df)} bars for {exchange}:{symbol}"
                    )
                    return df
                else:
                    logger.warning(
                        f"Empty response for {exchange}:{symbol} (attempt {attempt})"
                    )
            except Exception as e:
                logger.warning(
                    f"Error fetching {exchange}:{symbol} (attempt {attempt}): {e}"
                )
                if "session" in str(e).lower() or "connect" in str(e).lower():
                    logger.info("Reconnecting TvDatafeed session...")
                    try:
                        self._connect()
                    except Exception:
                        pass

            if attempt < TV_MAX_RETRIES:
                backoff = TV_BACKOFF_BASE ** attempt
                logger.info(f"Retrying in {backoff}s...")
                time.sleep(backoff)

        logger.error(f"All {TV_MAX_RETRIES} attempts failed for {exchange}:{symbol}")
        return None

    def get_latest(
        self,
        symbol: str,
        exchange: str,
    ) -> Optional[pd.DataFrame]:
        """Fetch latest available 1-min bars (convenience wrapper for updates)."""
        return self.get_hist(symbol, exchange, n_bars=TV_MAX_BARS)
