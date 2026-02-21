"""Multi-source bar data fetcher with orchestration logic.

Supports:
- tvDatafeed: Latest ~13 trading days (1-min bars)
- yfinance: US stocks, up to ~60 days of 1-min data (7-day windows)
- pykrx: Korean stocks, daily OHLCV (1-min not available via pykrx)

Strategy:
- US: yfinance for recent history + tvDatafeed for latest
- KR: tvDatafeed for latest + future integration with broker APIs
"""

import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from loguru import logger
from tqdm import tqdm

from config.settings import SYMBOLS_DIR
from src.collector.collection_tracker import CollectionTracker
from src.collector.storage import save_bars, validate_bars
from src.collector.tv_client import TVClient


def load_symbol_list(market: str) -> pd.DataFrame:
    """Load symbol list CSV for a given market.

    Args:
        market: 'kr', 'us_stocks', or 'us_etf_index'

    Returns:
        DataFrame with columns [ticker, name, tv_symbol, exchange, ...]
    """
    if market == "kr":
        path = SYMBOLS_DIR / "kr_symbols.csv"
    elif market == "us_stocks":
        path = SYMBOLS_DIR / "us_stocks.csv"
    elif market == "us_etf_index":
        path = SYMBOLS_DIR / "us_etf_index.csv"
    else:
        raise ValueError(f"Unknown market: {market}")

    return pd.read_csv(path)


def fetch_us_yfinance(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Fetch 1-minute bars from yfinance for US stocks.

    yfinance constraints:
    - 1-min data available for last ~60 days only
    - Must request in 7-day windows

    Args:
        ticker: US ticker symbol (e.g., 'AAPL')
        start_date: Start date (YYYY-MM-DD), defaults to 60 days ago
        end_date: End date (YYYY-MM-DD), defaults to today

    Returns:
        DataFrame with OHLCV columns indexed by datetime, or None on failure.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed")
        return None

    end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.now()
    start_dt = (
        datetime.strptime(start_date, "%Y-%m-%d")
        if start_date
        else end_dt - timedelta(days=59)
    )

    # yfinance allows max 7-day windows for 1-min data
    all_dfs = []
    current_start = start_dt

    while current_start < end_dt:
        current_end = min(current_start + timedelta(days=7), end_dt)

        try:
            stock = yf.Ticker(ticker)
            df = stock.history(
                start=current_start.strftime("%Y-%m-%d"),
                end=current_end.strftime("%Y-%m-%d"),
                interval="1m",
            )
            if df is not None and not df.empty:
                df = df.rename(columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                })
                df = df[["open", "high", "low", "close", "volume"]].copy()
                df.index = pd.to_datetime(df.index).tz_localize(None)
                df.index.name = "datetime"
                all_dfs.append(df)
        except Exception as e:
            logger.warning(f"yfinance error for {ticker} ({current_start}~{current_end}): {e}")

        current_start = current_end
        time.sleep(0.5)  # Be nice to yfinance

    if not all_dfs:
        return None

    combined = pd.concat(all_dfs)
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()
    logger.info(f"yfinance: fetched {len(combined)} bars for {ticker}")
    return combined


def fetch_kr_pykrx_daily(
    ticker: str,
    start_date: str,
    end_date: str,
) -> Optional[pd.DataFrame]:
    """Fetch daily OHLCV from pykrx for Korean stocks.

    Note: pykrx does NOT provide 1-minute bars.
    This is a fallback for daily data only.

    Args:
        ticker: Korean stock code (e.g., '005930')
        start_date: Start date (YYYYMMDD)
        end_date: End date (YYYYMMDD)

    Returns:
        DataFrame with OHLCV columns, or None on failure.
    """
    try:
        from pykrx import stock as pykrx_stock
    except ImportError:
        logger.error("pykrx not installed")
        return None

    try:
        df = pykrx_stock.get_market_ohlcv_by_date(start_date, end_date, ticker)
        if df is None or df.empty:
            return None

        df = df.rename(columns={
            "시가": "open",
            "고가": "high",
            "저가": "low",
            "종가": "close",
            "거래량": "volume",
        })
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df.index = pd.to_datetime(df.index)
        df.index.name = "datetime"
        logger.info(f"pykrx: fetched {len(df)} daily bars for {ticker}")
        return df
    except Exception as e:
        logger.warning(f"pykrx error for {ticker}: {e}")
        return None


class BarFetcher:
    """Orchestrates multi-source data collection for all symbols."""

    def __init__(
        self,
        tv_username: Optional[str] = None,
        tv_password: Optional[str] = None,
    ):
        self._tv_client: Optional[TVClient] = None
        self._tv_username = tv_username
        self._tv_password = tv_password
        self._tracker = CollectionTracker()

    @property
    def tv_client(self) -> TVClient:
        if self._tv_client is None:
            self._tv_client = TVClient(self._tv_username, self._tv_password)
        return self._tv_client

    @property
    def tracker(self) -> CollectionTracker:
        return self._tracker

    def collect_us_stocks(self, source: str = "yfinance") -> None:
        """Collect 1-min bars for all US stocks."""
        symbols_df = load_symbol_list("us_stocks")
        self._collect_batch(symbols_df, market="us", source=source)

    def collect_us_etfs(self, source: str = "yfinance") -> None:
        """Collect 1-min bars for US ETFs and index futures."""
        symbols_df = load_symbol_list("us_etf_index")
        etfs = symbols_df[symbols_df.get("type", "etf") == "etf"]
        self._collect_batch(etfs, market="us", source=source)

    def collect_kr_stocks(self, source: str = "tvdatafeed") -> None:
        """Collect 1-min bars for Korean stocks."""
        symbols_df = load_symbol_list("kr")
        self._collect_batch(symbols_df, market="kr", source=source)

    def collect_single(
        self,
        ticker: str,
        exchange: str,
        market: str,
        source: str = "tvdatafeed",
    ) -> Optional[pd.DataFrame]:
        """Collect data for a single symbol.

        Args:
            ticker: Symbol ticker
            exchange: Exchange name
            market: 'kr' or 'us'
            source: Data source to use

        Returns:
            Collected DataFrame or None.
        """
        df = None

        if source == "tvdatafeed":
            df = self.tv_client.get_hist(ticker, exchange)
        elif source == "yfinance":
            df = fetch_us_yfinance(ticker)
        else:
            logger.error(f"Unknown source: {source}")
            return None

        if df is not None and not df.empty:
            validation = validate_bars(df.reset_index() if "datetime" not in df.columns else df)
            if not validation["valid"]:
                logger.warning(f"Validation failed for {ticker}: {validation}")

            saved = save_bars(df, market=market, symbol=ticker)
            total_bars = sum(saved.values())

            self._tracker.upsert(
                symbol=ticker,
                exchange=exchange,
                market=market,
                source=source,
                start_date=str(df.index.min() if isinstance(df.index, pd.DatetimeIndex) else df["datetime"].min()),
                end_date=str(df.index.max() if isinstance(df.index, pd.DatetimeIndex) else df["datetime"].max()),
                bar_count=total_bars,
                status="complete",
            )
            return df
        else:
            self._tracker.upsert(
                symbol=ticker,
                exchange=exchange,
                market=market,
                source=source,
                status="error",
                error_message="No data returned",
            )
            return None

    def _collect_batch(
        self,
        symbols_df: pd.DataFrame,
        market: str,
        source: str,
    ) -> None:
        """Collect data for a batch of symbols with progress tracking."""
        total = len(symbols_df)
        logger.info(f"Starting batch collection: {total} symbols, market={market}, source={source}")

        for _, row in tqdm(symbols_df.iterrows(), total=total, desc=f"Collecting {market}"):
            ticker = str(row["ticker"])
            exchange = row.get("exchange", "")

            # Skip already completed
            status = self._tracker.get_status(ticker, exchange, source)
            if status and status["status"] == "complete":
                logger.debug(f"Skipping {ticker} (already complete)")
                continue

            try:
                self.collect_single(ticker, exchange, market, source)
            except Exception as e:
                logger.error(f"Failed to collect {ticker}: {e}")
                self._tracker.mark_error(ticker, exchange, source, str(e))

    def update_latest(self, market: str) -> None:
        """Update all symbols with latest data from tvDatafeed."""
        if market == "kr":
            symbols_df = load_symbol_list("kr")
        else:
            stocks = load_symbol_list("us_stocks")
            etfs = load_symbol_list("us_etf_index")
            symbols_df = pd.concat([stocks, etfs], ignore_index=True)

        logger.info(f"Updating latest bars for {len(symbols_df)} {market} symbols via tvDatafeed")

        for _, row in tqdm(symbols_df.iterrows(), total=len(symbols_df), desc=f"Updating {market}"):
            ticker = str(row["ticker"])
            tv_symbol = row.get("tv_symbol", "")
            if ":" in tv_symbol:
                exchange, symbol = tv_symbol.split(":", 1)
            else:
                exchange = row.get("exchange", "")
                symbol = ticker

            try:
                df = self.tv_client.get_hist(symbol, exchange)
                if df is not None and not df.empty:
                    save_bars(df, market=market, symbol=ticker)
            except Exception as e:
                logger.warning(f"Failed to update {ticker}: {e}")
