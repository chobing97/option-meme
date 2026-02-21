"""Multi-source bar data fetcher with orchestration logic.

Strategy (yfinance → tvDatafeed 순서):
1. yfinance로 먼저 수집 (~30-60일치 1분봉, base layer)
2. tvDatafeed로 최신 ~13일치를 덮어쓰기 (overlay)

tvDatafeed 데이터가 겹치는 구간에서 yfinance를 대체하므로,
최신 구간은 항상 tvDatafeed 품질로 유지됨.

Supports:
- yfinance: US/KR 공통, 최대 ~60일 1분봉 (7일 윈도우 반복)
- tvDatafeed: US/KR 공통, 최신 ~13거래일 1분봉
- pykrx: KR 일봉 전용 (1분봉 미지원, 보조용)
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


# ── Symbol list loaders ────────────────────────────────


def load_symbol_list(market: str) -> pd.DataFrame:
    """Load symbol list CSV for a given market.

    Args:
        market: 'kr', 'us_stocks', or 'us_etf_index'
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


# ── yfinance fetchers ─────────────────────────────────


def _yf_ticker(ticker: str, market: str) -> str:
    """Convert ticker to yfinance format.

    US: 그대로 ('AAPL')
    KR: KOSPI → '{ticker}.KS', KOSDAQ → '{ticker}.KQ'
        (기본 KOSPI 가정, 실패 시 KOSDAQ fallback)
    """
    if market == "us":
        return ticker
    # KR: 6자리 숫자 코드
    return f"{ticker}.KS"


def fetch_yfinance(
    ticker: str,
    market: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Fetch 1-minute bars from yfinance.

    yfinance constraints:
    - 1-min data available for last ~60 days only
    - Must request in 7-day windows

    Args:
        ticker: Ticker symbol (e.g., 'AAPL', '005930')
        market: 'kr' or 'us'
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

    yf_ticker = _yf_ticker(ticker, market)

    all_dfs = []
    current_start = start_dt

    while current_start < end_dt:
        current_end = min(current_start + timedelta(days=7), end_dt)

        try:
            stock = yf.Ticker(yf_ticker)
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
            logger.warning(f"yfinance error for {yf_ticker} ({current_start}~{current_end}): {e}")

        current_start = current_end
        time.sleep(0.5)

    # KR KOSPI 실패 시 KOSDAQ(.KQ) fallback
    if not all_dfs and market == "kr":
        yf_ticker_kq = f"{ticker}.KQ"
        logger.info(f"KOSPI failed for {ticker}, trying KOSDAQ ({yf_ticker_kq})")
        current_start = start_dt
        while current_start < end_dt:
            current_end = min(current_start + timedelta(days=7), end_dt)
            try:
                stock = yf.Ticker(yf_ticker_kq)
                df = stock.history(
                    start=current_start.strftime("%Y-%m-%d"),
                    end=current_end.strftime("%Y-%m-%d"),
                    interval="1m",
                )
                if df is not None and not df.empty:
                    df = df.rename(columns={
                        "Open": "open", "High": "high",
                        "Low": "low", "Close": "close", "Volume": "volume",
                    })
                    df = df[["open", "high", "low", "close", "volume"]].copy()
                    df.index = pd.to_datetime(df.index).tz_localize(None)
                    df.index.name = "datetime"
                    all_dfs.append(df)
            except Exception as e:
                logger.warning(f"yfinance KOSDAQ error for {yf_ticker_kq}: {e}")
            current_start = current_end
            time.sleep(0.5)

    if not all_dfs:
        return None

    combined = pd.concat(all_dfs)
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()
    logger.info(f"yfinance: fetched {len(combined)} bars for {ticker} ({market})")
    return combined


# ── pykrx fetcher (일봉, 보조용) ──────────────────────


def fetch_kr_pykrx_daily(
    ticker: str,
    start_date: str,
    end_date: str,
) -> Optional[pd.DataFrame]:
    """Fetch daily OHLCV from pykrx for Korean stocks.

    Note: pykrx does NOT provide 1-minute bars.
    This is a fallback for daily data only.
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
            "시가": "open", "고가": "high",
            "저가": "low", "종가": "close", "거래량": "volume",
        })
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df.index = pd.to_datetime(df.index)
        df.index.name = "datetime"
        logger.info(f"pykrx: fetched {len(df)} daily bars for {ticker}")
        return df
    except Exception as e:
        logger.warning(f"pykrx error for {ticker}: {e}")
        return None


# ── Helper: save + track ──────────────────────────────


def _save_and_track(
    df: pd.DataFrame,
    ticker: str,
    exchange: str,
    market: str,
    source: str,
    tracker: CollectionTracker,
) -> int:
    """Validate, save to Parquet, update tracker. Returns bar count."""
    normalized = df.reset_index() if "datetime" not in df.columns else df
    validation = validate_bars(normalized)
    if not validation["valid"]:
        logger.warning(f"Validation issues for {ticker} ({source}): {validation}")

    saved = save_bars(df, market=market, symbol=ticker)
    total_bars = sum(saved.values())

    dt_series = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df["datetime"])
    tracker.upsert(
        symbol=ticker,
        exchange=exchange,
        market=market,
        source=source,
        start_date=str(dt_series.min()),
        end_date=str(dt_series.max()),
        bar_count=total_bars,
        status="complete",
    )
    return total_bars


# ── Main orchestrator ─────────────────────────────────


class BarFetcher:
    """Orchestrates two-phase data collection: yfinance → tvDatafeed overlay."""

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

    # ── Public: 마켓별 전체 수집 ───────────────────────

    def collect_us_stocks(self) -> None:
        """Collect all US stocks: yfinance → tvDatafeed."""
        symbols_df = load_symbol_list("us_stocks")
        self._collect_batch(symbols_df, market="us")

    def collect_us_etfs(self) -> None:
        """Collect US ETFs: yfinance → tvDatafeed."""
        symbols_df = load_symbol_list("us_etf_index")
        etfs = symbols_df[symbols_df.get("type", "etf") == "etf"]
        self._collect_batch(etfs, market="us")

    def collect_kr_stocks(self) -> None:
        """Collect all KR stocks: yfinance → tvDatafeed."""
        symbols_df = load_symbol_list("kr")
        self._collect_batch(symbols_df, market="kr")

    # ── Public: 단일 종목 수집 ─────────────────────────

    def collect_single(
        self,
        ticker: str,
        exchange: str,
        market: str,
    ) -> Optional[pd.DataFrame]:
        """Collect a single symbol with two-phase strategy.

        Phase 1: yfinance (~60일치 base)
        Phase 2: tvDatafeed (~13일치 overlay, 겹치는 구간 덮어쓰기)

        Returns:
            The final merged DataFrame, or None if both sources fail.
        """
        tv_symbol = ticker
        tv_exchange = exchange
        # tv_symbol에서 exchange:symbol 분리
        if ":" in exchange:
            tv_exchange, tv_symbol = exchange.split(":", 1)

        # ── Phase 1: yfinance ──────────────────────────
        logger.info(f"[1/2] yfinance 수집: {ticker} ({market})")
        yf_df = fetch_yfinance(ticker, market)

        if yf_df is not None and not yf_df.empty:
            bars = _save_and_track(yf_df, ticker, exchange, market, "yfinance", self._tracker)
            logger.info(f"  yfinance: {bars} bars 저장 완료")
        else:
            logger.warning(f"  yfinance: {ticker} 데이터 없음")

        # ── Phase 2: tvDatafeed overlay ────────────────
        logger.info(f"[2/2] tvDatafeed 덮어쓰기: {ticker} ({market})")
        try:
            tv_df = self.tv_client.get_hist(tv_symbol, tv_exchange)
        except Exception as e:
            logger.warning(f"  tvDatafeed 실패: {ticker}: {e}")
            tv_df = None

        if tv_df is not None and not tv_df.empty:
            # save_bars가 내부적으로 기존 Parquet과 merge하되,
            # 중복 datetime은 keep="last"이므로 tv 데이터가 yf를 덮어씀
            bars = _save_and_track(tv_df, ticker, exchange, market, "tvdatafeed", self._tracker)
            logger.info(f"  tvDatafeed: {bars} bars 덮어쓰기 완료")
        else:
            logger.warning(f"  tvDatafeed: {ticker} 데이터 없음")

        # ── 최종 결과 ─────────────────────────────────
        if yf_df is None and tv_df is None:
            self._tracker.upsert(
                symbol=ticker, exchange=exchange, market=market,
                source="combined", status="error",
                error_message="Both yfinance and tvDatafeed returned no data",
            )
            return None

        # 최종 저장된 데이터 중 더 최신 것 반환
        return tv_df if (tv_df is not None and not tv_df.empty) else yf_df

    # ── Private: 배치 수집 ─────────────────────────────

    def _collect_batch(
        self,
        symbols_df: pd.DataFrame,
        market: str,
    ) -> None:
        """Collect data for a batch of symbols (yfinance → tvDatafeed per symbol)."""
        total = len(symbols_df)
        logger.info(f"배치 수집 시작: {total} 종목, market={market}")

        for _, row in tqdm(symbols_df.iterrows(), total=total, desc=f"Collecting {market}"):
            ticker = str(row["ticker"])
            exchange = row.get("exchange", "")

            # tv_symbol이 있으면 exchange 정보로 활용
            tv_symbol = row.get("tv_symbol", "")
            if tv_symbol and ":" in tv_symbol:
                exchange = tv_symbol  # "KRX:005930" 형태로 보존

            try:
                self.collect_single(ticker, exchange, market)
            except Exception as e:
                logger.error(f"수집 실패 {ticker}: {e}")
                self._tracker.mark_error(ticker, exchange, "combined", str(e))
