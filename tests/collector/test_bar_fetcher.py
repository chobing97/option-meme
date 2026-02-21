"""Tests for src.collector.bar_fetcher — orchestration logic."""

from unittest.mock import MagicMock, patch, PropertyMock

import pandas as pd
import pytest

from src.collector.bar_fetcher import _yf_ticker, BarFetcher


# ── _yf_ticker ──────────────────────────────────────────────


def test_yf_ticker_us():
    assert _yf_ticker("AAPL", "us") == "AAPL"


def test_yf_ticker_kr():
    assert _yf_ticker("005930", "kr") == "005930.KS"


# ── fetch_yfinance ──────────────────────────────────────────


def _make_yf_history(n: int = 5, tz: str = "America/New_York") -> pd.DataFrame:
    dates = pd.date_range("2025-01-02 09:30", periods=n, freq="min", tz=tz)
    return pd.DataFrame(
        {
            "Open": [100.0] * n,
            "High": [101.0] * n,
            "Low": [99.0] * n,
            "Close": [100.5] * n,
            "Volume": [1000] * n,
            "Dividends": [0.0] * n,
            "Stock Splits": [0.0] * n,
        },
        index=dates,
    )


def test_fetch_yfinance_success():
    """Normal yfinance fetch → lowercase columns, tz-naive index."""
    with (
        patch("yfinance.Ticker") as mock_ticker_cls,
        patch("src.collector.bar_fetcher.time") as mock_time,
    ):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = _make_yf_history(5)
        mock_ticker_cls.return_value = mock_ticker
        mock_time.sleep = MagicMock()

        from src.collector.bar_fetcher import fetch_yfinance

        result = fetch_yfinance("AAPL", "us", start_date="2025-01-02", end_date="2025-01-03")

        assert result is not None
        assert not result.empty
        assert list(result.columns) == ["open", "high", "low", "close", "volume"]
        assert result.index.tz is None


def test_fetch_yfinance_kr_kosdaq_fallback():
    """.KS returns empty → fallback to .KQ succeeds."""
    with (
        patch("yfinance.Ticker") as mock_ticker_cls,
        patch("src.collector.bar_fetcher.time") as mock_time,
    ):
        mock_ks = MagicMock()
        mock_ks.history.return_value = pd.DataFrame()  # empty

        mock_kq = MagicMock()
        mock_kq.history.return_value = _make_yf_history(3, tz="Asia/Seoul")

        # First call → .KS ticker (empty), second call → .KQ ticker (success)
        mock_ticker_cls.side_effect = [mock_ks, mock_kq]
        mock_time.sleep = MagicMock()

        from src.collector.bar_fetcher import fetch_yfinance

        result = fetch_yfinance("035720", "kr", start_date="2025-01-02", end_date="2025-01-03")

        assert result is not None
        assert len(result) == 3
        # Ticker was called with both .KS and .KQ
        ticker_calls = [c[0][0] for c in mock_ticker_cls.call_args_list]
        assert "035720.KS" in ticker_calls
        assert "035720.KQ" in ticker_calls


def test_fetch_yfinance_both_fail():
    """.KS and .KQ both empty → returns None."""
    with (
        patch("yfinance.Ticker") as mock_ticker_cls,
        patch("src.collector.bar_fetcher.time") as mock_time,
    ):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_ticker_cls.return_value = mock_ticker
        mock_time.sleep = MagicMock()

        from src.collector.bar_fetcher import fetch_yfinance

        result = fetch_yfinance("999999", "kr", start_date="2025-01-02", end_date="2025-01-03")

        assert result is None


# ── BarFetcher.collect_single ───────────────────────────────


def _make_ohlcv_df(n: int = 5) -> pd.DataFrame:
    dates = pd.date_range("2025-01-02 09:00", periods=n, freq="min")
    df = pd.DataFrame(
        {
            "open": [100.0] * n,
            "high": [101.0] * n,
            "low": [99.0] * n,
            "close": [100.5] * n,
            "volume": [1000] * n,
        },
        index=dates,
    )
    df.index.name = "datetime"
    return df


@pytest.fixture
def fetcher_mocked(tmp_path):
    """BarFetcher with tracker pointed to tmp_path and tv_client mocked."""
    with patch("src.collector.bar_fetcher.CollectionTracker") as mock_tracker_cls:
        mock_tracker = MagicMock()
        mock_tracker_cls.return_value = mock_tracker
        fetcher = BarFetcher()
        fetcher._tv_client = MagicMock()
        yield fetcher, mock_tracker


def test_collect_single_both_succeed(fetcher_mocked):
    fetcher, mock_tracker = fetcher_mocked
    yf_df = _make_ohlcv_df(10)
    tv_df = _make_ohlcv_df(5)

    with (
        patch("src.collector.bar_fetcher.fetch_yfinance", return_value=yf_df),
        patch("src.collector.bar_fetcher._save_and_track", return_value=5) as mock_save,
    ):
        fetcher._tv_client.get_hist.return_value = tv_df
        result = fetcher.collect_single("005930", "KRX", "kr")

        assert result is not None
        assert mock_save.call_count == 2  # yfinance + tvdatafeed
        # Returns tv_df since it's not None
        assert len(result) == 5


def test_collect_single_yfinance_only(fetcher_mocked):
    fetcher, mock_tracker = fetcher_mocked
    yf_df = _make_ohlcv_df(10)

    with (
        patch("src.collector.bar_fetcher.fetch_yfinance", return_value=yf_df),
        patch("src.collector.bar_fetcher._save_and_track", return_value=10) as mock_save,
    ):
        fetcher._tv_client.get_hist.return_value = None
        result = fetcher.collect_single("005930", "KRX", "kr")

        assert result is not None
        assert mock_save.call_count == 1  # yfinance only
        assert len(result) == 10


def test_collect_single_tvdatafeed_only(fetcher_mocked):
    fetcher, mock_tracker = fetcher_mocked
    tv_df = _make_ohlcv_df(5)

    with (
        patch("src.collector.bar_fetcher.fetch_yfinance", return_value=None),
        patch("src.collector.bar_fetcher._save_and_track", return_value=5) as mock_save,
    ):
        fetcher._tv_client.get_hist.return_value = tv_df
        result = fetcher.collect_single("005930", "KRX", "kr")

        assert result is not None
        assert mock_save.call_count == 1  # tvdatafeed only
        assert len(result) == 5


def test_collect_single_both_fail(fetcher_mocked):
    fetcher, mock_tracker = fetcher_mocked

    with (
        patch("src.collector.bar_fetcher.fetch_yfinance", return_value=None),
        patch("src.collector.bar_fetcher._save_and_track") as mock_save,
    ):
        fetcher._tv_client.get_hist.return_value = None
        result = fetcher.collect_single("005930", "KRX", "kr")

        assert result is None
        mock_save.assert_not_called()
        # Tracker should record error for combined source
        mock_tracker.upsert.assert_called_once()
        call_kwargs = mock_tracker.upsert.call_args
        assert call_kwargs.kwargs.get("status", call_kwargs[1].get("status")) == "error"


def test_collect_single_exchange_colon_split(fetcher_mocked):
    """'KRX:005930' in exchange field → splits into exchange='KRX', symbol='005930'."""
    fetcher, mock_tracker = fetcher_mocked
    tv_df = _make_ohlcv_df(3)

    with (
        patch("src.collector.bar_fetcher.fetch_yfinance", return_value=None),
        patch("src.collector.bar_fetcher._save_and_track", return_value=3),
    ):
        fetcher._tv_client.get_hist.return_value = tv_df
        fetcher.collect_single("005930", "KRX:005930", "kr")

        # tv_client.get_hist should have been called with the split values
        fetcher._tv_client.get_hist.assert_called_once_with("005930", "KRX")


# ── BarFetcher._collect_batch ───────────────────────────────


def test_collect_batch_iterates_all(fetcher_mocked):
    fetcher, mock_tracker = fetcher_mocked
    symbols_df = pd.DataFrame({
        "ticker": ["AAPL", "MSFT", "GOOG"],
        "exchange": ["NASDAQ", "NASDAQ", "NASDAQ"],
    })

    with patch.object(fetcher, "collect_single", return_value=_make_ohlcv_df(3)) as mock_cs:
        fetcher._collect_batch(symbols_df, market="us")
        assert mock_cs.call_count == 3


def test_collect_batch_error_continues(fetcher_mocked):
    fetcher, mock_tracker = fetcher_mocked
    symbols_df = pd.DataFrame({
        "ticker": ["AAPL", "MSFT"],
        "exchange": ["NASDAQ", "NASDAQ"],
    })

    with patch.object(
        fetcher,
        "collect_single",
        side_effect=[RuntimeError("API down"), _make_ohlcv_df(3)],
    ) as mock_cs:
        fetcher._collect_batch(symbols_df, market="us")

        assert mock_cs.call_count == 2
        # First symbol errored → mark_error should be called
        mock_tracker.mark_error.assert_called_once()
