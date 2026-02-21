"""Tests for src.collector.tv_client — TvDatafeed and time mocked."""

from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest


# ── Helpers ─────────────────────────────────────────────────


def _make_tv_response(n: int = 5) -> pd.DataFrame:
    """Create a fake TvDatafeed.get_hist() response."""
    dates = pd.date_range("2025-01-02 09:00", periods=n, freq="min")
    return pd.DataFrame(
        {
            "open": range(n),
            "high": range(n),
            "low": range(n),
            "close": range(n),
            "volume": range(n),
            "symbol": ["TEST"] * n,  # extra column from tvDatafeed
        },
        index=dates,
    )


# ── Tests ───────────────────────────────────────────────────


def test_get_hist_success():
    """Normal response → OHLCV columns only, DatetimeIndex."""
    with (
        patch("src.collector.tv_client.TvDatafeed") as mock_cls,
        patch("src.collector.tv_client.time") as mock_time,
    ):
        mock_instance = MagicMock()
        mock_instance.get_hist.return_value = _make_tv_response(5)
        mock_cls.return_value = mock_instance
        mock_time.time.return_value = 0.0
        mock_time.sleep = MagicMock()

        from src.collector.tv_client import TVClient

        client = TVClient()
        result = client.get_hist("005930", "KRX")

        assert result is not None
        assert list(result.columns) == ["open", "high", "low", "close", "volume"]
        assert isinstance(result.index, pd.DatetimeIndex)
        assert len(result) == 5


def test_get_hist_retries_on_empty():
    """None/empty response × 3 attempts → final None, backoff values 2 and 4 present."""
    with (
        patch("src.collector.tv_client.TvDatafeed") as mock_cls,
        patch("src.collector.tv_client.time") as mock_time,
    ):
        mock_instance = MagicMock()
        mock_instance.get_hist.return_value = None
        mock_cls.return_value = mock_instance
        mock_time.time.return_value = 0.0
        mock_time.sleep = MagicMock()

        from src.collector.tv_client import TVClient

        client = TVClient()
        result = client.get_hist("005930", "KRX")

        assert result is None
        assert mock_instance.get_hist.call_count == 3
        # 3 rate-limit sleeps + 2 backoff sleeps = 5 total
        all_sleeps = [c[0][0] for c in mock_time.sleep.call_args_list]
        assert len(all_sleeps) == 5
        # Backoff values (2^1=2, 2^2=4) must be present
        assert 2 in all_sleeps
        assert 4 in all_sleeps


def test_get_hist_exponential_backoff():
    """Exception × 3 → backoff sleeps of 2 and 4 seconds."""
    with (
        patch("src.collector.tv_client.TvDatafeed") as mock_cls,
        patch("src.collector.tv_client.time") as mock_time,
    ):
        mock_instance = MagicMock()
        mock_instance.get_hist.side_effect = RuntimeError("network error")
        mock_cls.return_value = mock_instance
        mock_time.time.return_value = 0.0
        mock_time.sleep = MagicMock()

        from src.collector.tv_client import TVClient

        client = TVClient()
        result = client.get_hist("005930", "KRX")

        assert result is None
        # Check backoff values: 2^1=2, 2^2=4
        all_sleeps = [c[0][0] for c in mock_time.sleep.call_args_list]
        assert 2 in all_sleeps
        assert 4 in all_sleeps


def test_session_error_reconnects():
    """Error containing 'session' → reconnect then retry succeeds."""
    with (
        patch("src.collector.tv_client.TvDatafeed") as mock_cls,
        patch("src.collector.tv_client.time") as mock_time,
    ):
        mock_instance = MagicMock()
        mock_instance.get_hist.side_effect = [
            RuntimeError("session expired"),
            _make_tv_response(3),
        ]
        mock_cls.return_value = mock_instance
        mock_time.time.return_value = 0.0
        mock_time.sleep = MagicMock()

        from src.collector.tv_client import TVClient

        client = TVClient()
        result = client.get_hist("005930", "KRX")

        assert result is not None
        assert len(result) == 3
        # TvDatafeed constructor called: 1 (init) + 1 (reconnect)
        assert mock_cls.call_count == 2


def test_rate_limit_sleep():
    """When elapsed < TV_RATE_LIMIT_SEC, sleep should fill the gap."""
    with (
        patch("src.collector.tv_client.TvDatafeed") as mock_cls,
        patch("src.collector.tv_client.time") as mock_time,
    ):
        mock_instance = MagicMock()
        mock_instance.get_hist.return_value = _make_tv_response(3)
        mock_cls.return_value = mock_instance

        # Simulate: last_request_time=10.0, now=11.0 → elapsed=1.0 < 2.0 → sleep(1.0)
        mock_time.time.side_effect = [11.0, 11.0]  # called in _rate_limit
        mock_time.sleep = MagicMock()

        from src.collector.tv_client import TVClient

        client = TVClient()
        client._last_request_time = 10.0
        client.get_hist("005930", "KRX")

        # First sleep call should be approximately 1.0 (rate limit)
        first_sleep = mock_time.sleep.call_args_list[0][0][0]
        assert 0.9 <= first_sleep <= 1.1


def test_n_bars_capped():
    """n_bars > 5000 → clamped to 5000 (TV_MAX_BARS)."""
    with (
        patch("src.collector.tv_client.TvDatafeed") as mock_cls,
        patch("src.collector.tv_client.time") as mock_time,
    ):
        mock_instance = MagicMock()
        mock_instance.get_hist.return_value = _make_tv_response(5)
        mock_cls.return_value = mock_instance
        mock_time.time.return_value = 0.0
        mock_time.sleep = MagicMock()

        from src.collector.tv_client import TVClient

        client = TVClient()
        client.get_hist("005930", "KRX", n_bars=99999)

        # Verify the actual call used n_bars=5000
        call_kwargs = mock_instance.get_hist.call_args
        assert call_kwargs.kwargs.get("n_bars", call_kwargs[1].get("n_bars")) == 5000
