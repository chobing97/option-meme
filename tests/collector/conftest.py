"""Collector-specific test fixtures."""

import sqlite3
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.collector.collection_tracker import CollectionTracker


@pytest.fixture
def mock_tv_client():
    """Create a TVClient instance with mocked TvDatafeed and time."""
    with (
        patch("src.collector.tv_client.TvDatafeed") as mock_tvdf_cls,
        patch("src.collector.tv_client.time") as mock_time,
    ):
        mock_tvdf_cls.return_value = MagicMock()
        mock_time.time.return_value = 0.0
        mock_time.sleep = MagicMock()

        from src.collector.tv_client import TVClient

        client = TVClient()
        yield client, mock_tvdf_cls, mock_time


@pytest.fixture
def tracker_in_memory(tmp_path):
    """CollectionTracker backed by a temp-file SQLite (avoids global DB_PATH)."""
    db_path = tmp_path / "test_collection.db"
    tracker = CollectionTracker(db_path=db_path)
    yield tracker
    tracker.close()


@pytest.fixture
def yfinance_response_df() -> pd.DataFrame:
    """yfinance .history() style DataFrame (uppercase columns, tz-aware index)."""
    dates = pd.date_range("2025-01-02 09:00", periods=5, freq="min", tz="America/New_York")
    rng = np.random.RandomState(7)
    return pd.DataFrame(
        {
            "Open": 100 + rng.randn(5),
            "High": 101 + rng.rand(5),
            "Low": 99 + rng.rand(5),
            "Close": 100 + rng.randn(5),
            "Volume": rng.randint(1000, 5000, size=5),
            "Dividends": 0.0,
            "Stock Splits": 0.0,
        },
        index=dates,
    )


@pytest.fixture
def pykrx_response_df() -> pd.DataFrame:
    """pykrx style DataFrame (Korean column names)."""
    dates = pd.date_range("2025-01-02", periods=3, freq="B")
    rng = np.random.RandomState(8)
    return pd.DataFrame(
        {
            "시가": 50000 + rng.randint(-500, 500, size=3),
            "고가": 50500 + rng.randint(0, 500, size=3),
            "저가": 49500 + rng.randint(-500, 0, size=3),
            "종가": 50000 + rng.randint(-500, 500, size=3),
            "거래량": rng.randint(100000, 500000, size=3),
        },
        index=dates,
    )
