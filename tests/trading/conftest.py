"""Trading test fixtures."""

from collections import deque
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.trading.broker.base import (
    OptionContract,
    Order,
    OrderSide,
    OrderStatus,
    Signal,
    SignalType,
)
from src.trading.broker.mock_broker import MockBroker
from src.trading.datafeed.base import DataFeed
from src.trading.signal_detector import BarAccumulator


# ── Stub DataFeed ────────────────────────────────────


class StubFeed(DataFeed):
    """Minimal DataFeed that replays a list of bars."""

    def __init__(self, bars: list[pd.Series], market: str = "kr", symbol: str = "TEST"):
        self.market = market
        self.symbol = symbol
        self.replay_date = "2026-01-05"
        self._bars = deque(bars)
        self._connected = False

    def connect(self) -> None:
        self._connected = True

    def get_latest_bar(self):
        if not self._bars:
            return None
        return self._bars.popleft()

    def get_history(self, n_days: int = 5) -> pd.DataFrame:
        return _make_history_df(n_days=3)

    def is_session_active(self) -> bool:
        return self._connected and len(self._bars) > 0

    def disconnect(self) -> None:
        self._connected = False


# ── Stub SignalDetector ──────────────────────────────


class StubDetector:
    """Fake SignalDetector that returns pre-configured signals per bar index."""

    def __init__(self, signals: list[Signal]):
        self._signals = list(signals)
        self._idx = 0
        self.model_type = "gbm"
        self.threshold = 0.5

    def detect(self, accumulator):
        if self._idx < len(self._signals):
            sig = self._signals[self._idx]
            self._idx += 1
            return sig
        return Signal(
            signal_type=SignalType.NONE,
            timestamp=datetime.now(),
            close_price=50000.0,
        )


# ── Helper functions ─────────────────────────────────


def _make_history_df(n_days: int = 3, bars_per_day: int = 10) -> pd.DataFrame:
    """Create a history DataFrame for BarAccumulator."""
    rng = np.random.RandomState(42)
    rows = []
    base_date = datetime(2026, 1, 2, 9, 0)
    for d in range(n_days):
        day = base_date + timedelta(days=d)
        for b in range(bars_per_day):
            dt = day + timedelta(minutes=b)
            rows.append({
                "datetime": dt,
                "date": day.date(),
                "open": 50000 + rng.randn() * 100,
                "high": 50200 + rng.rand() * 100,
                "low": 49800 + rng.rand() * 100,
                "close": 50000 + rng.randn() * 100,
                "volume": int(rng.randint(1000, 50000)),
                "minutes_from_open": b,
            })
    return pd.DataFrame(rows)


def make_bar(dt: datetime, close: float = 50000.0) -> pd.Series:
    """Create a single bar Series."""
    return pd.Series({
        "datetime": dt,
        "date": dt.date(),
        "open": close - 10,
        "high": close + 50,
        "low": close - 50,
        "close": close,
        "volume": 10000,
        "minutes_from_open": dt.minute - 0,
    })


def make_bars(n: int, start: datetime = None, base_close: float = 50000.0) -> list[pd.Series]:
    """Create n bars with 1-minute intervals."""
    if start is None:
        start = datetime(2026, 1, 5, 9, 0)
    return [
        make_bar(start + timedelta(minutes=i), base_close + i * 10)
        for i in range(n)
    ]


# ── Fixtures ─────────────────────────────────────────


@pytest.fixture
def broker():
    """Connected MockBroker with 10M capital."""
    b = MockBroker(capital=10_000_000)
    b.connect()
    return b


@pytest.fixture
def broker_small():
    """Connected MockBroker with very small capital for rejection tests."""
    b = MockBroker(capital=100)
    b.connect()
    return b


@pytest.fixture
def sample_contract():
    """ATM put contract."""
    return OptionContract(
        symbol="TEST",
        expiry=datetime(2026, 1, 31, 15, 30),
        strike=50000.0,
        option_type="put",
    )
