import pytest
from datetime import datetime

from src.backtest.strategy import StrategyConfig
from src.backtest.executor.base import OptionContract, Position


@pytest.fixture
def default_config():
    return StrategyConfig(threshold=0.3, tp_pct=0.10, sl_pct=-0.05)


@pytest.fixture
def sample_contract():
    return OptionContract(
        symbol="AAPL",
        strike=150.0,
        expiry=datetime(2026, 3, 28),
        option_type="put",
    )


@pytest.fixture
def open_position(sample_contract):
    """entry_price=3.0, current_price=3.0 (no PnL)."""
    return Position(
        contract=sample_contract,
        quantity=1,
        avg_entry_price=3.0,
        current_price=3.0,
        unrealized_pnl_pct=0.0,
    )


@pytest.fixture
def profitable_position(sample_contract):
    """unrealized_pnl_pct=+0.12 (above default TP of 0.10)."""
    return Position(
        contract=sample_contract,
        quantity=1,
        avg_entry_price=3.0,
        current_price=3.36,
        unrealized_pnl_pct=0.12,
    )


@pytest.fixture
def losing_position(sample_contract):
    """unrealized_pnl_pct=-0.06 (below default SL of -0.05)."""
    return Position(
        contract=sample_contract,
        quantity=1,
        avg_entry_price=3.0,
        current_price=2.82,
        unrealized_pnl_pct=-0.06,
    )


@pytest.fixture
def sample_bar():
    return {
        "close": 150.0,
        "peak_prob": 0.2,
        "trough_prob": 0.2,
        "minutes_from_open": 60,
    }


@pytest.fixture
def peak_bar():
    return {
        "close": 150.0,
        "peak_prob": 0.5,
        "trough_prob": 0.1,
        "minutes_from_open": 60,
    }


@pytest.fixture
def trough_bar():
    return {
        "close": 150.0,
        "peak_prob": 0.1,
        "trough_prob": 0.5,
        "minutes_from_open": 60,
    }


@pytest.fixture
def neutral_bar():
    return {
        "close": 150.0,
        "peak_prob": 0.1,
        "trough_prob": 0.1,
        "minutes_from_open": 60,
    }
