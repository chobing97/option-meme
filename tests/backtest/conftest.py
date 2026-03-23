import pytest
from datetime import datetime

import pandas as pd

from src.backtest.strategy import PutBuyConfig as StrategyConfig
from src.backtest.executor.base import OptionContract, Position
from src.backtest.types import PortfolioState


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


# Legacy dict-based bar fixtures (kept for backward compatibility)
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


# New pd.Series-based row fixtures
@pytest.fixture
def sample_row():
    return pd.Series({"symbol": "AAPL", "close": 150.0, "peak_prob": 0.2,
                       "trough_prob": 0.2, "minutes_from_open": 60})


@pytest.fixture
def peak_row():
    return pd.Series({"symbol": "AAPL", "close": 150.0, "peak_prob": 0.5,
                       "trough_prob": 0.1, "minutes_from_open": 60})


@pytest.fixture
def trough_row():
    return pd.Series({"symbol": "AAPL", "close": 150.0, "peak_prob": 0.1,
                       "trough_prob": 0.5, "minutes_from_open": 60})


@pytest.fixture
def neutral_row():
    return pd.Series({"symbol": "AAPL", "close": 150.0, "peak_prob": 0.1,
                       "trough_prob": 0.1, "minutes_from_open": 60})


# Portfolio fixtures
@pytest.fixture
def portfolio_with_position(open_position):
    return PortfolioState(cash=100_000, positions=[open_position], equity=100_300)


@pytest.fixture
def portfolio_with_profitable(profitable_position):
    return PortfolioState(cash=100_000, positions=[profitable_position], equity=100_336)


@pytest.fixture
def portfolio_with_losing(losing_position):
    return PortfolioState(cash=100_000, positions=[losing_position], equity=100_282)


@pytest.fixture
def empty_portfolio():
    return PortfolioState(cash=100_000, positions=[], equity=100_000)


@pytest.fixture
def default_context():
    return {"session_minutes": 390, "bar_index": 0, "timestamp": datetime(2026, 3, 21, 10, 30)}
