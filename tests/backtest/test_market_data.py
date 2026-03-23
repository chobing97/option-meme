"""Tests for BacktestMarketData — data loading and quote lookups."""

import pytest
from datetime import datetime

import pandas as pd

from src.backtest.executor.base import OptionContract
from src.backtest.market_data import BacktestMarketData
from src.backtest.types import Quote


# ── Fixtures ──────────────────────────────────────────


@pytest.fixture
def mock_options_dir(tmp_path):
    """Create mock options data structure for AAPL."""
    sym_dir = tmp_path / "us" / "AAPL"
    sym_dir.mkdir(parents=True)

    # contracts.parquet
    contracts = pd.DataFrame(
        {
            "symbol": ["AAPL  260109P00270000"],
            "underlying": ["AAPL"],
            "expiry": [pd.Timestamp("2026-01-09")],
            "cp": ["P"],
            "strike": [270.0],
            "period_start": [pd.Timestamp("2026-01-02")],
            "stock_close": [272.0],
        }
    )
    contracts.to_parquet(sym_dir / "contracts.parquet", index=False)

    # OHLCV parquet with 5 bars
    bars = pd.DataFrame(
        {
            "datetime": pd.date_range("2026-01-02 09:30", periods=5, freq="min"),
            "symbol": ["AAPL  260109P00270000"] * 5,
            "open": [2.50, 2.55, 2.60, 2.45, 2.40],
            "high": [2.60, 2.65, 2.70, 2.55, 2.50],
            "low": [2.40, 2.45, 2.50, 2.35, 2.30],
            "close": [2.55, 2.60, 2.65, 2.50, 2.45],
            "volume": [100, 200, 150, 0, 100],
            "source": ["databento"] * 5,
        }
    )
    bars.to_parquet(sym_dir / "2026.parquet", index=False)

    return tmp_path


@pytest.fixture
def market_data(mock_options_dir):
    """BacktestMarketData with AAPL data loaded."""
    md = BacktestMarketData(
        symbols=["AAPL"],
        market="us",
        data_dir=mock_options_dir,
    )
    md.load_data()
    return md


@pytest.fixture
def sample_contract():
    """The AAPL put contract matching mock data."""
    return OptionContract(
        symbol="AAPL",
        strike=270.0,
        expiry=datetime(2026, 1, 9),
        option_type="put",
        contract_id="AAPL  260109P00270000",
    )


# ── load_data ─────────────────────────────────────────


class TestLoadData:
    def test_load_data_success(self, market_data):
        """Contracts and OHLCV loaded, symbol present."""
        assert "AAPL" in market_data._contracts
        assert "AAPL" in market_data._ohlcv
        assert len(market_data._contracts["AAPL"]) == 1
        assert len(market_data._ohlcv["AAPL"]) == 5

    def test_load_missing_symbol(self, mock_options_dir):
        """Missing symbol logs warning, no error."""
        md = BacktestMarketData(
            symbols=["NONEXIST"],
            market="us",
            data_dir=mock_options_dir,
        )
        md.load_data()  # should not raise
        assert "NONEXIST" not in md._contracts
        assert "NONEXIST" not in md._ohlcv


# ── get_option_chain ──────────────────────────────────


class TestGetOptionChain:
    def test_chain_normal(self, market_data):
        """Active contract returned for valid timestamp."""
        ts = datetime(2026, 1, 5, 10, 0)
        chain = market_data.get_option_chain("AAPL", "put", ts)
        assert len(chain) == 1
        assert chain[0].strike == 270.0
        assert chain[0].option_type == "put"

    def test_chain_expired(self, market_data):
        """Expired contract not returned (timestamp.date >= expiry)."""
        ts = datetime(2026, 1, 10, 10, 0)
        chain = market_data.get_option_chain("AAPL", "put", ts)
        assert len(chain) == 0

    def test_chain_empty_before_period(self, market_data):
        """No active contracts before period_start."""
        ts = datetime(2025, 12, 31, 10, 0)
        chain = market_data.get_option_chain("AAPL", "put", ts)
        assert len(chain) == 0

    def test_chain_empty_unknown_symbol(self, market_data):
        """Unknown symbol returns empty list."""
        ts = datetime(2026, 1, 5, 10, 0)
        chain = market_data.get_option_chain("MSFT", "put", ts)
        assert chain == []

    def test_chain_wrong_option_type(self, market_data):
        """Only put contracts — asking for call returns empty."""
        ts = datetime(2026, 1, 5, 10, 0)
        chain = market_data.get_option_chain("AAPL", "call", ts)
        assert len(chain) == 0


# ── get_option_quote ──────────────────────────────────


class TestGetOptionQuote:
    def test_quote_asof(self, market_data, sample_contract):
        """Asof lookup returns Quote with correct fields."""
        ts = datetime(2026, 1, 2, 9, 31)
        quote = market_data.get_option_quote(sample_contract, ts)

        assert quote is not None
        assert isinstance(quote, Quote)
        assert quote.close == pytest.approx(2.60)
        assert quote.high == pytest.approx(2.65)
        assert quote.low == pytest.approx(2.45)
        assert quote.volume == 200

    def test_quote_no_data(self, market_data, sample_contract):
        """Returns None when no data before timestamp."""
        ts = datetime(2025, 1, 1, 9, 30)
        quote = market_data.get_option_quote(sample_contract, ts)
        assert quote is None

    def test_quote_unknown_contract(self, market_data):
        """Returns None for unknown contract."""
        unknown = OptionContract(
            symbol="AAPL",
            strike=999.0,
            expiry=datetime(2026, 1, 9),
            option_type="put",
            contract_id="NONEXIST",
        )
        ts = datetime(2026, 1, 2, 9, 31)
        quote = market_data.get_option_quote(unknown, ts)
        assert quote is None


# ── get_stock_quote ───────────────────────────────────


class TestGetStockQuote:
    def test_stock_quote_returns_none(self, market_data):
        """Backtesting has no stock data — always returns None."""
        ts = datetime(2026, 1, 2, 9, 31)
        quote = market_data.get_stock_quote("AAPL", ts)
        assert quote is None
