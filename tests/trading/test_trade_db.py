"""Tests for TradeDB SQLite persistence."""

import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from src.trading.broker.base import (
    OptionContract,
    Order,
    OrderSide,
    OrderStatus,
)
from src.trading.trade_db import TradeDB


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test_trading.db"


@pytest.fixture
def db(db_path):
    d = TradeDB(db_path)
    yield d
    d.close()


@pytest.fixture
def sample_order():
    contract = OptionContract(
        symbol="5930",
        expiry=datetime(2026, 2, 28, 15, 30),
        strike=70000.0,
        option_type="put",
    )
    return Order(
        side=OrderSide.BUY,
        contract=contract,
        quantity=2,
        fill_price=1200.0,
        fill_time=datetime(2026, 2, 21, 9, 15),
        status=OrderStatus.FILLED,
    )


class TestCreateTables:
    def test_tables_exist(self, db, db_path):
        conn = sqlite3.connect(str(db_path))
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {t[0] for t in tables}
        assert "trades" in table_names
        assert "daily_summary" in table_names
        conn.close()

    def test_creates_parent_dirs(self, tmp_path):
        deep_path = tmp_path / "a" / "b" / "c" / "test.db"
        d = TradeDB(deep_path)
        assert deep_path.exists()
        d.close()


class TestRecordTrade:
    def test_insert_and_read(self, db, db_path, sample_order):
        db.record_trade("2026-02-21", "kr", "5930", sample_order, "PEAK_SIGNAL")

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT * FROM trades").fetchall()
        conn.close()

        assert len(rows) == 1
        row = rows[0]
        assert row[1] == "2026-02-21"  # session_date
        assert row[2] == "kr"  # market
        assert row[3] == "5930"  # symbol
        assert row[5] == "BUY"  # side
        assert row[6] == 70000.0  # strike
        assert row[8] == 2  # quantity
        assert row[9] == 1200.0  # fill_price
        assert row[10] == "PEAK_SIGNAL"  # reason

    def test_multiple_trades(self, db, db_path, sample_order):
        db.record_trade("2026-02-21", "kr", "5930", sample_order, "PEAK_SIGNAL")
        db.record_trade("2026-02-21", "kr", "5930", sample_order, "STOP_LOSS")

        conn = sqlite3.connect(str(db_path))
        count = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        conn.close()
        assert count == 2

    def test_sell_order(self, db, db_path):
        contract = OptionContract(
            symbol="660", expiry=datetime(2026, 2, 28), strike=50000.0, option_type="put"
        )
        sell = Order(
            side=OrderSide.SELL, contract=contract, quantity=1,
            fill_price=1500.0, fill_time=datetime(2026, 2, 21, 9, 30),
            status=OrderStatus.FILLED,
        )
        db.record_trade("2026-02-21", "kr", "660", sell, "TROUGH_SIGNAL")

        conn = sqlite3.connect(str(db_path))
        row = conn.execute("SELECT side, reason FROM trades").fetchone()
        conn.close()
        assert row == ("SELL", "TROUGH_SIGNAL")


class TestRecordDailySummary:
    def test_insert_and_read(self, db, db_path):
        summary = {
            "buys": 3,
            "sells": 2,
            "net_pnl": -500.0,
            "peak_signals": 5,
            "trough_signals": 4,
        }
        db.record_daily_summary("2026-02-21", "kr", "5930", summary)

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT * FROM daily_summary").fetchall()
        conn.close()

        assert len(rows) == 1
        row = rows[0]
        assert row[1] == "2026-02-21"
        assert row[4] == 3  # buys
        assert row[5] == 2  # sells
        assert row[6] == -500.0  # net_pnl

    def test_missing_keys_default_zero(self, db, db_path):
        db.record_daily_summary("2026-02-21", "kr", "5930", {})

        conn = sqlite3.connect(str(db_path))
        row = conn.execute("SELECT buys, sells, net_pnl FROM daily_summary").fetchone()
        conn.close()
        assert row == (0, 0, 0.0)

    def test_multi_symbol_summaries(self, db, db_path):
        db.record_daily_summary("2026-02-21", "kr", "5930", {"buys": 1})
        db.record_daily_summary("2026-02-21", "kr", "660", {"buys": 2})

        conn = sqlite3.connect(str(db_path))
        count = conn.execute("SELECT COUNT(*) FROM daily_summary").fetchone()[0]
        conn.close()
        assert count == 2
