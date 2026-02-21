"""Tests for src.collector.stock_info_db — in-memory SQLite."""

import pytest

from src.collector.stock_info_db import StockInfoDB


@pytest.fixture
def db(tmp_path):
    """StockInfoDB backed by a temp-file SQLite."""
    db_path = tmp_path / "test_stock_info.db"
    sdb = StockInfoDB(db_path=db_path)
    yield sdb
    sdb.close()


# ── init ────────────────────────────────────────────────────


def test_init_creates_table(db):
    cur = db._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='stock_info'"
    )
    assert cur.fetchone() is not None


# ── upsert / get ────────────────────────────────────────────


def test_upsert_insert(db):
    db.upsert("AAPL", "us", {
        "name": "Apple Inc.",
        "exchange": "NMS",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "currency": "USD",
        "market_cap": 3_000_000_000_000,
    })
    row = db.get("AAPL", "us")
    assert row is not None
    assert row["name"] == "Apple Inc."
    assert row["sector"] == "Technology"
    assert row["market_cap"] == 3_000_000_000_000
    assert row["updated_at"] is not None


def test_upsert_update(db):
    db.upsert("AAPL", "us", {
        "name": "Apple Inc.",
        "sector": "Technology",
        "market_cap": 3_000_000_000_000,
    })
    db.upsert("AAPL", "us", {
        "name": "Apple Inc.",
        "sector": "Tech",
        "market_cap": 3_500_000_000_000,
    })
    row = db.get("AAPL", "us")
    assert row["sector"] == "Tech"
    assert row["market_cap"] == 3_500_000_000_000

    # Only 1 row
    cur = db._conn.execute("SELECT COUNT(*) FROM stock_info WHERE symbol='AAPL'")
    assert cur.fetchone()[0] == 1


def test_upsert_missing_fields_default(db):
    """Missing fields should default to empty string / 0."""
    db.upsert("TSLA", "us", {})
    row = db.get("TSLA", "us")
    assert row["name"] == ""
    assert row["sector"] == ""
    assert row["market_cap"] == 0


def test_get_nonexistent(db):
    assert db.get("ZZZZ", "us") is None


# ── get_all ─────────────────────────────────────────────────


def test_get_all_no_filter(db):
    db.upsert("AAPL", "us", {"name": "Apple"})
    db.upsert("005930", "kr", {"name": "Samsung"})
    all_rows = db.get_all()
    assert len(all_rows) == 2


def test_get_all_market_filter(db):
    db.upsert("AAPL", "us", {"name": "Apple"})
    db.upsert("MSFT", "us", {"name": "Microsoft"})
    db.upsert("005930", "kr", {"name": "Samsung"})

    us_rows = db.get_all(market="us")
    assert len(us_rows) == 2
    assert all(r["market"] == "us" for r in us_rows)

    kr_rows = db.get_all(market="kr")
    assert len(kr_rows) == 1
    assert kr_rows[0]["symbol"] == "005930"


# ── composite key ───────────────────────────────────────────


def test_same_symbol_different_market(db):
    """Same symbol in different markets should be separate rows."""
    db.upsert("SPY", "us", {"name": "SPDR S&P 500"})
    db.upsert("SPY", "kr", {"name": "SPY (KR listed)"})
    assert db.get("SPY", "us")["name"] == "SPDR S&P 500"
    assert db.get("SPY", "kr")["name"] == "SPY (KR listed)"
    assert len(db.get_all()) == 2
