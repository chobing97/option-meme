"""Tests for src.collector.collection_tracker — in-memory SQLite."""

import sqlite3

import pytest

from src.collector.collection_tracker import CollectionTracker


# ── init ────────────────────────────────────────────────────


def test_init_creates_table(tracker_in_memory):
    cur = tracker_in_memory._conn.cursor()
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='collection_log'"
    )
    assert cur.fetchone() is not None

    # Verify indices exist
    cur.execute("SELECT name FROM sqlite_master WHERE type='index'")
    index_names = {row[0] for row in cur.fetchall()}
    assert "idx_collection_symbol" in index_names
    assert "idx_collection_status" in index_names
    assert "idx_collection_market" in index_names


# ── upsert / get_status ────────────────────────────────────


def test_upsert_insert(tracker_in_memory):
    tracker_in_memory.upsert(
        symbol="005930", exchange="KRX", market="kr",
        source="yfinance", start_date="2025-01-01", end_date="2025-01-10",
        bar_count=500, status="partial",
    )
    status = tracker_in_memory.get_status("005930", "KRX", "yfinance")
    assert status is not None
    assert status["symbol"] == "005930"
    assert status["bar_count"] == 500
    assert status["status"] == "partial"


def test_upsert_update(tracker_in_memory):
    tracker_in_memory.upsert(
        symbol="005930", exchange="KRX", market="kr",
        source="yfinance", bar_count=500, status="partial",
    )
    tracker_in_memory.upsert(
        symbol="005930", exchange="KRX", market="kr",
        source="yfinance", bar_count=1000, status="complete",
    )
    status = tracker_in_memory.get_status("005930", "KRX", "yfinance")
    assert status["bar_count"] == 1000
    assert status["status"] == "complete"

    # Only 1 row for this key
    cur = tracker_in_memory._conn.cursor()
    cur.execute(
        "SELECT COUNT(*) FROM collection_log WHERE symbol='005930' AND source='yfinance'"
    )
    assert cur.fetchone()[0] == 1


def test_upsert_coalesce_null(tracker_in_memory):
    """When start_date=None on update, existing value should be preserved (COALESCE)."""
    tracker_in_memory.upsert(
        symbol="AAPL", exchange="NASDAQ", market="us",
        source="tvdatafeed", start_date="2025-01-01", end_date="2025-01-10",
        bar_count=100,
    )
    # Update with start_date=None → should keep "2025-01-01"
    tracker_in_memory.upsert(
        symbol="AAPL", exchange="NASDAQ", market="us",
        source="tvdatafeed", start_date=None, end_date="2025-01-15",
        bar_count=200,
    )
    status = tracker_in_memory.get_status("AAPL", "NASDAQ", "tvdatafeed")
    assert status["start_date"] == "2025-01-01"
    assert status["end_date"] == "2025-01-15"
    assert status["bar_count"] == 200


def test_get_status_nonexistent(tracker_in_memory):
    result = tracker_in_memory.get_status("ZZZZ", "NOWHERE", "yfinance")
    assert result is None


# ── mark_complete / mark_error ──────────────────────────────


def test_mark_complete(tracker_in_memory):
    tracker_in_memory.upsert(
        symbol="005930", exchange="KRX", market="kr",
        source="yfinance", status="partial",
    )
    tracker_in_memory.mark_complete("005930", "KRX", "yfinance")
    status = tracker_in_memory.get_status("005930", "KRX", "yfinance")
    assert status["status"] == "complete"


def test_mark_error(tracker_in_memory):
    tracker_in_memory.upsert(
        symbol="005930", exchange="KRX", market="kr",
        source="yfinance", status="partial",
    )
    tracker_in_memory.mark_error("005930", "KRX", "yfinance", "Connection timeout")
    status = tracker_in_memory.get_status("005930", "KRX", "yfinance")
    assert status["status"] == "error"
    assert status["error_message"] == "Connection timeout"


# ── get_pending_symbols ─────────────────────────────────────


def test_get_pending_symbols(tracker_in_memory):
    tracker_in_memory.upsert(
        symbol="005930", exchange="KRX", market="kr",
        source="yfinance", status="complete",
    )
    tracker_in_memory.upsert(
        symbol="000660", exchange="KRX", market="kr",
        source="yfinance", status="partial",
    )
    tracker_in_memory.upsert(
        symbol="035720", exchange="KRX", market="kr",
        source="yfinance", status="error",
    )

    pending = tracker_in_memory.get_pending_symbols(market="kr", source="yfinance")
    symbols = [r["symbol"] for r in pending]
    assert len(pending) == 2
    assert "000660" in symbols
    assert "035720" in symbols
    assert "005930" not in symbols  # complete → excluded


# ── summary ─────────────────────────────────────────────────


def test_summary_aggregation(tracker_in_memory):
    tracker_in_memory.upsert(
        symbol="005930", exchange="KRX", market="kr",
        source="yfinance", bar_count=500, status="complete",
    )
    tracker_in_memory.upsert(
        symbol="000660", exchange="KRX", market="kr",
        source="yfinance", bar_count=300, status="partial",
    )
    tracker_in_memory.upsert(
        symbol="AAPL", exchange="NASDAQ", market="us",
        source="tvdatafeed", bar_count=1000, status="complete",
    )

    summary = tracker_in_memory.summary()
    assert len(summary) >= 2  # at least 2 groups

    # Find kr/yfinance/complete group
    kr_yf_complete = [
        s for s in summary
        if s["market"] == "kr" and s["source"] == "yfinance" and s["status"] == "complete"
    ]
    assert len(kr_yf_complete) == 1
    assert kr_yf_complete[0]["cnt"] == 1
    assert kr_yf_complete[0]["total_bars"] == 500
