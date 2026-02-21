"""SQLite-based collection progress tracker for resumable data collection."""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

from config.settings import METADATA_DIR

DB_PATH = METADATA_DIR / "collection.db"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS collection_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    exchange TEXT NOT NULL,
    market TEXT NOT NULL,          -- 'kr' or 'us'
    source TEXT NOT NULL,          -- 'tvdatafeed', 'yfinance', 'pykrx'
    start_date TEXT,               -- earliest date in collected data
    end_date TEXT,                 -- latest date in collected data
    bar_count INTEGER DEFAULT 0,
    last_collected_at TEXT,
    status TEXT DEFAULT 'pending', -- 'pending', 'partial', 'complete', 'error'
    error_message TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    UNIQUE(symbol, exchange, source)
);
"""

CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_collection_symbol ON collection_log(symbol, exchange);
CREATE INDEX IF NOT EXISTS idx_collection_status ON collection_log(status);
CREATE INDEX IF NOT EXISTS idx_collection_market ON collection_log(market);
"""


class CollectionTracker:
    """Track data collection progress per symbol in SQLite."""

    def __init__(self, db_path: Path = DB_PATH):
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        cur = self._conn.cursor()
        cur.executescript(CREATE_TABLE_SQL + CREATE_INDEX_SQL)
        self._conn.commit()
        logger.debug(f"Collection tracker DB initialized at {self._db_path}")

    def get_status(
        self, symbol: str, exchange: str, source: str
    ) -> Optional[dict]:
        """Get collection status for a symbol/source combination."""
        cur = self._conn.cursor()
        cur.execute(
            "SELECT * FROM collection_log WHERE symbol=? AND exchange=? AND source=?",
            (symbol, exchange, source),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def upsert(
        self,
        symbol: str,
        exchange: str,
        market: str,
        source: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        bar_count: int = 0,
        status: str = "partial",
        error_message: Optional[str] = None,
    ) -> None:
        """Insert or update collection record."""
        now = datetime.utcnow().isoformat()
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO collection_log
                (symbol, exchange, market, source, start_date, end_date,
                 bar_count, last_collected_at, status, error_message, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, exchange, source) DO UPDATE SET
                start_date = COALESCE(?, start_date),
                end_date = COALESCE(?, end_date),
                bar_count = ?,
                last_collected_at = ?,
                status = ?,
                error_message = ?,
                updated_at = ?
            """,
            (
                symbol, exchange, market, source, start_date, end_date,
                bar_count, now, status, error_message, now,
                start_date, end_date, bar_count, now, status, error_message, now,
            ),
        )
        self._conn.commit()

    def mark_complete(self, symbol: str, exchange: str, source: str) -> None:
        """Mark a symbol as fully collected."""
        cur = self._conn.cursor()
        now = datetime.utcnow().isoformat()
        cur.execute(
            """UPDATE collection_log SET status='complete', updated_at=?
               WHERE symbol=? AND exchange=? AND source=?""",
            (now, symbol, exchange, source),
        )
        self._conn.commit()

    def mark_error(
        self, symbol: str, exchange: str, source: str, error: str
    ) -> None:
        """Mark a symbol as errored."""
        cur = self._conn.cursor()
        now = datetime.utcnow().isoformat()
        cur.execute(
            """UPDATE collection_log SET status='error', error_message=?, updated_at=?
               WHERE symbol=? AND exchange=? AND source=?""",
            (error, now, symbol, exchange, source),
        )
        self._conn.commit()

    def get_pending_symbols(self, market: str, source: str) -> list[dict]:
        """Get symbols that haven't been fully collected yet."""
        cur = self._conn.cursor()
        cur.execute(
            """SELECT * FROM collection_log
               WHERE market=? AND source=? AND status != 'complete'
               ORDER BY symbol""",
            (market, source),
        )
        return [dict(r) for r in cur.fetchall()]

    def get_all_symbols(self, market: Optional[str] = None) -> list[dict]:
        """Get all tracked symbols, optionally filtered by market."""
        cur = self._conn.cursor()
        if market:
            cur.execute(
                "SELECT * FROM collection_log WHERE market=? ORDER BY symbol",
                (market,),
            )
        else:
            cur.execute("SELECT * FROM collection_log ORDER BY market, symbol")
        return [dict(r) for r in cur.fetchall()]

    def summary(self) -> dict:
        """Get collection summary statistics."""
        cur = self._conn.cursor()
        cur.execute(
            """SELECT market, source, status, COUNT(*) as cnt, SUM(bar_count) as total_bars
               FROM collection_log GROUP BY market, source, status"""
        )
        return [dict(r) for r in cur.fetchall()]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
