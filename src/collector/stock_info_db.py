"""SQLite storage for stock basic information (sector, industry, etc.)."""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from loguru import logger

from config.settings import RAW_STOCK_DIR

DB_PATH = RAW_STOCK_DIR / "stock_info.db"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS stock_info (
    symbol TEXT NOT NULL,
    market TEXT NOT NULL,
    name TEXT DEFAULT '',
    exchange TEXT DEFAULT '',
    sector TEXT DEFAULT '',
    industry TEXT DEFAULT '',
    currency TEXT DEFAULT '',
    market_cap INTEGER DEFAULT 0,
    updated_at TEXT,
    PRIMARY KEY (symbol, market)
);
"""


class StockInfoDB:
    """Manage stock basic info in data/raw/stock_info.db."""

    def __init__(self, db_path: Path = DB_PATH):
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        self._conn.executescript(CREATE_TABLE_SQL)
        self._conn.commit()
        logger.debug(f"StockInfoDB initialized at {self._db_path}")

    def upsert(self, symbol: str, market: str, info_dict: dict) -> None:
        """Insert or update stock info."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT INTO stock_info
                (symbol, market, name, exchange, sector, industry, currency, market_cap, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, market) DO UPDATE SET
                name = ?,
                exchange = ?,
                sector = ?,
                industry = ?,
                currency = ?,
                market_cap = ?,
                updated_at = ?
            """,
            (
                symbol,
                market,
                info_dict.get("name", ""),
                info_dict.get("exchange", ""),
                info_dict.get("sector", ""),
                info_dict.get("industry", ""),
                info_dict.get("currency", ""),
                info_dict.get("market_cap", 0),
                now,
                # ON CONFLICT UPDATE values
                info_dict.get("name", ""),
                info_dict.get("exchange", ""),
                info_dict.get("sector", ""),
                info_dict.get("industry", ""),
                info_dict.get("currency", ""),
                info_dict.get("market_cap", 0),
                now,
            ),
        )
        self._conn.commit()

    def get(self, symbol: str, market: str) -> Optional[dict]:
        """Get stock info for a symbol."""
        cur = self._conn.execute(
            "SELECT * FROM stock_info WHERE symbol = ? AND market = ?",
            (symbol, market),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def get_all(self, market: Optional[str] = None) -> list[dict]:
        """Get all stock info, optionally filtered by market."""
        if market:
            cur = self._conn.execute(
                "SELECT * FROM stock_info WHERE market = ? ORDER BY symbol",
                (market,),
            )
        else:
            cur = self._conn.execute(
                "SELECT * FROM stock_info ORDER BY market, symbol"
            )
        return [dict(r) for r in cur.fetchall()]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
