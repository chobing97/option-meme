"""SQLite trade history and daily summary storage."""

import sqlite3
from datetime import datetime
from pathlib import Path

from config.settings import TRADE_DB_DIR
from src.trading.broker.base import Order


class TradeDB:
    """Persist trade records and daily summaries to SQLite."""

    def __init__(self, db_path: Path | None = None):
        if db_path is None:
            db_path = TRADE_DB_DIR / "trading.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._conn = sqlite3.connect(str(db_path))
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY,
                session_date TEXT,
                market TEXT,
                symbol TEXT,
                timestamp TEXT,
                side TEXT,
                strike REAL,
                expiry TEXT,
                quantity INTEGER,
                fill_price REAL,
                reason TEXT
            );

            CREATE TABLE IF NOT EXISTS daily_summary (
                id INTEGER PRIMARY KEY,
                session_date TEXT,
                market TEXT,
                symbol TEXT,
                buys INTEGER,
                sells INTEGER,
                net_pnl REAL,
                peak_signals INTEGER,
                trough_signals INTEGER
            );
            """
        )
        self._conn.commit()

    def record_trade(
        self,
        session_date: str,
        market: str,
        symbol: str,
        order: Order,
        reason: str,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO trades
                (session_date, market, symbol, timestamp, side,
                 strike, expiry, quantity, fill_price, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_date,
                market,
                symbol,
                order.fill_time.isoformat() if order.fill_time else "",
                order.side.value,
                order.contract.strike,
                order.contract.expiry.strftime("%Y-%m-%d"),
                order.quantity,
                order.fill_price,
                reason,
            ),
        )
        self._conn.commit()

    def record_daily_summary(
        self,
        session_date: str,
        market: str,
        symbol: str,
        summary: dict,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO daily_summary
                (session_date, market, symbol, buys, sells,
                 net_pnl, peak_signals, trough_signals)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_date,
                market,
                symbol,
                summary.get("buys", 0),
                summary.get("sells", 0),
                summary.get("net_pnl", 0.0),
                summary.get("peak_signals", 0),
                summary.get("trough_signals", 0),
            ),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
