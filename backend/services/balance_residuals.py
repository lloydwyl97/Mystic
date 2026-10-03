"""Fiat balance residuals outside the trading universe.

A small USD remainder is left after a fiat deposit is converted to USDT. It is
not tradable in the fixed universe and owns no position, so it is classified
once in ``documented_balance_residuals`` and compared against that record on
every balance sync. Any change from the recorded quantity is still reported.
"""

from __future__ import annotations

import logging
import sqlite3

logger = logging.getLogger(__name__)

FIAT_RESIDUAL_ASSETS = frozenset({"USD"})
# Larger fiat balances are unconverted deposits, not a residual.
FIAT_RESIDUAL_MAX = 1.0
FIAT_RESIDUAL_TOLERANCE = 1e-6
FIAT_RESIDUAL_NOTE = "fiat conversion remainder; outside the trading universe; owns no position"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documented_balance_residuals (
    symbol TEXT PRIMARY KEY,
    exchange_qty REAL NOT NULL,
    position_qty REAL NOT NULL,
    residual_qty REAL NOT NULL,
    note TEXT NOT NULL,
    updated_at TEXT DEFAULT (datetime('now'))
)
"""


def classify_fiat_residual(conn: sqlite3.Connection, asset: str, exchange_qty: float) -> str:
    """Return MATCHED, CHANGED, or "" when ``asset`` is not a fiat residual asset.

    First sight of a small fiat balance records it (MATCHED). Afterwards only the
    recorded quantity matches; any other quantity is CHANGED.
    """
    code = str(asset or "").upper()
    if code not in FIAT_RESIDUAL_ASSETS:
        return ""
    qty = float(exchange_qty or 0.0)
    conn.execute(_SCHEMA)
    row = conn.execute("SELECT residual_qty FROM documented_balance_residuals WHERE symbol=?", (code,)).fetchone()
    if row is None:
        if qty > FIAT_RESIDUAL_MAX:
            return "CHANGED"
        conn.execute(
            "INSERT INTO documented_balance_residuals(symbol, exchange_qty, position_qty, residual_qty, note) VALUES (?,?,?,?,?)",
            (code, qty, 0.0, qty, FIAT_RESIDUAL_NOTE),
        )
        logger.info("BALANCE_RESIDUAL_CLASSIFIED asset=%s qty=%.8f note=%s", code, qty, FIAT_RESIDUAL_NOTE)
        return "MATCHED"
    return "MATCHED" if abs(qty - float(row[0] or 0.0)) <= FIAT_RESIDUAL_TOLERANCE else "CHANGED"
