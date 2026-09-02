"""Isolate every test from the production/canonical trading database.

Must run before backend.database_schema is imported so DATABASE_PATH is the
temporary file. Does not weaken assertions. Does not open Ocean or Local books.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

_FORBIDDEN = {
    Path("/home/mystic/mystic/mystic_trading.db").resolve(),
}

_ISOLATED = Path(os.environ.get("MYSTIC_TEST_DB") or "/tmp/mystic_pytest_isolated.db").resolve()
if _ISOLATED in _FORBIDDEN:
    _ISOLATED = Path("/tmp/mystic_pytest_isolated.db").resolve()

os.environ["DATABASE_PATH"] = str(_ISOLATED)
os.environ["DATABASE_URL"] = f"sqlite:///{_ISOLATED}"
os.environ["MYSTIC_DB_PATH"] = str(_ISOLATED)
os.environ["TRADING_DB_PATH"] = str(_ISOLATED)
os.environ["MYSTIC_TRADING_DB_PATH"] = str(_ISOLATED)
os.environ.setdefault("JWT_SECRET", "pytest-isolated-jwt")
# Match the paper-test contract the suite assumes when a worktree has no .env.
# Do not load the canonical .env (it points DATABASE_PATH at the live book).
os.environ.setdefault("SCALP_PAPER_ENABLED", "true")
os.environ.setdefault("SCALP_FEE_MODEL_VERIFIED", "true")
os.environ.setdefault("SCALP_LIVE", "false")
os.environ.setdefault("SCALP_PAPER_AUTO_ARM", "true")


def _ensure_isolated_schema(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    from backend.database_schema import create_paper_trades_table, ensure_paper_trades_columns

    create_paper_trades_table(str(path))
    conn = sqlite3.connect(str(path))
    try:
        ensure_paper_trades_columns(conn)
        existing = {row[1] for row in conn.execute("PRAGMA table_info(paper_trades)")}
        extras = (
            ("explainability_json", "TEXT"),
            ("is_synthetic", "INTEGER DEFAULT 0"),
            ("fees_paid", "REAL DEFAULT 0"),
            ("slippage_cost", "REAL DEFAULT 0"),
            ("mode", "TEXT"),
        )
        for col_name, col_type in extras:
            if col_name not in existing:
                conn.execute(f"ALTER TABLE paper_trades ADD COLUMN {col_name} {col_type}")
        conn.commit()
    finally:
        conn.close()


_ensure_isolated_schema(_ISOLATED)

if Path(os.environ["DATABASE_PATH"]).resolve() in _FORBIDDEN:
    raise RuntimeError("tests refused to use the production/canonical trading database")
