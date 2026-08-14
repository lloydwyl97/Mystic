"""Scalp SQLite schema — scalp_* tables only; never mutates DAY tables."""

from __future__ import annotations

import sqlite3
from pathlib import Path

DAY_TABLES = (
    "portfolio_engine_ledger",
    "portfolio_engine_positions",
    "portfolio_engine_scoreboard_daily",
    "paper_trades",
    "trade_learning_outcomes",
)

SCALP_TABLES = (
    "scalp_paper_ledger",
    "scalp_paper_positions",
    "scalp_paper_trades",
    "scalp_rejects",
    "scalp_scoreboard_daily",
    "scalp_trade_audit",
    "scalp_position_reviews",
    "scalp_outcome_attribution",
    "scalp_post_trade_feature_reviews",
    "scalp_strategy_score_weights",
    "scalp_gate_counters",
    "scalp_shadow_rejects",
)

SCHEMA_VERSION = 3
OPEN_POSITION_UNIQUE_INDEX = "idx_scalp_paper_positions_symbol_open"


def _positions_ddl(*, with_symbol_unique: bool) -> str:
    symbol_col = "symbol TEXT NOT NULL UNIQUE," if with_symbol_unique else "symbol TEXT NOT NULL,"
    return f"""
            CREATE TABLE IF NOT EXISTS scalp_paper_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                {symbol_col}
                exchange TEXT NOT NULL DEFAULT 'binance_us',
                strategy_id TEXT NOT NULL DEFAULT 'scalp',
                quantity REAL NOT NULL,
                entry_price REAL NOT NULL,
                entry_time TEXT NOT NULL,
                entry_time_epoch REAL NOT NULL,
                trade_id TEXT NOT NULL UNIQUE,
                paper_order_id TEXT,
                status TEXT NOT NULL DEFAULT 'OPEN',
                reprice_count INTEGER NOT NULL DEFAULT 0,
                diagnostics_json TEXT
            );
            """


def _has_open_symbol_unique_index(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='index' AND name=?",
        (OPEN_POSITION_UNIQUE_INDEX,),
    ).fetchone()
    return row is not None


def _symbol_has_table_unique(conn: sqlite3.Connection) -> bool:
    rows = conn.execute("PRAGMA index_list(scalp_paper_positions)").fetchall()
    for row in rows:
        # row: seq, name, unique, origin(c/u/p), partial
        name = str(row[1])
        unique = int(row[2])
        origin = str(row[3]) if len(row) > 3 else ""
        if unique != 1 or origin != "u":
            continue
        if name.startswith("sqlite_autoindex"):
            cols = conn.execute(f"PRAGMA index_info({name!r})").fetchall()
            col_names = [str(c[2]) for c in cols]
            if col_names == ["symbol"]:
                return True
    return False


def migrate_scalp_positions_open_unique(conn: sqlite3.Connection) -> bool:
    """
    Migrate scalp_paper_positions to allow multiple CLOSED rows per symbol.
    Adds partial unique index: one OPEN row per symbol.
    Returns True if migration ran, False if already applied.
    """
    if _has_open_symbol_unique_index(conn) and not _symbol_has_table_unique(conn):
        return False

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS scalp_paper_positions_v2 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            exchange TEXT NOT NULL DEFAULT 'binance_us',
            strategy_id TEXT NOT NULL DEFAULT 'scalp',
            quantity REAL NOT NULL,
            entry_price REAL NOT NULL,
            entry_time TEXT NOT NULL,
            entry_time_epoch REAL NOT NULL,
            trade_id TEXT NOT NULL UNIQUE,
            paper_order_id TEXT,
            status TEXT NOT NULL DEFAULT 'OPEN',
            reprice_count INTEGER NOT NULL DEFAULT 0,
            diagnostics_json TEXT
        );

        INSERT INTO scalp_paper_positions_v2 (
            id, symbol, exchange, strategy_id, quantity, entry_price,
            entry_time, entry_time_epoch, trade_id, paper_order_id, status,
            reprice_count, diagnostics_json
        )
        SELECT
            id, symbol, exchange, strategy_id, quantity, entry_price,
            entry_time, entry_time_epoch, trade_id, paper_order_id, status,
            reprice_count, diagnostics_json
        FROM scalp_paper_positions;

        DROP TABLE scalp_paper_positions;
        ALTER TABLE scalp_paper_positions_v2 RENAME TO scalp_paper_positions;
        """
    )
    conn.execute(
        f"""
        CREATE UNIQUE INDEX IF NOT EXISTS {OPEN_POSITION_UNIQUE_INDEX}
        ON scalp_paper_positions(symbol)
        WHERE status = 'OPEN'
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_scalp_paper_positions_status ON scalp_paper_positions(status)")
    return True


def _current_schema_version(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT schema_version FROM scalp_meta WHERE id = 1").fetchone()
    if row is None:
        return 0
    return int(row[0])


def migrate_exit_manager_v3(conn: sqlite3.Connection) -> bool:
    """Add exit-manager columns and scalp_position_reviews table."""
    cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(scalp_paper_positions)")}
    added = False
    for name, ddl in (
        ("state", "TEXT NOT NULL DEFAULT 'OPEN'"),
        ("max_favorable_pct", "REAL NOT NULL DEFAULT 0"),
        ("max_adverse_pct", "REAL NOT NULL DEFAULT 0"),
        ("stale_review_count", "INTEGER NOT NULL DEFAULT 0"),
        ("last_review_ts", "TEXT"),
        ("last_state_reason", "TEXT"),
        ("session_low_bid", "REAL"),
    ):
        if name not in cols:
            conn.execute(f"ALTER TABLE scalp_paper_positions ADD COLUMN {name} {ddl}")
            added = True
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS scalp_position_reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            hold_seconds REAL NOT NULL,
            current_bid REAL NOT NULL,
            entry_price REAL NOT NULL,
            executable_net_pct REAL NOT NULL,
            max_favorable_pct REAL NOT NULL DEFAULT 0,
            max_adverse_pct REAL NOT NULL DEFAULT 0,
            recovery_from_low_pct REAL NOT NULL DEFAULT 0,
            bid_change_15s REAL NOT NULL DEFAULT 0,
            bid_change_30s REAL NOT NULL DEFAULT 0,
            bid_change_60s REAL NOT NULL DEFAULT 0,
            higher_lows INTEGER NOT NULL DEFAULT 0,
            spread_pct REAL NOT NULL DEFAULT 0,
            decision TEXT NOT NULL,
            state TEXT NOT NULL,
            reason TEXT NOT NULL,
            diagnostics_json TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_scalp_position_reviews_trade ON scalp_position_reviews(trade_id)")
    return added


def apply_scalp_migrations(conn: sqlite3.Connection) -> list[str]:
    applied: list[str] = []
    version = _current_schema_version(conn)
    if version < 3:
        if migrate_exit_manager_v3(conn):
            applied.append("migrate_exit_manager_v3")
        else:
            applied.append("ensure_exit_manager_v3")
        conn.execute("UPDATE scalp_meta SET schema_version = ? WHERE id = 1", (SCHEMA_VERSION,))
    if version < 2:
        if migrate_scalp_positions_open_unique(conn):
            applied.append("migrate_scalp_positions_open_unique_v2")
        conn.execute(
            "UPDATE scalp_meta SET schema_version = ? WHERE id = 1",
            (SCHEMA_VERSION,),
        )
        if conn.execute("SELECT changes()").fetchone()[0] == 0:
            conn.execute(
                "INSERT INTO scalp_meta (id, schema_version) VALUES (1, ?)",
                (SCHEMA_VERSION,),
            )
    elif not _has_open_symbol_unique_index(conn):
        migrate_scalp_positions_open_unique(conn)
        applied.append("repair_open_symbol_partial_index")
    return applied


def init_scalp_schema(db_path: str | Path, *, principal: float = 1000.0) -> list[str]:
    path = Path(db_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS scalp_meta (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                schema_version INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS scalp_paper_ledger (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                principal REAL NOT NULL,
                cash_balance REAL NOT NULL,
                positions_value REAL NOT NULL DEFAULT 0,
                realized_pnl REAL NOT NULL DEFAULT 0,
                unrealized_pnl REAL NOT NULL DEFAULT 0,
                total_equity REAL NOT NULL,
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS scalp_paper_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT NOT NULL UNIQUE,
                symbol TEXT NOT NULL,
                exchange TEXT NOT NULL DEFAULT 'binance_us',
                strategy_id TEXT NOT NULL DEFAULT 'scalp',
                side TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                notional REAL NOT NULL,
                fee_usd REAL NOT NULL DEFAULT 0,
                slippage_usd REAL NOT NULL DEFAULT 0,
                pnl_usd REAL,
                pnl_pct REAL,
                entry_price REAL,
                exit_reason TEXT,
                diagnostics_json TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS scalp_rejects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                exchange TEXT NOT NULL DEFAULT 'binance_us',
                strategy_id TEXT NOT NULL DEFAULT 'scalp',
                side TEXT NOT NULL,
                reason TEXT NOT NULL,
                detail TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS scalp_scoreboard_daily (
                day TEXT PRIMARY KEY,
                trades INTEGER NOT NULL DEFAULT 0,
                wins INTEGER NOT NULL DEFAULT 0,
                losses INTEGER NOT NULL DEFAULT 0,
                net_pnl REAL NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS scalp_trade_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT NOT NULL,
                action TEXT NOT NULL,
                symbol TEXT NOT NULL,
                exchange TEXT NOT NULL DEFAULT 'binance_us',
                strategy_id TEXT NOT NULL DEFAULT 'scalp',
                qty REAL,
                price REAL,
                pre_ledger_json TEXT,
                post_ledger_json TEXT,
                reason TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            """
        )
        # Fresh installs: positions table without symbol UNIQUE + partial index.
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='scalp_paper_positions'").fetchone():
            conn.executescript(_positions_ddl(with_symbol_unique=False))
            conn.execute(
                f"""
                CREATE UNIQUE INDEX IF NOT EXISTS {OPEN_POSITION_UNIQUE_INDEX}
                ON scalp_paper_positions(symbol)
                WHERE status = 'OPEN'
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_scalp_paper_positions_status ON scalp_paper_positions(status)")

        applied: list[str] = []
        row = conn.execute("SELECT 1 FROM scalp_meta WHERE id = 1").fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO scalp_meta (id, schema_version) VALUES (1, ?)",
                (SCHEMA_VERSION,),
            )
        ledger = conn.execute("SELECT 1 FROM scalp_paper_ledger WHERE id = 1").fetchone()
        if ledger is None:
            conn.execute(
                """
                INSERT INTO scalp_paper_ledger
                (id, principal, cash_balance, positions_value, realized_pnl,
                 unrealized_pnl, total_equity)
                VALUES (1, ?, ?, 0, 0, 0, ?)
                """,
                (principal, principal, principal),
            )
        else:
            ledger_row = conn.execute(
                """
                SELECT principal, cash_balance, positions_value, realized_pnl,
                       unrealized_pnl, total_equity
                FROM scalp_paper_ledger WHERE id = 1
                """
            ).fetchone()
            trade_count = int(conn.execute("SELECT COUNT(*) FROM scalp_paper_trades").fetchone()[0] or 0)
            position_count = int(conn.execute("SELECT COUNT(*) FROM scalp_paper_positions").fetchone()[0] or 0)
            if ledger_row is not None and trade_count == 0 and position_count == 0:
                stored_principal = float(ledger_row[0] or 0)
                canonical_principal = stored_principal if stored_principal > 0 else float(principal)
                cash = float(ledger_row[1] or 0)
                positions_value = float(ledger_row[2] or 0)
                realized_pnl = float(ledger_row[3] or 0)
                unrealized_pnl = float(ledger_row[4] or 0)
                total_equity = float(ledger_row[5] or 0)
                clean_empty = all(abs(value) < 0.01 for value in (positions_value, realized_pnl, unrealized_pnl))
                basis_mismatch = abs(cash - canonical_principal) >= 0.01 or abs(total_equity - canonical_principal) >= 0.01
                if clean_empty and basis_mismatch:
                    conn.execute(
                        """
                        UPDATE scalp_paper_ledger
                        SET principal=?, cash_balance=?, positions_value=0,
                            realized_pnl=0, unrealized_pnl=0, total_equity=?,
                            updated_at=datetime('now')
                        WHERE id=1
                        """,
                        (canonical_principal, canonical_principal, canonical_principal),
                    )
                    applied.append("repair_empty_ledger_basis")
        applied.extend(apply_scalp_migrations(conn))
        conn.commit()
        try:
            from backend.services.scalp_outcome_attribution import ensure_scalp_outcome_attribution_table
            from backend.services.scalp_post_trade_feature_review import ensure_scalp_post_trade_review_table
            from backend.services.scalp_strategy_score_weight_writer import ensure_scalp_strategy_score_weights_table

            ensure_scalp_outcome_attribution_table(str(path))
            ensure_scalp_post_trade_review_table(str(path))
            ensure_scalp_strategy_score_weights_table(str(path))
            applied.append("ensure_scalp_intelligence_tables")
        except Exception:
            pass
        try:
            from backend.services.scalp_gate_telemetry import ensure_scalp_gate_schema

            ensure_scalp_gate_schema(str(path))
            applied.append("ensure_scalp_gate_tables")
        except Exception:
            pass
        return applied


def assert_scalp_sql_only(statement: str) -> None:
    """Guard: scalp engine must never mutate Mystic DAY tables."""
    lower = statement.lower()
    for table in DAY_TABLES:
        if table in lower:
            raise RuntimeError(f"scalp engine blocked SQL touching DAY table {table!r}")


def verify_scalp_tables(db_path: str | Path) -> dict[str, int]:
    path = Path(db_path).resolve()
    with sqlite3.connect(path) as conn:
        out: dict[str, int] = {}
        for table in SCALP_TABLES:
            row = conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchone()
            out[table] = int(row[0]) if row else 0
        return out


def verify_open_position_constraints(db_path: str | Path) -> dict[str, object]:
    path = Path(db_path).resolve()
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        indexes = [
            {
                "name": str(row[1]),
                "unique": int(row[2]),
                "origin": str(row[3]) if len(row) > 3 else "",
            }
            for row in conn.execute("PRAGMA index_list(scalp_paper_positions)").fetchall()
        ]
        return {
            "schema_version": _current_schema_version(conn),
            "has_partial_open_index": _has_open_symbol_unique_index(conn),
            "symbol_table_unique": _symbol_has_table_unique(conn),
            "indexes": indexes,
            "open_rows": [dict(r) for r in conn.execute("SELECT id, symbol, status FROM scalp_paper_positions WHERE status='OPEN'").fetchall()],
            "all_rows": [dict(r) for r in conn.execute("SELECT id, symbol, status, trade_id FROM scalp_paper_positions ORDER BY id").fetchall()],
        }
