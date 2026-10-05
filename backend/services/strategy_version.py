"""Current strategy versions, trade provenance, and the current-version boundary.

A closed trade belongs to the CURRENT version of its engine only when its BUY
was stamped with the engine's current strategy and entry contract and it
closed under the current exit contract. Rows written before stamping existed
carry no version: they are LEGACY. Legacy rows are retained for accounting and
forensics and are never used as current performance, current learning, or a
permission input. Nothing here blocks a trade.
"""

from __future__ import annotations

import contextlib
import functools
import os
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.services.day_v2.winner_contract import DAY_EXIT_CONTRACT_RUNNER

DAY_ENGINE = "DAY_V2"
SCALP_ENGINE = "SCALP_V2"

DAY_STRATEGY_VERSION = "DAY_V2_FIVE_SETUP_RUNNER_V1"
DAY_ENTRY_CONTRACT_VERSION = "DAY_V2_ENTRY_PRIOR_STRUCTURE_V1"
DAY_EXIT_CONTRACT_VERSION = DAY_EXIT_CONTRACT_RUNNER

SCALP_STRATEGY_VERSION = "SCALP_V2_SHORT_HORIZON_V1"
SCALP_ENTRY_CONTRACT_VERSION = "SCALP_V2_STRATEGY_PASS_NET_EDGE_V1"
SCALP_EXIT_CONTRACT_VERSION = "SCALP_V2_TARGET_STOP_HORIZON_V1"

ACCOUNTING_CONTRACT_VERSION = "TWO_ENGINE_FIFO_NET_V1"
ADAPTIVE_STATE_VERSION = "ADAPTIVE_ONLINE_V2"

# Economic anchors: the first commit after which an engine's entry signal, cost
# model, markout definitions and exit contract are all unchanged. Only evidence
# decided at or after the anchor may move economic state; earlier rows stay
# forensic. DAY: 7ab0f13 removed the DAY clock sells (closes before it carry the
# current exit version string but a retired exit path). SCALP: a222109 follows
# the claim contract (30a9a4f), the stale-book exit fix (7e45d86) and the
# order-book staleness fix (0b60a7e) that changed micro features and costs.
ECONOMIC_ANCHORS: dict[str, tuple[str, str]] = {
    DAY_ENGINE: ("7ab0f13", "2026-10-01T21:41:15Z"),
    SCALP_ENGINE: ("a222109", "2026-10-05T02:02:29Z"),
}

VERSION_COLUMNS: tuple[str, ...] = (
    "strategy_version",
    "entry_contract_version",
    "exit_contract_version",
    "code_sha",
    "accounting_contract_version",
)

_ENGINE_VERSIONS: dict[str, dict[str, str]] = {
    DAY_ENGINE: {
        "strategy_version": DAY_STRATEGY_VERSION,
        "entry_contract_version": DAY_ENTRY_CONTRACT_VERSION,
        "exit_contract_version": DAY_EXIT_CONTRACT_VERSION,
    },
    SCALP_ENGINE: {
        "strategy_version": SCALP_STRATEGY_VERSION,
        "entry_contract_version": SCALP_ENTRY_CONTRACT_VERSION,
        "exit_contract_version": SCALP_EXIT_CONTRACT_VERSION,
    },
}

_NON_STRATEGY_EXITS = ("DUST_WRITEOFF", "HUMAN_MANUAL_SELL", "MANUAL_UNMATCHED")

BOUNDARY_TABLE = "strategy_version_boundaries"


@functools.lru_cache(maxsize=1)
def current_code_sha() -> str:
    env_sha = os.getenv("MYSTIC_CODE_SHA", "").strip()
    if env_sha:
        return env_sha[:40]
    repo = Path(__file__).resolve().parents[2]
    with contextlib.suppress(Exception):
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=False)
        sha = out.stdout.strip()
        if out.returncode == 0 and sha:
            return sha[:40]
    return "unknown"


def engine_versions(engine_id: str) -> dict[str, str] | None:
    versions = _ENGINE_VERSIONS.get(str(engine_id or "").strip().upper())
    return dict(versions) if versions else None


def economic_anchor(engine_id: str) -> dict[str, Any] | None:
    """{"commit", "at", "epoch"} for the engine's current economic anchor, else None."""
    anchor = ECONOMIC_ANCHORS.get(str(engine_id or "").strip().upper())
    if not anchor:
        return None
    commit, at = anchor
    return {"commit": commit, "at": at, "epoch": datetime.fromisoformat(at.replace("Z", "+00:00")).timestamp()}


def economic_version(engine_id: str) -> str:
    """Tag carried by every row of economic state: contracts, anchor and learner format.

    State written under any other tag is never read, so a contract change, a new
    anchor or a new learner format starts from evidence of that version only.
    """
    engine = str(engine_id or "").strip().upper()
    versions = _ENGINE_VERSIONS.get(engine)
    anchor = ECONOMIC_ANCHORS.get(engine)
    if not versions or not anchor:
        return ""
    return "|".join((versions["strategy_version"], versions["entry_contract_version"], versions["exit_contract_version"], anchor[0], ADAPTIVE_STATE_VERSION))


def version_provenance(engine_id: str) -> dict[str, str]:
    """Full provenance for a trade opened now by ``engine_id`` ({} for other engines)."""
    versions = engine_versions(engine_id)
    if versions is None:
        return {}
    return {
        "engine_id": str(engine_id).upper(),
        **versions,
        "code_sha": current_code_sha(),
        "accounting_contract_version": ACCOUNTING_CONTRACT_VERSION,
        "adaptive_state_version": ADAPTIVE_STATE_VERSION,
    }


def is_current_version(engine_id: str, row: dict[str, Any] | sqlite3.Row | None) -> bool:
    versions = engine_versions(engine_id)
    if versions is None or row is None:
        return False
    try:
        return all(str(row[key] or "") == value for key, value in versions.items())
    except (KeyError, IndexError):
        return False


def ensure_version_columns(conn: sqlite3.Connection, table: str) -> None:
    cols = {str(r[1]) for r in conn.execute(f"PRAGMA table_info({table})")}
    for col in VERSION_COLUMNS:
        if col not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} TEXT DEFAULT ''")


def stamp_buy_version(conn: sqlite3.Connection, trade_id: str, engine_id: str) -> dict[str, str]:
    """Stamp the opening versions on a BUY row in the caller's transaction."""
    prov = version_provenance(engine_id)
    if not trade_id or not prov:
        return {}
    ensure_version_columns(conn, "paper_trades")
    conn.execute(
        "UPDATE paper_trades SET strategy_version=?, entry_contract_version=?, exit_contract_version=?, code_sha=?, accounting_contract_version=? WHERE trade_id=?",
        (*(prov[c] for c in VERSION_COLUMNS), trade_id),
    )
    return prov


def buy_versions(conn: sqlite3.Connection, buy_trade_id: str) -> dict[str, str]:
    if not buy_trade_id:
        return {}
    ensure_version_columns(conn, "paper_trades")
    row = conn.execute(
        "SELECT strategy_version, entry_contract_version FROM paper_trades WHERE trade_id=? AND UPPER(side)='BUY' ORDER BY rowid DESC LIMIT 1",
        (buy_trade_id,),
    ).fetchone()
    if not row:
        return {}
    return {"strategy_version": str(row[0] or ""), "entry_contract_version": str(row[1] or "")}


def stamp_sell_version(conn: sqlite3.Connection, sell_trade_id: str, buy_trade_id: str, engine_id: str) -> dict[str, str]:
    """SELL inherits the BUY's strategy/entry versions; exit contract and SHA are the closing code's.

    A lot opened before versioning keeps blank entry versions and is legacy.
    """
    prov = version_provenance(engine_id)
    if not sell_trade_id or not prov:
        return {}
    opened = buy_versions(conn, buy_trade_id)
    stamped = {
        "strategy_version": opened.get("strategy_version", ""),
        "entry_contract_version": opened.get("entry_contract_version", ""),
        "exit_contract_version": prov["exit_contract_version"],
        "code_sha": prov["code_sha"],
        "accounting_contract_version": prov["accounting_contract_version"],
    }
    conn.execute(
        "UPDATE paper_trades SET strategy_version=?, entry_contract_version=?, exit_contract_version=?, code_sha=?, accounting_contract_version=? WHERE trade_id=?",
        (*(stamped[c] for c in VERSION_COLUMNS), sell_trade_id),
    )
    stamped["version_current"] = "1" if is_current_version(engine_id, stamped) else "0"
    return stamped


def lot_versions(db_path: str, engine_id: str, buy_trade_id: str) -> dict[str, Any]:
    """Versions of a lot being closed, for learning rows. Read-only."""
    prov = version_provenance(engine_id)
    out: dict[str, Any] = {
        "strategy_version": "",
        "entry_contract_version": "",
        "exit_contract_version": prov.get("exit_contract_version", ""),
        "code_sha": prov.get("code_sha", ""),
        "version_current": False,
    }
    with contextlib.suppress(Exception), sqlite3.connect(db_path, timeout=5) as conn:
        out.update(buy_versions(conn, buy_trade_id))
    out["version_current"] = is_current_version(engine_id, out)
    return out


def learning_version_filter(engine_id: str) -> tuple[str, tuple[str, ...]]:
    """SQL predicate + params selecting only current-version learning rows for an engine."""
    versions = engine_versions(engine_id)
    if versions is None:
        return "0", ()
    return (
        "strategy_version = ? AND entry_contract_version = ? AND exit_contract_version = ?",
        (versions["strategy_version"], versions["entry_contract_version"], versions["exit_contract_version"]),
    )


def register_version_boundaries(db_path: str) -> dict[str, str]:
    """Record the first time each current engine version ran. History is untouched."""
    now = datetime.now(timezone.utc).isoformat()
    sha = current_code_sha()
    with sqlite3.connect(db_path, timeout=15) as conn:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {BOUNDARY_TABLE} (
                engine_id TEXT NOT NULL,
                strategy_version TEXT NOT NULL,
                entry_contract_version TEXT NOT NULL,
                exit_contract_version TEXT NOT NULL,
                code_sha TEXT NOT NULL,
                started_at_utc TEXT NOT NULL,
                PRIMARY KEY (engine_id, strategy_version, entry_contract_version, exit_contract_version)
            )
            """
        )
        for engine_id, versions in _ENGINE_VERSIONS.items():
            conn.execute(
                f"INSERT OR IGNORE INTO {BOUNDARY_TABLE} VALUES (?, ?, ?, ?, ?, ?)",
                (engine_id, versions["strategy_version"], versions["entry_contract_version"], versions["exit_contract_version"], sha, now),
            )
        conn.commit()
    return {engine_id: current_version_start(db_path, engine_id) or "" for engine_id in _ENGINE_VERSIONS}


def current_version_start(db_path: str, engine_id: str) -> str | None:
    versions = engine_versions(engine_id)
    if versions is None:
        return None
    with contextlib.suppress(Exception), sqlite3.connect(db_path, timeout=5) as conn:
        row = conn.execute(
            f"SELECT started_at_utc FROM {BOUNDARY_TABLE} WHERE engine_id=? AND strategy_version=? AND entry_contract_version=? AND exit_contract_version=?",
            (engine_id, versions["strategy_version"], versions["entry_contract_version"], versions["exit_contract_version"]),
        ).fetchone()
        return str(row[0]) if row else None
    return None


def _stats(pnls: list[float]) -> dict[str, Any]:
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gross_win = sum(wins)
    gross_loss = -sum(losses)
    n = len(pnls)
    return {
        "round_trips": n,
        "wins": len(wins),
        "losses": len(losses),
        "net_usd": round(sum(pnls), 4),
        "profit_factor": (round(gross_win / gross_loss, 4) if gross_loss > 0 else None),
        "expectancy_usd": round(sum(pnls) / n, 4) if n else None,
        "avg_win_usd": round(gross_win / len(wins), 4) if wins else None,
        "avg_loss_usd": round(-gross_loss / len(losses), 4) if losses else None,
    }


def performance_by_version(db_path: str) -> dict[str, Any]:
    """Current-version and legacy closed-trade performance, per engine.

    Strategy closes only: dust write-offs, manual sells and unmatched closes are
    excluded from both buckets. Counts carry no permission consequence.
    """
    out: dict[str, Any] = {"engines": {}, "legacy_engines": {}}
    with sqlite3.connect(db_path, timeout=10) as conn:
        conn.row_factory = sqlite3.Row
        cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(paper_trades)")}
        pnl_expr = "COALESCE(pnl_usd_net, pnl)" if "pnl_usd_net" in cols else "pnl"
        realized = "AND COALESCE(counts_toward_realized, 1) = 1" if "counts_toward_realized" in cols else ""
        engine_col = "COALESCE(NULLIF(engine_id, ''), 'LEGACY_DAY_LIVE')" if "engine_id" in cols else "'LEGACY_DAY_LIVE'"
        version_cols = ", ".join((c if c in cols else f"'' AS {c}") for c in ("strategy_version", "entry_contract_version", "exit_contract_version"))
        placeholders = ",".join("?" for _ in _NON_STRATEGY_EXITS)
        rows = conn.execute(
            f"""
            SELECT {engine_col} AS engine, {pnl_expr} AS pnl_net, {version_cols}
            FROM paper_trades
            WHERE UPPER(side) = 'SELL' AND LOWER(COALESCE(mode, '')) = 'live'
              AND UPPER(COALESCE(exit_reason, '')) NOT IN ({placeholders})
              AND {pnl_expr} IS NOT NULL {realized}
            """,
            _NON_STRATEGY_EXITS,
        ).fetchall()
    buckets: dict[tuple[str, bool], list[float]] = {}
    for row in rows:
        engine = str(row["engine"] or "").upper()
        current = is_current_version(engine, row)
        buckets.setdefault((engine, current), []).append(float(row["pnl_net"]))
    for engine_id, versions in _ENGINE_VERSIONS.items():
        out["engines"][engine_id] = {
            **versions,
            "current_version_start_utc": current_version_start(db_path, engine_id),
            "current": _stats(buckets.get((engine_id, True), [])),
            "legacy": _stats(buckets.get((engine_id, False), [])),
        }
    for (engine, current), pnls in buckets.items():
        if engine not in _ENGINE_VERSIONS and not current:
            out["legacy_engines"][engine] = _stats(pnls)
    out["accounting_contract_version"] = ACCOUNTING_CONTRACT_VERSION
    out["adaptive_state_version"] = ADAPTIVE_STATE_VERSION
    out["code_sha"] = current_code_sha()
    return out


__all__ = [
    "ACCOUNTING_CONTRACT_VERSION",
    "DAY_ENTRY_CONTRACT_VERSION",
    "DAY_EXIT_CONTRACT_VERSION",
    "DAY_STRATEGY_VERSION",
    "SCALP_ENTRY_CONTRACT_VERSION",
    "SCALP_EXIT_CONTRACT_VERSION",
    "SCALP_STRATEGY_VERSION",
    "VERSION_COLUMNS",
    "current_code_sha",
    "current_version_start",
    "engine_versions",
    "is_current_version",
    "learning_version_filter",
    "lot_versions",
    "performance_by_version",
    "register_version_boundaries",
    "stamp_buy_version",
    "stamp_sell_version",
    "version_provenance",
]
