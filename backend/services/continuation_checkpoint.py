"""Exact recovery of the installed hold-advantage surface.

The historical fold log stored only an identity, not the feature vector,
weight, or ridge statistics, so the live surface cannot be rebuilt from
that log. The recovery contract is the verified live checkpoint plus every
later fold event. It does not retrain and it does not change the update
equation.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from typing import Any

from backend.services.adaptive_learning import PRIOR_STRENGTH, current_economic_version
from backend.services.continuation_surface import ADVANTAGE_VERSION, FEATURE_NAMES, SURFACE_MODEL

DAY_ENGINE = "DAY_V2"
RECOVERY_OK = "ok"
RECOVERY_FAILED = "CONTINUATION_RECOVERY_FAILED"
_FLOAT_TOL = 1e-9


def feature_contract() -> str:
    raw = json.dumps({"names": list(FEATURE_NAMES), "version": ADVANTAGE_VERSION, "prior": PRIOR_STRENGTH}, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def _canon(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True, default=str)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canon(value).encode()).hexdigest()


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_event_columns(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS continuation_advantage_log (
            engine_id TEXT NOT NULL,
            trade_id TEXT NOT NULL,
            snapshot_time REAL NOT NULL,
            horizon INTEGER NOT NULL,
            PRIMARY KEY (engine_id, trade_id, snapshot_time, horizon)
        )
        """
    )
    have = {str(row[1]) for row in conn.execute("PRAGMA table_info(continuation_advantage_log)")}
    for name, decl in (
        ("symbol", "TEXT"),
        ("setup", "TEXT"),
        ("regime", "TEXT"),
        ("features_json", "TEXT"),
        ("advantage", "REAL"),
        ("weight", "REAL"),
        ("label_time", "REAL"),
        ("feature_version", "TEXT"),
    ):
        if name not in have:
            conn.execute(f"ALTER TABLE continuation_advantage_log ADD COLUMN {name} {decl}")


def ensure_checkpoint_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS continuation_checkpoint (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            engine_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            continuation_version TEXT NOT NULL,
            feature_contract TEXT NOT NULL,
            economic_version TEXT NOT NULL,
            last_event_id INTEGER NOT NULL,
            state_hash TEXT NOT NULL,
            key_count INTEGER NOT NULL,
            effective_weight REAL NOT NULL,
            payload_json TEXT NOT NULL,
            checksum TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS continuation_recovery (
            engine_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            detail TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )


def _rows(conn: sqlite3.Connection, engine: str) -> list[dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    try:
        found = conn.execute(
            """
            SELECT economic_version, symbol, setup, regime, metric, n, ewma, m2, updated_at
            FROM adaptive_metric_state
            WHERE engine_id=? AND metric LIKE 'hold_adv_%'
            ORDER BY economic_version, symbol, setup, regime, metric
            """,
            (engine,),
        ).fetchall()
    except sqlite3.Error:
        return []
    return [dict(row) for row in found]


def _ridge(conn: sqlite3.Connection, engine: str) -> dict[str, Any]:
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT model, payload, updated_at FROM adaptive_linear_model WHERE engine_id=? AND model LIKE ? ORDER BY model",
            (engine, f"{SURFACE_MODEL}%"),
        ).fetchone()
    except sqlite3.Error:
        return {}
    if row is None:
        return {}
    try:
        payload = json.loads(row["payload"] or "{}")
    except (TypeError, ValueError):
        payload = {}
    return {"model": str(row["model"]), "updated_at": str(row["updated_at"] or ""), "payload": payload if isinstance(payload, dict) else {}}


def _meta(conn: sqlite3.Connection, engine: str) -> dict[str, Any] | None:
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT * FROM continuation_learning_meta WHERE engine_id=? AND economic_version=?",
            (engine, current_economic_version(engine)),
        ).fetchone()
    except sqlite3.Error:
        return None
    return None if row is None else dict(row)


def live_document(conn: sqlite3.Connection, engine: str = DAY_ENGINE) -> dict[str, Any]:
    rows = _rows(conn, engine)
    return {
        "engine_id": engine,
        "continuation_version": ADVANTAGE_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "prior_strength": PRIOR_STRENGTH,
        "economic_version": current_economic_version(engine),
        "rows": rows,
        "ridge": _ridge(conn, engine),
        "meta": _meta(conn, engine),
        "key_count": len(rows),
        "effective_weight": sum(float(row["n"] or 0.0) for row in rows),
    }


def state_hash(document: dict[str, Any]) -> str:
    body = {
        "rows": document.get("rows") or [],
        "ridge": document.get("ridge") or {},
        "feature_names": document.get("feature_names") or [],
        "prior_strength": document.get("prior_strength"),
        "continuation_version": document.get("continuation_version"),
        "economic_version": document.get("economic_version"),
    }
    return _sha(body)


def _max_event_id(conn: sqlite3.Connection) -> int:
    ensure_event_columns(conn)
    row = conn.execute("SELECT MAX(rowid) FROM continuation_advantage_log").fetchone()
    return int(row[0] or 0) if row else 0


def write_checkpoint(db_path: str, engine: str = DAY_ENGINE) -> dict[str, Any]:
    """Snapshot the installed surface. Metric rows and the ridge are not modified."""
    conn = _connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        ensure_checkpoint_tables(conn)
        document = live_document(conn, engine)
        document["last_event_id"] = _max_event_id(conn)
        digest = state_hash(document)
        payload = _canon(document)
        checksum = hashlib.sha256(payload.encode()).hexdigest()
        created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        cur = conn.execute(
            """
            INSERT INTO continuation_checkpoint (
                engine_id, created_at, continuation_version, feature_contract, economic_version,
                last_event_id, state_hash, key_count, effective_weight, payload_json, checksum
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                engine,
                created,
                ADVANTAGE_VERSION,
                feature_contract(),
                str(document["economic_version"]),
                int(document["last_event_id"]),
                digest,
                int(document["key_count"]),
                float(document["effective_weight"]),
                payload,
                checksum,
            ),
        )
        conn.commit()
        return {"id": int(cur.lastrowid), "state_hash": digest, "key_count": document["key_count"], "effective_weight": document["effective_weight"], "checksum": checksum}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def latest_checkpoint(conn: sqlite3.Connection, engine: str = DAY_ENGINE) -> dict[str, Any] | None:
    ensure_checkpoint_tables(conn)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM continuation_checkpoint WHERE engine_id=? ORDER BY id DESC LIMIT 1", (engine,)).fetchone()
    if row is None:
        return None
    payload_text = str(row["payload_json"])
    checksum = hashlib.sha256(payload_text.encode()).hexdigest()
    try:
        document = json.loads(payload_text)
    except (TypeError, ValueError):
        document = None
    return {
        "id": int(row["id"]),
        "created_at": str(row["created_at"]),
        "continuation_version": str(row["continuation_version"]),
        "feature_contract": str(row["feature_contract"]),
        "economic_version": str(row["economic_version"]),
        "last_event_id": int(row["last_event_id"]),
        "state_hash": str(row["state_hash"]),
        "key_count": int(row["key_count"]),
        "effective_weight": float(row["effective_weight"]),
        "checksum": str(row["checksum"]),
        "checksum_ok": checksum == str(row["checksum"]),
        "document": document if isinstance(document, dict) else None,
    }


def _near(left: float, right: float) -> bool:
    return abs(float(left) - float(right)) <= _FLOAT_TOL * max(1.0, abs(float(left)), abs(float(right)))


def documents_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if int(left.get("key_count") or 0) != int(right.get("key_count") or 0):
        return False
    if not _near(float(left.get("effective_weight") or 0.0), float(right.get("effective_weight") or 0.0)):
        return False
    lrows = {(r["symbol"], r["setup"], r["regime"], r["metric"]): r for r in left.get("rows") or []}
    rrows = {(r["symbol"], r["setup"], r["regime"], r["metric"]): r for r in right.get("rows") or []}
    if set(lrows) != set(rrows):
        return False
    for key, row in lrows.items():
        other = rrows[key]
        if not (_near(row["n"], other["n"]) and _near(row["ewma"], other["ewma"]) and _near(row["m2"], other["m2"])):
            return False
    lridge = ((left.get("ridge") or {}).get("payload") or {}).get("horizons") or {}
    rridge = ((right.get("ridge") or {}).get("payload") or {}).get("horizons") or {}
    if set(lridge) != set(rridge):
        return False
    for horizon, stats in lridge.items():
        other = rridge[horizon]
        if not _near(float(stats.get("n") or 0.0), float(other.get("n") or 0.0)):
            return False
        for key in ("xty", "sum_x", "sum_x2"):
            a, b = stats.get(key) or [], other.get(key) or []
            if len(a) != len(b) or any(not _near(x, y) for x, y in zip(a, b, strict=True)):
                return False
        a_xtx, b_xtx = stats.get("xtx") or [], other.get("xtx") or []
        if len(a_xtx) != len(b_xtx):
            return False
        for a_row, b_row in zip(a_xtx, b_xtx, strict=True):
            if len(a_row) != len(b_row) or any(not _near(x, y) for x, y in zip(a_row, b_row, strict=True)):
                return False
    return True


def _install_document(conn: sqlite3.Connection, document: dict[str, Any]) -> None:
    engine = str(document["engine_id"])
    conn.execute("DELETE FROM adaptive_metric_state WHERE engine_id=? AND metric LIKE 'hold_adv_%'", (engine,))
    for row in document.get("rows") or []:
        conn.execute(
            """
            INSERT INTO adaptive_metric_state
            (engine_id, economic_version, symbol, setup, regime, metric, n, ewma, m2, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (
                engine,
                row["economic_version"],
                row["symbol"],
                row["setup"],
                row["regime"],
                row["metric"],
                row["n"],
                row["ewma"],
                row["m2"],
                row["updated_at"],
            ),
        )
    ridge = document.get("ridge") or {}
    if ridge.get("model"):
        conn.execute("DELETE FROM adaptive_linear_model WHERE engine_id=? AND model=?", (engine, ridge["model"]))
        conn.execute(
            "INSERT INTO adaptive_linear_model (engine_id, model, payload, updated_at) VALUES (?,?,?,?)",
            (engine, ridge["model"], json.dumps(ridge.get("payload") or {}, separators=(",", ":")), ridge.get("updated_at") or ""),
        )
    meta = document.get("meta")
    if isinstance(meta, dict) and meta.get("engine_id"):
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS continuation_learning_meta (
                engine_id TEXT NOT NULL,
                economic_version TEXT NOT NULL,
                learning_version TEXT NOT NULL,
                source TEXT NOT NULL,
                anchor_commit TEXT NOT NULL,
                anchor_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                observations INTEGER NOT NULL,
                PRIMARY KEY (engine_id, economic_version)
            )
            """
        )
        conn.execute(
            """
            INSERT INTO continuation_learning_meta
            (engine_id, economic_version, learning_version, source, anchor_commit, anchor_at, updated_at, observations)
            VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(engine_id, economic_version) DO UPDATE SET
                learning_version=excluded.learning_version, source=excluded.source,
                anchor_commit=excluded.anchor_commit, anchor_at=excluded.anchor_at,
                updated_at=excluded.updated_at, observations=excluded.observations
            """,
            (
                meta["engine_id"],
                meta["economic_version"],
                meta["learning_version"],
                meta["source"],
                meta.get("anchor_commit") or "",
                meta.get("anchor_at") or "",
                meta.get("updated_at") or "",
                int(meta.get("observations") or 0),
            ),
        )


def events_after(conn: sqlite3.Connection, last_event_id: int, engine: str = DAY_ENGINE) -> list[dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    ensure_event_columns(conn)
    rows = conn.execute(
        """
        SELECT rowid AS event_id, engine_id, trade_id, snapshot_time, horizon, symbol, setup, regime,
               features_json, advantage, weight, label_time, feature_version
        FROM continuation_advantage_log
        WHERE rowid>? AND engine_id=? AND features_json IS NOT NULL
        ORDER BY rowid
        """,
        (int(last_event_id), engine),
    ).fetchall()
    return [dict(row) for row in rows]


def apply_event(db_path: str, event: dict[str, Any], conn: sqlite3.Connection | None = None) -> None:
    from backend.services.continuation_surface import record_advantage

    features = json.loads(event["features_json"] or "{}")
    record_advantage(
        db_path,
        engine=str(event["engine_id"]),
        symbol=str(event["symbol"] or ""),
        setup=str(event["setup"] or ""),
        regime=str(event["regime"] or ""),
        horizon=int(event["horizon"]),
        features=features if isinstance(features, dict) else {},
        advantage=float(event["advantage"]),
        weight=float(event["weight"]),
        now=float(event["label_time"] or event["snapshot_time"]),
        conn=conn,
    )


def materialize(db_path: str, document: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    """Load a checkpoint and apply later events into ``db_path``. Returns the resulting document."""
    from backend.services.adaptive_learning import _connect as connect_learning

    connect_learning(db_path).close()
    conn = _connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        _install_document(conn, document)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    for event in events:
        apply_event(db_path, event)
    conn = _connect(db_path)
    try:
        return live_document(conn, str(document["engine_id"]))
    finally:
        conn.close()


def _mark(conn: sqlite3.Connection, engine: str, status: str, detail: str) -> None:
    ensure_checkpoint_tables(conn)
    conn.execute(
        """
        INSERT INTO continuation_recovery (engine_id, status, detail, updated_at)
        VALUES (?,?,?,?)
        ON CONFLICT(engine_id) DO UPDATE SET status=excluded.status, detail=excluded.detail, updated_at=excluded.updated_at
        """,
        (engine, status, detail[:500], time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
    )


def day_entry_block_reason(db_path: str) -> str:
    """Empty when DAY entries may proceed. A failed recovery is an integrity block."""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error:
        return ""
    try:
        row = conn.execute("SELECT status FROM continuation_recovery WHERE engine_id=?", (DAY_ENGINE,)).fetchone()
    except sqlite3.Error:
        return ""
    finally:
        conn.close()
    if row is None:
        return ""
    return "" if str(row[0]) == RECOVERY_OK else RECOVERY_FAILED


def recover_continuation(db_path: str, engine: str = DAY_ENGINE) -> dict[str, Any]:
    """Verify the live surface against its checkpoint and later events.

    A missing checkpoint of a live surface becomes that checkpoint. A corrupt
    or incompatible checkpoint does not replace the installed rows.
    """
    import tempfile
    from pathlib import Path

    from backend.services.adaptive_learning import _connect as connect_learning

    connect_learning(db_path).close()
    conn = _connect(db_path)
    try:
        ensure_checkpoint_tables(conn)
        current = live_document(conn, engine)
        saved = latest_checkpoint(conn, engine)
        if saved is None:
            conn.close()
            written = write_checkpoint(db_path, engine)
            conn = _connect(db_path)
            _mark(conn, engine, RECOVERY_OK, f"checkpoint {written['id']} created from the installed surface")
            conn.commit()
            return {"status": RECOVERY_OK, "checkpoint_id": written["id"], "created": True, "state_hash": written["state_hash"]}
        if not saved["checksum_ok"] or saved["document"] is None:
            _mark(conn, engine, RECOVERY_FAILED, f"checkpoint {saved['id']} checksum failed")
            conn.commit()
            return {"status": RECOVERY_FAILED, "reason": "checksum"}
        if saved["feature_contract"] != feature_contract() or saved["continuation_version"] != ADVANTAGE_VERSION:
            _mark(conn, engine, RECOVERY_FAILED, f"checkpoint {saved['id']} feature contract mismatch")
            conn.commit()
            return {"status": RECOVERY_FAILED, "reason": "feature_contract"}
        events = events_after(conn, int(saved["last_event_id"]), engine)
    finally:
        conn.close()
    scratch_dir = Path(tempfile.mkdtemp(prefix="continuation_recover_"))
    scratch = scratch_dir / "state.db"
    try:
        rebuilt = materialize(str(scratch), saved["document"], events)
    finally:
        for suffix in ("", "-wal", "-shm"):
            try:
                Path(str(scratch) + suffix).unlink()
            except OSError:
                pass
        try:
            scratch_dir.rmdir()
        except OSError:
            pass
    if documents_match(current, rebuilt):
        conn = _connect(db_path)
        try:
            _mark(conn, engine, RECOVERY_OK, f"checkpoint {saved['id']} matches live plus {len(events)} events")
            conn.commit()
        finally:
            conn.close()
        return {"status": RECOVERY_OK, "checkpoint_id": saved["id"], "events": len(events), "state_hash": state_hash(current)}
    if int(current["key_count"]) == 0 and int(rebuilt["key_count"]) > 0:
        conn = _connect(db_path)
        try:
            conn.execute("BEGIN IMMEDIATE")
            _install_document(conn, rebuilt)
            _mark(conn, engine, RECOVERY_OK, f"restored checkpoint {saved['id']} plus {len(events)} events")
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return {"status": RECOVERY_OK, "restored": True, "checkpoint_id": saved["id"], "state_hash": state_hash(rebuilt)}
    conn = _connect(db_path)
    try:
        _mark(conn, engine, RECOVERY_FAILED, f"checkpoint {saved['id']} does not match the installed surface")
        conn.commit()
    finally:
        conn.close()
    return {"status": RECOVERY_FAILED, "reason": "mismatch", "checkpoint_id": saved["id"]}
