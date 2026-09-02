"""DAY gate counters, shadow rejects, decision records, and attribution helpers."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.services.day_gate_registry import CONFIG_VERSION, DECISION_POLICY_VERSION, get_gate

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS day_gate_counters (
    date TEXT NOT NULL,
    gate_id TEXT NOT NULL,
    symbol TEXT NOT NULL DEFAULT '',
    setup TEXT NOT NULL DEFAULT '',
    regime TEXT NOT NULL DEFAULT '',
    strategy_version TEXT NOT NULL DEFAULT '',
    model_version TEXT NOT NULL DEFAULT '',
    config_version TEXT NOT NULL DEFAULT '',
    mode TEXT NOT NULL DEFAULT 'paper',
    evaluated INTEGER NOT NULL DEFAULT 0,
    passed INTEGER NOT NULL DEFAULT 0,
    hard_blocked INTEGER NOT NULL DEFAULT 0,
    sized_down INTEGER NOT NULL DEFAULT 0,
    rank_penalized INTEGER NOT NULL DEFAULT 0,
    errors INTEGER NOT NULL DEFAULT 0,
    shadow_evals INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (date, gate_id, symbol, setup, regime, strategy_version, model_version, config_version, mode)
);

CREATE TABLE IF NOT EXISTS day_shadow_rejects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    decision_id TEXT,
    symbol TEXT NOT NULL,
    setup TEXT,
    gate_id TEXT NOT NULL,
    bar_timestamp INTEGER,
    entry_price REAL,
    stop_price REAL,
    target_price REAL,
    fee_model_version TEXT,
    policy_version TEXT,
    detail TEXT,
    diag_json TEXT,
    bar_closed INTEGER NOT NULL DEFAULT 1,
    hyp_exit_price REAL,
    hyp_mfe_pct REAL,
    hyp_mae_pct REAL,
    hyp_net_pnl REAL,
    hyp_resolved_at TEXT,
    hyp_exit_reason TEXT,
    hyp_gross_return REAL,
    hyp_fees REAL,
    hyp_slip REAL,
    prevented_profitable INTEGER
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_day_shadow_decision_gate
    ON day_shadow_rejects(decision_id, gate_id) WHERE decision_id IS NOT NULL AND decision_id != '';
CREATE INDEX IF NOT EXISTS idx_day_shadow_rejects_gate ON day_shadow_rejects(gate_id, created_at);
CREATE INDEX IF NOT EXISTS idx_day_shadow_unresolved ON day_shadow_rejects(hyp_resolved_at);

CREATE TABLE IF NOT EXISTS day_decision_records (
    decision_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    symbol TEXT NOT NULL,
    aw_valid INTEGER NOT NULL DEFAULT 0,
    setup TEXT,
    regime TEXT,
    gates_json TEXT,
    first_hard_block TEXT,
    other_blocking_gates_json TEXT,
    ml_score REAL,
    ml_rank_adjustment REAL,
    ml_size_adjustment REAL,
    requested_size REAL,
    approved_size REAL,
    final_decision TEXT NOT NULL,
    strategy_version TEXT,
    model_version TEXT,
    feature_version TEXT,
    artifact_version TEXT,
    config_version TEXT,
    policy_version TEXT,
    mode TEXT NOT NULL DEFAULT 'paper',
    detail_json TEXT
);
"""


def _utc_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _migrate_columns(conn: sqlite3.Connection, table: str, columns: list[tuple[str, str]]) -> None:
    existing = {str(r[1]) for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    for name, decl in columns:
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")


def ensure_day_gate_schema(db_path: str | Path) -> None:
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        conn.executescript(SCHEMA_SQL)
        # Backward-compatible column adds for older DBs created with narrower PK
        try:
            _migrate_columns(
                conn,
                "day_gate_counters",
                [
                    ("strategy_version", "TEXT NOT NULL DEFAULT ''"),
                    ("model_version", "TEXT NOT NULL DEFAULT ''"),
                    ("config_version", "TEXT NOT NULL DEFAULT ''"),
                    ("mode", "TEXT NOT NULL DEFAULT 'paper'"),
                    ("errors", "INTEGER NOT NULL DEFAULT 0"),
                    ("shadow_evals", "INTEGER NOT NULL DEFAULT 0"),
                ],
            )
            _migrate_columns(
                conn,
                "day_shadow_rejects",
                [
                    ("hyp_gross_return", "REAL"),
                    ("hyp_fees", "REAL"),
                    ("hyp_slip", "REAL"),
                    ("prevented_profitable", "INTEGER"),
                ],
            )
        except Exception:
            logger.debug("day_gate schema migrate skipped", exc_info=True)
        conn.commit()
    finally:
        conn.close()


def record_gate_event(
    db_path: str | Path,
    *,
    gate_id: str,
    symbol: str = "",
    outcome: str = "evaluated",
    setup: str = "",
    regime: str = "",
    decision_id: str = "",
    detail: str = "",
    strategy_version: str = "",
    model_version: str = "",
    config_version: str = "",
    mode: str = "paper",
) -> None:
    """Increment day_gate_counters for one gate outcome."""
    try:
        ensure_day_gate_schema(db_path)
        gid = str(gate_id or "").strip()
        if not gid:
            return
        spec = get_gate(gid)
        if spec is not None and spec.status == "disabled":
            return
        day = _utc_today()
        sym = str(symbol or "")
        setup_s = str(setup or "")
        regime_s = str(regime or "")
        cfg_v = str(config_version or CONFIG_VERSION or "")
        mode_s = str(mode or "paper")
        cols = {
            "evaluated": 1,
            "passed": 1 if outcome == "passed" else 0,
            "hard_blocked": 1 if outcome == "hard_blocked" else 0,
            "sized_down": 1 if outcome == "sized_down" else 0,
            "rank_penalized": 1 if outcome == "rank_penalized" else 0,
            "errors": 1 if outcome == "error" else 0,
            "shadow_evals": 1 if outcome == "shadow" or (spec and spec.status == "shadow_only") else 0,
        }
        conn = sqlite3.connect(str(db_path), timeout=30)
        try:
            # Prefer extended PK; fall back to legacy 5-col PK if migrate incomplete
            try:
                conn.execute(
                    """
                    INSERT INTO day_gate_counters(
                        date, gate_id, symbol, setup, regime,
                        strategy_version, model_version, config_version, mode,
                        evaluated, passed, hard_blocked, sized_down, rank_penalized, errors, shadow_evals, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(date, gate_id, symbol, setup, regime, strategy_version, model_version, config_version, mode) DO UPDATE SET
                        evaluated = evaluated + excluded.evaluated,
                        passed = passed + excluded.passed,
                        hard_blocked = hard_blocked + excluded.hard_blocked,
                        sized_down = sized_down + excluded.sized_down,
                        rank_penalized = rank_penalized + excluded.rank_penalized,
                        errors = errors + excluded.errors,
                        shadow_evals = shadow_evals + excluded.shadow_evals,
                        updated_at = excluded.updated_at
                    """,
                    (
                        day,
                        gid,
                        sym,
                        setup_s,
                        regime_s,
                        str(strategy_version or ""),
                        str(model_version or ""),
                        cfg_v,
                        mode_s,
                        cols["evaluated"],
                        cols["passed"],
                        cols["hard_blocked"],
                        cols["sized_down"],
                        cols["rank_penalized"],
                        cols["errors"],
                        cols["shadow_evals"],
                        _utc_now(),
                    ),
                )
            except sqlite3.OperationalError:
                conn.execute(
                    """
                    INSERT INTO day_gate_counters(
                        date, gate_id, symbol, setup, regime,
                        evaluated, passed, hard_blocked, sized_down, rank_penalized, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(date, gate_id, symbol, setup, regime) DO UPDATE SET
                        evaluated = evaluated + excluded.evaluated,
                        passed = passed + excluded.passed,
                        hard_blocked = hard_blocked + excluded.hard_blocked,
                        sized_down = sized_down + excluded.sized_down,
                        rank_penalized = rank_penalized + excluded.rank_penalized,
                        updated_at = excluded.updated_at
                    """,
                    (
                        day,
                        gid,
                        sym,
                        setup_s,
                        regime_s,
                        cols["evaluated"],
                        cols["passed"],
                        cols["hard_blocked"],
                        cols["sized_down"],
                        cols["rank_penalized"],
                        _utc_now(),
                    ),
                )
            conn.commit()
        finally:
            conn.close()
        logger.info(
            "DAY_GATE_EVENT gate=%s outcome=%s symbol=%s decision_id=%s setup=%s regime=%s detail=%s",
            gid,
            outcome,
            sym,
            decision_id,
            setup_s,
            regime_s,
            detail,
        )
    except Exception:
        logger.warning("record_gate_event failed gate=%s", gate_id, exc_info=True)


def _candidate_fields(candidate: Any) -> dict[str, Any]:
    dd = getattr(candidate, "decision_data", None) or {}
    if not isinstance(dd, dict):
        dd = {}
    price = float(getattr(candidate, "price", 0.0) or dd.get("price") or 0.0)
    stop = float(dd.get("thesis_invalid_level") or dd.get("stop_price") or 0.0)
    target = float(dd.get("thesis_target_level") or dd.get("take_profit_price") or 0.0)
    return {
        "decision_id": str(getattr(candidate, "decision_id", "") or dd.get("decision_id") or ""),
        "symbol": str(getattr(candidate, "symbol", "") or dd.get("symbol") or ""),
        "setup": str(dd.get("setup_type") or dd.get("entry_thesis") or dd.get("allweather_setup") or ""),
        "entry_price": price,
        "stop_price": stop,
        "target_price": target,
        "regime": str(dd.get("allweather_regime") or dd.get("aw_regime") or dd.get("day_route_regime") or ""),
    }


def record_shadow_reject(
    db_path: str | Path,
    *,
    candidate: Any,
    gate_id: str,
    bar_timestamp: int | None = None,
    detail: str = "",
    diag: dict[str, Any] | None = None,
) -> None:
    """Persist a rejected candidate for counterfactual / opportunity tracking.

    Idempotent on (decision_id, gate_id). Never reserves cash/slots or places orders.
    """
    try:
        ensure_day_gate_schema(db_path)
        fields = _candidate_fields(candidate)
        if not fields["symbol"]:
            return
        if diag and not fields["entry_price"]:
            try:
                fields["entry_price"] = float(diag.get("close") or 0.0)
            except Exception:
                pass
        did = fields["decision_id"]
        conn = sqlite3.connect(str(db_path), timeout=30)
        try:
            if did:
                hit = conn.execute(
                    "SELECT id FROM day_shadow_rejects WHERE decision_id=? AND gate_id=? LIMIT 1",
                    (did, str(gate_id)),
                ).fetchone()
                if hit:
                    return
            conn.execute(
                """
                INSERT INTO day_shadow_rejects(
                    created_at, decision_id, symbol, setup, gate_id, bar_timestamp,
                    entry_price, stop_price, target_price, fee_model_version, policy_version,
                    detail, diag_json, bar_closed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    _utc_now(),
                    did or None,
                    fields["symbol"],
                    fields["setup"],
                    str(gate_id),
                    int(bar_timestamp or 0) or None,
                    fields["entry_price"] or None,
                    fields["stop_price"] or None,
                    fields["target_price"] or None,
                    "binance_us_taker_v1",
                    DECISION_POLICY_VERSION,
                    detail or "",
                    json.dumps(diag or {}, default=str)[:8000],
                ),
            )
            conn.commit()
        finally:
            conn.close()
        record_gate_event(
            db_path,
            gate_id=str(gate_id),
            symbol=fields["symbol"],
            outcome="shadow",
            setup=fields["setup"],
            regime=fields.get("regime") or "",
            decision_id=did,
            detail=detail,
        )
    except Exception:
        logger.warning("record_shadow_reject failed gate=%s", gate_id, exc_info=True)


def record_day_decision(
    db_path: str | Path,
    *,
    decision_id: str,
    symbol: str,
    aw_valid: bool,
    setup: str = "",
    regime: str = "",
    gates: list[dict[str, Any]] | None = None,
    first_hard_block: str = "",
    other_blocking_gates: list[str] | None = None,
    ml_score: float | None = None,
    ml_rank_adjustment: float | None = None,
    ml_size_adjustment: float | None = None,
    requested_size: float | None = None,
    approved_size: float | None = None,
    final_decision: str = "reject",
    strategy_version: str = "",
    model_version: str = "",
    feature_version: str = "",
    artifact_version: str = "",
    config_version: str = "",
    mode: str | None = None,
    detail: dict[str, Any] | None = None,
) -> None:
    """Upsert a structured DAY decision record (idempotent on decision_id)."""
    try:
        if mode is None:
            from backend.services.day_decision_observability import runtime_account_execution_mode

            mode = runtime_account_execution_mode()
        ensure_day_gate_schema(db_path)
        did = str(decision_id or "").strip()
        if not did:
            return
        gates_list = list(gates or [])
        first = str(first_hard_block or "")
        if not first:
            for g in gates_list:
                if str(g.get("outcome") or "") == "hard_blocked":
                    first = str(g.get("gate_id") or "")
                    break
        others = list(other_blocking_gates or [])
        if not others:
            others = [str(g.get("gate_id")) for g in gates_list if str(g.get("outcome")) == "hard_blocked" and str(g.get("gate_id")) != first]
        conn = sqlite3.connect(str(db_path), timeout=30)
        try:
            conn.execute(
                """
                INSERT INTO day_decision_records(
                    decision_id, created_at, symbol, aw_valid, setup, regime, gates_json,
                    first_hard_block, other_blocking_gates_json,
                    ml_score, ml_rank_adjustment, ml_size_adjustment,
                    requested_size, approved_size, final_decision,
                    strategy_version, model_version, feature_version, artifact_version,
                    config_version, policy_version, mode, detail_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(decision_id) DO UPDATE SET
                    final_decision=excluded.final_decision,
                    approved_size=COALESCE(excluded.approved_size, day_decision_records.approved_size),
                    first_hard_block=COALESCE(NULLIF(excluded.first_hard_block,''), day_decision_records.first_hard_block),
                    gates_json=excluded.gates_json,
                    detail_json=excluded.detail_json
                """,
                (
                    did,
                    _utc_now(),
                    str(symbol or ""),
                    1 if aw_valid else 0,
                    setup or None,
                    regime or None,
                    json.dumps(gates_list, default=str)[:16000],
                    first or None,
                    json.dumps(others, default=str),
                    ml_score,
                    ml_rank_adjustment,
                    ml_size_adjustment,
                    requested_size,
                    approved_size,
                    str(final_decision or "reject"),
                    strategy_version or None,
                    model_version or None,
                    feature_version or None,
                    artifact_version or None,
                    config_version or CONFIG_VERSION,
                    DECISION_POLICY_VERSION,
                    mode or "unknown",
                    json.dumps(detail or {}, default=str)[:8000],
                ),
            )
            conn.commit()
        finally:
            conn.close()
        logger.info(
            "DAY_DECISION decision_id=%s symbol=%s aw_valid=%s final=%s first_block=%s setup=%s regime=%s",
            did,
            symbol,
            aw_valid,
            final_decision,
            first,
            setup,
            regime,
        )
    except Exception:
        logger.warning("record_day_decision failed decision_id=%s", decision_id, exc_info=True)


def counters_today(db_path: str | Path, *, date: str | None = None) -> list[dict[str, Any]]:
    ensure_day_gate_schema(db_path)
    day = date or _utc_today()
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT gate_id,
                   SUM(evaluated) AS evaluated,
                   SUM(passed) AS passed,
                   SUM(hard_blocked) AS hard_blocked,
                   SUM(sized_down) AS sized_down,
                   SUM(rank_penalized) AS rank_penalized
            FROM day_gate_counters
            WHERE date = ?
            GROUP BY gate_id
            ORDER BY hard_blocked DESC, evaluated DESC
            """,
            (day,),
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            spec = get_gate(str(d["gate_id"]))
            d["layer"] = spec.layer if spec else None
            d["behavior"] = spec.behavior if spec else None
            d["purpose"] = spec.purpose if spec else None
            out.append(d)
        return out
    finally:
        conn.close()


def shadow_rejects_summary(db_path: str | Path, *, limit: int = 50) -> dict[str, Any]:
    ensure_day_gate_schema(db_path)
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        by_gate = conn.execute(
            """
            SELECT gate_id, COUNT(*) AS n,
                   SUM(CASE WHEN hyp_resolved_at IS NULL THEN 1 ELSE 0 END) AS unresolved,
                   SUM(CASE WHEN hyp_net_pnl IS NOT NULL THEN hyp_net_pnl ELSE 0 END) AS hyp_net_sum
            FROM day_shadow_rejects
            GROUP BY gate_id
            ORDER BY n DESC
            """
        ).fetchall()
        recent = conn.execute(
            """
            SELECT id, created_at, symbol, setup, gate_id, entry_price, hyp_net_pnl, hyp_exit_reason, hyp_resolved_at
            FROM day_shadow_rejects
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
        return {
            "by_gate": [dict(r) for r in by_gate],
            "recent": [dict(r) for r in recent],
        }
    finally:
        conn.close()


def attribution_report(db_path: str | Path, *, date: str | None = None) -> dict[str, Any]:
    """Executed PnL attribution + gate opportunity from shadows."""
    ensure_day_gate_schema(db_path)
    day = date or _utc_today()
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        executed = conn.execute(
            """
            SELECT COUNT(*) AS trades,
                   COALESCE(SUM(pnl), 0) AS net_pnl,
                   COALESCE(SUM(fees_paid), 0) AS fees,
                   COALESCE(SUM(slippage_cost), 0) AS slip,
                   SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS wins
            FROM paper_trades
            WHERE UPPER(side)='SELL' AND date(timestamp)=?
              AND COALESCE(is_synthetic,0)=0
            """,
            (day,),
        ).fetchone()
        by_exit = conn.execute(
            """
            SELECT COALESCE(exit_type, 'UNKNOWN') AS exit_type, COUNT(*) AS n, COALESCE(SUM(pnl),0) AS net_pnl
            FROM paper_trades
            WHERE UPPER(side)='SELL' AND date(timestamp)=?
              AND COALESCE(is_synthetic,0)=0
            GROUP BY 1
            ORDER BY n DESC
            """,
            (day,),
        ).fetchall()
        shadow = conn.execute(
            """
            SELECT gate_id, COUNT(*) AS rejects,
                   SUM(CASE WHEN hyp_net_pnl IS NOT NULL AND hyp_net_pnl > 0 THEN 1 ELSE 0 END) AS hyp_wins,
                   SUM(CASE WHEN hyp_net_pnl IS NOT NULL AND hyp_net_pnl <= 0 THEN 1 ELSE 0 END) AS hyp_losses,
                   COALESCE(SUM(hyp_net_pnl), 0) AS hyp_net_sum,
                   SUM(CASE WHEN hyp_resolved_at IS NULL THEN 1 ELSE 0 END) AS unresolved
            FROM day_shadow_rejects
            WHERE date(created_at)=?
            GROUP BY gate_id
            ORDER BY rejects DESC
            """,
            (day,),
        ).fetchall()
        no_signal = conn.execute(
            "SELECT COALESCE(SUM(hard_blocked),0) FROM day_gate_counters WHERE date=? AND gate_id='AW_NO_SIGNAL'",
            (day,),
        ).fetchone()[0]
        eval_err = conn.execute(
            "SELECT COALESCE(SUM(hard_blocked),0) FROM day_gate_counters WHERE date=? AND gate_id='AW_EVAL_ERROR'",
            (day,),
        ).fetchone()[0]
        # Gate saved (blocked losers) vs destroyed (blocked winners)
        saved_destroyed = []
        for r in shadow:
            d = dict(r)
            hyp = float(d.get("hyp_net_sum") or 0.0)
            d["gate_destroyed_expectancy"] = max(0.0, hyp)  # positive hyp = missed winners
            d["gate_saved_expectancy"] = max(0.0, -hyp)  # negative hyp = avoided losers
            d["net_opportunity_effect"] = -hyp  # positive = gate helped
            saved_destroyed.append(d)
        overlap = conn.execute(
            """
            SELECT a.gate_id AS gate_a, b.gate_id AS gate_b, COUNT(*) AS n
            FROM day_shadow_rejects a
            JOIN day_shadow_rejects b
              ON a.decision_id = b.decision_id AND a.gate_id < b.gate_id
            WHERE date(a.created_at)=? AND a.decision_id IS NOT NULL AND a.decision_id != ''
            GROUP BY 1, 2
            ORDER BY n DESC
            LIMIT 20
            """,
            (day,),
        ).fetchall()
        return {
            "date": day,
            "policy_version": DECISION_POLICY_VERSION,
            "executed": dict(executed) if executed else {},
            "by_exit_type": [dict(r) for r in by_exit],
            "gate_opportunity": saved_destroyed,
            "gate_overlap": [dict(r) for r in overlap],
            "no_signal_count": int(no_signal or 0),
            "eval_error_count": int(eval_err or 0),
            "gate_counters": counters_today(db_path, date=day),
        }
    finally:
        conn.close()


def _hyp_net_usd(entry: float, exit_px: float, *, notional: float = 100.0, fee_rt: float = 0.002) -> float:
    """Round-trip fee model on a fixed notional for comparable shadow opportunity."""
    if entry <= 0:
        return 0.0
    gross = (exit_px - entry) / entry * notional
    fees = notional * fee_rt * 2.0
    return float(gross - fees)


def _walk_aw_brackets(
    bars: list[dict[str, Any]],
    *,
    entry: float,
    stop: float,
    target: float,
    max_bars: int = 48,
) -> dict[str, Any] | None:
    """Closed-bar walk: stop / target / time. bars are dicts with ts/high/low/close."""
    if not bars or entry <= 0:
        return None
    mfe = 0.0
    mae = 0.0
    for i, b in enumerate(bars[:max_bars]):
        high = float(b.get("high") or 0.0)
        low = float(b.get("low") or 0.0)
        close = float(b.get("close") or 0.0)
        if high > 0:
            mfe = max(mfe, (high - entry) / entry)
        if low > 0:
            mae = min(mae, (low - entry) / entry)
        # Conservative intrabar: stop before target if both touched
        hit_stop = stop > 0 and low > 0 and low <= stop
        hit_tgt = target > entry and high >= target
        if hit_stop and hit_tgt:
            exit_px = stop
            reason = "SHADOW_STOP_BEFORE_TARGET"
            return {
                "exit_price": exit_px,
                "exit_reason": reason,
                "mfe_pct": mfe * 100.0,
                "mae_pct": mae * 100.0,
                "net_pnl": _hyp_net_usd(entry, exit_px),
            }
        if hit_stop:
            return {
                "exit_price": stop,
                "exit_reason": "SHADOW_ATR_STOP",
                "mfe_pct": mfe * 100.0,
                "mae_pct": mae * 100.0,
                "net_pnl": _hyp_net_usd(entry, stop),
            }
        if hit_tgt:
            return {
                "exit_price": target,
                "exit_reason": "SHADOW_ATR_TARGET",
                "mfe_pct": mfe * 100.0,
                "mae_pct": mae * 100.0,
                "net_pnl": _hyp_net_usd(entry, target),
            }
        if i == max_bars - 1 and close > 0:
            return {
                "exit_price": close,
                "exit_reason": "SHADOW_TIME_STOP",
                "mfe_pct": mfe * 100.0,
                "mae_pct": mae * 100.0,
                "net_pnl": _hyp_net_usd(entry, close),
            }
    return None


def resolve_shadow_rejects_closed_bar(db_path: str | Path, *, max_rows: int = 100) -> dict[str, Any]:
    """Expire unresolved shadows after 72h when async kline resolver has not filled them."""
    ensure_day_gate_schema(db_path)
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    resolved = 0
    try:
        rows = conn.execute(
            """
            SELECT id, created_at, entry_price, stop_price, target_price, symbol
            FROM day_shadow_rejects
            WHERE hyp_resolved_at IS NULL AND entry_price IS NOT NULL AND entry_price > 0
            ORDER BY id ASC
            LIMIT ?
            """,
            (int(max_rows),),
        ).fetchall()
        now = _utc_now()
        for r in rows:
            entry = float(r["entry_price"])
            stop = float(r["stop_price"] or 0.0)
            try:
                created = datetime.fromisoformat(str(r["created_at"]).replace("Z", "+00:00"))
                age_h = (datetime.now(timezone.utc) - created).total_seconds() / 3600.0
            except Exception:
                age_h = 0.0
            if age_h < 72.0:
                continue
            if stop > 0 and stop < entry:
                hyp_pnl = _hyp_net_usd(entry, stop)
                reason = "SHADOW_EXPIRE_ASSUME_STOP"
                exit_px = stop
            else:
                hyp_pnl = 0.0
                reason = "SHADOW_EXPIRE_FLAT"
                exit_px = entry
            conn.execute(
                """
                UPDATE day_shadow_rejects
                SET hyp_resolved_at=?, hyp_exit_reason=?, hyp_net_pnl=?, hyp_exit_price=?
                WHERE id=?
                """,
                (now, reason, float(hyp_pnl), float(exit_px), int(r["id"])),
            )
            resolved += 1
        conn.commit()
        return {"resolved": resolved, "scanned": len(rows)}
    finally:
        conn.close()


async def resolve_shadow_rejects_async(db_path: str | Path, *, max_rows: int = 40) -> dict[str, Any]:
    """Resolve shadows via closed 1h bars (stop/target/time). Does not place orders."""
    ensure_day_gate_schema(db_path)
    try:
        from backend.services.allweather_breakout_pullback_adapter import _closed_primary_bars
        from backend.services.allweather_signal_engine import normalize_bars
        from backend.services.live_market_data import live_market_data_service
    except Exception as exc:
        logger.debug("shadow async deps unavailable: %s", exc)
        return resolve_shadow_rejects_closed_bar(db_path, max_rows=max_rows)

    def _api_sym(sym: str) -> str:
        s = str(sym or "").strip().upper().replace("-", "/")
        if "/" not in s and s.endswith("USDT") and len(s) > 4:
            s = s[:-4] + "/USDT"
        return s

    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    resolved = 0
    scanned = 0
    try:
        rows = conn.execute(
            """
            SELECT id, created_at, entry_price, stop_price, target_price, symbol, bar_timestamp
            FROM day_shadow_rejects
            WHERE hyp_resolved_at IS NULL AND entry_price IS NOT NULL AND entry_price > 0
              AND stop_price IS NOT NULL AND stop_price > 0
            ORDER BY id ASC
            LIMIT ?
            """,
            (int(max_rows),),
        ).fetchall()
        scanned = len(rows)
        now = _utc_now()
        for r in rows:
            symbol = str(r["symbol"] or "")
            entry = float(r["entry_price"])
            stop = float(r["stop_price"] or 0.0)
            target = float(r["target_price"] or 0.0)
            bar_ts = int(r["bar_timestamp"] or 0)
            api_sym = _api_sym(symbol)
            try:
                fetch_meta = await live_market_data_service.get_ohlcv_with_meta(api_sym, "1h", limit=120)
                raw = fetch_meta.get("rows") if isinstance(fetch_meta, dict) else None
                bars = normalize_bars(raw)
                bars, _, _ = _closed_primary_bars(bars)
            except Exception:
                continue
            if bar_ts > 0:
                # bar_timestamp may be seconds or ms
                ts_sec = bar_ts // 1000 if bar_ts > 10_000_000_000 else bar_ts
                future = [b for b in bars if int(b.get("ts") or 0) > ts_sec]
            else:
                try:
                    created = datetime.fromisoformat(str(r["created_at"]).replace("Z", "+00:00"))
                    ts_sec = int(created.timestamp())
                except Exception:
                    ts_sec = 0
                future = [b for b in bars if int(b.get("ts") or 0) > ts_sec] if ts_sec else bars[-48:]
            if len(future) < 2:
                continue
            walked = _walk_aw_brackets(future, entry=entry, stop=stop, target=target)
            if not walked:
                continue
            hyp_net = float(walked["net_pnl"])
            fees = abs(entry) * 0.002 * 2.0 if entry else 0.0
            gross = (float(walked["exit_price"]) - entry) / entry * 100.0 if entry else 0.0
            conn.execute(
                """
                UPDATE day_shadow_rejects
                SET hyp_resolved_at=?, hyp_exit_reason=?, hyp_net_pnl=?, hyp_exit_price=?,
                    hyp_mfe_pct=?, hyp_mae_pct=?, bar_closed=1,
                    hyp_gross_return=?, hyp_fees=?, prevented_profitable=?
                WHERE id=?
                """,
                (
                    now,
                    walked["exit_reason"],
                    hyp_net,
                    float(walked["exit_price"]),
                    float(walked["mfe_pct"]),
                    float(walked["mae_pct"]),
                    float(gross),
                    float(fees),
                    1 if hyp_net > 0 else 0,
                    int(r["id"]),
                ),
            )
            resolved += 1
        conn.commit()
    finally:
        conn.close()
    expired = resolve_shadow_rejects_closed_bar(db_path, max_rows=max_rows)
    return {"resolved": resolved, "scanned": scanned, "expired": expired}


__all__ = [
    "attribution_report",
    "counters_today",
    "ensure_day_gate_schema",
    "record_day_decision",
    "record_gate_event",
    "record_shadow_reject",
    "resolve_shadow_rejects_async",
    "resolve_shadow_rejects_closed_bar",
    "shadow_rejects_summary",
]
