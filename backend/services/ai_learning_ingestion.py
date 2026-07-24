"""
Continuous learning ingestion — separates LEARNING from TRADE EXECUTION.

The trade engine stays selective (top-4 DAY only). This module makes the
learning loop non-selective by capturing labels from every meaningful market
decision, not just closed sells:

  Stream 1: candidate snapshots   — every evaluated bar candidate with its
            decision (BUY / REJECT / BLOCK / NO_TRADE) and reason code.
  Stream 2: forward labeling      — every snapshot is followed forward and
            labeled with future returns, MFE/MAE, target reachability,
            invalidation breach, and a verdict (missed opportunity vs
            correct reject).
  Stream 3: position heartbeats   — open positions emit learning rows while
            holding (MFE/MAE path, thesis validity, would-sell-now result).
  Stream 5: missed-move stats     — bounded ranking adjustments derived from
            labeled snapshots (never opens trades by itself).
  Tier B/C: training row extraction joining snapshots/heartbeats to the
            inference-time feature vectors in ai_inference_log.

Closed-trade outcomes (Tier A) and walk-forward candle anchors (Tier D)
remain in ai_outcome_training_rows / the training cache; this module only
adds the missing streams and never contaminates realized-PnL metrics.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any

from backend.config.trading_economics import (
    ESTIMATED_ROUNDTRIP_COST,
    MIN_NET_PROFIT_TO_SELL,
    TAKER_FEE,
)
from backend.database_schema import DATABASE_PATH

logger = logging.getLogger(__name__)

SNAPSHOT_RETENTION_DAYS = 45
HEARTBEAT_RETENTION_DAYS = 45
LABEL_FINAL_HORIZON_SEC = 24 * 3600
LABEL_GIVEUP_AGE_SEC = 72 * 3600
LABEL_BATCH_LIMIT = int(os.getenv("AI_SNAPSHOT_LABEL_BATCH_LIMIT", "1000") or "1000")

# Tier weights (Tier A = OUTCOME_ROW_WEIGHT 5.0, Tier D = SELF_SUPERVISED 1.0 live elsewhere)
TIER_B_WEIGHT = 2.0
TIER_C_WEIGHT = 0.8
MAX_TIER_BC_SHARE = 0.5

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ai_candidate_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc TEXT NOT NULL,
    epoch_ms INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    strategy_id TEXT NOT NULL DEFAULT 'day',
    decision_id TEXT,
    decision TEXT NOT NULL,
    reason_code TEXT,
    rank INTEGER DEFAULT 0,
    rank_score REAL DEFAULT 0,
    confidence REAL DEFAULT 0,
    side TEXT,
    entry_thesis TEXT,
    thesis_score REAL DEFAULT 0,
    thesis_invalid_level REAL DEFAULT 0,
    thesis_target_level REAL DEFAULT 0,
    regime TEXT,
    trend_state TEXT,
    relative_volume REAL DEFAULT 0,
    spread_pct REAL DEFAULT 0,
    cost_estimate_pct REAL DEFAULT 0,
    price REAL NOT NULL DEFAULT 0,
    open_position_json TEXT,
    label_status TEXT NOT NULL DEFAULT 'PENDING',
    fwd_ret_15m REAL,
    fwd_ret_30m REAL,
    fwd_ret_1h REAL,
    fwd_ret_4h REAL,
    fwd_ret_24h REAL,
    mfe_pct REAL,
    mae_pct REAL,
    would_hit_target INTEGER,
    would_breach_invalidation INTEGER,
    verdict TEXT,
    labeled_at_utc TEXT
);
CREATE INDEX IF NOT EXISTS ix_cand_snap_status ON ai_candidate_snapshots(label_status, epoch_ms);
CREATE INDEX IF NOT EXISTS ix_cand_snap_sym ON ai_candidate_snapshots(symbol, epoch_ms);
CREATE TABLE IF NOT EXISTS ai_position_heartbeats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc TEXT NOT NULL,
    epoch_ms INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    trade_id TEXT,
    entry_price REAL DEFAULT 0,
    mark REAL DEFAULT 0,
    unrealized_pct REAL DEFAULT 0,
    net_unrealized_pct REAL DEFAULT 0,
    highest_since_entry REAL DEFAULT 0,
    lowest_since_entry REAL DEFAULT 0,
    mfe_pct REAL DEFAULT 0,
    mae_pct REAL DEFAULT 0,
    dist_to_target_pct REAL,
    dist_to_invalid_pct REAL,
    thesis_valid INTEGER,
    exit_allowed INTEGER,
    would_sell_now_net_usd REAL DEFAULT 0,
    hold_seconds REAL DEFAULT 0,
    entry_thesis TEXT,
    quantity REAL DEFAULT 0,
    gross_unrealized_pnl REAL DEFAULT 0,
    net_unrealized_pnl REAL DEFAULT 0,
    fee_estimate REAL DEFAULT 0,
    heartbeat_calc_version INTEGER DEFAULT 1
);
CREATE INDEX IF NOT EXISTS ix_pos_hb_trade ON ai_position_heartbeats(trade_id, epoch_ms);
CREATE INDEX IF NOT EXISTS ix_pos_hb_sym ON ai_position_heartbeats(symbol, epoch_ms);
"""

HEARTBEAT_CALC_VERSION = 2

_HEARTBEAT_MIGRATION_COLUMNS: tuple[tuple[str, str], ...] = (
    ("quantity", "REAL DEFAULT 0"),
    ("gross_unrealized_pnl", "REAL DEFAULT 0"),
    ("net_unrealized_pnl", "REAL DEFAULT 0"),
    ("fee_estimate", "REAL DEFAULT 0"),
    ("heartbeat_calc_version", "INTEGER DEFAULT 1"),
)

_tables_ready = False


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _migrate_heartbeat_columns(conn: sqlite3.Connection) -> None:
    existing = {r[1] for r in conn.execute("PRAGMA table_info(ai_position_heartbeats)").fetchall()}
    for col_name, col_def in _HEARTBEAT_MIGRATION_COLUMNS:
        if col_name not in existing:
            conn.execute(f"ALTER TABLE ai_position_heartbeats ADD COLUMN {col_name} {col_def}")


def ensure_learning_ingestion_tables(db_path: str = DATABASE_PATH) -> None:
    global _tables_ready
    with sqlite3.connect(db_path) as conn:
        # CREATE IF NOT EXISTS is idempotent — always run so alternate db_path works in tests.
        for stmt in _SCHEMA.strip().split(";"):
            s = stmt.strip()
            if s:
                conn.execute(s)
        _migrate_heartbeat_columns(conn)
        conn.commit()
    _tables_ready = True


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        if raw is None or str(raw).strip() == "":
            return default
        v = float(raw)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


# =========================================================================
# Stream 1: candidate snapshots
# =========================================================================


def record_candidate_snapshot(
    *,
    symbol: str,
    decision: str,
    reason_code: str = "",
    strategy_id: str = "day",
    decision_id: str = "",
    rank: int = 0,
    rank_score: float = 0.0,
    confidence: float = 0.0,
    side: str = "",
    price: float = 0.0,
    decision_data: dict[str, Any] | None = None,
    open_position_state: dict[str, Any] | None = None,
    db_path: str = DATABASE_PATH,
) -> None:
    """Persist one candidate decision snapshot (sync; call via to_thread from async paths)."""
    ensure_learning_ingestion_tables(db_path)
    dd = decision_data or {}
    try:
        with sqlite3.connect(db_path, timeout=10) as conn:
            conn.execute(
                """
                INSERT INTO ai_candidate_snapshots (
                    ts_utc, epoch_ms, symbol, strategy_id, decision_id, decision, reason_code,
                    rank, rank_score, confidence, side, entry_thesis, thesis_score,
                    thesis_invalid_level, thesis_target_level, regime, trend_state,
                    relative_volume, spread_pct, cost_estimate_pct, price, open_position_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    _now_iso(),
                    int(time.time() * 1000),
                    symbol,
                    (strategy_id or "day").strip().lower(),
                    decision_id or "",
                    (decision or "NO_TRADE").strip().upper(),
                    reason_code or "",
                    int(rank or 0),
                    _safe_float(rank_score),
                    _safe_float(confidence),
                    (side or "").strip().lower(),
                    str(dd.get("entry_thesis") or dd.get("setup_type") or ""),
                    _safe_float(dd.get("thesis_score")),
                    _safe_float(dd.get("thesis_invalid_level")),
                    _safe_float(dd.get("thesis_target_level")),
                    str(dd.get("regime_label") or dd.get("regime") or dd.get("ctx_market_regime") or ""),
                    str(dd.get("price_structure_regime") or ""),
                    _safe_float(dd.get("relative_volume"), _safe_float(dd.get("ctx_relative_volume"))),
                    _safe_float(dd.get("spread_pct")),
                    float(ESTIMATED_ROUNDTRIP_COST),
                    _safe_float(price),
                    json.dumps(open_position_state or {}, separators=(",", ":")) if open_position_state else None,
                ),
            )
            conn.commit()
    except sqlite3.Error as e:
        logger.debug("record_candidate_snapshot failed for %s: %s", symbol, e)


# =========================================================================
# Stream 3: position heartbeats
# =========================================================================


def _compute_gross_unrealized_pct(*, entry_price: float, mark: float) -> float:
    """Mark-based gross move fraction for long positions (0.0025 = +0.25%)."""
    if entry_price <= 0 or mark <= 0:
        return 0.0
    return (mark - entry_price) / entry_price


def _compute_gross_unrealized_usd(*, entry_price: float, mark: float, quantity: float) -> float:
    if entry_price <= 0 or mark <= 0 or quantity <= 0:
        return 0.0
    return quantity * (mark - entry_price)


def _compute_fee_estimate(
    *,
    mark: float,
    quantity: float,
    entry_fee: float = 0.0,
    sell_fee_rate: float | None = None,
) -> float:
    """Entry fee plus estimated sell-side fee at mark."""
    if mark <= 0 or quantity <= 0:
        return float(entry_fee or 0.0)
    rate = float(TAKER_FEE if sell_fee_rate is None else sell_fee_rate)
    return float(entry_fee or 0.0) + (quantity * mark * rate)


def _compute_mark_sell_net_usd(
    *,
    entry_price: float,
    mark: float,
    quantity: float,
    entry_fee: float = 0.0,
    sell_fee_rate: float | None = None,
) -> float:
    """Dollar net if sold at mark (entry cost basis + sell-side fee)."""
    if entry_price <= 0 or mark <= 0 or quantity <= 0:
        return 0.0
    gross_pnl = _compute_gross_unrealized_usd(entry_price=entry_price, mark=mark, quantity=quantity)
    return gross_pnl - _compute_fee_estimate(
        mark=mark,
        quantity=quantity,
        entry_fee=entry_fee,
        sell_fee_rate=sell_fee_rate,
    )


def _resolve_thesis_valid(
    *,
    thesis_valid: bool | None,
    entry_thesis: str,
    thesis_score: float,
    thesis_invalid_level: float,
    thesis_target_level: float,
    entry_vwap: float,
    entry_price: float,
    mark: float,
) -> bool | None:
    if thesis_valid is not None:
        return thesis_valid
    from backend.services.day_trade_thesis import EXIT_THESIS_WARNING, evaluate_thesis_exit

    thesis_eval = evaluate_thesis_exit(
        entry_thesis=entry_thesis,
        thesis_score=float(thesis_score or 0.0),
        thesis_invalid_level=float(thesis_invalid_level or 0.0),
        thesis_target_level=float(thesis_target_level or 0.0),
        entry_vwap=float(entry_vwap or 0.0),
        entry_price=entry_price,
        mark=mark,
        bundle=None,
    )
    action = str(thesis_eval.get("action") or "")
    reason = str(thesis_eval.get("reason") or "")
    if action == "default" and reason == "no_thesis":
        return None
    if action == "warn" or reason.startswith(EXIT_THESIS_WARNING):
        return False
    if thesis_invalid_level > 0 and entry_price > 0:
        return mark >= thesis_invalid_level
    if action in ("hold", "sell"):
        return True
    return None


def record_position_heartbeat(
    *,
    symbol: str,
    trade_id: str,
    entry_price: float,
    mark: float,
    entry_time_epoch: float,
    thesis_invalid_level: float = 0.0,
    thesis_target_level: float = 0.0,
    entry_thesis: str = "",
    quantity: float = 0.0,
    entry_fee: float = 0.0,
    thesis_valid: bool | None = None,
    thesis_score: float = 0.0,
    entry_vwap: float = 0.0,
    db_path: str = DATABASE_PATH,
) -> None:
    """Persist one open-position learning heartbeat (sync)."""
    ensure_learning_ingestion_tables(db_path)
    if entry_price <= 0 or mark <= 0:
        return
    unrealized = _compute_gross_unrealized_pct(entry_price=entry_price, mark=mark)
    net_unrealized = unrealized - float(ESTIMATED_ROUNDTRIP_COST)
    hold_sec = max(0.0, time.time() - float(entry_time_epoch or time.time()))
    try:
        with sqlite3.connect(db_path, timeout=10) as conn:
            row = conn.execute(
                "SELECT highest_since_entry, lowest_since_entry FROM ai_position_heartbeats WHERE trade_id = ? ORDER BY epoch_ms DESC LIMIT 1",
                (trade_id,),
            ).fetchone()
            prev_hi = float(row[0]) if row and row[0] else entry_price
            prev_lo = float(row[1]) if row and row[1] else entry_price
            hi = max(prev_hi, mark)
            lo = min(prev_lo, mark)
            mfe = (hi - entry_price) / entry_price
            mae = (entry_price - lo) / entry_price
            dist_target = ((thesis_target_level - mark) / mark) if thesis_target_level > 0 else None
            dist_invalid = ((mark - thesis_invalid_level) / mark) if thesis_invalid_level > 0 else None
            exit_allowed = net_unrealized >= float(MIN_NET_PROFIT_TO_SELL)
            gross_unrealized_pnl = _compute_gross_unrealized_usd(
                entry_price=entry_price,
                mark=mark,
                quantity=quantity,
            )
            fee_estimate = _compute_fee_estimate(
                mark=mark,
                quantity=quantity,
                entry_fee=entry_fee,
            )
            net_unrealized_pnl = _compute_mark_sell_net_usd(
                entry_price=entry_price,
                mark=mark,
                quantity=quantity,
                entry_fee=entry_fee,
            )
            would_sell_net_usd = net_unrealized_pnl
            thesis_valid = _resolve_thesis_valid(
                thesis_valid=thesis_valid,
                entry_thesis=entry_thesis,
                thesis_score=thesis_score,
                thesis_invalid_level=thesis_invalid_level,
                thesis_target_level=thesis_target_level,
                entry_vwap=entry_vwap,
                entry_price=entry_price,
                mark=mark,
            )
            conn.execute(
                """
                INSERT INTO ai_position_heartbeats (
                    ts_utc, epoch_ms, symbol, trade_id, entry_price, mark, unrealized_pct,
                    net_unrealized_pct, highest_since_entry, lowest_since_entry, mfe_pct, mae_pct,
                    dist_to_target_pct, dist_to_invalid_pct, thesis_valid, exit_allowed,
                    would_sell_now_net_usd, hold_seconds, entry_thesis,
                    quantity, gross_unrealized_pnl, net_unrealized_pnl, fee_estimate,
                    heartbeat_calc_version
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    _now_iso(),
                    int(time.time() * 1000),
                    symbol,
                    trade_id,
                    entry_price,
                    mark,
                    unrealized,
                    net_unrealized,
                    hi,
                    lo,
                    mfe,
                    mae,
                    dist_target,
                    dist_invalid,
                    None if thesis_valid is None else int(bool(thesis_valid)),
                    int(exit_allowed),
                    would_sell_net_usd,
                    hold_sec,
                    entry_thesis or "",
                    float(quantity or 0.0),
                    gross_unrealized_pnl,
                    net_unrealized_pnl,
                    fee_estimate,
                    HEARTBEAT_CALC_VERSION,
                ),
            )
            conn.commit()
    except sqlite3.Error as e:
        logger.debug("record_position_heartbeat failed for %s: %s", symbol, e)


# =========================================================================
# Stream 2: forward labeling of snapshots
# =========================================================================


def _bus(symbol: str) -> str:
    return (symbol or "").replace("/", "").upper()


def _load_series(r: Any, sym_bus: str) -> dict[str, list[list[float]]]:
    """Load cached OHLCV series (epoch SECONDS or MS in col 0 normalized to ms).

    Primary source is the cached day-active bundle (1m≈16h, 5m≈41h, 15m≈100h,
    1h≈41d). Shallow klines:* caches fill the freshest minutes.
    """
    out: dict[str, list[list[float]]] = {}
    with contextlib.suppress(Exception):
        raw_b = r.get(f"day_active_bundle:{sym_bus}")
        if isinstance(raw_b, bytes):
            raw_b = raw_b.decode("utf-8")
        if raw_b:
            bd = json.loads(raw_b)
            if isinstance(bd, dict):
                inner = bd.get("bundle") if isinstance(bd.get("bundle"), dict) else bd
                for tf in ("1m", "5m", "15m", "1h", "4h"):
                    rows = inner.get(tf)
                    if isinstance(rows, list) and rows and isinstance(rows[0], (list, tuple)):
                        out[tf] = [list(map(float, x[:6])) for x in rows if len(x) >= 6]
    for tf in ("1m", "5m", "15m"):
        if tf in out:
            continue
        with contextlib.suppress(Exception):
            raw = r.get(f"klines:{sym_bus}:{tf}")
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            if raw:
                rows = json.loads(raw)
                if isinstance(rows, list) and rows:
                    out[tf] = [list(map(float, x[:6])) for x in rows if len(x) >= 6]
    # Normalize timestamps to milliseconds
    for tf, rows in out.items():
        if rows and float(rows[-1][0]) < 1e12:
            out[tf] = [[float(x[0]) * 1000.0, *x[1:6]] for x in rows]
    return out


def _fetch_historical_klines_rest(
    sym_bus: str,
    interval: str,
    start_ms: float,
    end_ms: float,
    *,
    timeout_sec: float = 10.0,
) -> list[list[float]]:
    """
    Durable fallback: fetch a bounded historical OHLCV window directly from
    Binance.US public REST (no auth, no rate-limit risk at this call volume).

    Used only when Redis-cached series (day_active_bundle/klines:*, TTL on
    the order of ~2-4 minutes for the live bundle) do not cover a labeling
    window that a snapshot needs — e.g. after a Redis restart/flush or a
    service-downtime gap. Binance.US retains kline history indefinitely, so
    this makes forward labeling resilient to Redis cache TTL/continuity
    without ever persisting a second copy of full market data.
    """
    try:
        import httpx

        url = f"https://api.binance.us/api/v3/klines?symbol={sym_bus}&interval={interval}&startTime={int(start_ms)}&endTime={int(end_ms)}&limit=1000"
        with httpx.Client(timeout=timeout_sec) as client:
            resp = client.get(url)
            resp.raise_for_status()
            rows = resp.json()
        if not isinstance(rows, list):
            return []
        # Binance kline row: [open_time_ms, open, high, low, close, volume, ...]
        return [[float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5])] for x in rows if isinstance(x, (list, tuple)) and len(x) >= 6]
    except Exception as exc:
        logger.debug("FORWARD_LABEL_REST_FALLBACK_FAILED %s %s: %s", sym_bus, interval, exc)
        return []


def _ensure_series_covers_window(
    series: dict[str, list[list[float]]],
    *,
    sym_bus: str,
    t0_ms: float,
    end_ms: float,
) -> dict[str, list[list[float]]]:
    """
    If the Redis-derived series does not cover [t0_ms, end_ms] for any of the
    timeframes used by the labeler, fetch the missing window directly from
    Binance.US REST (durable fallback) and merge it in-memory for this run
    only. Never writes the fetched rows back to Redis or a new table.
    """
    for tf in ("1h", "15m", "5m"):
        rows = series.get(tf) or []
        covers_start = bool(rows) and float(rows[0][0]) <= t0_ms
        covers_end = bool(rows) and float(rows[-1][0]) >= end_ms
        if covers_start and covers_end:
            continue
        # Pad the request window slightly so boundary bars are included.
        pad_ms = 3 * 3600 * 1000.0 if tf == "1h" else 15 * 60 * 1000.0
        fetched = _fetch_historical_klines_rest(sym_bus, tf, t0_ms - pad_ms, end_ms + pad_ms)
        if fetched:
            merged = {row[0]: row for row in rows}
            for row in fetched:
                merged[row[0]] = row
            series[tf] = sorted(merged.values(), key=lambda x: x[0])
            logger.info(
                "FORWARD_LABEL_REST_FALLBACK_USED symbol=%s tf=%s fetched_rows=%d reason=redis_series_gap",
                sym_bus,
                tf,
                len(fetched),
            )
    return series


def _close_at(series: list[list[float]], target_ms: float) -> float | None:
    """Close of the last bar opening at/before target_ms (None outside coverage)."""
    if not series:
        return None
    if target_ms < float(series[0][0]) or target_ms > float(series[-1][0]) + 2 * (float(series[-1][0]) - float(series[-2][0]) if len(series) > 1 else 60000):
        return None
    best = None
    for row in series:
        if float(row[0]) <= target_ms:
            best = row
        else:
            break
    return float(best[4]) if best else None


def _window_extremes(series: list[list[float]], t0_ms: float, t1_ms: float) -> tuple[float, float] | None:
    hi = -math.inf
    lo = math.inf
    found = False
    for row in series:
        ts = float(row[0])
        if t0_ms <= ts <= t1_ms:
            hi = max(hi, float(row[2]))
            lo = min(lo, float(row[3]))
            found = True
        elif ts > t1_ms:
            break
    return (hi, lo) if found else None


_HORIZONS = (
    ("fwd_ret_15m", 15 * 60),
    ("fwd_ret_30m", 30 * 60),
    ("fwd_ret_1h", 3600),
    ("fwd_ret_4h", 4 * 3600),
    ("fwd_ret_24h", 24 * 3600),
)

# Invalidation stored on NO_TRADE/REJECT snapshots can sit within a few bps of
# entry (thesis level from bar eval, not an open position). Path labels must
# not treat a micro dip through that razor-thin level as a full invalidation
# when MAE is otherwise negligible vs the DAY default invalidation floor.
_MIN_PATH_INVALIDATION_MAE = 0.0125
_TIGHT_INVALIDATION_DIST = 0.002


def _path_invalidation_breached(p0: float, mae: float, inv_level: float) -> bool:
    """True only when the forward path materially breached invalidation."""
    if p0 <= 0:
        return False
    mae = max(0.0, float(mae or 0.0))
    if inv_level <= 0:
        return mae >= _MIN_PATH_INVALIDATION_MAE
    inv_dist = (p0 - inv_level) / p0
    if inv_dist <= _TIGHT_INVALIDATION_DIST:
        return mae >= _MIN_PATH_INVALIDATION_MAE
    return (p0 * (1.0 - mae)) < inv_level


def label_pending_snapshots(db_path: str = DATABASE_PATH) -> dict[str, int]:
    """Label PENDING/PARTIAL snapshots whose horizons have elapsed. Returns counters."""
    ensure_learning_ingestion_tables(db_path)
    counters = {"labeled": 0, "partial": 0, "unlabelable": 0, "scanned": 0}
    try:
        from backend.config.redis_config import get_redis_client

        r = get_redis_client()
    except Exception:
        r = None
    if r is None:
        return counters

    now_ms = time.time() * 1000.0
    cutoff_ms = now_ms - 15 * 60 * 1000
    with sqlite3.connect(db_path, timeout=15) as conn:
        conn.row_factory = sqlite3.Row
        # Split the batch so PARTIAL completion work cannot starve first-horizon PENDING.
        half = max(1, int(LABEL_BATCH_LIMIT) // 2)
        _cols = (
            "id, symbol, epoch_ms, price, decision, thesis_invalid_level, thesis_target_level, "
            "fwd_ret_15m, fwd_ret_30m, fwd_ret_1h, fwd_ret_4h, fwd_ret_24h"
        )
        pending_rows = conn.execute(
            f"""
            SELECT {_cols}
            FROM ai_candidate_snapshots
            WHERE label_status='PENDING' AND epoch_ms <= ?
            ORDER BY epoch_ms ASC
            LIMIT ?
            """,
            (cutoff_ms, half),
        ).fetchall()
        remain = max(0, int(LABEL_BATCH_LIMIT) - len(pending_rows))
        partial_rows = conn.execute(
            f"""
            SELECT {_cols}
            FROM ai_candidate_snapshots
            WHERE label_status='PARTIAL' AND epoch_ms <= ?
            ORDER BY epoch_ms ASC
            LIMIT ?
            """,
            (cutoff_ms, remain),
        ).fetchall()
        if len(pending_rows) + len(partial_rows) < int(LABEL_BATCH_LIMIT):
            need = int(LABEL_BATCH_LIMIT) - len(pending_rows) - len(partial_rows)
            if len(pending_rows) < half:
                extra = conn.execute(
                    f"""
                    SELECT {_cols}
                    FROM ai_candidate_snapshots
                    WHERE label_status='PARTIAL' AND epoch_ms <= ?
                    ORDER BY epoch_ms ASC
                    LIMIT ? OFFSET ?
                    """,
                    (cutoff_ms, need, len(partial_rows)),
                ).fetchall()
                partial_rows = list(partial_rows) + list(extra)
            else:
                extra = conn.execute(
                    f"""
                    SELECT {_cols}
                    FROM ai_candidate_snapshots
                    WHERE label_status='PENDING' AND epoch_ms <= ?
                    ORDER BY epoch_ms ASC
                    LIMIT ? OFFSET ?
                    """,
                    (cutoff_ms, need, len(pending_rows)),
                ).fetchall()
                pending_rows = list(pending_rows) + list(extra)
        rows = list(pending_rows) + list(partial_rows)
        counters["scanned"] = len(rows)
        counters["scanned_pending"] = len(pending_rows)
        counters["scanned_partial"] = len(partial_rows)
        if not rows:
            _prune_old_rows(conn)
            return counters

        series_cache: dict[str, dict[str, list[list[float]]]] = {}
        for row in rows:
            sym_bus = _bus(row["symbol"])
            if sym_bus not in series_cache:
                series_cache[sym_bus] = _load_series(r, sym_bus)
            series = series_cache[sym_bus]
            t0 = float(row["epoch_ms"])
            p0 = float(row["price"] or 0.0)
            age_sec = (now_ms - t0) / 1000.0
            if p0 <= 0:
                conn.execute(
                    "UPDATE ai_candidate_snapshots SET label_status='UNLABELABLE', labeled_at_utc=? WHERE id=?",
                    (_now_iso(), row["id"]),
                )
                counters["unlabelable"] += 1
                continue

            def _missing_horizon_cols() -> list[tuple[str, int]]:
                return [(col, horizon_sec) for col, horizon_sec in _HORIZONS if row[col] is None and age_sec >= horizon_sec]

            updates: dict[str, float] = {}
            still_missing = _missing_horizon_cols()
            if still_missing:
                for col, horizon_sec in still_missing:
                    target = t0 + horizon_sec * 1000.0
                    px = None
                    for tf in ("1m", "5m", "15m", "1h"):
                        px = _close_at(series.get(tf) or [], target)
                        if px is not None:
                            break
                    if px is not None and px > 0:
                        updates[col] = (px - p0) / p0
                # Durable REST fallback: only fire once redis-derived coverage
                # genuinely fails a needed lookup AND the row is old enough that
                # routine retry-next-cycle is unlikely to help (avoids hammering
                # the public REST endpoint every ~2min for ordinary in-flight rows).
                remaining_missing = [c for c, h in still_missing if c not in updates]
                if remaining_missing and age_sec >= (LABEL_GIVEUP_AGE_SEC * 0.5):
                    max_needed_horizon = max(h for _, h in still_missing)
                    series = _ensure_series_covers_window(
                        series,
                        sym_bus=sym_bus,
                        t0_ms=t0,
                        end_ms=t0 + max_needed_horizon * 1000.0,
                    )
                    series_cache[sym_bus] = series
                    for col, horizon_sec in still_missing:
                        if col in updates:
                            continue
                        target = t0 + horizon_sec * 1000.0
                        px = None
                        for tf in ("1h", "15m", "5m", "1m"):
                            px = _close_at(series.get(tf) or [], target)
                            if px is not None:
                                break
                        if px is not None and px > 0:
                            updates[col] = (px - p0) / p0

            full_done = age_sec >= LABEL_FINAL_HORIZON_SEC
            mfe = mae = None
            if full_done:
                window_end = t0 + LABEL_FINAL_HORIZON_SEC * 1000.0
                for tf in ("5m", "15m", "1h"):
                    ext = _window_extremes(series.get(tf) or [], t0, window_end)
                    if ext:
                        mfe = (ext[0] - p0) / p0
                        mae = (p0 - ext[1]) / p0
                        break
                if mfe is None and age_sec >= (LABEL_GIVEUP_AGE_SEC * 0.5):
                    # Same durable REST fallback for the MFE/MAE window as used
                    # above for point-in-time forward returns.
                    series = _ensure_series_covers_window(series, sym_bus=sym_bus, t0_ms=t0, end_ms=window_end)
                    series_cache[sym_bus] = series
                    for tf in ("5m", "15m", "1h"):
                        ext = _window_extremes(series.get(tf) or [], t0, window_end)
                        if ext:
                            mfe = (ext[0] - p0) / p0
                            mae = (p0 - ext[1]) / p0
                            break

            have_24h = updates.get("fwd_ret_24h") is not None or row["fwd_ret_24h"] is not None
            if full_done and (mfe is not None or have_24h):
                inv_level = float(row["thesis_invalid_level"] or 0.0)
                would_hit = None
                breach = None
                if mfe is not None:
                    would_hit = int((mfe - float(ESTIMATED_ROUNDTRIP_COST)) >= float(MIN_NET_PROFIT_TO_SELL))
                    breach = int(_path_invalidation_breached(p0, float(mae or 0.0), inv_level))
                decision = str(row["decision"] or "").upper()
                verdict = "NEUTRAL"
                if would_hit is not None:
                    opportunity = bool(would_hit) and not bool(breach)
                    if decision == "BUY":
                        verdict = "GOOD_BUY" if opportunity else "BAD_BUY"
                    else:
                        verdict = "MISSED_OPPORTUNITY" if opportunity else "CORRECT_REJECT"
                set_parts = ["label_status='LABELED'", "labeled_at_utc=?", "mfe_pct=?", "mae_pct=?", "would_hit_target=?", "would_breach_invalidation=?", "verdict=?"]
                params: list[Any] = [_now_iso(), mfe, mae, would_hit, breach, verdict]
                for col, val in updates.items():
                    set_parts.append(f"{col}=?")
                    params.append(val)
                params.append(row["id"])
                conn.execute(f"UPDATE ai_candidate_snapshots SET {', '.join(set_parts)} WHERE id=?", params)
                counters["labeled"] += 1
            elif age_sec >= LABEL_GIVEUP_AGE_SEC:
                conn.execute(
                    "UPDATE ai_candidate_snapshots SET label_status='UNLABELABLE', labeled_at_utc=? WHERE id=?",
                    (_now_iso(), row["id"]),
                )
                counters["unlabelable"] += 1
            elif updates:
                set_parts = ["label_status='PARTIAL'"]
                params = []
                for col, val in updates.items():
                    set_parts.append(f"{col}=?")
                    params.append(val)
                params.append(row["id"])
                conn.execute(f"UPDATE ai_candidate_snapshots SET {', '.join(set_parts)} WHERE id=?", params)
                counters["partial"] += 1
            else:
                # No progress this pass — leave status, but do not let ancient
                # no-progress PENDING monopolize the next ORDER BY batch forever.
                counters["deferred"] = int(counters.get("deferred") or 0) + 1
        _prune_old_rows(conn)
        conn.commit()
    counters["reconciled"] = _reconcile_verdict_batch(db_path, limit=100)
    return counters


def _reconcile_verdict_batch(db_path: str = DATABASE_PATH, limit: int = 100) -> int:
    """Re-apply breach logic to already-labeled rows (bounded batch after rule fixes)."""
    updated = 0
    try:
        with sqlite3.connect(db_path, timeout=10) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT id, price, decision, mfe_pct, mae_pct, thesis_invalid_level, would_hit_target, verdict
                FROM ai_candidate_snapshots
                WHERE label_status='LABELED' AND would_hit_target=1 AND mfe_pct IS NOT NULL
                ORDER BY id DESC LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
            for row in rows:
                p0 = float(row["price"] or 0.0)
                inv = float(row["thesis_invalid_level"] or 0.0)
                mae = float(row["mae_pct"] or 0.0)
                breach = _path_invalidation_breached(p0, mae, inv)
                decision = str(row["decision"] or "").upper()
                opportunity = bool(int(row["would_hit_target"] or 0)) and not breach
                if decision == "BUY":
                    new_verdict = "GOOD_BUY" if opportunity else "BAD_BUY"
                else:
                    new_verdict = "MISSED_OPPORTUNITY" if opportunity else "CORRECT_REJECT"
                if new_verdict != row["verdict"]:
                    conn.execute(
                        "UPDATE ai_candidate_snapshots SET verdict=?, would_breach_invalidation=? WHERE id=?",
                        (new_verdict, int(breach), row["id"]),
                    )
                    updated += 1
            conn.commit()
    except sqlite3.Error as e:
        logger.debug("verdict reconcile skipped: %s", e)
    return updated


def _prune_old_rows(conn: sqlite3.Connection) -> None:
    cutoff_snap = (time.time() - SNAPSHOT_RETENTION_DAYS * 86400) * 1000
    cutoff_hb = (time.time() - HEARTBEAT_RETENTION_DAYS * 86400) * 1000
    with contextlib.suppress(sqlite3.Error):
        conn.execute("DELETE FROM ai_candidate_snapshots WHERE epoch_ms < ?", (cutoff_snap,))
        conn.execute("DELETE FROM ai_position_heartbeats WHERE epoch_ms < ?", (cutoff_hb,))


# =========================================================================
# Stream 5: missed-move ranking adjustments (bounded; never opens trades)
# =========================================================================


def missed_move_rank_adjustments(
    lookback_days: int = 21,
    db_path: str = DATABASE_PATH,
) -> dict[str, float]:
    """Per-symbol bounded rank delta from labeled non-BUY snapshots.

    High missed-opportunity rate nudges rank up (max +0.06); a symbol whose
    rejections keep proving correct gets a small negative nudge (max -0.04).
    Execution gates are unchanged — this only reorders ranking among top-4.
    """
    ensure_learning_ingestion_tables(db_path)
    out: dict[str, float] = {}
    cutoff = (time.time() - lookback_days * 86400) * 1000
    try:
        with sqlite3.connect(db_path, timeout=10) as conn:
            rows = conn.execute(
                """
                SELECT symbol,
                       SUM(CASE WHEN verdict='MISSED_OPPORTUNITY' THEN 1 ELSE 0 END) AS missed,
                       SUM(CASE WHEN verdict='CORRECT_REJECT' THEN 1 ELSE 0 END) AS correct
                FROM ai_candidate_snapshots
                WHERE label_status='LABELED' AND decision != 'BUY' AND epoch_ms >= ?
                GROUP BY symbol
                """,
                (cutoff,),
            ).fetchall()
        for sym, missed, correct in rows:
            total = int(missed or 0) + int(correct or 0)
            if total < 5:
                continue
            rate = float(missed or 0) / total
            # Stronger feedback so missed-ops actually move top-4 ranking.
            delta = (rate - 0.28) * 0.18
            out[str(sym)] = max(-0.04, min(0.06, round(delta, 5)))
    except sqlite3.Error as e:
        logger.debug("missed_move_rank_adjustments failed: %s", e)
    return out


# =========================================================================
# Tier B / Tier C training row extraction
# =========================================================================


def _features_for_decision_ids(
    conn: sqlite3.Connection,
    decision_ids: list[str],
    feature_dim: int,
    min_feature_version: int,
) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    if not decision_ids:
        return out
    chunk = 400
    for i in range(0, len(decision_ids), chunk):
        ids = decision_ids[i : i + chunk]
        q = ",".join("?" for _ in ids)
        rows = conn.execute(
            f"""
            SELECT decision_id, features_json, feature_version FROM ai_inference_log
            WHERE decision_id IN ({q}) AND features_json IS NOT NULL
            """,
            ids,
        ).fetchall()
        for did, fj, fv in rows:
            try:
                if int(fv or 0) < min_feature_version:
                    continue
                feats = json.loads(fj)
                if isinstance(feats, list) and len(feats) == feature_dim:
                    try:
                        from backend.services.day_feature_health import zero_learning_blocked_feature_dims

                        out[str(did)] = zero_learning_blocked_feature_dims([float(x) for x in feats])
                    except Exception:
                        out[str(did)] = [float(x) for x in feats]
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
    return out


def tier_c_training_rows(
    *,
    strategy_id: str,
    symbol: str,
    feature_dim: int = 145,
    min_feature_version: int = 5,
    db_path: str = DATABASE_PATH,
) -> tuple[list[list[float]], list[int]]:
    """Labeled candidate snapshots → (X, y). Label 1 = forward path reached
    the net-profit target without breaching invalidation."""
    ensure_learning_ingestion_tables(db_path)
    sid = (strategy_id or "day").strip().lower()
    xs: list[list[float]] = []
    ys: list[int] = []
    try:
        with sqlite3.connect(db_path, timeout=15) as conn:
            rows = conn.execute(
                """
                SELECT decision_id, would_hit_target, would_breach_invalidation
                FROM ai_candidate_snapshots
                WHERE label_status='LABELED' AND strategy_id=? AND symbol IN (?, ?)
                  AND decision_id != '' AND would_hit_target IS NOT NULL
                ORDER BY epoch_ms DESC LIMIT 3000
                """,
                (sid, symbol, _bus(symbol)),
            ).fetchall()
            dids = [str(r[0]) for r in rows]
            feats_map = _features_for_decision_ids(conn, dids, feature_dim, min_feature_version)
            for did, hit, breach in rows:
                feats = feats_map.get(str(did))
                if feats is None:
                    continue
                label = 1 if (int(hit or 0) == 1 and int(breach or 0) == 0) else 0
                xs.append(feats)
                ys.append(label)
    except sqlite3.Error as e:
        logger.debug("tier_c_training_rows failed for %s: %s", symbol, e)
    return xs, ys


def tier_b_training_rows(
    *,
    strategy_id: str,
    symbol: str,
    feature_dim: int = 145,
    min_feature_version: int = 5,
    min_hold_seconds: float = 4 * 3600,
    db_path: str = DATABASE_PATH,
) -> tuple[list[list[float]], list[int]]:
    """Open-trade MFE/MAE path labels → (X, y). For each trade with enough
    hold time, label 1 if the path's MFE cleared the net-profit floor after
    costs. Features come from the BUY snapshot's inference-time vector."""
    ensure_learning_ingestion_tables(db_path)
    (strategy_id or "day").strip().lower()
    xs: list[list[float]] = []
    ys: list[int] = []
    try:
        with sqlite3.connect(db_path, timeout=15) as conn:
            rows = conn.execute(
                """
                SELECT h.trade_id, MAX(h.mfe_pct) AS mfe, MAX(h.hold_seconds) AS hold_sec,
                       MIN(h.epoch_ms) AS first_ms, h.symbol
                FROM ai_position_heartbeats h
                WHERE h.symbol IN (?, ?)
                GROUP BY h.trade_id
                HAVING hold_sec >= ?
                ORDER BY first_ms DESC LIMIT 500
                """,
                (symbol, _bus(symbol), float(min_hold_seconds)),
            ).fetchall()
            for _trade_id, mfe, _hold, first_ms, sym in rows:
                snap = conn.execute(
                    """
                    SELECT decision_id FROM ai_candidate_snapshots
                    WHERE symbol IN (?, ?) AND decision='BUY' AND decision_id != ''
                      AND ABS(epoch_ms - ?) < 1800000
                    ORDER BY ABS(epoch_ms - ?) ASC LIMIT 1
                    """,
                    (sym, _bus(sym), float(first_ms or 0), float(first_ms or 0)),
                ).fetchone()
                if not snap:
                    continue
                feats_map = _features_for_decision_ids(conn, [str(snap[0])], feature_dim, min_feature_version)
                feats = feats_map.get(str(snap[0]))
                if feats is None:
                    continue
                label = 1 if (float(mfe or 0.0) - float(ESTIMATED_ROUNDTRIP_COST)) >= float(MIN_NET_PROFIT_TO_SELL) else 0
                xs.append(feats)
                ys.append(label)
    except sqlite3.Error as e:
        logger.debug("tier_b_training_rows failed for %s: %s", symbol, e)
    return xs, ys


def tiered_holdout_eval_rows(
    *,
    strategy_id: str,
    symbol: str,
    feature_dim: int = 145,
    min_feature_version: int = 5,
    db_path: str = DATABASE_PATH,
) -> tuple[list[list[float]], list[int]]:
    """Synthetic (Tier C) holdout for promotion fallback when real closed-trade
    holdout is too scarce. Never mixed into real-PnL metrics."""
    return tier_c_training_rows(
        strategy_id=strategy_id,
        symbol=symbol,
        feature_dim=feature_dim,
        min_feature_version=min_feature_version,
        db_path=db_path,
    )


# =========================================================================
# Stream 8: learning health summary
# =========================================================================


def learning_health_summary(db_path: str = DATABASE_PATH) -> dict[str, Any]:
    ensure_learning_ingestion_tables(db_path)
    out: dict[str, Any] = {
        "generated_at": _now_iso(),
        "totals": {},
        "per_symbol": {},
        "warnings": [],
    }
    try:
        with sqlite3.connect(db_path, timeout=10) as conn:
            t = out["totals"]
            for name, q in (
                ("closed_outcome_rows", "SELECT COUNT(*) FROM ai_outcome_training_rows"),
                ("trade_learning_outcomes", "SELECT COUNT(*) FROM trade_learning_outcomes"),
                ("candidate_snapshots", "SELECT COUNT(*) FROM ai_candidate_snapshots"),
                ("candidate_snapshots_pending", "SELECT COUNT(*) FROM ai_candidate_snapshots WHERE label_status IN ('PENDING','PARTIAL')"),
                ("candidate_snapshots_labeled", "SELECT COUNT(*) FROM ai_candidate_snapshots WHERE label_status='LABELED'"),
                ("position_heartbeats", "SELECT COUNT(*) FROM ai_position_heartbeats"),
                ("missed_opportunities", "SELECT COUNT(*) FROM ai_candidate_snapshots WHERE verdict='MISSED_OPPORTUNITY'"),
            ):
                with contextlib.suppress(sqlite3.Error):
                    t[name] = int(conn.execute(q).fetchone()[0])

            per_sym = out["per_symbol"]
            with contextlib.suppress(sqlite3.Error):
                for sym, cnt in conn.execute("SELECT symbol, COUNT(*) FROM ai_candidate_snapshots GROUP BY symbol").fetchall():
                    per_sym.setdefault(str(sym), {})["snapshots"] = int(cnt)
            with contextlib.suppress(sqlite3.Error):
                for sym, cnt in conn.execute("SELECT symbol, COUNT(*) FROM ai_candidate_snapshots WHERE label_status='LABELED' GROUP BY symbol").fetchall():
                    per_sym.setdefault(str(sym), {})["labeled_snapshots"] = int(cnt)
            with contextlib.suppress(sqlite3.Error):
                for sym, cnt in conn.execute("SELECT symbol, COUNT(*) FROM ai_position_heartbeats GROUP BY symbol").fetchall():
                    per_sym.setdefault(str(sym), {})["heartbeats"] = int(cnt)
            with contextlib.suppress(sqlite3.Error):
                for sym, cnt in conn.execute("SELECT symbol, COUNT(*) FROM ai_outcome_training_rows WHERE strategy_id='day' GROUP BY symbol").fetchall():
                    per_sym.setdefault(str(sym), {})["closed_outcomes"] = int(cnt)
            with contextlib.suppress(sqlite3.Error):
                for sym, reason, at in conn.execute(
                    """
                    SELECT symbol, reason, MAX(created_at) FROM ai_model_promotion_events
                    WHERE event_type IN ('reject','promote') GROUP BY symbol
                    """
                ).fetchall():
                    per_sym.setdefault(str(sym), {})["last_promotion_event"] = f"{reason} @ {at}"

        # Model artifact freshness
        with contextlib.suppress(Exception):
            from pathlib import Path

            from backend.config.trading_universe import TRADING_SYMBOLS
            from backend.utils.path_helpers import ensure_model_directories

            active = Path(ensure_model_directories()["active"]) / "day"
            for sym in TRADING_SYMBOLS:
                p = active / f"{sym}_direction.pkl"
                if p.exists():
                    age_days = (time.time() - p.stat().st_mtime) / 86400.0
                    per = out["per_symbol"].setdefault(sym, {})
                    per["model_active_date"] = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%d")
                    per["model_age_days"] = round(age_days, 1)
                    if age_days > 7:
                        out["warnings"].append(f"{sym}: active model {age_days:.0f}d old")

        closed = int(out["totals"].get("closed_outcome_rows") or 0)
        labeled = int(out["totals"].get("candidate_snapshots_labeled") or 0)
        if closed < 200 and labeled < 200:
            out["warnings"].append("DATA_STARVATION: closed outcomes and labeled snapshots both sparse")

        out["heartbeat_schema"] = _heartbeat_schema_health(db_path)
    except sqlite3.Error as e:
        out["error"] = str(e)
    return out


def _heartbeat_schema_health(db_path: str = DATABASE_PATH) -> dict[str, Any]:
    """
    Heartbeat telemetry schema observability (repair-all continuation, Phase 3).

    record_position_heartbeat() is the sole producer of ai_position_heartbeats
    and always stamps HEARTBEAT_CALC_VERSION on every write, so there is
    exactly one *active* writer schema at any time. Older rows (a lower
    heartbeat_calc_version) are retained for history but must never be
    reported as current runtime evidence — "active" freshness here is
    computed only from the current-version rows.
    """
    health: dict[str, Any] = {
        "current_version": HEARTBEAT_CALC_VERSION,
        "version_counts": {},
        "current_version_row_count": 0,
        "historical_row_count": 0,
        "latest_current_version_heartbeat_utc": None,
        "latest_current_version_age_sec": None,
        "active_health": "UNKNOWN",
    }
    try:
        with sqlite3.connect(db_path, timeout=10) as conn:
            for ver, cnt in conn.execute("SELECT heartbeat_calc_version, COUNT(*) FROM ai_position_heartbeats GROUP BY heartbeat_calc_version").fetchall():
                health["version_counts"][str(ver)] = int(cnt)
            health["current_version_row_count"] = int(health["version_counts"].get(str(HEARTBEAT_CALC_VERSION), 0))
            health["historical_row_count"] = sum(health["version_counts"].values()) - health["current_version_row_count"]

            latest = conn.execute(
                "SELECT ts_utc, epoch_ms FROM ai_position_heartbeats WHERE heartbeat_calc_version = ? ORDER BY epoch_ms DESC LIMIT 1",
                (HEARTBEAT_CALC_VERSION,),
            ).fetchone()
            if latest:
                health["latest_current_version_heartbeat_utc"] = latest[0]
                age_sec = max(0.0, time.time() - float(latest[1] or 0) / 1000.0)
                health["latest_current_version_age_sec"] = round(age_sec, 1)
                # Positions heartbeat while genuinely open; a healthy runtime with
                # open positions emits one roughly every few minutes. No open
                # positions -> no recent heartbeat is expected and not unhealthy.
                health["active_health"] = "FRESH" if age_sec <= 900 else "STALE"
            elif health["historical_row_count"] > 0:
                health["active_health"] = "NO_CURRENT_VERSION_DATA"
            else:
                health["active_health"] = "NO_DATA_YET"
    except sqlite3.Error as e:
        health["error"] = str(e)
    return health


_SCALP_OUTCOMES_SCHEMA = """
CREATE TABLE IF NOT EXISTS scalp_learning_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ingested_at TEXT NOT NULL,
    source_id INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    setup_name TEXT,
    entry_timestamp REAL,
    exit_timestamp REAL,
    entry_price REAL,
    exit_price REAL,
    quantity REAL,
    fees_paid REAL,
    slippage_cost REAL,
    net_pnl_usd REAL,
    net_pnl_pct REAL,
    hold_seconds REAL,
    exit_reason TEXT,
    UNIQUE(source_id)
);
CREATE INDEX IF NOT EXISTS ix_scalp_lo_symbol ON scalp_learning_outcomes(symbol, exit_timestamp);
CREATE INDEX IF NOT EXISTS ix_scalp_lo_setup ON scalp_learning_outcomes(setup_name, exit_timestamp);
"""

_scalp_outcomes_tables_ready = False


def _ensure_scalp_outcomes_table(db_path: str) -> None:
    global _scalp_outcomes_tables_ready
    if _scalp_outcomes_tables_ready:
        return
    with sqlite3.connect(db_path) as conn:
        for stmt in _SCALP_OUTCOMES_SCHEMA.strip().split(";"):
            s = stmt.strip()
            if s:
                conn.execute(s)
        conn.commit()
    _scalp_outcomes_tables_ready = True


def ingest_scalp_outcomes(db_path: str = DATABASE_PATH) -> dict[str, int]:
    """Ingest closed SCALP paper trade outcomes into scalp_learning_outcomes.

    Reads from trade_learning_outcomes (rows written by paper_engine._after_commit
    with engine='binance_scalp_paper' in extra_json) and normalizes them into a
    separate scalp_learning_outcomes table so the AI training pipeline can consume
    SCALP outcomes without mixing the DAY and SCALP datasets.

    Returns counters: {"ingested": N, "skipped": N, "errors": N}
    """
    _ensure_scalp_outcomes_table(db_path)
    counters = {"ingested": 0, "skipped": 0, "errors": 0}
    try:
        with sqlite3.connect(db_path, timeout=15) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT id, symbol, entry_timestamp, exit_timestamp,
                       entry_price, exit_price, quantity,
                       fees_paid, slippage_cost,
                       net_profit_usd, net_profit_pct,
                       hold_seconds, close_reason, extra_json
                FROM trade_learning_outcomes
                WHERE extra_json LIKE '%binance_scalp_paper%'
                  AND exit_timestamp IS NOT NULL
                ORDER BY id ASC
                """
            ).fetchall()

            existing_ids = {
                r[0]
                for r in conn.execute("SELECT source_id FROM scalp_learning_outcomes").fetchall()
            }

            for row in rows:
                source_id = int(row["id"])
                if source_id in existing_ids:
                    counters["skipped"] += 1
                    continue
                try:
                    extra: dict = {}
                    with contextlib.suppress(Exception):
                        raw = row["extra_json"]
                        if raw:
                            extra = json.loads(raw) if isinstance(raw, str) else {}
                    setup_name = str(extra.get("setup") or "")
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO scalp_learning_outcomes
                        (ingested_at, source_id, symbol, setup_name,
                         entry_timestamp, exit_timestamp, entry_price, exit_price,
                         quantity, fees_paid, slippage_cost,
                         net_pnl_usd, net_pnl_pct, hold_seconds, exit_reason)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            _now_iso(),
                            source_id,
                            str(row["symbol"] or ""),
                            setup_name,
                            float(row["entry_timestamp"] or 0),
                            float(row["exit_timestamp"] or 0),
                            float(row["entry_price"] or 0),
                            float(row["exit_price"] or 0),
                            float(row["quantity"] or 0),
                            float(row["fees_paid"] or 0),
                            float(row["slippage_cost"] or 0),
                            float(row["net_profit_usd"] or 0),
                            float(row["net_profit_pct"] or 0),
                            float(row["hold_seconds"] or 0),
                            str(row["close_reason"] or ""),
                        ),
                    )
                    counters["ingested"] += 1
                except Exception as e:
                    logger.debug("ingest_scalp_outcomes row %s failed: %s", source_id, e)
                    counters["errors"] += 1
            conn.commit()
    except sqlite3.Error as e:
        logger.debug("ingest_scalp_outcomes failed: %s", e)
    return counters


def scalp_outcomes_summary(db_path: str = DATABASE_PATH) -> dict[str, Any]:
    """Return basic stats from scalp_learning_outcomes for health reporting."""
    _ensure_scalp_outcomes_table(db_path)
    out: dict[str, Any] = {"total": 0, "wins": 0, "losses": 0, "net_pnl": 0.0, "per_setup": {}}
    try:
        with sqlite3.connect(db_path, timeout=10) as conn:
            row = conn.execute(
                """
                SELECT COUNT(*),
                       SUM(CASE WHEN net_pnl_usd > 0 THEN 1 ELSE 0 END),
                       SUM(CASE WHEN net_pnl_usd <= 0 THEN 1 ELSE 0 END),
                       COALESCE(SUM(net_pnl_usd), 0)
                FROM scalp_learning_outcomes
                """
            ).fetchone()
            if row:
                out["total"] = int(row[0] or 0)
                out["wins"] = int(row[1] or 0)
                out["losses"] = int(row[2] or 0)
                out["net_pnl"] = float(row[3] or 0)
            for setup, cnt, wins, pnl in conn.execute(
                """
                SELECT setup_name,
                       COUNT(*),
                       SUM(CASE WHEN net_pnl_usd > 0 THEN 1 ELSE 0 END),
                       COALESCE(SUM(net_pnl_usd), 0)
                FROM scalp_learning_outcomes
                GROUP BY setup_name
                """
            ).fetchall():
                out["per_setup"][str(setup or "")] = {
                    "count": int(cnt or 0),
                    "wins": int(wins or 0),
                    "net_pnl": float(pnl or 0),
                }
    except sqlite3.Error as e:
        logger.debug("scalp_outcomes_summary failed: %s", e)
    return out


__all__ = [
    "LABEL_BATCH_LIMIT",
    "MAX_TIER_BC_SHARE",
    "TIER_B_WEIGHT",
    "TIER_C_WEIGHT",
    "ensure_learning_ingestion_tables",
    "ingest_scalp_outcomes",
    "label_pending_snapshots",
    "learning_health_summary",
    "missed_move_rank_adjustments",
    "record_candidate_snapshot",
    "record_position_heartbeat",
    "scalp_outcomes_summary",
    "tier_b_training_rows",
    "tier_c_training_rows",
    "tiered_holdout_eval_rows",
]
