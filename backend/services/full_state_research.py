"""Research-only full-market-state rows. Nothing here places or sizes an order.

Every decision interval, for each of BTC, ETH, SOL and XRP, one causal state
is stored whether or not a legacy setup fired. The setup name is metadata
(``NO_SETUP`` when none fired). Future labels are market opportunity only:
bid at the horizon over ask at the decision, never a live-policy P&L.

DAY and SCALP keep separate label queues. A 12h DAY horizon cannot occupy
the SCALP resolver. There is no row-count gate and no live permission change.
"""

from __future__ import annotations

import contextlib
import json
import math
import sqlite3
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from backend.services.horizon_alignment import (
    BAR_CLOSE_SOURCE,
    DAY_HORIZONS,
    LABEL_VERSION,
    SCALP_HORIZONS,
    align_observation,
    bar_close_known_at,
    executable_gross,
)

FULL_STATE_VERSION = "FULL_STATE_V1"
NO_SETUP = "NO_SETUP"
MARKET_KIND = "MARKET_OPPORTUNITY"
DAY_ENGINE = "DAY_V2"
SCALP_ENGINE = "SCALP_V2"
DAY_SAMPLE_SEC = 900
SCALP_SAMPLE_SEC = 30
VECTOR_MAX_AGE_SEC = 300.0
HALF_SPREAD = 0.00006
STATES = "research_market_states"
LABELS = "research_market_labels"

SCALP_FEATURE_KEYS: tuple[str, ...] = (
    "obi_l1",
    "obi_l5",
    "obi_l10",
    "obi_l20",
    "ofi_1s",
    "ofi_5s",
    "ofi_30s",
    "spread_pct",
    "microprice_pressure",
    "agg_flow_imbalance_5s",
    "bid_cancelled_5s",
    "ask_cancelled_5s",
    "obi_l10_persistence_5s",
    "obi_l10_reversal_freq_5s",
    "data_age_sec",
)

_last_scalp_bucket: dict[str, int] = {}
_last_mid: dict[str, float] = {}

Observe = Callable[[str, str, float, int], list[tuple[float, float, str]]]


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"""CREATE TABLE IF NOT EXISTS {STATES} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            engine TEXT NOT NULL,
            symbol TEXT NOT NULL,
            decision_ts REAL NOT NULL,
            feature_ts REAL,
            features_json TEXT NOT NULL,
            setup_label TEXT NOT NULL,
            context_json TEXT NOT NULL,
            entry_ask REAL,
            entry_bid REAL,
            entry_obs_ts REAL,
            entry_source TEXT,
            cost REAL NOT NULL,
            version TEXT NOT NULL,
            labels_done INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL,
            UNIQUE (engine, symbol, decision_ts)
        )"""
    )
    conn.execute(
        f"""CREATE TABLE IF NOT EXISTS {LABELS} (
            state_id INTEGER NOT NULL,
            decision_ts REAL NOT NULL,
            horizon_sec INTEGER NOT NULL,
            status TEXT NOT NULL,
            obs_ts REAL,
            source TEXT,
            timing_error_sec REAL,
            gross REAL,
            net REAL,
            kind TEXT NOT NULL CHECK (kind = '{MARKET_KIND}'),
            version TEXT NOT NULL,
            PRIMARY KEY (state_id, horizon_sec)
        )"""
    )


def _base(symbol: str) -> str:
    return str(symbol or "").upper().replace("/", "").replace("-", "").replace("USDT", "")


def _variants(symbol: str) -> tuple[str, ...]:
    raw = str(symbol or "").upper().replace("/", "").replace("-", "")
    return (raw, raw.replace("USDT", "/USDT"), raw.replace("USDT", "-USDT"))


def _epoch(ts: Any) -> float | None:
    if isinstance(ts, int | float):
        return float(ts)
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.timestamp()


def _cost_beyond_spread() -> float:
    from backend.config.trading_economics import SLIPPAGE_BUFFER, TAKER_FEE

    return 2.0 * float(TAKER_FEE) + 2.0 * float(SLIPPAGE_BUFFER)


def record_state(
    db_path: str,
    *,
    engine: str,
    symbol: str,
    decision_ts: float,
    feature_ts: float | None,
    features: dict[str, float],
    setup_label: str,
    context: dict[str, Any],
    entry_ask: float | None,
    entry_bid: float | None,
    entry_obs_ts: float | None,
    entry_source: str | None,
    cost: float,
) -> None:
    """Insert one research state. A setup name never decides whether the row exists."""
    label = str(setup_label or NO_SETUP)
    with contextlib.closing(_connect(db_path)) as conn:
        ensure_tables(conn)
        conn.execute(
            f"""INSERT OR IGNORE INTO {STATES} (
                engine, symbol, decision_ts, feature_ts, features_json, setup_label, context_json,
                entry_ask, entry_bid, entry_obs_ts, entry_source, cost, version, created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                engine,
                str(symbol).upper().replace("/", "").replace("-", ""),
                float(decision_ts),
                feature_ts,
                json.dumps(features, separators=(",", ":")),
                label,
                json.dumps(context, separators=(",", ":")),
                entry_ask,
                entry_bid,
                entry_obs_ts,
                entry_source,
                float(cost),
                FULL_STATE_VERSION,
                time.time(),
            ),
        )
        conn.commit()


def annotate_setup(db_path: str, engine: str, symbol: str, decision_ts: float, setup: str) -> None:
    """Attach a legacy setup name. The row already exists with ``NO_SETUP``."""
    if not setup or setup == NO_SETUP:
        return
    sym = str(symbol).upper().replace("/", "").replace("-", "")
    with contextlib.closing(_connect(db_path)) as conn, contextlib.suppress(sqlite3.Error):
        ensure_tables(conn)
        conn.execute(
            f"UPDATE {STATES} SET setup_label=? WHERE engine=? AND symbol=? AND decision_ts=? AND setup_label=?",
            (str(setup), engine, sym, float(decision_ts), NO_SETUP),
        )
        conn.commit()


def _latest_vector(conn: sqlite3.Connection, symbol: str, as_of: float) -> tuple[float, list[float]] | None:
    hi = datetime.fromtimestamp(as_of, tz=UTC).isoformat()
    lo = datetime.fromtimestamp(as_of - VECTOR_MAX_AGE_SEC, tz=UTC).isoformat()
    for variant in _variants(symbol):
        try:
            row = conn.execute(
                "SELECT ts_utc, features_json FROM ai_inference_log WHERE strategy_id='day' AND feature_dim=145 AND symbol=? AND ts_utc<=? AND ts_utc>=? ORDER BY ts_utc DESC LIMIT 1",
                (variant, hi, lo),
            ).fetchone()
        except sqlite3.Error:
            return None
        if row is None:
            continue
        stamp = _epoch(row["ts_utc"])
        try:
            vec = json.loads(row["features_json"])
        except (TypeError, ValueError):
            return None
        if stamp is None or not isinstance(vec, list):
            return None
        return stamp, [float(x or 0.0) for x in vec]
    return None


def _latest_book(conn: sqlite3.Connection, symbol: str, as_of: float, max_age: float) -> tuple[float, float, float] | None:
    try:
        row = conn.execute(
            "SELECT ts_utc, features_json FROM microstructure_feature_snapshots WHERE symbol=? AND ts_utc<=? AND ts_utc>=? ORDER BY ts_utc DESC LIMIT 1",
            (_base(symbol), float(as_of), float(as_of) - max_age),
        ).fetchone()
    except sqlite3.Error:
        return None
    if row is None:
        return None
    try:
        feat = json.loads(row["features_json"] or "{}")
    except (TypeError, ValueError):
        return None
    bid, ask = float(feat.get("best_bid") or 0), float(feat.get("best_ask") or 0)
    if bid <= 0 or ask < bid:
        return None
    return float(row["ts_utc"]), bid, ask


def capture_day_boundary(db_path: str, as_of: float, symbols: tuple[str, ...] | None = None) -> int:
    """One DAY research row per symbol at the 15m boundary. Setup defaults to NO_SETUP."""
    from backend.services.day_feature_catalog import causal_features
    from backend.services.day_v2.config import DAY_V2_UNIVERSE

    moment = float(as_of)
    names = tuple(symbols or DAY_V2_UNIVERSE)
    with contextlib.closing(_connect(db_path)) as conn:
        vectors = {sym: _latest_vector(conn, sym, moment) for sym in names}
        books = {sym: _latest_book(conn, sym, moment, 5.0) for sym in names}
    cross = {}
    for sym, got in vectors.items():
        if got is not None and got[1] and got[1][0] > 0:
            cross[_base(sym)] = math.log(got[1][0])
    cost = _cost_beyond_spread()
    for sym in names:
        got = vectors.get(sym)
        feats: dict[str, float] = {}
        feature_ts = None
        if got is not None and len(got[1]) == 145:
            feature_ts = got[0]
            with contextlib.suppress(ValueError):
                feats = causal_features(got[1])
        for other, log_px in cross.items():
            if other != _base(sym):
                feats[f"x_log_price_{other}"] = log_px
        book = books.get(sym)
        ask = bid = obs = source = None
        if book is not None:
            obs, bid, ask = book
            source = "book_ask"
        record_state(
            db_path,
            engine=DAY_ENGINE,
            symbol=sym,
            decision_ts=moment,
            feature_ts=feature_ts,
            features=feats,
            setup_label=NO_SETUP,
            context={"cross_log_price": cross, "feature_age_sec": None if feature_ts is None else moment - feature_ts},
            entry_ask=ask,
            entry_bid=bid,
            entry_obs_ts=obs,
            entry_source=source,
            cost=cost,
        )
    return len(names)


def _scalp_features(symbol: str, book: dict[str, Any], db_path: str, now: float) -> dict[str, float]:
    feats = {k: float(book.get(k) or 0.0) for k in SCALP_FEATURE_KEYS}
    bid, ask = float(book.get("best_bid") or 0), float(book.get("best_ask") or 0)
    mid = (bid + ask) / 2.0 if bid > 0 and ask > bid else 0.0
    micro = float(book.get("microprice") or 0)
    feats["micro_displacement"] = (micro / mid - 1.0) if mid > 0 and micro > 0 else 0.0
    prev = _last_mid.get(symbol)
    feats["mid_ret_prev"] = (mid / prev - 1.0) if prev and mid > 0 else 0.0
    if mid > 0:
        _last_mid[symbol] = mid
    for other, other_mid in list(_last_mid.items()):
        if other != symbol and other_mid > 0 and mid > 0:
            feats[f"x_mid_ratio_{other}"] = math.log(mid / other_mid)
    with contextlib.suppress(Exception):
        from backend.services.book_queue_capture import queue_features_asof

        for key, value in queue_features_asof(db_path, symbol, now, 30.0).items():
            feats[f"q_{key}"] = float(value)
    return feats


def capture_scalp_cycle(db_path: str, symbols: list[str] | tuple[str, ...], now: float) -> int:
    """One SCALP research row per symbol per 30s. No setup is required."""
    from backend.services.microstructure_engine import compute_features

    bucket = int(float(now) // SCALP_SAMPLE_SEC) * SCALP_SAMPLE_SEC
    written = 0
    cost = _cost_beyond_spread()
    for raw in symbols:
        sym = str(raw).upper().replace("/", "").replace("-", "")
        if _last_scalp_bucket.get(sym) == bucket:
            continue
        _last_scalp_bucket[sym] = bucket
        try:
            book = compute_features(sym) or {}
        except Exception:
            book = {}
        bid, ask = float(book.get("best_bid") or 0), float(book.get("best_ask") or 0)
        ask_ok = ask if ask > bid > 0 else None
        record_state(
            db_path,
            engine=SCALP_ENGINE,
            symbol=sym,
            decision_ts=float(bucket),
            feature_ts=float(book.get("ts") or now) if book else None,
            features=_scalp_features(sym, book, db_path, float(now)) if book else {},
            setup_label=NO_SETUP,
            context={"sample_sec": SCALP_SAMPLE_SEC},
            entry_ask=ask_ok,
            entry_bid=bid if ask_ok else None,
            entry_obs_ts=float(book.get("ts") or now) if ask_ok else None,
            entry_source="book_ask" if ask_ok else None,
            cost=cost,
        )
        written += 1
    return written


def _bar_points(conn: sqlite3.Connection, symbol: str, target: float, early: float) -> list[tuple[float, float, str]]:
    lo = datetime.fromtimestamp(target - early - 120.0, tz=UTC).strftime("%Y-%m-%d %H:%M:%S")
    hi = datetime.fromtimestamp(target + 1.0, tz=UTC).strftime("%Y-%m-%d %H:%M:%S")
    points = []
    for variant in _variants(symbol):
        try:
            rows = conn.execute("SELECT ts, close FROM feature_ohlcv WHERE symbol=? AND interval='1m' AND ts>=? AND ts<=? ORDER BY ts", (variant, lo, hi)).fetchall()
        except sqlite3.Error:
            return []
        for ts, close in rows:
            opened = _epoch(ts)
            if opened is None or close is None:
                continue
            known = bar_close_known_at(opened, 60.0)
            if known <= target:
                points.append((known, float(close) * (1.0 - HALF_SPREAD), BAR_CLOSE_SOURCE))
        if points:
            break
    return points


def _book_points(conn: sqlite3.Connection, symbol: str, target: float, early: float) -> list[tuple[float, float, str]]:
    try:
        rows = conn.execute(
            "SELECT ts_utc, features_json FROM microstructure_feature_snapshots WHERE symbol=? AND ts_utc<=? AND ts_utc>=? ORDER BY ts_utc",
            (_base(symbol), float(target), float(target) - early),
        ).fetchall()
    except sqlite3.Error:
        return []
    points = []
    for ts, raw in rows:
        try:
            feat = json.loads(raw or "{}")
        except (TypeError, ValueError):
            continue
        bid = float(feat.get("best_bid") or 0)
        if bid > 0 and float(ts) <= target:
            points.append((float(ts), bid, "book_bid"))
    return points


def day_observations(db_path: str, symbol: str, target: float, horizon: int) -> list[tuple[float, float, str]]:
    from backend.services.horizon_alignment import max_early_sec

    with contextlib.closing(_connect(db_path)) as conn:
        early = max_early_sec(horizon)
        return _book_points(conn, symbol, target, early) + _bar_points(conn, symbol, target, early)


def scalp_observations(db_path: str, symbol: str, target: float, horizon: int) -> list[tuple[float, float, str]]:
    from backend.services.horizon_alignment import max_early_sec

    with contextlib.closing(_connect(db_path)) as conn:
        early = max_early_sec(horizon)
        points = _book_points(conn, symbol, target, early)
        if horizon >= 120:
            points.extend(_bar_points(conn, symbol, target, early))
        return points


def resolve_market_labels(db_path: str, engine: str, now: float, observe: Observe, *, limit: int = 40) -> int:
    """Fill due market labels for one engine. The other engine's rows are not read."""
    horizons = DAY_HORIZONS if engine == DAY_ENGINE else SCALP_HORIZONS
    pending: list[tuple] = []
    with contextlib.closing(_connect(db_path)) as conn:
        ensure_tables(conn)
        conn.commit()
        for horizon in horizons:
            if len(pending) >= limit:
                break
            rows = conn.execute(
                f"""SELECT s.id, s.symbol, s.decision_ts, s.entry_ask, s.cost FROM {STATES} s
                    WHERE s.engine=? AND s.labels_done=0 AND s.decision_ts + ? <= ?
                    AND NOT EXISTS (SELECT 1 FROM {LABELS} l WHERE l.state_id=s.id AND l.horizon_sec=?)
                    ORDER BY s.decision_ts LIMIT ?""",
                (engine, int(horizon), float(now), int(horizon), limit - len(pending)),
            ).fetchall()
            pending.extend((int(r["id"]), str(r["symbol"]), float(r["decision_ts"]), r["entry_ask"], float(r["cost"]), int(horizon)) for r in rows)
    writes: list[tuple] = []
    for state_id, symbol, decision_ts, entry, cost, horizon in pending:
        target = decision_ts + horizon
        aligned = align_observation(observe(db_path, symbol, target, horizon), target, horizon)
        gross = net = None
        status = aligned["status"]
        if status == "OK" and entry is not None and float(entry) > 0 and aligned["price"] is not None:
            gross = executable_gross(float(entry), float(aligned["price"]))
            net = None if gross is None else gross - float(cost)
        else:
            status = "MISSING"
        writes.append((state_id, decision_ts, horizon, status, aligned["obs_ts"], aligned["source"], aligned["timing_error_sec"], gross, net, MARKET_KIND, LABEL_VERSION))
    if not writes:
        return 0
    with contextlib.closing(_connect(db_path)) as conn:
        conn.executemany(
            f"""INSERT OR REPLACE INTO {LABELS} (state_id, decision_ts, horizon_sec, status, obs_ts, source, timing_error_sec, gross, net, kind, version)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            writes,
        )
        ids = sorted({w[0] for w in writes})
        for state_id in ids:
            done = conn.execute(f"SELECT COUNT(*) FROM {LABELS} WHERE state_id=?", (state_id,)).fetchone()[0]
            if int(done) >= len(horizons):
                conn.execute(f"UPDATE {STATES} SET labels_done=1 WHERE id=?", (state_id,))
        conn.commit()
    return len(writes)


def tick_day_research(db_path: str, as_of: float, now: float | None = None) -> None:
    capture_day_boundary(db_path, as_of)
    resolve_market_labels(db_path, DAY_ENGINE, float(now if now is not None else time.time()), day_observations)


def tick_scalp_research(db_path: str, symbols: list[str] | tuple[str, ...], now: float) -> None:
    capture_scalp_cycle(db_path, symbols, now)
    resolve_market_labels(db_path, SCALP_ENGINE, float(now), scalp_observations)


def storage_plan() -> dict[str, float]:
    """Expected volume of the 30s SCALP sample and the 15m DAY sample."""
    scalp_rows_per_hour = 4 * (3600 / SCALP_SAMPLE_SEC)
    day_rows_per_hour = 4 * (3600 / DAY_SAMPLE_SEC)
    scalp_bytes = scalp_rows_per_hour * 1200.0
    day_bytes = day_rows_per_hour * 2500.0
    return {
        "scalp_rows_per_hour": scalp_rows_per_hour,
        "day_rows_per_hour": day_rows_per_hour,
        "mb_per_hour": (scalp_bytes + day_bytes) / 1e6,
        "gb_per_day": (scalp_bytes + day_bytes) * 24 / 1e9,
    }


__all__ = [
    "DAY_ENGINE",
    "DAY_SAMPLE_SEC",
    "FULL_STATE_VERSION",
    "LABELS",
    "MARKET_KIND",
    "NO_SETUP",
    "SCALP_ENGINE",
    "SCALP_SAMPLE_SEC",
    "STATES",
    "annotate_setup",
    "capture_day_boundary",
    "capture_scalp_cycle",
    "day_observations",
    "ensure_tables",
    "record_state",
    "resolve_market_labels",
    "scalp_observations",
    "storage_plan",
    "tick_day_research",
    "tick_scalp_research",
]
