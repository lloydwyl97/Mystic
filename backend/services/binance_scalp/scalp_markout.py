"""Short-horizon executable markout dataset for SCALP.

In-memory schedule + batched SQLite persist on the SCALP DB.
Never writes on every depth packet. Never shares the breaker hot-path lock
for high-frequency inserts — uses a short timeout and skips on lock.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any

from backend.services.binance_scalp.orderbook_book import walk_sell_qty
from backend.services.binance_scalp.scalp_micro_contract import MARKOUT_HORIZONS_SEC, version_stamps

logger = logging.getLogger(__name__)

_PENDING: list["_PendingMarkout"] = []
_LAST_FLUSH_TS = 0.0
_FLUSH_INTERVAL_SEC = 2.0
_TABLE_READY: set[str] = set()


@dataclass
class _PendingMarkout:
    kind: str  # candidate | entry
    symbol: str
    side: str
    t0: float
    mid0: float
    entry_px: float
    qty: float
    notional: float
    fee_pct: float
    slip_pct: float
    horizons: tuple[int, ...] = MARKOUT_HORIZONS_SEC
    remaining: set[int] = field(default_factory=lambda: set(MARKOUT_HORIZONS_SEC))
    points: dict[int, dict[str, float]] = field(default_factory=dict)
    mfe: float = 0.0
    mae: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)
    done: bool = False


def schedule_markout(
    *,
    kind: str,
    symbol: str,
    side: str,
    mid: float,
    entry_px: float,
    qty: float,
    notional: float,
    fee_pct: float,
    slip_pct: float,
    extra: dict[str, Any] | None = None,
    now: float | None = None,
) -> None:
    if mid <= 0 or entry_px <= 0:
        return
    _PENDING.append(
        _PendingMarkout(
            kind=kind,
            symbol=symbol.upper().replace("/", ""),
            side=side.upper(),
            t0=float(now if now is not None else time.time()),
            mid0=float(mid),
            entry_px=float(entry_px),
            qty=float(qty),
            notional=float(notional),
            fee_pct=float(fee_pct),
            slip_pct=float(slip_pct),
            extra=dict(extra or {}),
        )
    )


def _mid(bid: float, ask: float) -> float:
    if bid > 0 and ask > 0:
        return 0.5 * (bid + ask)
    return max(bid, ask, 0.0)


def _executable_exit_px(side: str, bids, asks, qty: float, bid: float, ask: float) -> float:
    if side == "BUY":
        walk = walk_sell_qty(list(bids or []), max(qty, 1e-12), bid)
        if walk.expected_avg_fill > 0:
            return float(walk.expected_avg_fill)
        return float(bid or 0.0)
    # short not used; treat as lift ask
    return float(ask or 0.0)


def observe_book(
    symbol: str,
    *,
    bid: float,
    ask: float,
    bids=None,
    asks=None,
    now: float | None = None,
) -> list[dict[str, Any]]:
    """Advance pending markouts. Returns newly completed rows (not yet flushed)."""
    t = float(now if now is not None else time.time())
    mid = _mid(bid, ask)
    if mid <= 0:
        return []
    completed: list[dict[str, Any]] = []
    for p in _PENDING:
        if p.done or p.symbol != symbol.upper().replace("/", ""):
            continue
        elapsed = t - p.t0
        if p.side == "BUY":
            ret = (mid - p.mid0) / p.mid0
        else:
            ret = (p.mid0 - mid) / p.mid0
        p.mfe = max(p.mfe, ret)
        p.mae = min(p.mae, ret)
        due = [h for h in list(p.remaining) if elapsed + 1e-9 >= h]
        for h in due:
            exit_px = _executable_exit_px(p.side, bids, asks, p.qty, bid, ask)
            if p.side == "BUY":
                gross = (exit_px - p.entry_px) / p.entry_px if p.entry_px > 0 else 0.0
            else:
                gross = (p.entry_px - exit_px) / p.entry_px if p.entry_px > 0 else 0.0
            mid_m = (mid - p.mid0) / p.mid0 if p.side == "BUY" else (p.mid0 - mid) / p.mid0
            fee_adj = gross - p.fee_pct
            slip_adj = fee_adj - p.slip_pct
            p.points[h] = {
                "mid_markout": round(mid_m, 8),
                "gross_markout": round(gross, 8),
                "fee_adj_markout": round(fee_adj, 8),
                "slip_adj_markout": round(slip_adj, 8),
                "executable_net_markout": round(slip_adj, 8),
                "exit_px": round(exit_px, 8),
            }
            p.remaining.discard(h)
        if not p.remaining:
            p.done = True
            completed.append(_row(p))
    return completed


def _row(p: _PendingMarkout) -> dict[str, Any]:
    return {
        "kind": p.kind,
        "symbol": p.symbol,
        "side": p.side,
        "t0": p.t0,
        "mid0": p.mid0,
        "entry_px": p.entry_px,
        "qty": p.qty,
        "notional": p.notional,
        "mfe": round(p.mfe, 8),
        "mae": round(p.mae, 8),
        "points": p.points,
        "extra": p.extra,
        **version_stamps(),
    }


def _ensure_table(db_path: str) -> None:
    if db_path in _TABLE_READY:
        return
    with sqlite3.connect(db_path, timeout=2.0) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scalp_micro_markouts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kind TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                t0 REAL NOT NULL,
                mid0 REAL,
                entry_px REAL,
                qty REAL,
                notional REAL,
                mfe REAL,
                mae REAL,
                points_json TEXT,
                extra_json TEXT,
                feature_version TEXT,
                microstructure_version TEXT,
                model_version TEXT,
                created_at REAL NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_scalp_micro_markouts_t0 ON scalp_micro_markouts(symbol, t0)")
        conn.commit()
    _TABLE_READY.add(db_path)


def flush_completed(db_path: str, *, force: bool = False, now: float | None = None) -> int:
    """Persist completed markouts. Skip silently on lock — breaker stays free."""
    global _LAST_FLUSH_TS
    t = float(now if now is not None else time.time())
    if not force and (t - _LAST_FLUSH_TS) < _FLUSH_INTERVAL_SEC:
        return 0
    done = [p for p in _PENDING if p.done]
    if not done:
        return 0
    try:
        _ensure_table(db_path)
        with sqlite3.connect(db_path, timeout=1.0) as conn:
            conn.execute("BEGIN")
            for p in done:
                row = _row(p)
                conn.execute(
                    """
                    INSERT INTO scalp_micro_markouts (
                        kind, symbol, side, t0, mid0, entry_px, qty, notional,
                        mfe, mae, points_json, extra_json,
                        feature_version, microstructure_version, model_version, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["kind"],
                        row["symbol"],
                        row["side"],
                        row["t0"],
                        row["mid0"],
                        row["entry_px"],
                        row["qty"],
                        row["notional"],
                        row["mfe"],
                        row["mae"],
                        json.dumps(row["points"]),
                        json.dumps(row["extra"], default=str),
                        row.get("feature_version"),
                        row.get("microstructure_version"),
                        row.get("model_version"),
                        t,
                    ),
                )
            conn.commit()
        for p in done:
            _PENDING.remove(p)
        _LAST_FLUSH_TS = t
        return len(done)
    except sqlite3.OperationalError as exc:
        logger.debug("scalp_markout flush skipped (lock-safe): %s", exc)
        return 0
    except Exception as exc:
        logger.debug("scalp_markout flush failed: %s", exc)
        return 0


def pending_count() -> int:
    return len(_PENDING)


def reset_markouts() -> None:
    _PENDING.clear()
    _TABLE_READY.clear()


def compute_markout_point(
    *,
    side: str,
    mid0: float,
    entry_px: float,
    mid_t: float,
    exit_px: float,
    fee_pct: float,
    slip_pct: float,
) -> dict[str, float]:
    if side.upper() == "BUY":
        mid_m = (mid_t - mid0) / mid0 if mid0 else 0.0
        gross = (exit_px - entry_px) / entry_px if entry_px else 0.0
    else:
        mid_m = (mid0 - mid_t) / mid0 if mid0 else 0.0
        gross = (entry_px - exit_px) / entry_px if entry_px else 0.0
    fee_adj = gross - fee_pct
    slip_adj = fee_adj - slip_pct
    return {
        "mid_markout": mid_m,
        "gross_markout": gross,
        "fee_adj_markout": fee_adj,
        "slip_adj_markout": slip_adj,
        "executable_net_markout": slip_adj,
    }


__all__ = [
    "compute_markout_point",
    "flush_completed",
    "observe_book",
    "pending_count",
    "reset_markouts",
    "schedule_markout",
]
