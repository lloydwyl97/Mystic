"""DAY CHURN_GUARD recovery: same frozen 20-SELL window cannot re-lock after 4h parole.

Does not change the 0.50 ratio limit or the 4h parole duration.
Persists activation identity in operational_state so a restart keeps the lock
and the parole fingerprint.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

CHURN_RATIO_LIMIT = 0.50
CHURN_TRADE_WINDOW = 20
CHURN_GUARD_MIN_SAMPLES = 5
CHURN_GUARD_MIN_GROSS_WIN_SUM = 15.0
CHURN_GUARD_MAX_SEC = 14400
OPERATIONAL_KEY = "day_churn_guard"


@dataclass
class ChurnGuardState:
    active: bool = False
    activated_at: float = 0.0
    parole_window_fp: str = ""
    ratio: float = 0.0
    last_window_fp: str = ""


def fingerprint_sell_ids(sell_ids: Sequence[Any]) -> str:
    """Deterministic identity of the evaluated SELL window (audit ids in query order)."""
    return "|".join(str(int(i)) for i in sell_ids if i is not None and str(i) != "")


def compute_churn_ratio(rows: Iterable[Sequence[Any]]) -> tuple[float, float, float, int]:
    """rows: (roundtrip_fees, roundtrip_slippage, winning_invariant_or_0, sell_id)."""
    seq = list(rows)
    total_fees = sum(float(r[0] or 0) for r in seq)
    total_slip = sum(float(r[1] or 0) for r in seq)
    gross_profit = sum(float(r[2] or 0) for r in seq)
    costs = total_fees + total_slip
    if gross_profit > 0:
        ratio = costs / gross_profit
    else:
        ratio = float("inf") if seq else 0.0
    return costs, gross_profit, ratio, len(seq)


def evaluate_churn_transition(
    *,
    now: float,
    rows: Sequence[Sequence[Any]],
    state: ChurnGuardState,
    ratio_limit: float = CHURN_RATIO_LIMIT,
    max_sec: float = CHURN_GUARD_MAX_SEC,
    min_samples: int = CHURN_GUARD_MIN_SAMPLES,
    min_gross_win: float = CHURN_GUARD_MIN_GROSS_WIN_SUM,
) -> ChurnGuardState:
    """Apply one evaluation. ``state`` is not mutated; a new state is returned."""
    costs, gross_profit, ratio, n = compute_churn_ratio(rows)
    fp = fingerprint_sell_ids([r[3] for r in rows]) if n else ""
    out = ChurnGuardState(
        active=state.active,
        activated_at=state.activated_at,
        parole_window_fp=state.parole_window_fp,
        ratio=ratio if n else 0.0,
        last_window_fp=fp,
    )

    if n < min_samples or gross_profit < min_gross_win:
        if out.active:
            logger.info(
                "REGIME: CHURN_GUARD cleared (samples=%s min=%s gross_profit=%.4f)",
                n,
                min_samples,
                gross_profit,
            )
        out.active = False
        out.activated_at = 0.0
        return out

    if ratio <= ratio_limit:
        out.active = False
        out.activated_at = 0.0
        return out

    # ratio > limit
    if out.active:
        elapsed = now - out.activated_at if out.activated_at > 0 else 0.0
        if elapsed > max_sec:
            logger.info(
                "REGIME: CHURN_GUARD auto-cleared after %.0fh (max=%dh) — parole; identical window cannot re-arm",
                elapsed / 3600.0,
                int(max_sec // 3600),
            )
            out.active = False
            out.activated_at = 0.0
            # Keep parole_window_fp from the activation that just expired.
            return out
        out.active = True
        return out

    # Inactive, ratio still excessive
    if fp and fp == out.parole_window_fp:
        logger.info(
            "REGIME: CHURN_GUARD parole holds — identical frozen SELL window fp=%s ratio=%.2f; not re-arming",
            fp[:48],
            ratio,
        )
        out.active = False
        out.activated_at = 0.0
        return out

    out.active = True
    out.activated_at = now
    out.parole_window_fp = fp
    logger.warning(
        "REGIME: CHURN_GUARD activated - fees=%.2f profit=%.2f ratio=%.2f > %.2f (max %dh) fp=%s",
        costs,
        gross_profit,
        ratio,
        ratio_limit,
        int(max_sec // 3600),
        fp[:48],
    )
    return out


def state_to_json(state: ChurnGuardState) -> str:
    return json.dumps(
        {
            "active": bool(state.active),
            "activated_at": float(state.activated_at or 0.0),
            "parole_window_fp": str(state.parole_window_fp or ""),
            "ratio": float(state.ratio) if state.ratio != float("inf") else None,
            "last_window_fp": str(state.last_window_fp or ""),
        },
        separators=(",", ":"),
    )


def state_from_json(blob: str | None) -> ChurnGuardState:
    if not blob:
        return ChurnGuardState()
    try:
        d = json.loads(blob)
    except Exception:
        return ChurnGuardState()
    if not isinstance(d, dict):
        return ChurnGuardState()
    ratio = d.get("ratio")
    return ChurnGuardState(
        active=bool(d.get("active")),
        activated_at=float(d.get("activated_at") or 0.0),
        parole_window_fp=str(d.get("parole_window_fp") or ""),
        ratio=float(ratio) if ratio is not None else 0.0,
        last_window_fp=str(d.get("last_window_fp") or ""),
    )


def _ensure_operational_state(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS operational_state (
            key TEXT PRIMARY KEY,
            value_json TEXT,
            updated_ts REAL
        )
        """
    )


def load_churn_guard_state(db_path: str) -> ChurnGuardState:
    try:
        conn = sqlite3.connect(db_path, timeout=5.0)
        try:
            _ensure_operational_state(conn)
            row = conn.execute(
                "SELECT value_json FROM operational_state WHERE key=?",
                (OPERATIONAL_KEY,),
            ).fetchone()
            return state_from_json(row[0] if row else None)
        finally:
            conn.close()
    except Exception:
        logger.debug("load_churn_guard_state failed", exc_info=True)
        return ChurnGuardState()


def persist_churn_guard_state(db_path: str, state: ChurnGuardState) -> None:
    try:
        conn = sqlite3.connect(db_path, timeout=5.0)
        try:
            _ensure_operational_state(conn)
            conn.execute(
                """
                INSERT INTO operational_state(key, value_json, updated_ts)
                VALUES(?, ?, strftime('%s','now'))
                ON CONFLICT(key) DO UPDATE SET
                  value_json=excluded.value_json,
                  updated_ts=excluded.updated_ts
                """,
                (OPERATIONAL_KEY, state_to_json(state)),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        logger.debug("persist_churn_guard_state failed", exc_info=True)
