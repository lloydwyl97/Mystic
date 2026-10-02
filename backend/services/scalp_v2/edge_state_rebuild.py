"""Rebuild SCALP derived edge/risk state from current-version candidate markouts.

Derived estimators only: ``edge_residual``, ``markout_mae`` (gross path MAE, the
live risk estimate) and the ``micro_edge`` model for SCALP_V2. Every other
metric, every DAY row, ownership, accounting and trade history are untouched.
Both state tables are copied to backup tables before anything is deleted.

The replay is chronological and causal: a candidate's label is folded in at
``evaluated_at + label_horizon``, and an optional ``on_decision`` callback sees
the state exactly as it stood at each candidate's own decision time.
"""

from __future__ import annotations

import bisect
import calendar
import json
import math
import sqlite3
import time
from collections.abc import Callable
from typing import Any

from backend.services import adaptive_learning as al
from backend.services.scalp_v2.raw_move_source import is_directional, normalize_raw_move_source

REBUILT_METRICS = ("edge_residual", "edge_residual_strategy", "markout_mae")
MICRO_MODEL = "micro_edge"


def _bar_epoch(ts: str) -> float | None:
    try:
        return float(calendar.timegm(time.strptime(str(ts)[:19].replace("T", " "), "%Y-%m-%d %H:%M:%S")))
    except (TypeError, ValueError):
        return None


class OhlcvPath:
    """In-memory 1m bars for reconstruction. Same windows as the live helpers."""

    def __init__(self, db_path: str, symbols: list[str]) -> None:
        self._bars: dict[str, list[tuple[float, float, float, float, float]]] = {}
        conn = sqlite3.connect(db_path, timeout=15)
        try:
            for sym in symbols:
                raw = al._norm_symbol(sym)
                rows: list[tuple[float, float, float, float, float]] = []
                for variant in (raw, raw.replace("USDT", "-USDT"), raw.replace("USDT", "/USDT")):
                    for ts, o, h, lo, c in conn.execute(
                        "SELECT ts, open, high, low, close FROM feature_ohlcv WHERE symbol=? AND interval='1m' ORDER BY ts",
                        (variant,),
                    ):
                        t = _bar_epoch(ts)
                        if t is not None and None not in (o, h, lo, c):
                            rows.append((t, float(o), float(h), float(lo), float(c)))
                    if rows:
                        break
                rows.sort()
                self._bars[raw] = rows
        finally:
            conn.close()
        self._ts = {k: [b[0] for b in v] for k, v in self._bars.items()}

    def raw_expected_move(self, symbol: str, at: float) -> float | None:
        """ATR-only SCALP expected move from bars closed by ``at`` (ranking's opinion-reject path)."""
        from backend.services.binance_scalp.strategies.common import estimate_expected_move_pct

        key = al._norm_symbol(symbol)
        bars = self._bars.get(key) or []
        i = bisect.bisect_right(self._ts.get(key) or [], float(at) - 60.0)
        window = bars[max(0, i - 15) : i]
        if len(window) < 15:
            return None
        move = estimate_expected_move_pct([{"high": b[2], "low": b[3], "close": b[4]} for b in window], structural=0.0)
        return move if move > 0 else None

    def low_between(self, symbol: str, start: float, end: float) -> float | None:
        key = al._norm_symbol(symbol)
        ts = self._ts.get(key) or []
        bars = self._bars.get(key) or []
        lo_i = bisect.bisect_left(ts, math.ceil(float(start) / 60.0) * 60.0)
        hi_i = bisect.bisect_left(ts, float(end))
        lows = [b[3] for b in bars[lo_i:hi_i]]
        return min(lows) if lows else None


def backup_state(db_path: str, suffix: str) -> dict[str, str]:
    names = {"metric_state": f"adaptive_metric_state_bak_{suffix}", "linear_model": f"adaptive_linear_model_bak_{suffix}"}
    with al._connect(db_path) as conn:
        conn.execute(f"CREATE TABLE {names['metric_state']} AS SELECT * FROM adaptive_metric_state")
        conn.execute(f"CREATE TABLE {names['linear_model']} AS SELECT * FROM adaptive_linear_model")
        conn.commit()
    return names


def rebuild_scalp_edge_state(
    db_path: str,
    *,
    raw_move_for: Callable[[sqlite3.Row], tuple[float | None, str]],
    path_low: Callable[[str, float, float], float | None] | None,
    backup_suffix: str | None,
    on_decision: Callable[[sqlite3.Row, dict[str, Any], float | None, str], None] | None = None,
) -> dict[str, Any]:
    """Rebuild SCALP edge residuals, gross markout_mae and micro_edge from clean rows.

    ``raw_move_for(row)`` returns the decision-time (raw expected move, source)
    for rows recorded before the raw move was stored. It returns (None, ...)
    when the exact value is not recoverable; that row then teaches risk only.
    """
    engine = al.SCALP_ENGINE
    version = al.current_strategy_version(engine)
    backups = backup_state(db_path, backup_suffix) if backup_suffix else {}
    with al._connect(db_path) as conn:
        conn.execute(
            f"DELETE FROM adaptive_metric_state WHERE engine_id=? AND metric IN ({','.join('?' * len(REBUILT_METRICS))})",
            (engine, *REBUILT_METRICS),
        )
        conn.execute("DELETE FROM adaptive_linear_model WHERE engine_id=? AND model=?", (engine, MICRO_MODEL))
        conn.commit()
        rows = conn.execute(
            "SELECT * FROM adaptive_candidate_markouts WHERE engine_id=? AND strategy_version=? AND learned=1 ORDER BY id",
            (engine, version),
        ).fetchall()
        legacy = conn.execute(
            "SELECT COUNT(*) FROM adaptive_candidate_markouts WHERE engine_id=? AND strategy_version!=?",
            (engine, version),
        ).fetchone()[0]

    events: list[tuple[float, int, int]] = []
    raw_by_id: dict[int, tuple[float | None, str]] = {}
    for idx, row in enumerate(rows):
        cols = row.keys()
        stored = row["raw_expected_move"] if "raw_expected_move" in cols else None
        if stored is not None and float(stored) > 0:
            source = row["raw_move_source"] if "raw_move_source" in cols else None
            raw_by_id[idx] = (float(stored), normalize_raw_move_source(source))
        else:
            raw_by_id[idx] = raw_move_for(row)
        horizon = float(row["label_horizon"] or 0) or 600.0
        events.append((float(row["evaluated_at"]), 1, idx))
        events.append((float(row["evaluated_at"]) + horizon, 0, idx))
    events.sort()

    stats = {"rows": len(rows), "legacy_rows_ignored": int(legacy), "residual_obs": 0, "risk_obs": 0, "micro_updates": 0, "no_raw": 0, "no_label": 0}
    for moment, kind, idx in events:
        row = rows[idx]
        raw, source = raw_by_id[idx]
        feats = {}
        try:
            feats = json.loads(row["features_json"] or "{}")
        except (TypeError, ValueError):
            feats = {}
        if kind == 1:
            if on_decision is not None:
                on_decision(row, al.scalp_decision(db_path, row["symbol"], row["setup"], row["regime"], feats), raw, source)
            continue
        forward = al._forward_from_stored(row)
        if forward is None:
            stats["no_label"] += 1
            continue
        marks = json.loads(row["markouts_json"] or "{}")
        horizon = float(row["label_horizon"] or 0) or 600.0
        path = [v for h in al.SCALP_HORIZONS_SEC if h <= horizon + 1e-9 and (v := al._mark_at(marks, h)) is not None]
        low = path_low(row["symbol"], float(row["evaluated_at"]), float(row["evaluated_at"]) + horizon) if path_low else None
        mae = al.scalp_gross_path_mae(ref_price=float(row["ref_price"]), roundtrip_cost=float(row["roundtrip_cost"] or 0), marks=path, path_low=low)
        common = {"engine": engine, "symbol": row["symbol"], "setup": row["setup"], "regime": row["regime"], "strategy_version": row["strategy_version"], "now": moment}
        if mae is not None and al.observe(db_path, metric="markout_mae", value=mae, **common):
            stats["risk_obs"] += 1
        if raw is None:
            stats["no_raw"] += 1
            continue
        residual = al.scalp_edge_residual(forward_net=forward, raw_expected_move=raw, roundtrip_cost=float(row["roundtrip_cost"] or 0))
        if al.observe(db_path, metric=al.residual_metric(source), value=residual, **common):
            stats["residual_obs"] += 1
        if isinstance(feats, dict) and feats and is_directional(source):
            al.update_linear_model(db_path, engine, MICRO_MODEL, feats, residual)
            stats["micro_updates"] += 1
    stats["backups"] = backups
    return stats


__all__ = ["OhlcvPath", "backup_state", "rebuild_scalp_edge_state"]
