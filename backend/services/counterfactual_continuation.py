"""Counterfactual executable continuation states. Research only.

For a hypothetical long entered at the executable ask at ``entry_t``, walk
the executable exit price (bid) forward and, at each state time ``s``, record
the live continuation surface's state features (``continuation_surface
.state_features``) and, per horizon ``h``, the hold advantage
``exit_net(s + h) - exit_net(s)``. Every input of a state is known at ``s``;
only the label reads ``s + h``.

Rows are tagged ``COUNTERFACTUAL_EXECUTABLE`` and written only to their own
table. They are never a fill, never accounting and no live learner reads them.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from backend.services.continuation_surface import ADVANTAGE_VERSION, DAY_ADVANTAGE_HORIZONS, SCALP_ADVANTAGE_HORIZONS, state_features
from backend.services.horizon_alignment import max_early_sec

CF_KIND = "COUNTERFACTUAL_EXECUTABLE"
REAL_KIND = "REAL_FILL"
TABLE = "counterfactual_continuation_states"


@dataclass(frozen=True)
class Path:
    """Executable exit prices (bid) at increasing times, plus the entry ask."""

    ts: np.ndarray
    exit_px: np.ndarray


def states_for(
    path: Path,
    *,
    engine: str,
    symbol: str,
    entry_t: float,
    entry_px: float,
    cost: float,
    step_sec: float,
    max_age_sec: float,
    horizons: Sequence[int],
    match_tol_sec: float,
) -> list[dict[str, Any]]:
    if entry_px <= 0 or len(path.ts) == 0:
        return []
    ts, px = path.ts, path.exit_px
    i0 = int(np.searchsorted(ts, entry_t, side="right"))
    out: list[dict[str, Any]] = []
    prev_net: float | None = None
    k = 1
    while k * step_sec <= max_age_sec:
        s = entry_t + k * step_sec
        j = int(np.searchsorted(ts, s, side="right")) - 1
        k += 1
        if j < i0 or s - ts[j] > match_tol_sec:
            prev_net = None
            continue
        seg = px[i0 : j + 1]
        mark = float(px[j])
        net = mark / entry_px - 1.0 - cost
        hi, lo = float(seg.max()), float(seg.min())
        feats = state_features(entry=entry_px, mark=mark, net=net, mfe=hi / entry_px - 1.0, mae=max(0.0, 1.0 - lo / entry_px), high_water=hi, prev_net=prev_net, age_sec=s - entry_t)
        prev_net = net
        labels: dict[str, float] = {}
        for h in horizons:
            target = s + h
            f = int(np.searchsorted(ts, target, side="right")) - 1
            # Last print at or before the horizon, inside the early tolerance.
            # A print after the horizon is not a fallback.
            early = min(float(match_tol_sec), max_early_sec(h))
            if f >= 0 and ts[f] <= target and target - float(ts[f]) <= early:
                labels[str(h)] = float(px[f] / entry_px - 1.0 - cost) - net
        if labels:
            out.append(
                {
                    "kind": CF_KIND,
                    "engine_id": engine,
                    "symbol": symbol,
                    "entry_t": float(entry_t),
                    "state_t": float(s),
                    "entry_px": float(entry_px),
                    "features": feats,
                    "advantage": labels,
                    "version": ADVANTAGE_VERSION,
                }
            )
    return out


def scalp_states(book: dict[str, np.ndarray], symbol: str, entries: Iterable[float], *, cost: float, step_sec: float = 60.0, max_age_sec: float = 1200.0, tol: float = 10.0) -> list[dict[str, Any]]:
    """SCALP: entry at the snapshot ask at or before t, exit at later snapshot bids."""
    path = Path(np.asarray(book["ts"], dtype=float), np.asarray(book["bid"], dtype=float))
    out = []
    for t in entries:
        i = int(np.searchsorted(path.ts, t, side="right")) - 1
        if i < 0 or t - path.ts[i] > 5.0:
            continue
        out.extend(
            states_for(
                path,
                engine="SCALP_V2",
                symbol=symbol,
                entry_t=float(t),
                entry_px=float(book["ask"][i]),
                cost=cost,
                step_sec=step_sec,
                max_age_sec=max_age_sec,
                horizons=SCALP_ADVANTAGE_HORIZONS,
                match_tol_sec=tol,
            )
        )
    return out


def day_states(
    bars_1m: Sequence[tuple[float, float, float, float, float]],
    symbol: str,
    entries: Iterable[tuple[float, float]],
    *,
    cost: float,
    half_spread: float,
    step_sec: float = 900.0,
    max_age_sec: float = 43200.0,
) -> list[dict[str, Any]]:
    """DAY: exit price is a closed 1m bar's close minus the half spread, stamped at bar close."""
    ts = np.array([b[0] + 60.0 for b in bars_1m], dtype=float)
    px = np.array([b[4] for b in bars_1m], dtype=float) * (1.0 - half_spread)
    path = Path(ts, px)
    out = []
    for t, ask in entries:
        out.extend(
            states_for(
                path,
                engine="DAY_V2",
                symbol=symbol,
                entry_t=float(t),
                entry_px=float(ask),
                cost=cost,
                step_sec=step_sec,
                max_age_sec=max_age_sec,
                horizons=DAY_ADVANTAGE_HORIZONS,
                match_tol_sec=120.0,
            )
        )
    return out


DDL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL CHECK (kind = '{CF_KIND}'),
    engine_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    entry_t REAL NOT NULL,
    state_t REAL NOT NULL,
    entry_px REAL NOT NULL,
    features_json TEXT NOT NULL,
    advantage_json TEXT NOT NULL,
    version TEXT NOT NULL
)
"""


def write_states(db_path: str, rows: Sequence[dict[str, Any]]) -> int:
    if any(r.get("kind") != CF_KIND for r in rows):
        raise ValueError("only COUNTERFACTUAL_EXECUTABLE states are written here")
    with contextlib.closing(sqlite3.connect(db_path)) as conn:
        conn.execute(DDL)
        conn.executemany(
            f"INSERT INTO {TABLE} (kind, engine_id, symbol, entry_t, state_t, entry_px, features_json, advantage_json, version) VALUES (?,?,?,?,?,?,?,?,?)",
            [
                (
                    r["kind"],
                    r["engine_id"],
                    r["symbol"],
                    r["entry_t"],
                    r["state_t"],
                    r["entry_px"],
                    json.dumps(r["features"], separators=(",", ":")),
                    json.dumps(r["advantage"], separators=(",", ":")),
                    r["version"],
                )
                for r in rows
            ],
        )
        conn.commit()
    return len(rows)


__all__ = ["CF_KIND", "REAL_KIND", "TABLE", "day_states", "scalp_states", "states_for", "write_states"]
