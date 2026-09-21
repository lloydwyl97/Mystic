#!/usr/bin/env python3
"""Production-conformant DAY entry replay vs Ocean fills.

Loads Ocean DAY env keys (no secrets) before importing economics, seeds
completed 4H from Binance.US, and prints the conformance report.
Analysis only — does not write to production.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Ocean dump 2026-09-01 — same keys production .env actually set. No secrets.
# Applied before backend imports so trading_economics cannot bind module defaults.
_OCEAN_DAY_ENV = {
    "DAY_AI_SIGNAL_LOOP_SEC": "60",
    "DAY_BASE_NOTIONAL_PER_SLOT_USD": "2000",
    "DAY_NOTIONAL_MULT": "1.1",
    "DAY_TARGET_NOTIONAL_PER_SLOT_USD": "4000",
    "DAY_MAX_DEPLOYED_USD": "9000",
    "DAY_MAX_BUYS_PER_BAR": "4",
    "DAY_PATH_AWARE_EXIT": "true",
    "DAY_STALL_EXIT_ENABLED": "false",
    "DAY_GIVEBACK_EXIT_ENABLED": "false",
    "ESTIMATED_ROUNDTRIP_COST": "0.0006",
    "ESTIMATED_ROUNDTRIP_COST_PCT": "0.0006",
    "MAX_OPEN_POSITIONS": "4",
    "DAY_MAX_OPEN_SLOTS": "4",
    "POST_SELL_COOLDOWN_BARS": "3",
    "TRADING_MODE": "live",
    "EXECUTION_MODE": "live",
    "SCALP_LIVE": "false",
}

from backend.services.day_production_env import apply_env_map, snapshot_day_env

apply_env_map(_OCEAN_DAY_ENV, override=True)

from backend.config.execution_cost_model import LEGACY_SELL_ROUNDTRIP_PCT
from backend.services.day_asof_4h import FOURH_KEEP_MAX, FourHAsOfTracker, merge_seed_and_1m_completed
from backend.services.day_entry_conformance import (
    load_actual_entries,
    load_actual_exits,
    load_rejects,
    match_entries,
    report_to_dict,
)
from backend.services.day_production_lifecycle_replay import (
    SYMBOLS,
    ReplayCandidate,
    decision_bars,
    load_1m_bars,
    load_inferences,
    run_arm,
)

WINDOW_START = "2026-08-25"
WINDOW_END = "2026-09-02"
START_EPOCH = int(datetime(2026, 8, 25, tzinfo=timezone.utc).timestamp())
END_EPOCH = int(datetime(2026, 9, 2, tzinfo=timezone.utc).timestamp())
STARTING_CASH = 239.02654324
WINDOW_DAYS = 8.0


def _admit_path_ev(inf: dict) -> tuple[bool, float]:
    ev = float(inf.get("path_ev") or 0.0)
    return ev > 0.0, ev


def fetch_1m_gap(symbol: str, start_epoch: int, end_epoch: int) -> list[tuple[int, float, float, float, float, float]]:
    """Completed 1m klines production had live but Ocean did not persist before 22:27."""
    out: list[tuple[int, float, float, float, float, float]] = []
    cursor = int(start_epoch * 1000)
    end_ms = int(end_epoch * 1000)
    while cursor < end_ms:
        url = f"https://api.binance.us/api/v3/klines?symbol={symbol}&interval=1m&startTime={cursor}&endTime={end_ms}&limit=1000"
        with urllib.request.urlopen(url, timeout=30) as resp:
            raw = json.loads(resp.read().decode())
        if not raw:
            break
        for row in raw:
            open_ms = int(row[0])
            if open_ms >= end_ms:
                continue
            out.append((open_ms // 1000, float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])))
        cursor = int(raw[-1][0]) + 60_000
        if len(raw) < 1000:
            break
    return out


def fetch_completed_4h(symbol: str, end_epoch: float, limit: int = 500) -> list[list[float]]:
    end_ms = int(end_epoch * 1000)
    url = f"https://api.binance.us/api/v3/klines?symbol={symbol}&interval=4h&endTime={end_ms}&limit={limit}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        raw = json.loads(resp.read().decode())
    out: list[list[float]] = []
    for row in raw:
        open_ms = int(row[0])
        close_ms = int(row[6])
        if close_ms > end_ms:
            continue
        out.append([float(open_ms // 1000), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])])
    return out


def main() -> int:
    db = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/ocean_day_replay_slim.db")
    conn = sqlite3.connect(str(db))
    actual = load_actual_entries(conn, start=WINDOW_START, end=WINDOW_END)
    exits = load_actual_exits(conn, start=WINDOW_START, end=WINDOW_END)
    rejects = load_rejects(conn, start=WINDOW_START, end=WINDOW_END)
    bars = load_1m_bars(conn)
    for sym in SYMBOLS:
        existing = bars.get(sym) or []
        first_stored = int(existing[0][0]) if existing else END_EPOCH
        if first_stored > START_EPOCH:
            gap = fetch_1m_gap(sym, START_EPOCH, first_stored)
            merged = {int(b[0]): b for b in gap}
            for b in existing:
                merged[int(b[0])] = b
            bars[sym] = [merged[k] for k in sorted(merged)]
    inferences = load_inferences(conn)
    events = decision_bars(inferences)
    first_1m = min((int(b[0][0]) for b in bars.values() if b), default=START_EPOCH)
    trackers: dict[str, FourHAsOfTracker] = {}
    fourh: dict[str, list[list[float]]] = {}
    for sym in SYMBOLS:
        seed = fetch_completed_4h(sym, first_1m)
        fourh[sym] = merge_seed_and_1m_completed(seed, bars.get(sym, []))
        trk = FourHAsOfTracker(bars_1m=bars.get(sym, []), keep=FOURH_KEEP_MAX)
        trk.seed_completed(fourh[sym])
        trackers[sym] = trk
    candidates: list[ReplayCandidate] = []
    skip: dict[str, int] = {}
    closed, accepted, rejected = run_arm(
        name="current_production",
        events=events,
        bars=bars,
        fourh=fourh,
        admit=_admit_path_ev,
        start_epoch=START_EPOCH,
        end_epoch=END_EPOCH,
        sell_cost=LEGACY_SELL_ROUNDTRIP_PCT,
        starting_cash=STARTING_CASH,
        trackers=trackers,
        skip_reasons=skip,
        candidates=candidates,
    )
    replay_accepted = [c for c in candidates if c.accepted]
    evidence = [
        "1m feature_ohlcv starts 2026-08-25 22:27 UTC — first 9 Aug-25 fills have no as-of 1m",
        "ai_inference_log has no final_selection_score; rank falls back to p_buy",
        "day_decision_records.mode=paper and reject first_hard_block=EXECUTION_GATE",
        "current d319c50 code no longer hard-blocks late_4h_rise / same_4h_rise_rebuy",
        "pending live limit orders and unfilled PROTECTED_LIMIT_BUY cannot be reconstructed from 1m",
        f"stored live DAY BUY fills in window={len(actual)}; briefing denominator=53",
    ]
    report = match_entries(
        actual,
        replay_accepted,
        rejects=rejects,
        replay_all=candidates,
        actual_exits=exits,
        replay_exit_reasons={r: sum(1 for t in closed if t.exit_reason == r) for r in {t.exit_reason for t in closed}},
        replay_holds=[float(t.hold_sec) for t in closed],
        window_days=WINDOW_DAYS,
        evidence_limits=evidence,
    )
    payload = report_to_dict(report)
    payload["env_snapshot"] = snapshot_day_env()
    payload["replay_skip_reasons"] = skip
    payload["replay_closed_trades"] = len(closed)
    payload["replay_admit_pass"] = accepted
    payload["replay_admit_veto"] = rejected
    payload["first_1m_epoch"] = first_1m
    payload["first_1m_utc"] = datetime.fromtimestamp(first_1m, tz=timezone.utc).isoformat()
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
