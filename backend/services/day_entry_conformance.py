"""Match replay-admitted DAY entries against stored Ocean fills.

Observability / analysis only. Does not change production ranking or gates.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from backend.services.day_production_lifecycle_replay import ReplayCandidate, parse_epoch

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
EXACT_SEC = 1
LOOP_CADENCE_SEC = 60
BAR_CADENCE_SEC = 900


def _api(symbol: str) -> str:
    s = str(symbol or "").replace("/", "").replace("-", "").replace("_", "").upper()
    if s.endswith("USD") and not s.endswith("USDT"):
        s += "T"
    return s


@dataclass
class ActualEntry:
    epoch: int
    timestamp: str
    symbol: str
    decision_id: str
    quantity: float
    price: float
    notional: float


@dataclass
class Mismatch:
    kind: str
    symbol: str
    actual_epoch: int | None
    replay_epoch: int | None
    actual_decision_id: str
    first_divergence: str
    detail: str


@dataclass
class ConformanceReport:
    actual_entries: int
    replay_entries: int
    exact_matches: int
    loop_cadence_matches: int
    bar_cadence_matches: int
    false_positives: int
    missed: int
    precision: float
    recall: float
    symbol_distribution_actual: dict[str, int]
    symbol_distribution_replay: dict[str, int]
    trades_per_day_actual: float
    trades_per_day_replay: float
    hold_time_actual: dict[str, Any]
    hold_time_replay: dict[str, Any]
    exit_reason_actual: dict[str, int]
    exit_reason_replay: dict[str, int]
    mismatches: list[Mismatch] = field(default_factory=list)
    evidence_limits: list[str] = field(default_factory=list)
    stored_live_buy_count: int = 0
    briefing_actual_entries: int = 53


def load_actual_entries(conn: sqlite3.Connection, *, start: str, end: str) -> list[ActualEntry]:
    rows = conn.execute(
        """
        SELECT timestamp, symbol, decision_id, quantity, price
        FROM paper_trades
        WHERE UPPER(side)='BUY'
          AND COALESCE(strategy_id,'')='day'
          AND COALESCE(mode,'')='live'
          AND timestamp>=? AND timestamp<?
        ORDER BY timestamp
        """,
        (start, end),
    ).fetchall()
    out: list[ActualEntry] = []
    for ts, symbol, decision_id, qty, price in rows:
        ep = parse_epoch(ts)
        if ep is None:
            continue
        q = float(qty or 0)
        p = float(price or 0)
        out.append(
            ActualEntry(
                epoch=ep,
                timestamp=str(ts),
                symbol=_api(symbol),
                decision_id=str(decision_id or ""),
                quantity=q,
                price=p,
                notional=q * p,
            )
        )
    return out


def load_actual_exits(conn: sqlite3.Connection, *, start: str, end: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT timestamp, symbol, exit_reason, hold_time_seconds, quantity, price, pnl
        FROM paper_trades
        WHERE UPPER(side)='SELL'
          AND COALESCE(strategy_id,'')='day'
          AND COALESCE(mode,'')='live'
          AND timestamp>=? AND timestamp<?
        ORDER BY timestamp
        """,
        (start, end),
    ).fetchall()
    out: list[dict[str, Any]] = []
    for ts, symbol, reason, hold, qty, price, pnl in rows:
        out.append(
            {
                "timestamp": ts,
                "epoch": parse_epoch(ts),
                "symbol": _api(symbol),
                "exit_reason": str(reason or ""),
                "hold_sec": float(hold or 0),
                "notional": float(qty or 0) * float(price or 0),
                "pnl": float(pnl or 0),
            }
        )
    return out


def load_rejects(conn: sqlite3.Connection, *, start: str, end: str) -> list[tuple[int, str, str, str]]:
    cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(portfolio_engine_rejects)")}
    ts_col = "ts" if "ts" in cols else "timestamp"
    reason_col = "reason" if "reason" in cols else "reject_reason"
    filter_col = "filter_name" if "filter_name" in cols else "code"
    rows = conn.execute(
        f"""
        SELECT {ts_col}, symbol, {reason_col}, {filter_col}
        FROM portfolio_engine_rejects
        WHERE {ts_col}>=? AND {ts_col}<?
        ORDER BY {ts_col}
        """,
        (start, end),
    ).fetchall()
    out: list[tuple[int, str, str, str]] = []
    for ts, symbol, reason, filt in rows:
        ep = parse_epoch(ts)
        if ep is None:
            continue
        out.append((ep, _api(symbol), str(reason or ""), str(filt or "")))
    return out


def _nearest_reject(
    rejects: list[tuple[int, str, str, str]],
    *,
    symbol: str,
    epoch: int,
    window: int = BAR_CADENCE_SEC,
) -> str | None:
    best = None
    best_dt = window + 1
    for ep, sym, reason, filt in rejects:
        if sym != symbol:
            continue
        dt = abs(ep - epoch)
        if dt < best_dt:
            best_dt = dt
            best = f"{filt}:{reason}" if filt else reason
    return best


def _hold_dist(holds: list[float]) -> dict[str, Any]:
    xs = sorted(float(h) for h in holds if h is not None)
    n = len(xs)

    def pct(lim: float) -> float:
        return round(100.0 * sum(1 for h in xs if h <= lim) / n, 2) if n else 0.0

    return {
        "n": n,
        "median_s": xs[n // 2] if n else None,
        "p10_s": xs[int(0.10 * (n - 1))] if n > 1 else (xs[0] if n else None),
        "p90_s": xs[int(0.90 * (n - 1))] if n > 1 else (xs[0] if n else None),
        "pct_15m": pct(900),
        "pct_30m": pct(1800),
        "pct_4h": pct(14400),
    }


def match_entries(
    actual: list[ActualEntry],
    replay_accepted: list[ReplayCandidate],
    *,
    rejects: list[tuple[int, str, str, str]] | None = None,
    replay_all: list[ReplayCandidate] | None = None,
    actual_exits: list[dict[str, Any]] | None = None,
    replay_exit_reasons: dict[str, int] | None = None,
    replay_holds: list[float] | None = None,
    window_days: float = 8.0,
    evidence_limits: list[str] | None = None,
) -> ConformanceReport:
    rejects = rejects or []
    used_replay: set[int] = set()

    exact = loop = bar = 0
    missed: list[Mismatch] = []
    matched_actual = 0
    for act in actual:
        pool = [(i, c) for i, c in enumerate(replay_accepted) if c.symbol == act.symbol and i not in used_replay]
        if not pool:
            first = _first_replay_reason(replay_all, act.symbol, act.epoch)
            reject = _nearest_reject(rejects, symbol=act.symbol, epoch=act.epoch)
            missed.append(
                Mismatch(
                    kind="actual_entry_missed_by_replay",
                    symbol=act.symbol,
                    actual_epoch=act.epoch,
                    replay_epoch=None,
                    actual_decision_id=act.decision_id,
                    first_divergence=first or "NO_REPLAY_CANDIDATE_ON_SYMBOL",
                    detail=f"stored_reject_near={reject}; decision_id={act.decision_id}",
                )
            )
            continue
        i, cand = min(pool, key=lambda row: abs(row[1].epoch - act.epoch))
        dt = abs(cand.epoch - act.epoch)
        if dt <= EXACT_SEC:
            exact += 1
            used_replay.add(i)
            matched_actual += 1
        elif dt <= LOOP_CADENCE_SEC:
            loop += 1
            used_replay.add(i)
            matched_actual += 1
        elif dt <= BAR_CADENCE_SEC:
            bar += 1
            used_replay.add(i)
            matched_actual += 1
        else:
            first = _first_replay_reason(replay_all, act.symbol, act.epoch)
            reject = _nearest_reject(rejects, symbol=act.symbol, epoch=act.epoch)
            missed.append(
                Mismatch(
                    kind="actual_entry_missed_by_replay",
                    symbol=act.symbol,
                    actual_epoch=act.epoch,
                    replay_epoch=cand.epoch,
                    actual_decision_id=act.decision_id,
                    first_divergence=first or f"NEAREST_REPLAY_ENTRY_DT={dt}s",
                    detail=f"stored_reject_near={reject}; decision_id={act.decision_id}",
                )
            )

    extras: list[Mismatch] = []
    for i, cand in enumerate(replay_accepted):
        if i in used_replay:
            continue
        reject = _nearest_reject(rejects, symbol=cand.symbol, epoch=cand.epoch)
        extras.append(
            Mismatch(
                kind="replay_only_extra_entry",
                symbol=cand.symbol,
                actual_epoch=None,
                replay_epoch=cand.epoch,
                actual_decision_id="",
                first_divergence=reject or "NO_STORED_REJECT_NEAR_REPLAY_ENTRY",
                detail=f"replay_reason={cand.first_reason}; cash={cand.cash}; slots={cand.slot_occupancy}",
            )
        )

    n_act = len(actual)
    n_rep = len(replay_accepted)
    cadence_matches = exact + loop + bar
    precision = cadence_matches / n_rep if n_rep else 0.0
    recall = cadence_matches / n_act if n_act else 0.0
    act_syms: dict[str, int] = defaultdict(int)
    rep_syms: dict[str, int] = defaultdict(int)
    for a in actual:
        act_syms[a.symbol] += 1
    for c in replay_accepted:
        rep_syms[c.symbol] += 1
    exit_act: dict[str, int] = defaultdict(int)
    holds_act: list[float] = []
    for row in actual_exits or []:
        exit_act[str(row.get("exit_reason") or "NONE")] += 1
        holds_act.append(float(row.get("hold_sec") or 0))
    return ConformanceReport(
        actual_entries=n_act,
        replay_entries=n_rep,
        exact_matches=exact,
        loop_cadence_matches=loop,
        bar_cadence_matches=bar,
        false_positives=len(extras),
        missed=len(missed),
        precision=round(precision, 4),
        recall=round(recall, 4),
        symbol_distribution_actual=dict(act_syms),
        symbol_distribution_replay=dict(rep_syms),
        trades_per_day_actual=round(n_act / max(1e-9, window_days), 3),
        trades_per_day_replay=round(n_rep / max(1e-9, window_days), 3),
        hold_time_actual=_hold_dist(holds_act),
        hold_time_replay=_hold_dist(replay_holds or []),
        exit_reason_actual=dict(exit_act),
        exit_reason_replay=dict(replay_exit_reasons or {}),
        mismatches=missed + extras,
        evidence_limits=list(evidence_limits or []),
        stored_live_buy_count=n_act,
    )


def _first_replay_reason(replay_all: list[ReplayCandidate] | None, symbol: str, epoch: int) -> str:
    if not replay_all:
        return ""
    best = None
    best_dt = BAR_CADENCE_SEC + 1
    for cand in replay_all:
        if cand.symbol != symbol:
            continue
        dt = abs(cand.epoch - epoch)
        if dt < best_dt:
            best_dt = dt
            best = cand
    if best is None:
        return "NO_REPLAY_CANDIDATE_WITHIN_BAR"
    return f"{best.first_reason}|dt={best_dt}s|accepted={best.accepted}"


def report_to_dict(report: ConformanceReport) -> dict[str, Any]:
    return {
        "actual_entries": report.actual_entries,
        "briefing_actual_entries": report.briefing_actual_entries,
        "stored_live_buy_count": report.stored_live_buy_count,
        "replay_entries": report.replay_entries,
        "exact_symbol_timestamp_matches": report.exact_matches,
        "matches_within_60s_loop": report.loop_cadence_matches,
        "matches_within_900s_bar": report.bar_cadence_matches,
        "event_level_matches": report.exact_matches + report.loop_cadence_matches + report.bar_cadence_matches,
        "false_positive_replay_entries": report.false_positives,
        "missed_actual_entries": report.missed,
        "precision": report.precision,
        "recall": report.recall,
        "symbol_distribution_actual": report.symbol_distribution_actual,
        "symbol_distribution_replay": report.symbol_distribution_replay,
        "trades_per_day_actual": report.trades_per_day_actual,
        "trades_per_day_replay": report.trades_per_day_replay,
        "hold_time_actual": report.hold_time_actual,
        "hold_time_replay": report.hold_time_replay,
        "exit_reason_actual": report.exit_reason_actual,
        "exit_reason_replay": report.exit_reason_replay,
        "mismatches": [
            {
                "kind": m.kind,
                "symbol": m.symbol,
                "actual_epoch": m.actual_epoch,
                "actual_utc": datetime.fromtimestamp(m.actual_epoch, tz=timezone.utc).isoformat() if m.actual_epoch else None,
                "replay_epoch": m.replay_epoch,
                "replay_utc": datetime.fromtimestamp(m.replay_epoch, tz=timezone.utc).isoformat() if m.replay_epoch else None,
                "actual_decision_id": m.actual_decision_id,
                "first_divergence": m.first_divergence,
                "detail": m.detail,
            }
            for m in report.mismatches
        ],
        "evidence_limits": report.evidence_limits,
    }
