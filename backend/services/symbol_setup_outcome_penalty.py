"""
Outcome-driven ranking discipline for symbol/setup/regime churn.

Reads closed paper_trades (not opinions) and applies ranking/EV penalties only
when rolling realized outcomes are negative — never a hard global symbol block.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.utils.symbols import normalize_symbol

logger = logging.getLogger(__name__)

# Post STOP_LOSS-cleanup round trips (clean mark + exit path active).
CLEAN_INFRA_MIN_SELL_ID = 2987
# First buy after outcome-penalty deploy (passive-watch epoch).
POST_PENALTY_MIN_BUY_ID = 3022
# First buy after v3 final-selection ranking deploy (recovery epoch).
POST_V3_MIN_BUY_ID = 3103

XRP_PENALTY_SETUPS = frozenset({"FAILED_BREAKDOWN_REVERSAL", "RANGE_BOUNCE"})
XRP_PENALTY_REGIMES = frozenset({"bear", "range", "sideways", "range_bound", "neutral"})

SOL_CREDIT_SETUPS = frozenset({"FAILED_BREAKDOWN_REVERSAL"})
SOL_CREDIT_REGIMES = frozenset({"bear", "range", "sideways", "range_bound", "neutral"})

BEAR_RANGE_ALIASES = frozenset({"bear", "range", "sideways", "range_bound"})

# Prior generation (21-trade sample) — for audit diffs only.
XRP_PENALTY_V1_RANK_DELTA = -0.16
XRP_PENALTY_V1_EV_FACTOR = 0.68
XRP_PENALTY_V1_SIZE_FACTOR = 0.63

# Strengthened generation (ranking discipline only, not a hard block).
XRP_PENALTY_V2_RANK_BASE = -0.28
XRP_PENALTY_V2_RANK_EXTRA = -0.04
XRP_PENALTY_V2_EV_FLOOR = 0.45
XRP_PENALTY_V2_EV_BASE = 0.50
XRP_PENALTY_V2_SIZE_FLOOR = 0.40
XRP_PENALTY_V2_SIZE_BASE = 0.50

SOL_CREDIT_RANK_MAX = 0.06
SOL_CREDIT_MIN_TRADES_FOR_FULL = 10
XRP_RECOVERY_MIN_TRADES = 10

# v3 final-selection discipline (ranking only — no hard blocks).
XRP_PENALTY_V3_RANK_BASE = -0.38
XRP_PENALTY_V3_RANK_EXTRA = -0.05
XRP_PENALTY_V3_EV_FLOOR = 0.35
XRP_PENALTY_V3_EV_BASE = 0.42
XRP_PENALTY_V3_SIZE_FLOOR = 0.40
XRP_PENALTY_V3_SIZE_BASE = 0.50
XRP_PENALTY_V3_FINAL_SCORE = -0.10

BTC_PENALTY_SETUPS = frozenset({"FAILED_BREAKDOWN_REVERSAL"})
BTC_PENALTY_V3_RANK = -0.08
BTC_PENALTY_V3_EV_FACTOR = 0.85
BTC_PENALTY_V3_FINAL_SCORE = -0.03

SOL_V3_RANK_MAX = 0.10
SOL_V3_FINAL_SCORE_CREDIT = 0.04

ETH_CREDIT_SETUPS = frozenset({"FAILED_BREAKDOWN_REVERSAL", "RANGE_BOUNCE"})
ETH_V3_RANK_MAX = 0.04
ETH_V3_FINAL_SCORE_CREDIT = 0.02


@dataclass
class ClosedTradeRow:
    sell_id: int
    symbol: str
    setup: str
    regime: str
    exit_reason: str
    pnl: float
    hold_min: float
    selected_ev: float | None
    timestamp: str


def _parse_explain(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        return json.loads(raw) if isinstance(raw, str) else dict(raw)
    except Exception:
        return {}


def _extract_setup_regime_ev(explain: dict[str, Any]) -> tuple[str, str, float | None]:
    setup = str(explain.get("setup_type") or explain.get("entry_thesis") or "")
    regime = str(explain.get("day_route_regime") or explain.get("signal_regime_label") or explain.get("regime") or explain.get("adaptive_regime") or "neutral").strip().lower()
    sc = explain.get("score_components_json")
    if isinstance(sc, str):
        try:
            scj = json.loads(sc)
            regime = str(scj.get("adaptive_regime") or regime).strip().lower()
        except Exception:
            pass
    ev_raw = explain.get("selected_net_expected_value")
    try:
        selected_ev = float(ev_raw) if ev_raw is not None else None
    except Exception:
        selected_ev = None
    return setup, regime, selected_ev


def _load_closed_trades(db_path: str | Path, *, min_sell_id: int = CLEAN_INFRA_MIN_SELL_ID) -> list[ClosedTradeRow]:
    rows: list[ClosedTradeRow] = []
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            """
            SELECT id, symbol, pnl, exit_reason, timestamp, entry_price, price,
                   explainability_json, hold_time_seconds
            FROM paper_trades
            WHERE side = 'SELL'
              AND id >= ?
              AND COALESCE(is_synthetic, 0) = 0
            ORDER BY id ASC
            """,
            (min_sell_id,),
        )
        for r in cur.fetchall():
            explain = _parse_explain(r["explainability_json"])
            setup, regime, selected_ev = _extract_setup_regime_ev(explain)
            hold_sec = float(r["hold_time_seconds"] or explain.get("hold_time_seconds") or explain.get("time_in_trade_sec") or 0)
            hold_min = hold_sec / 60.0 if hold_sec > 0 else 75.0
            rows.append(
                ClosedTradeRow(
                    sell_id=int(r["id"]),
                    symbol=normalize_symbol(r["symbol"]),
                    setup=setup,
                    regime=regime,
                    exit_reason=str(r["exit_reason"] or ""),
                    pnl=float(r["pnl"] or 0.0),
                    hold_min=hold_min,
                    selected_ev=selected_ev,
                    timestamp=str(r["timestamp"] or ""),
                )
            )
    return rows


def _metrics(trades: list[ClosedTradeRow]) -> dict[str, Any]:
    if not trades:
        return {
            "count": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "realized_pnl": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "avg_net_pnl": 0.0,
            "expectancy": 0.0,
            "profit_factor": 0.0,
            "time_stop_rate": 0.0,
            "time_stop_pnl": 0.0,
            "avg_hold_min": 0.0,
            "positive_ev_negative_outcome": 0,
        }
    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p >= 0]
    losses = [p for p in pnls if p < 0]
    gross_win = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0
    ts = [t for t in trades if "TIME_STOP" in (t.exit_reason or "").upper()]
    pos_ev_neg = sum(1 for t in trades if (t.selected_ev or 0) > 0 and t.pnl < 0)
    return {
        "count": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(trades),
        "realized_pnl": sum(pnls),
        "avg_win": (sum(wins) / len(wins)) if wins else 0.0,
        "avg_loss": (sum(losses) / len(losses)) if losses else 0.0,
        "avg_net_pnl": sum(pnls) / len(trades),
        "expectancy": sum(pnls) / len(trades),
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0),
        "time_stop_rate": len(ts) / len(trades),
        "time_stop_pnl": sum(t.pnl for t in ts),
        "avg_hold_min": sum(t.hold_min for t in trades) / len(trades),
        "positive_ev_negative_outcome": pos_ev_neg,
    }


def _exit_reason_counts(trades: list[ClosedTradeRow]) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for t in trades:
        out[str(t.exit_reason or "UNKNOWN")] += 1
    return dict(out)


def _setup_breakdown(trades: list[ClosedTradeRow]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[ClosedTradeRow]] = defaultdict(list)
    for t in trades:
        groups[t.setup or "UNKNOWN"].append(t)
    return {k: _metrics(v) for k, v in groups.items()}


def _regime_bear_range_count(trades: list[ClosedTradeRow]) -> int:
    return sum(1 for t in trades if t.regime in BEAR_RANGE_ALIASES or "bear" in t.regime or "range" in t.regime)


def _churn_protection_buckets(trades: list[ClosedTradeRow]) -> dict[str, Any]:
    """Rolling churn metrics keyed by symbol/setup/regime."""
    buckets: dict[str, list[ClosedTradeRow]] = defaultdict(list)
    for t in trades:
        key = f"{t.symbol}|{t.setup or 'UNKNOWN'}|{t.regime or 'neutral'}"
        buckets[key].append(t)
    out: dict[str, Any] = {}
    for key, bucket_trades in sorted(buckets.items()):
        sym, setup, regime = key.split("|", 2)
        out[key] = {
            "symbol": sym,
            "setup": setup,
            "regime": regime,
            "metrics_all": _metrics(bucket_trades),
            "metrics_last_5": _metrics(bucket_trades[-5:]),
            "metrics_last_10": _metrics(bucket_trades[-10:]),
            "exit_reason_counts": _exit_reason_counts(bucket_trades),
        }
    return out


def _simulate_ranking_with_penalty(
    *,
    xrp_raw_ev: float = 0.01284342,
    btc_raw_ev: float = 0.01152369,
    sol_raw_ev: float = 0.00396396,
    xrp_rank_score: float = 0.40,
    btc_rank_score: float = 0.43,
    sol_rank_score: float = 0.44,
) -> dict[str, Any]:
    """Illustrate pre/post penalty ordering using typical top-4 EV snapshots."""
    from backend.database_schema import DATABASE_PATH

    pen = evaluate_outcome_penalty("XRP/USDT", "FAILED_BREAKDOWN_REVERSAL", "bear", db_path=DATABASE_PATH)
    peer_ceiling = float(pen.get("peer_ev_ceiling") or 0.012)
    xrp_adj_ev = min(xrp_raw_ev * float(pen.get("ev_factor") or 1.0), peer_ceiling * 0.98)
    xrp_adj_rank = max(0.0, xrp_rank_score + float(pen.get("rank_delta") or 0.0))
    candidates = [
        ("BTC/USDT", btc_raw_ev, btc_rank_score, False),
        ("XRP/USDT", xrp_raw_ev, xrp_rank_score, True),
        ("SOL/USDT", sol_raw_ev, sol_rank_score, False),
    ]
    pre = sorted(candidates, key=lambda x: (x[2], x[1]), reverse=True)
    post = sorted(
        [
            (
                sym,
                min(ev * (pen.get("ev_factor") if penalized else 1.0), peer_ceiling * 0.98) if penalized else ev,
                rs + (pen.get("rank_delta") if penalized else 0.0),
                penalized,
            )
            for sym, ev, rs, penalized in candidates
        ],
        key=lambda x: (x[2], x[1]),
        reverse=True,
    )
    return {
        "xrp_penalty_applied": bool(pen.get("applied")),
        "setups_regimes_affected": [{"setup": s, "regime": "bear/range", "symbol": "XRP/USDT"} for s in sorted(XRP_PENALTY_SETUPS)],
        "old_xrp_selected_ev": xrp_raw_ev,
        "new_xrp_adjusted_ev": round(xrp_adj_ev, 8),
        "old_xrp_rank_score_proxy": xrp_rank_score,
        "new_xrp_rank_score_proxy": round(xrp_adj_rank, 4),
        "xrp_still_ranks_1_pre_penalty": pre[0][0] == "XRP/USDT",
        "xrp_still_ranks_1_post_penalty": post[0][0] == "XRP/USDT",
        "preferred_symbol_post_penalty": post[0][0],
        "pre_penalty_order": [c[0] for c in pre],
        "post_penalty_order": [c[0] for c in post],
        "no_hard_xrp_global_block": True,
        "no_strategy_threshold_changes": True,
        "no_exit_changes": True,
        "no_ledger_reset": True,
        "penalty_detail": pen,
    }


def build_churn_audit(db_path: str | Path, *, min_sell_id: int = CLEAN_INFRA_MIN_SELL_ID) -> dict[str, Any]:
    all_trades = _load_closed_trades(db_path, min_sell_id=min_sell_id)
    by_symbol: dict[str, list[ClosedTradeRow]] = defaultdict(list)
    for t in all_trades:
        by_symbol[t.symbol].append(t)

    symbol_reports: dict[str, Any] = {}
    for sym, trades in sorted(by_symbol.items()):
        symbol_reports[sym] = {
            "total_trades": len(trades),
            "realized_pnl": round(sum(t.pnl for t in trades), 2),
            "metrics": _metrics(trades),
            "metrics_last_5": _metrics(trades[-5:]),
            "metrics_last_10": _metrics(trades[-10:]),
            "exit_reason_counts": _exit_reason_counts(trades),
            "setup_breakdown": _setup_breakdown(trades),
            "bear_range_entry_count": _regime_bear_range_count(trades),
            "positive_ev_negative_outcome_count": sum(1 for t in trades if (t.selected_ev or 0) > 0 and t.pnl < 0),
        }

    xrp = symbol_reports.get("XRP/USDT", {})
    xrp_setups = xrp.get("setup_breakdown", {})
    penalty_eval = evaluate_outcome_penalty("XRP/USDT", "FAILED_BREAKDOWN_REVERSAL", "bear", db_path=db_path)
    penalty_rb = evaluate_outcome_penalty("XRP/USDT", "RANGE_BOUNCE", "bear", db_path=db_path)

    return {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "clean_infra_min_sell_id": min_sell_id,
        "symbols": symbol_reports,
        "churn_protection_by_bucket": _churn_protection_buckets(all_trades),
        "xrp_focus": {
            "total_trades": xrp.get("total_trades", 0),
            "realized_pnl": xrp.get("realized_pnl", 0.0),
            "win_rate": xrp.get("metrics", {}).get("win_rate", 0.0),
            "avg_win": xrp.get("metrics", {}).get("avg_win", 0.0),
            "avg_loss": xrp.get("metrics", {}).get("avg_loss", 0.0),
            "exit_reason_counts": xrp.get("exit_reason_counts", {}),
            "setup_breakdown": xrp.get("setup_breakdown", {}),
            "failed_breakdown_reversal_pnl": xrp_setups.get("FAILED_BREAKDOWN_REVERSAL", {}).get("realized_pnl", 0.0),
            "range_bounce_pnl": xrp_setups.get("RANGE_BOUNCE", {}).get("realized_pnl", 0.0),
            "htf_trend_pullback_pnl": xrp_setups.get("HTF_TREND_PULLBACK", {}).get("realized_pnl", 0.0),
            "time_stop_pnl": _metrics(by_symbol.get("XRP/USDT", [])).get("time_stop_pnl", 0.0),
            "avg_hold_min": xrp.get("metrics", {}).get("avg_hold_min", 0.0),
            "bear_range_entries": xrp.get("bear_range_entry_count", 0),
            "positive_ev_negative_outcome": xrp.get("positive_ev_negative_outcome_count", 0),
            "metrics_last_5": xrp.get("metrics_last_5", {}),
            "metrics_last_10": xrp.get("metrics_last_10", {}),
        },
        "penalty_preview": penalty_eval,
        "penalty_preview_range_bounce": penalty_rb,
        "penalty_verification": _simulate_ranking_with_penalty(
            xrp_raw_ev=0.10718585,
            xrp_rank_score=0.18040309,
            btc_raw_ev=0.01152369,
            btc_rank_score=0.42929475,
            sol_raw_ev=0.01270612,
            sol_rank_score=0.44436028,
        ),
        "passive_watch": {
            "note": "Watch next 3-5 buys after penalty deploy; append rows as trades close.",
            "xrp_3020_status": "TIME_STOP_EXIT closed sell_id=3021 pnl=-28.95 (engine exit, no manual close)",
            "entries": [],
            "watch_target_count": 5,
        },
        "peer_comparison": {
            sym: {
                "realized_pnl": symbol_reports[sym]["realized_pnl"],
                "expectancy": symbol_reports[sym]["metrics"]["expectancy"],
                "win_rate": symbol_reports[sym]["metrics"]["win_rate"],
            }
            for sym in ("BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT")
            if sym in symbol_reports
        },
    }


def _peer_selected_ev_ceiling(all_trades: list[ClosedTradeRow], *, below_median: bool = False) -> float:
    """Peer EV anchor from recent selected EV medians."""
    import statistics

    peer_medians: list[float] = []
    for sym in ("BTC/USDT", "ETH/USDT", "SOL/USDT"):
        evs = [float(t.selected_ev) for t in [x for x in all_trades if x.symbol == sym][-5:] if t.selected_ev is not None and t.selected_ev > 0]
        if evs:
            peer_medians.append(float(statistics.median(evs)))
    if not peer_medians:
        return 0.012
    if below_median:
        return min(peer_medians)
    return max(peer_medians)


def _regime_matches_bear_range(regime_l: str) -> bool:
    if regime_l in BEAR_RANGE_ALIASES:
        return True
    return any(x in regime_l for x in ("bear", "range", "sideways"))


def _filter_bucket_trades(
    trades: list[ClosedTradeRow],
    *,
    symbol: str,
    setup: str | None = None,
    post_penalty_only: bool = False,
    min_buy_id: int = POST_PENALTY_MIN_BUY_ID,
    db_path: str | Path | None = None,
) -> list[ClosedTradeRow]:
    sym = normalize_symbol(symbol)
    setup_u = str(setup or "").strip().upper()
    out = [t for t in trades if t.symbol == sym and (not setup_u or (t.setup or "").upper() == setup_u)]
    if not post_penalty_only or db_path is None:
        return out
    # Match sells to buys >= min_buy_id for post-penalty epoch.
    buy_ids: set[int] = set()
    with sqlite3.connect(str(db_path)) as conn:
        for row in conn.execute(
            "SELECT id FROM paper_trades WHERE side='BUY' AND id >= ? AND symbol = ?",
            (min_buy_id, sym),
        ):
            buy_ids.add(int(row[0]))
    sell_to_buy: dict[int, int] = {}
    with sqlite3.connect(str(db_path)) as conn:
        for bid in sorted(buy_ids):
            sell = conn.execute(
                """
                SELECT id FROM paper_trades
                WHERE side='SELL' AND id > ? AND symbol = ?
                ORDER BY id LIMIT 1
                """,
                (bid, sym),
            ).fetchone()
            if sell:
                sell_to_buy[int(sell[0])] = bid
    return [t for t in out if sell_to_buy.get(t.sell_id, 0) >= min_buy_id]


def _xrp_penalty_recovery_met(
    key_trades: list[ClosedTradeRow],
    *,
    db_path: str | Path,
    min_buy_id: int = POST_V3_MIN_BUY_ID,
) -> bool:
    post_xrp = _filter_bucket_trades(
        key_trades,
        symbol="XRP/USDT",
        post_penalty_only=True,
        min_buy_id=min_buy_id,
        db_path=db_path,
    )
    m = _metrics(post_xrp)
    if m["count"] < XRP_RECOVERY_MIN_TRADES:
        return False
    if m["expectancy"] <= 0:
        return False
    if m["profit_factor"] <= 1.2:
        return False
    avg_win = float(m["avg_win"] or 0.0)
    avg_loss = abs(float(m["avg_loss"] or 0.0))
    if avg_win > 0 and avg_loss > (2.0 * avg_win):
        return False
    bad_exits = sum(1 for t in post_xrp if "TIME_STOP" in (t.exit_reason or "").upper() or "STOP_LOSS" in (t.exit_reason or "").upper())
    if bad_exits / max(len(post_xrp), 1) > 0.45:
        return False
    return True


def evaluate_outcome_penalty(
    symbol: str,
    setup: str,
    regime: str,
    *,
    db_path: str | Path | None = None,
    min_sell_id: int = CLEAN_INFRA_MIN_SELL_ID,
) -> dict[str, Any]:
    """
    Return outcome-based ranking penalty for a candidate.
    Only applies to XRP + FAILED_BREAKDOWN_REVERSAL/RANGE_BOUNCE + bear/range regimes.
    """
    sym = normalize_symbol(symbol)
    setup_u = str(setup or "").strip().upper()
    regime_l = str(regime or "neutral").strip().lower()

    base = {
        "applied": False,
        "symbol": sym,
        "setup": setup_u,
        "regime": regime_l,
        "rank_delta": 0.0,
        "ev_factor": 1.0,
        "size_factor": 1.0,
        "reason": "not_xrp_penalty_scope",
        "hard_block": False,
    }

    if sym != "XRP/USDT":
        return base
    if setup_u not in {s.upper() for s in XRP_PENALTY_SETUPS}:
        base["reason"] = "setup_not_in_penalty_scope"
        return base
    if regime_l not in XRP_PENALTY_REGIMES and not _regime_matches_bear_range(regime_l):
        base["reason"] = "regime_not_in_penalty_scope"
        return base

    if db_path is None:
        from backend.database_schema import DATABASE_PATH

        db_path = DATABASE_PATH

    all_trades = _load_closed_trades(db_path, min_sell_id=min_sell_id)
    sym_trades = [t for t in all_trades if t.symbol == sym]
    key_trades = [t for t in sym_trades if (t.setup or "").upper() == setup_u]
    if not key_trades:
        key_trades = sym_trades

    m5 = _metrics(key_trades[-5:])
    m10 = _metrics(key_trades[-10:])
    mall = _metrics(key_trades)

    # Compare to best peer expectancy (BTC/ETH/SOL)
    peer_exp: list[float] = []
    for peer in ("BTC/USDT", "ETH/USDT", "SOL/USDT"):
        pt = [t for t in all_trades if t.symbol == peer]
        if pt:
            peer_exp.append(_metrics(pt)["expectancy"])
    best_peer = max(peer_exp) if peer_exp else 0.0

    negative = mall["count"] >= 3 and mall["expectancy"] < 0
    repeated_time_stop = m5["count"] >= 3 and m5["time_stop_rate"] >= 0.6 and m5["expectancy"] < 0
    worse_than_peers = mall["expectancy"] < best_peer - 1.0

    if _xrp_penalty_recovery_met(key_trades, db_path=db_path):
        base["reason"] = "xrp_recovery_met_penalty_eased"
        base["recovery_met"] = True
        base["metrics_last_5"] = m5
        base["metrics_last_10"] = m10
        base["metrics_all"] = mall
        return base

    if not (negative and (repeated_time_stop or worse_than_peers)):
        base["reason"] = "outcomes_not_bad_enough_for_penalty"
        base["metrics_last_5"] = m5
        base["metrics_last_10"] = m10
        base["metrics_all"] = mall
        base["best_peer_expectancy"] = best_peer
        return base

    severity = min(1.0, abs(mall["expectancy"]) / 8.0 + m5["time_stop_rate"] * 0.35)
    rank_delta = XRP_PENALTY_V3_RANK_BASE - abs(XRP_PENALTY_V3_RANK_EXTRA) * severity
    ev_factor = max(XRP_PENALTY_V3_EV_FLOOR, XRP_PENALTY_V3_EV_BASE - 0.05 * severity)
    size_factor = max(XRP_PENALTY_V3_SIZE_FLOOR, XRP_PENALTY_V3_SIZE_BASE - 0.10 * severity)
    xrp_exp_positive = mall["expectancy"] > 0
    peer_ev_ceiling = _peer_selected_ev_ceiling(all_trades, below_median=not xrp_exp_positive)
    ev_cap_mult = 0.88 if not xrp_exp_positive else 0.96
    final_score_penalty = XRP_PENALTY_V3_FINAL_SCORE * (0.75 + 0.25 * severity)

    return {
        "applied": True,
        "symbol": sym,
        "setup": setup_u,
        "regime": regime_l,
        "rank_delta": round(rank_delta, 4),
        "ev_factor": round(ev_factor, 4),
        "size_factor": round(size_factor, 4),
        "final_score_adjustment": round(final_score_penalty, 4),
        "peer_ev_ceiling": round(peer_ev_ceiling, 8),
        "peer_ev_cap_multiplier": ev_cap_mult,
        "penalty_generation": "v3_final_selection",
        "reason": "negative_expectancy_time_stop_churn",
        "hard_block": False,
        "recovery_met": False,
        "metrics_last_5": m5,
        "metrics_last_10": m10,
        "metrics_all": mall,
        "best_peer_expectancy": best_peer,
        "severity": round(severity, 4),
    }


def evaluate_sol_outcome_credit(
    symbol: str,
    setup: str,
    regime: str,
    *,
    db_path: str | Path | None = None,
    min_sell_id: int = CLEAN_INFRA_MIN_SELL_ID,
) -> dict[str, Any]:
    """Conservative positive rank credit for SOL FBR bear/range when outcomes are strong."""
    sym = normalize_symbol(symbol)
    setup_u = str(setup or "").strip().upper()
    regime_l = str(regime or "neutral").strip().lower()

    base = {
        "applied": False,
        "symbol": sym,
        "setup": setup_u,
        "regime": regime_l,
        "rank_delta": 0.0,
        "ev_factor": 1.0,
        "size_factor": 1.0,
        "reason": "not_sol_credit_scope",
    }

    if sym != "SOL/USDT":
        return base
    if setup_u not in {s.upper() for s in SOL_CREDIT_SETUPS}:
        base["reason"] = "setup_not_in_credit_scope"
        return base
    if not _regime_matches_bear_range(regime_l) and regime_l not in SOL_CREDIT_REGIMES:
        base["reason"] = "regime_not_in_credit_scope"
        return base

    if db_path is None:
        from backend.database_schema import DATABASE_PATH

        db_path = DATABASE_PATH

    all_trades = _load_closed_trades(db_path, min_sell_id=min_sell_id)
    bucket = [t for t in all_trades if t.symbol == sym and (t.setup or "").upper() == setup_u]
    if not bucket:
        bucket = [t for t in all_trades if t.symbol == sym]
    post_bucket = _filter_bucket_trades(bucket, symbol=sym, setup=setup_u, post_penalty_only=True, db_path=db_path)
    ref = post_bucket if len(post_bucket) >= 2 else bucket
    m = _metrics(ref)

    if m["count"] < 2 or m["expectancy"] <= 0 or m["realized_pnl"] <= 0:
        base["reason"] = "sol_outcomes_not_strong_enough"
        base["metrics_all"] = m
        return base

    scale = min(1.0, m["count"] / float(SOL_CREDIT_MIN_TRADES_FOR_FULL))
    pf_boost = min(1.0, max(0.0, (float(m["profit_factor"]) - 1.0) / 2.0))
    rank_delta = round(SOL_V3_RANK_MAX * scale * max(0.35, pf_boost), 4)
    if rank_delta <= 0.005:
        base["reason"] = "sol_credit_too_small"
        base["metrics_all"] = m
        return base

    final_credit = round(SOL_V3_FINAL_SCORE_CREDIT * scale * max(0.35, pf_boost), 4)
    return {
        "applied": True,
        "symbol": sym,
        "setup": setup_u,
        "regime": regime_l,
        "rank_delta": rank_delta,
        "ev_factor": 1.0,
        "size_factor": 1.0,
        "final_score_adjustment": final_credit,
        "credit_amount": rank_delta,
        "reason": "sol_fbr_bear_positive_outcomes",
        "metrics_all": m,
        "credit_scale": round(scale, 4),
        "credit_generation": "v3_final_selection",
    }


def evaluate_btc_outcome_penalty(
    symbol: str,
    setup: str,
    regime: str,
    *,
    db_path: str | Path | None = None,
    min_sell_id: int = CLEAN_INFRA_MIN_SELL_ID,
) -> dict[str, Any]:
    """Mild BTC FBR bear/range penalty when rolling expectancy is negative."""
    sym = normalize_symbol(symbol)
    setup_u = str(setup or "").strip().upper()
    regime_l = str(regime or "neutral").strip().lower()

    base = {
        "applied": False,
        "symbol": sym,
        "setup": setup_u,
        "regime": regime_l,
        "rank_delta": 0.0,
        "ev_factor": 1.0,
        "size_factor": 1.0,
        "final_score_adjustment": 0.0,
        "reason": "not_btc_penalty_scope",
        "hard_block": False,
    }

    if sym != "BTC/USDT":
        return base
    if setup_u not in {s.upper() for s in BTC_PENALTY_SETUPS}:
        base["reason"] = "setup_not_in_btc_penalty_scope"
        return base
    if not _regime_matches_bear_range(regime_l):
        base["reason"] = "regime_not_in_btc_penalty_scope"
        return base

    if db_path is None:
        from backend.database_schema import DATABASE_PATH

        db_path = DATABASE_PATH

    all_trades = _load_closed_trades(db_path, min_sell_id=min_sell_id)
    bucket = [t for t in all_trades if t.symbol == sym and (t.setup or "").upper() == setup_u]
    if not bucket:
        bucket = [t for t in all_trades if t.symbol == sym]
    m = _metrics(bucket[-10:] if len(bucket) >= 10 else bucket)

    if m["count"] < 3 or m["expectancy"] >= 0:
        base["reason"] = "btc_outcomes_not_negative_enough"
        base["metrics_all"] = m
        return base

    return {
        "applied": True,
        "symbol": sym,
        "setup": setup_u,
        "regime": regime_l,
        "rank_delta": BTC_PENALTY_V3_RANK,
        "ev_factor": BTC_PENALTY_V3_EV_FACTOR,
        "size_factor": 1.0,
        "final_score_adjustment": BTC_PENALTY_V3_FINAL_SCORE,
        "reason": "btc_fbr_bear_negative_expectancy",
        "hard_block": False,
        "metrics_all": m,
        "penalty_generation": "v3_final_selection",
    }


def evaluate_eth_outcome_credit(
    symbol: str,
    setup: str,
    regime: str,
    *,
    db_path: str | Path | None = None,
    min_sell_id: int = CLEAN_INFRA_MIN_SELL_ID,
) -> dict[str, Any]:
    """Small ETH watch credit when rolling bucket outcomes are positive and stable."""
    sym = normalize_symbol(symbol)
    setup_u = str(setup or "").strip().upper()
    regime_l = str(regime or "neutral").strip().lower()

    base = {
        "applied": False,
        "symbol": sym,
        "setup": setup_u,
        "regime": regime_l,
        "rank_delta": 0.0,
        "ev_factor": 1.0,
        "size_factor": 1.0,
        "final_score_adjustment": 0.0,
        "reason": "not_eth_credit_scope",
    }

    if sym != "ETH/USDT":
        return base
    if setup_u not in {s.upper() for s in ETH_CREDIT_SETUPS}:
        base["reason"] = "setup_not_in_eth_credit_scope"
        return base
    if not _regime_matches_bear_range(regime_l):
        base["reason"] = "regime_not_in_eth_credit_scope"
        return base

    if db_path is None:
        from backend.database_schema import DATABASE_PATH

        db_path = DATABASE_PATH

    all_trades = _load_closed_trades(db_path, min_sell_id=min_sell_id)
    bucket = [t for t in all_trades if t.symbol == sym and (t.setup or "").upper() == setup_u]
    if len(bucket) < 2:
        bucket = [t for t in all_trades if t.symbol == sym]
    ref = bucket[-5:] if len(bucket) >= 5 else bucket
    m = _metrics(ref)

    if m["count"] < 2 or m["expectancy"] <= 0 or m["realized_pnl"] <= 0:
        base["reason"] = "eth_outcomes_not_strong_enough"
        base["metrics_all"] = m
        return base

    avg_win = abs(float(m["avg_win"] or 0.0))
    recent_losses = [t for t in ref if t.pnl < 0]
    if recent_losses:
        last_loss = abs(float(recent_losses[-1].pnl))
        if avg_win > 0 and last_loss > (2.0 * avg_win):
            base["reason"] = "eth_recent_large_loss"
            base["metrics_all"] = m
            return base

    scale = min(1.0, m["count"] / 5.0)
    rank_delta = round(ETH_V3_RANK_MAX * scale, 4)
    if rank_delta <= 0.005:
        base["reason"] = "eth_credit_too_small"
        base["metrics_all"] = m
        return base

    return {
        "applied": True,
        "symbol": sym,
        "setup": setup_u,
        "regime": regime_l,
        "rank_delta": rank_delta,
        "ev_factor": 1.0,
        "size_factor": 1.0,
        "final_score_adjustment": round(ETH_V3_FINAL_SCORE_CREDIT * scale, 4),
        "credit_amount": rank_delta,
        "reason": "eth_positive_bucket_watch_credit",
        "metrics_all": m,
        "credit_generation": "v3_final_selection",
    }


def compute_final_selection_score(
    *,
    adjusted_ev: float,
    outcome_adjusted_rank: float,
    raw_rank_score: float,
    buy_margin: float | None,
    final_score_adjustment: float = 0.0,
) -> float:
    """v3 primary sort key: outcome-adjusted EV + rank dominate buy_margin."""
    bm_norm = 0.0
    if buy_margin is not None:
        try:
            bm_v = float(buy_margin)
            bm_norm = max(-0.04, min(0.04, (bm_v - 0.015) * 0.35))
        except (TypeError, ValueError):
            bm_norm = 0.0
    score = float(adjusted_ev) * 0.55 + float(outcome_adjusted_rank) * 0.35 + float(raw_rank_score) * 0.05 + bm_norm + float(final_score_adjustment)
    return round(score, 8)


def assign_v3_selection_ranks(candidates: list[Any]) -> None:
    """After sort by final_selection_score, stamp rank / peer / why on each candidate."""
    for i, cand in enumerate(candidates):
        dd = dict(getattr(cand, "decision_data", None) or {})
        dd["final_selected_rank"] = i + 1
        if i == 0:
            if len(candidates) > 1:
                peer = candidates[1]
                peer_dd = dict(getattr(peer, "decision_data", None) or {})
                peer_sym = str(getattr(peer, "symbol", "") or peer_dd.get("symbol") or "")
                peer_score = float(peer_dd.get("final_selection_score") or peer_dd.get("selection_score") or 0.0)
                win_score = float(dd.get("final_selection_score") or dd.get("selection_score") or 0.0)
                dd["best_rejected_peer"] = peer_sym
                dd["selected_over_symbol"] = peer_sym
                dd["selected_over_score"] = round(peer_score, 8)
                dd["why_selected"] = f"final_selection_score {win_score:.6f} > {peer_sym} {peer_score:.6f}"
            else:
                dd["best_rejected_peer"] = ""
                dd["selected_over_symbol"] = ""
                dd["selected_over_score"] = 0.0
                dd["why_selected"] = "solo_candidate_no_peer"
        setattr(cand, "decision_data", dd)


def evaluate_outcome_penalty_for_candidate(decision_data: dict[str, Any], symbol: str) -> dict[str, Any]:
    dd = dict(decision_data or {})
    setup = str(dd.get("setup_type") or dd.get("entry_thesis") or "")
    regime = str(dd.get("day_route_regime") or dd.get("day_regime") or dd.get("regime") or "neutral")
    return evaluate_outcome_penalty(symbol, setup, regime)


def apply_v3_outcome_ranking_to_decision_data(
    decision_data: dict[str, Any],
    symbol: str,
    *,
    raw_rank_score: float,
    buy_margin: float | None = None,
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    """Apply v3 outcome ranking: penalties/credits hit final_selection_score directly."""
    dd = dict(decision_data or {})
    setup = str(dd.get("setup_type") or dd.get("entry_thesis") or "")
    regime = str(dd.get("day_route_regime") or dd.get("day_regime") or dd.get("regime") or "neutral")

    if db_path is None:
        from backend.database_schema import DATABASE_PATH

        db_path = DATABASE_PATH

    xrp_pen = evaluate_outcome_penalty(symbol, setup, regime, db_path=db_path)
    btc_pen = evaluate_btc_outcome_penalty(symbol, setup, regime, db_path=db_path)
    sol_cred = evaluate_sol_outcome_credit(symbol, setup, regime, db_path=db_path)
    eth_cred = evaluate_eth_outcome_credit(symbol, setup, regime, db_path=db_path)

    dd["outcome_churn_penalty_eval"] = xrp_pen
    dd["outcome_btc_penalty_eval"] = btc_pen
    dd["outcome_sol_credit_eval"] = sol_cred
    dd["outcome_eth_credit_eval"] = eth_cred
    dd["v3_ranking_fix_applied"] = True

    raw_ev = float(dd.get("selected_net_expected_value_raw") or dd.get("selected_net_expected_value") or dd.get("net_expected_value") or 0.0)
    dd["raw_ev"] = round(raw_ev, 8)
    if dd.get("selected_net_expected_value_raw") is None:
        dd["selected_net_expected_value_raw"] = raw_ev

    dd["raw_rank_score"] = round(float(raw_rank_score), 6)

    outcome_rank_delta = 0.0
    final_score_adjustment = 0.0
    ev_mult = 1.0
    size_mult = 1.0
    penalty_reasons: list[str] = []
    penalty_applied = False
    credit_applied = False

    active_pen = None
    if xrp_pen.get("applied"):
        active_pen = xrp_pen
    elif btc_pen.get("applied"):
        active_pen = btc_pen

    if active_pen is not None:
        penalty_applied = True
        dd["outcome_churn_penalty_applied"] = active_pen is xrp_pen and bool(xrp_pen.get("applied"))
        dd["outcome_btc_penalty_applied"] = active_pen is btc_pen and bool(btc_pen.get("applied"))
        outcome_rank_delta += float(active_pen.get("rank_delta") or 0.0)
        ev_mult *= float(active_pen.get("ev_factor") or 1.0)
        size_mult *= float(active_pen.get("size_factor") or 1.0)
        final_score_adjustment += float(active_pen.get("final_score_adjustment") or 0.0)
        penalty_reasons.append(str(active_pen.get("reason") or "outcome_penalty"))
        if xrp_pen.get("applied"):
            peer_ceiling = float(xrp_pen.get("peer_ev_ceiling") or 0.012)
            cap_mult = float(xrp_pen.get("peer_ev_cap_multiplier") or 0.88)
            scaled_ev = raw_ev * ev_mult
            dd["adjusted_ev"] = round(min(scaled_ev, peer_ceiling * cap_mult), 8)
            dd["outcome_churn_rank_penalty"] = float(xrp_pen.get("rank_delta") or 0.0)
            dd["outcome_churn_ev_factor"] = float(xrp_pen.get("ev_factor") or 1.0)
            dd["outcome_churn_peer_ev_ceiling"] = peer_ceiling
        else:
            dd["adjusted_ev"] = round(raw_ev * ev_mult, 8)
            dd["outcome_btc_rank_penalty"] = float(btc_pen.get("rank_delta") or 0.0)
    else:
        dd["outcome_churn_penalty_applied"] = False
        dd["outcome_btc_penalty_applied"] = False
        dd["adjusted_ev"] = round(raw_ev * ev_mult, 8)

    if sol_cred.get("applied"):
        credit_applied = True
        dd["outcome_sol_credit_applied"] = True
        outcome_rank_delta += float(sol_cred.get("rank_delta") or 0.0)
        final_score_adjustment += float(sol_cred.get("final_score_adjustment") or 0.0)
        dd["outcome_sol_rank_credit"] = float(sol_cred.get("rank_delta") or 0.0)
        dd["outcome_sol_credit_amount"] = float(sol_cred.get("credit_amount") or 0.0)
        penalty_reasons.append(str(sol_cred.get("reason") or "sol_credit"))
    else:
        dd["outcome_sol_credit_applied"] = False

    if eth_cred.get("applied"):
        credit_applied = True
        dd["outcome_eth_credit_applied"] = True
        outcome_rank_delta += float(eth_cred.get("rank_delta") or 0.0)
        final_score_adjustment += float(eth_cred.get("final_score_adjustment") or 0.0)
        dd["outcome_eth_rank_credit"] = float(eth_cred.get("rank_delta") or 0.0)
        penalty_reasons.append(str(eth_cred.get("reason") or "eth_credit"))
    else:
        dd["outcome_eth_credit_applied"] = False

    dd["outcome_penalty_applied"] = penalty_applied
    dd["outcome_credit_applied"] = credit_applied
    dd["penalty_reason"] = "; ".join(penalty_reasons) if penalty_reasons else ""
    dd["outcome_penalty_or_credit"] = round(outcome_rank_delta + final_score_adjustment, 6)
    dd["outcome_rank_delta"] = round(outcome_rank_delta, 4)
    dd["outcome_adjusted_rank_score"] = round(max(0.0, min(1.0, float(raw_rank_score) + outcome_rank_delta)), 6)
    dd["outcome_final_score_adjustment"] = round(final_score_adjustment, 6)

    dd["selected_net_expected_value"] = dd["adjusted_ev"]
    dd["raw_score"] = round(float(raw_rank_score), 6)
    dd["adjusted_score"] = dd["outcome_adjusted_rank_score"]

    if size_mult != 1.0:
        dd["thesis_size_factor"] = round(float(dd.get("thesis_size_factor") or 1.0) * size_mult, 4)

    bm = buy_margin
    if bm is None:
        try:
            bm = float(dd.get("buy_margin") or dd.get("redis_buy_margin_key") or dd.get("buy_margin_raw") or 0.0)
        except (TypeError, ValueError):
            bm = None

    dd["buy_margin_at_rank"] = bm
    dd["final_selection_score"] = compute_final_selection_score(
        adjusted_ev=float(dd["adjusted_ev"]),
        outcome_adjusted_rank=float(dd["outcome_adjusted_rank_score"]),
        raw_rank_score=float(raw_rank_score),
        buy_margin=bm,
        final_score_adjustment=final_score_adjustment,
    )
    dd["selection_score"] = dd["final_selection_score"]
    dd["adjusted_rank_used_in_final_selection"] = True
    return dd


def apply_outcome_penalty_to_decision_data(decision_data: dict[str, Any], symbol: str) -> dict[str, Any]:
    """Backward-compatible entry; requires raw_rank_score on decision_data if present."""
    dd = dict(decision_data or {})
    raw_rank = float(dd.get("raw_rank_score") or dd.get("raw_score") or 0.5)
    bm = dd.get("buy_margin_at_rank") or dd.get("buy_margin")
    return apply_v3_outcome_ranking_to_decision_data(dd, symbol, raw_rank_score=raw_rank, buy_margin=bm)


def build_ranking_adjustment_report(
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    """Snapshot of active outcome-based ranking adjustments for audit artifact."""
    if db_path is None:
        from backend.database_schema import DATABASE_PATH

        db_path = DATABASE_PATH

    xrp_pen = evaluate_outcome_penalty("XRP/USDT", "FAILED_BREAKDOWN_REVERSAL", "bear", db_path=db_path)
    sol_cred = evaluate_sol_outcome_credit("SOL/USDT", "FAILED_BREAKDOWN_REVERSAL", "bear", db_path=db_path)
    btc_pen = evaluate_btc_outcome_penalty("BTC/USDT", "FAILED_BREAKDOWN_REVERSAL", "bear", db_path=db_path)
    eth_cred = evaluate_eth_outcome_credit("ETH/USDT", "FAILED_BREAKDOWN_REVERSAL", "bear", db_path=db_path)

    return {
        "v3_ranking_fix_applied": True,
        "xrp_penalty_strengthened": bool(xrp_pen.get("applied") and xrp_pen.get("penalty_generation") == "v3_final_selection"),
        "xrp_final_score_penalty_active": bool(xrp_pen.get("applied")),
        "xrp_old_rank_delta": XRP_PENALTY_V1_RANK_DELTA,
        "xrp_new_rank_delta": float(xrp_pen.get("rank_delta") or 0.0),
        "xrp_old_ev_factor": XRP_PENALTY_V1_EV_FACTOR,
        "xrp_new_ev_factor": float(xrp_pen.get("ev_factor") or 1.0),
        "xrp_old_size_factor": XRP_PENALTY_V1_SIZE_FACTOR,
        "xrp_new_size_factor": float(xrp_pen.get("size_factor") or 1.0),
        "sol_positive_credit_applied": bool(sol_cred.get("applied")),
        "sol_credit_active": bool(sol_cred.get("applied")),
        "sol_credit_amount": float(sol_cred.get("credit_amount") or 0.0),
        "btc_mild_penalty_active": bool(btc_pen.get("applied")),
        "btc_changed": bool(btc_pen.get("applied")),
        "eth_watch_credit_active": bool(eth_cred.get("applied")),
        "eth_changed": bool(eth_cred.get("applied")),
        "adjusted_rank_used_in_final_selection": True,
        "no_hard_xrp_block": True,
        "no_strategy_changes": True,
        "xrp_recovery_rule": {
            "min_trades": XRP_RECOVERY_MIN_TRADES,
            "post_v3_min_buy_id": POST_V3_MIN_BUY_ID,
            "requires_positive_expectancy": True,
            "requires_profit_factor_above": 1.2,
            "requires_avg_loss_not_above_2x_avg_win": True,
            "recovery_met": bool(xrp_pen.get("recovery_met")),
        },
        "passive_watch_next_target": 20,
        "passive_watch_baseline_trade_count": 21,
    }


def write_churn_audit_artifact(db_path: str | Path, out_path: str | Path) -> dict[str, Any]:
    audit = build_churn_audit(db_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(audit, indent=2, default=str) + "\n", encoding="utf-8")
    return audit
