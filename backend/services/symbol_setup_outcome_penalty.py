"""
Outcome-driven ranking discipline for symbol/setup/regime churn.

Reads closed paper_trades (not opinions) and applies ranking/EV penalties only
when rolling realized outcomes are negative — never a hard global symbol block.
"""

from __future__ import annotations

import json
import logging
import os
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

# Universal low-MFE STALL demotion (rank/EV only — never a hard block).
DAY_UNIVERSAL_PENALTY_SYMBOLS = frozenset({"BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"})
DAY_UNIVERSAL_PENALTY_SETUPS = frozenset(
    {
        "RANGE_BOUNCE",
        "FAILED_BREAKDOWN_REVERSAL",
        "HTF_TREND_PULLBACK",
        "TREND_PULLBACK",  # alias / legacy label for HTF pullback thesis
        "BREAKOUT",
    }
)
# Setups that keep selecting into STALL_DEAD clusters after soft demotion.
P1C_TOXIC_STALL_SETUPS = frozenset(
    {
        "HTF_TREND_PULLBACK",
        "TREND_PULLBACK",
        "FAILED_BREAKDOWN_REVERSAL",
    }
)
# Profit path: never starve these via low-MFE fill defer (rank demotion still applies).
DAY_PREFERRED_FILL_SETUPS = frozenset(
    {
        "RANGE_BOUNCE",
        "BREAKOUT",
        "BREAKOUT_CONTINUATION",
        "VWAP_REVERSION",
    }
)
# Match STALL dead floors for learning demotion (do not change exit floors).
LOW_MFE_STALL_MAX_MFE_PCT = 0.0050
LOW_MFE_STALL_MIN_MAE_PCT = 0.0025
LOW_MFE_STALL_MIN_COUNT = 2
# Legacy v1 constants retained for audit diffs / older tests.
LOW_MFE_STALL_RANK_BASE = -0.12
LOW_MFE_STALL_EV_FACTOR = 0.72
LOW_MFE_STALL_FINAL_SCORE = -0.05
# Ocean/paper IDs are far below CLEAN_INFRA_MIN_SELL_ID (2987); low-MFE path
# must see recent post-clean sells or demotion never fires.
LOW_MFE_STALL_MIN_SELL_ID = 1

# P1C soft rank/EV calibration (non-blocking). Stronger HTF/FBR demotion.
P1B_LOOKBACK = 12
P1B_SETUP_LOOKBACK = 16
P1B_RANK_FLOOR = -0.75
P1B_EV_FLOOR = 0.22
P1B_FSS_FLOOR = -0.40
P1B_GIVEBACK_MFE_MAX = 0.0035
P1B_MFE_SEVERE = 0.0010
P1B_MFE_MODERATE = 0.0020
P1B_MAE_SEVERE = 0.0040
P1B_HOLD_DEAD_MIN = 100.0
P1B_SETUP_STALL_MIN = 3
P1B_SOFTEN_PF = 1.25
P1C_PENALTY_GENERATION = "low_mfe_stall_p1c"
# Defer fills whose post-demotion score is still negative (capacity discipline).
P1C_DEFER_FSS_MAX = 0.0
P1C_DEFER_MIN_STALL = 2


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
    mfe_pct: float | None = None
    mae_pct: float | None = None
    raw_exit_reason: str = ""
    worth_taking: int | None = None


def _parse_explain(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        return json.loads(raw) if isinstance(raw, str) else dict(raw)
    except Exception:
        return {}


def _extract_setup_regime_ev(explain: dict[str, Any]) -> tuple[str, str, float | None]:
    from backend.services.day_trade_thesis import resolve_setup_identity

    identity = resolve_setup_identity(explain)
    setup = identity["setup_type_canonical"] or str(explain.get("setup_type") or explain.get("entry_thesis") or "")
    regime = identity["day_route_regime"] or "neutral"
    # Do NOT overwrite regime with score_components adaptive_regime hybrids
    # like "trending_up::HTF_TREND_PULLBACK" — that polluted learning buckets.
    ev_raw = explain.get("selected_net_expected_value")
    try:
        selected_ev = float(ev_raw) if ev_raw is not None else None
    except Exception:
        selected_ev = None
    return setup, regime, selected_ev


def _extract_mfe_mae(explain: dict[str, Any]) -> tuple[float | None, float | None]:
    def _f(v: Any) -> float | None:
        if v is None or v == "":
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    mfe = _f(explain.get("mfe_pct"))
    if mfe is None:
        mfe = _f(explain.get("max_favorable_excursion"))
    mae = _f(explain.get("mae_pct"))
    if mae is None:
        mae = _f(explain.get("max_adverse_excursion"))
    if mae is not None and mae < 0:
        mae = abs(mae)
    return mfe, mae


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
            mfe, mae = _extract_mfe_mae(explain)
            raw_exit = str(
                explain.get("raw_exit_reason")
                or explain.get("exit_reason_raw")
                or explain.get("exit_trigger")
                or r["exit_reason"]
                or ""
            )
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
                    mfe_pct=mfe,
                    mae_pct=mae,
                    raw_exit_reason=raw_exit,
                    worth_taking=None,
                )
            )
    return rows


def _is_stall_dead_loss(t: ClosedTradeRow) -> bool:
    """True STALL_EXIT_DEAD_NO_MFE (or equivalent) with negative PnL."""
    if t.pnl >= 0:
        return False
    reason = (t.raw_exit_reason or t.exit_reason or "").upper()
    if "STALL_EXIT_DEAD" in reason or "DEAD_NO_MFE" in reason:
        return True
    # Measured STALL with low-MFE / elevated-MAE profile.
    if "STALL" not in reason:
        return False
    mfe = float(t.mfe_pct) if t.mfe_pct is not None else None
    mae = float(t.mae_pct) if t.mae_pct is not None else None
    if mfe is None:
        return False
    if mfe >= LOW_MFE_STALL_MAX_MFE_PCT:
        return False
    if mae is not None and mae < LOW_MFE_STALL_MIN_MAE_PCT:
        return False
    if t.worth_taking is not None and int(t.worth_taking) != 0:
        return False
    return True


def _is_secondary_giveback_loss(t: ClosedTradeRow) -> bool:
    """GIVEBACK with low MFE and negative PnL — secondary weakness only."""
    if t.pnl >= 0:
        return False
    reason = (t.raw_exit_reason or t.exit_reason or "").upper()
    if "GIVEBACK" not in reason:
        return False
    if "STALL_EXIT_DEAD" in reason or "DEAD_NO_MFE" in reason:
        return False
    mfe = float(t.mfe_pct) if t.mfe_pct is not None else None
    if mfe is None:
        return False
    return mfe < P1B_GIVEBACK_MFE_MAX


def _is_low_mfe_stall_loss(t: ClosedTradeRow) -> bool:
    """Backward-compatible: STALL dead OR (legacy) low-MFE GIVEBACK counted as loss."""
    return _is_stall_dead_loss(t) or _is_secondary_giveback_loss(t)


def _trade_quality_weight(t: ClosedTradeRow) -> float:
    """0..1 quality severity for a single dead/weak trade (higher = worse)."""
    mfe = float(t.mfe_pct) if t.mfe_pct is not None else None
    mae = float(t.mae_pct) if t.mae_pct is not None else None
    hold = float(t.hold_min or 0.0)
    w = 0.35
    if mfe is not None:
        if mfe < P1B_MFE_SEVERE:
            w += 0.30
        elif mfe < P1B_MFE_MODERATE:
            w += 0.18
        elif mfe < LOW_MFE_STALL_MAX_MFE_PCT:
            w += 0.08
    if mae is not None and mae >= P1B_MAE_SEVERE:
        w += 0.20
    if hold >= P1B_HOLD_DEAD_MIN and (mfe is None or mfe < P1B_MFE_MODERATE):
        w += 0.15  # dead-on-arrival near stall horizon
    return min(1.0, w)


def _p1b_count_tier_rank_ev(stall_count: int) -> tuple[float, float, float]:
    """Base rank_delta / ev_factor / fss_adj from STALL_DEAD count (P1C)."""
    if stall_count >= 5:
        return -0.52, 0.30, -0.26
    if stall_count >= 4:
        return -0.44, 0.36, -0.22
    if stall_count >= 3:
        return -0.36, 0.42, -0.18
    if stall_count >= 2:
        return -0.24, 0.52, -0.12
    return 0.0, 1.0, 0.0


def _normalize_penalty_setup(setup_u: str) -> str:
    """Collapse legacy TREND_PULLBACK into HTF bucket for outcome history."""
    s = str(setup_u or "").strip().upper()
    if s == "TREND_PULLBACK":
        return "HTF_TREND_PULLBACK"
    return s


def _setup_matches_penalty_bucket(trade_setup: str, target_setup: str) -> bool:
    a = _normalize_penalty_setup(trade_setup)
    b = _normalize_penalty_setup(target_setup)
    return bool(a) and a == b


def day_fbr_fills_enabled() -> bool:
    """DAY FBR fills off by default — Ocean evidence: FBR is the main bleed bucket."""
    return os.getenv("DAY_FBR_FILLS_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")


def day_htf_fills_enabled() -> bool:
    """DAY HTF/TREND_PULLBACK fills off by default — post-FBR-cut bleed bucket."""
    return os.getenv("DAY_HTF_FILLS_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")


def should_defer_day_fbr_fill(decision_data: dict[str, Any] | None) -> bool:
    """Skip DAY FAILED_BREAKDOWN_REVERSAL fills unless explicitly re-enabled."""
    if day_fbr_fills_enabled():
        return False
    dd = dict(decision_data or {})
    setup = _normalize_penalty_setup(
        str(dd.get("setup_type_canonical") or dd.get("setup_type") or dd.get("entry_thesis") or "")
    )
    return setup == "FAILED_BREAKDOWN_REVERSAL"


def should_defer_day_htf_fill(decision_data: dict[str, Any] | None) -> bool:
    """Skip DAY HTF_TREND_PULLBACK / TREND_PULLBACK fills unless explicitly re-enabled."""
    if day_htf_fills_enabled():
        return False
    dd = dict(decision_data or {})
    setup = _normalize_penalty_setup(
        str(dd.get("setup_type_canonical") or dd.get("setup_type") or dd.get("entry_thesis") or "")
    )
    return setup in ("HTF_TREND_PULLBACK", "TREND_PULLBACK")


def should_defer_low_mfe_stall_fill(decision_data: dict[str, Any] | None) -> bool:
    """
    Soft capacity discipline: do not consume an open slot with a deeply
    demoted toxic stall cluster whose final_selection_score is still < 0.

    RANGE/BREAKOUT/VWAP are never fill-deferred here — those are the active
    profit path while HTF/FBR fills are gated off. Rank/EV demotion still applies.
    """
    dd = dict(decision_data or {})
    setup = _normalize_penalty_setup(
        str(dd.get("setup_type_canonical") or dd.get("setup_type") or dd.get("entry_thesis") or "")
    )
    if setup in DAY_PREFERRED_FILL_SETUPS or setup.startswith("BREAKOUT"):
        return False
    reason = str(dd.get("penalty_reason") or "")
    low_mfe_on = bool(dd.get("outcome_low_mfe_stall_penalty_applied")) or (
        "repeated_low_mfe_stall_losses" in reason or "low_mfe" in reason.lower()
    )
    if not low_mfe_on and not bool(dd.get("outcome_penalty_applied")):
        return False
    if not low_mfe_on:
        return False
    try:
        fss = float(dd.get("final_selection_score") or 0.0)
    except (TypeError, ValueError):
        return False
    if fss >= P1C_DEFER_FSS_MAX:
        return False
    try:
        stall = int(dd.get("low_mfe_stall_count") or 0)
    except (TypeError, ValueError):
        stall = 0
    if setup in P1C_TOXIC_STALL_SETUPS and stall >= P1C_DEFER_MIN_STALL:
        return True
    return stall >= 3


def _bucket_pnl_pf(trades: list[ClosedTradeRow]) -> tuple[float, float, int, int]:
    """Return net_pnl, profit_factor, net_profit_count, stall_dead_count."""
    if not trades:
        return 0.0, 0.0, 0, 0
    net = sum(t.pnl for t in trades)
    wins = [t.pnl for t in trades if t.pnl > 0]
    losses = [t.pnl for t in trades if t.pnl <= 0]
    gw = sum(wins) if wins else 0.0
    gl = abs(sum(losses)) if losses else 0.0
    pf = (gw / gl) if gl > 0 else (999.0 if gw > 0 else 0.0)
    np_count = sum(
        1
        for t in trades
        if "NET_PROFIT" in (t.raw_exit_reason or t.exit_reason or "").upper()
    )
    stall_n = sum(1 for t in trades if _is_stall_dead_loss(t))
    return float(net), float(pf), int(np_count), int(stall_n)


def evaluate_low_mfe_stall_penalty(
    symbol: str,
    setup: str,
    regime: str,
    *,
    db_path: str | Path | None = None,
    min_sell_id: int = LOW_MFE_STALL_MIN_SELL_ID,
) -> dict[str, Any]:
    """
    P1B non-blocking rank/EV demotion for repeated low-MFE dead-trade history.

    Symbol+setup bucket is primary; setup-wide cluster adds a smaller demotion.
    GIVEBACK low-MFE negatives add secondary weakness only.
    Profitable/recovering buckets are softened. Never hard-blocks a candidate.
    """
    from backend.services.day_trade_thesis import resolve_setup_identity

    sym = normalize_symbol(symbol)
    identity = resolve_setup_identity({"setup_type": setup, "day_route_regime": regime})
    setup_raw = (identity["setup_type_canonical"] or str(setup or "")).strip().upper()
    setup_u = _normalize_penalty_setup(setup_raw)
    regime_l = (identity["day_route_regime"] or str(regime or "neutral")).strip().lower()
    toxic_setup = setup_u in P1C_TOXIC_STALL_SETUPS

    base: dict[str, Any] = {
        "applied": False,
        "symbol": sym,
        "setup": setup_u,
        "regime": regime_l,
        "rank_delta": 0.0,
        "ev_factor": 1.0,
        "size_factor": 1.0,
        "final_score_adjustment": 0.0,
        "reason": "low_mfe_stall_not_applicable",
        "hard_block": False,
        "candidate_eligible": True,
        "eligible": True,
        "penalty_generation": P1C_PENALTY_GENERATION,
        "low_mfe_stall_count": 0,
        "giveback_weak_count": 0,
        "bucket_net_pnl": 0.0,
        "bucket_profit_factor": 0.0,
    }

    if sym not in DAY_UNIVERSAL_PENALTY_SYMBOLS:
        base["reason"] = "symbol_not_in_day_penalty_scope"
        return base
    if setup_raw not in DAY_UNIVERSAL_PENALTY_SETUPS and setup_u not in DAY_UNIVERSAL_PENALTY_SETUPS:
        base["reason"] = "setup_not_in_day_penalty_scope"
        return base

    if db_path is None:
        from backend.database_schema import DATABASE_PATH

        db_path = DATABASE_PATH

    all_trades = _load_closed_trades(db_path, min_sell_id=min_sell_id)
    pair_bucket = [
        t for t in all_trades if t.symbol == sym and _setup_matches_penalty_bucket(t.setup, setup_u)
    ]
    recent = pair_bucket[-P1B_LOOKBACK:]
    stall_dead = [t for t in recent if _is_stall_dead_loss(t)]
    giveback_weak = [t for t in recent if _is_secondary_giveback_loss(t)]
    # Legacy combined count for min-threshold (stall primary; giveback can help reach 2).
    combined_weak = stall_dead + [t for t in giveback_weak if t not in stall_dead]

    setup_bucket = [
        t for t in all_trades if _setup_matches_penalty_bucket(t.setup, setup_u)
    ][-P1B_SETUP_LOOKBACK:]
    setup_stall = [t for t in setup_bucket if _is_stall_dead_loss(t)]
    setup_net, setup_pf, _, setup_stall_n = _bucket_pnl_pf(setup_bucket)

    pair_net, pair_pf, pair_np, pair_stall_n = _bucket_pnl_pf(recent)

    stall_count = len(stall_dead)
    gb_count = len(giveback_weak)
    trigger_count = max(stall_count, len(combined_weak) if stall_count >= 1 else len(combined_weak))

    setup_cluster = setup_stall_n >= P1B_SETUP_STALL_MIN and setup_net < 0

    giveback_only = stall_count == 0 and gb_count >= LOW_MFE_STALL_MIN_COUNT
    if (
        stall_count < LOW_MFE_STALL_MIN_COUNT
        and not giveback_only
        and not setup_cluster
    ):
        base["reason"] = "low_mfe_stall_history_insufficient"
        base["low_mfe_stall_count"] = stall_count
        base["giveback_weak_count"] = gb_count
        base["bucket_count"] = len(recent)
        base["bucket_net_pnl"] = round(pair_net, 4)
        base["bucket_profit_factor"] = round(pair_pf, 4)
        base["setup_stall_count"] = setup_stall_n
        return base

    # --- Primary symbol/setup demotion from STALL_DEAD count ---
    effective_stall = stall_count
    rank_delta = 0.0
    ev_factor = 1.0
    fss_adj = 0.0
    quality = 0.0
    toxic_boost_applied = False

    if effective_stall >= LOW_MFE_STALL_MIN_COUNT:
        rank_delta, ev_factor, fss_adj = _p1b_count_tier_rank_ev(effective_stall)
        quality_src = stall_dead
        quality = (
            sum(_trade_quality_weight(t) for t in quality_src) / len(quality_src)
            if quality_src
            else 0.0
        )
        # Scale magnitude by quality (up to +30% stronger on P1C).
        q_scale = 1.0 + 0.30 * quality
        rank_delta *= q_scale
        fss_adj *= q_scale
        ev_factor = 1.0 - (1.0 - ev_factor) * q_scale

        # PnL damage boost when bucket net negative with 2+ stall.
        if pair_stall_n >= 2 and pair_net < 0:
            damage = min(1.0, abs(pair_net) / 35.0)
            rank_delta -= 0.06 * damage
            fss_adj -= 0.03 * damage
            ev_factor *= max(0.84, 1.0 - 0.12 * damage)

        # Extra HTF/FBR rescope when the pair bucket is still bleeding.
        if toxic_setup and pair_net < 0 and pair_stall_n >= 2:
            toxic_boost_applied = True
            t_scale = min(1.0, 0.45 + 0.15 * pair_stall_n)
            rank_delta -= 0.12 * t_scale
            fss_adj -= 0.06 * t_scale
            ev_factor *= max(0.80, 1.0 - 0.12 * t_scale)

    # Secondary GIVEBACK weakness (small) — never uses stall-tier severity alone.
    if gb_count > 0:
        gb_rank = max(-0.12, -0.045 * gb_count)
        gb_ev = max(0.82, 1.0 - 0.055 * gb_count)
        rank_delta += gb_rank
        fss_adj -= 0.018 * min(gb_count, 3)
        ev_factor *= gb_ev

    # Setup-wide cluster (smaller than symbol/setup).
    setup_rank = 0.0
    setup_ev = 1.0
    setup_fss = 0.0
    if setup_cluster:
        setup_rank = -0.10 if toxic_setup else -0.08
        setup_ev = 0.88 if toxic_setup else 0.90
        setup_fss = -0.05 if toxic_setup else -0.04
        if setup_stall_n >= 5:
            setup_rank = -0.16 if toxic_setup else -0.12
            setup_ev = 0.80 if toxic_setup else 0.85
            setup_fss = -0.08 if toxic_setup else -0.06
        rank_delta += setup_rank
        fss_adj += setup_fss
        ev_factor *= setup_ev

    # Soften profitable / recovering buckets — never erase a bleeding toxic cluster.
    soften = 1.0
    soften_reasons: list[str] = []
    severe_bleed = effective_stall >= 3 and pair_net < 0
    if (
        pair_pf > P1B_SOFTEN_PF
        and pair_np >= max(1, pair_stall_n + (1 if toxic_setup else 0))
        and pair_net > 0
    ):
        soften *= 0.45
        soften_reasons.append("bucket_pf_and_net_profit_offset")
    last3 = recent[-3:]
    last3_stall = sum(1 for t in last3 if _is_stall_dead_loss(t))
    # P1C: do not soften severe bleeders on a lucky latest-3; require zero stalls in last3.
    if (
        not severe_bleed
        and len(last3) >= 3
        and sum(t.pnl for t in last3) > 0
        and last3_stall == 0
    ):
        soften *= 0.70
        soften_reasons.append("latest_3_net_positive")
    # Floor: keep most of the demotion for 3+ stall-dead; toxic bleeders keep more.
    if severe_bleed and toxic_setup:
        soften = max(soften, 0.85)
    elif effective_stall >= 3:
        soften = max(soften, 0.70)
    elif effective_stall >= 2:
        soften = max(soften, 0.50)
    elif giveback_only:
        soften = max(soften, 0.30)
    else:
        soften = max(soften, 0.40)

    if soften < 0.999:
        rank_delta *= soften
        fss_adj *= soften
        ev_factor = 1.0 - (1.0 - ev_factor) * soften

    # Caps / floors.
    rank_delta = max(P1B_RANK_FLOOR, min(0.0, rank_delta))
    ev_factor = max(P1B_EV_FLOOR, min(1.0, ev_factor))
    fss_adj = max(P1B_FSS_FLOOR, min(0.0, fss_adj))

    # If somehow no material demotion, do not claim applied.
    if rank_delta > -0.01 and ev_factor > 0.99 and fss_adj > -0.005:
        base["reason"] = "penalty_softened_to_noop"
        base["low_mfe_stall_count"] = stall_count
        base["giveback_weak_count"] = gb_count
        base["bucket_net_pnl"] = round(pair_net, 4)
        base["bucket_profit_factor"] = round(pair_pf, 4)
        base["soften"] = round(soften, 4)
        return base

    reasons = ["repeated_low_mfe_stall_losses"]
    if gb_count > 0:
        reasons.append("low_mfe_giveback_secondary")
    if setup_cluster:
        reasons.append("setup_wide_stall_cluster")
    if toxic_boost_applied:
        reasons.append("toxic_htf_fbr_boost")
    if soften_reasons:
        reasons.append("softened:" + ",".join(soften_reasons))

    return {
        "applied": True,
        "symbol": sym,
        "setup": setup_u,
        "regime": regime_l,
        "rank_delta": round(rank_delta, 4),
        "ev_factor": round(ev_factor, 4),
        "size_factor": 1.0,  # demote rank/EV only — never shrink size as a soft gate
        "final_score_adjustment": round(fss_adj, 4),
        "reason": "repeated_low_mfe_stall_losses",
        "outcome_penalty_reason": "; ".join(reasons),
        "hard_block": False,
        "candidate_eligible": True,
        "eligible": True,
        "penalty_generation": P1C_PENALTY_GENERATION,
        "low_mfe_stall_count": stall_count,
        "giveback_weak_count": gb_count,
        "combined_weak_count": len(combined_weak),
        "bucket_count": len(recent),
        "bucket_net_pnl": round(pair_net, 4),
        "bucket_profit_factor": round(pair_pf, 4),
        "bucket_net_profit_count": pair_np,
        "setup_stall_count": setup_stall_n,
        "setup_net_pnl": round(setup_net, 4),
        "setup_profit_factor": round(setup_pf, 4),
        "setup_cluster_applied": setup_cluster,
        "setup_rank_delta": round(setup_rank, 4),
        "setup_ev_factor": round(setup_ev, 4),
        "quality_severity": round(quality, 4),
        "soften": round(soften, 4),
        "soften_reasons": soften_reasons,
        "trigger_count": trigger_count,
        "effective_stall_count": effective_stall,
        "toxic_setup_boost": toxic_boost_applied,
    }


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


def _candidate_final_selection_score(cand: Any) -> float:
    dd = dict(getattr(cand, "decision_data", None) or {})
    try:
        return float(dd.get("final_selection_score") or dd.get("selection_score") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _candidate_nev(cand: Any) -> float:
    dd = dict(getattr(cand, "decision_data", None) or {})
    for key in ("selected_net_expected_value", "adjusted_ev", "raw_ev", "net_expected_value"):
        if dd.get(key) not in (None, ""):
            try:
                return float(dd[key])
            except (TypeError, ValueError):
                pass
    return 0.0


def _candidate_rank_score(cand: Any) -> float:
    dd = dict(getattr(cand, "decision_data", None) or {})
    try:
        if dd.get("rank_score") not in (None, ""):
            return float(dd["rank_score"])
    except (TypeError, ValueError):
        pass
    rank_fn = getattr(cand, "rank_score", None)
    if callable(rank_fn):
        try:
            return float(rank_fn())
        except Exception:
            return 0.0
    return 0.0


def _score_eps() -> float:
    return 1e-9


def build_truthful_selection_reason(
    selected: Any,
    ordered_candidates: list[Any],
    *,
    open_symbols: set[str] | frozenset[str] | None = None,
) -> dict[str, Any]:
    """Build honest why_selected / structured audit fields for the executable winner.

    ``ordered_candidates`` must already be sorted by the primary ranking key
    (final_selection_score descending). Open/capacity unavailability is reported
    explicitly — never as a false score victory.
    """
    open_symbols = set(open_symbols or ())
    sel_sym = str(getattr(selected, "symbol", "") or "")
    win_score = _candidate_final_selection_score(selected)
    win_nev = _candidate_nev(selected)
    win_rank = _candidate_rank_score(selected)

    higher_skipped: list[dict[str, Any]] = []
    score_peers: list[Any] = []
    for cand in ordered_candidates:
        sym = str(getattr(cand, "symbol", "") or "")
        if not sym or sym == sel_sym:
            continue
        score_peers.append(cand)
        c_score = _candidate_final_selection_score(cand)
        if c_score > win_score + _score_eps():
            reason = "same_symbol_already_open" if sym in open_symbols else "unavailable"
            higher_skipped.append(
                {
                    "symbol": sym,
                    "final_selection_score": round(c_score, 8),
                    "selected_net_expected_value": round(_candidate_nev(cand), 8),
                    "skipped_reason": reason,
                }
            )

    runner = None
    if higher_skipped:
        # Best unavailable peer is the score leader we could not execute.
        runner = higher_skipped[0]
        selection_key = "open_symbol_skipped_capacity" if runner["skipped_reason"] == "same_symbol_already_open" else "best_available_after_skip"
        why = f"{selection_key}: selected {sel_sym} final_selection_score={win_score:.6f}; higher_score {runner['symbol']}={runner['final_selection_score']:.6f} skipped ({runner['skipped_reason']})"
        skipped_reason = runner["skipped_reason"]
        runner_up_symbol = str(runner["symbol"])
        runner_up_score = float(runner["final_selection_score"])
    elif score_peers:
        peer = score_peers[0]
        peer_sym = str(getattr(peer, "symbol", "") or "")
        peer_score = _candidate_final_selection_score(peer)
        peer_nev = _candidate_nev(peer)
        peer_rank = _candidate_rank_score(peer)
        runner_up_symbol = peer_sym
        runner_up_score = peer_score
        skipped_reason = ""
        if win_score > peer_score + _score_eps():
            selection_key = "highest_final_selection_score"
            why = f"highest_final_selection_score: {sel_sym} {win_score:.6f} > {peer_sym} {peer_score:.6f}"
        elif abs(win_score - peer_score) <= _score_eps() and win_nev > peer_nev + _score_eps():
            selection_key = "higher_nev_tiebreak"
            why = f"higher_nev_tiebreak: {sel_sym} nev={win_nev:.6f} > {peer_sym} nev={peer_nev:.6f}"
        elif abs(win_score - peer_score) <= _score_eps() and abs(win_nev - peer_nev) <= _score_eps() and win_rank > peer_rank + _score_eps():
            selection_key = "higher_rank_score_tiebreak"
            why = f"higher_rank_score_tiebreak: {sel_sym} rank={win_rank:.6f} > {peer_sym} rank={peer_rank:.6f}"
        elif abs(win_score - peer_score) <= _score_eps():
            selection_key = "deterministic_tiebreak"
            why = f"deterministic_tiebreak: {sel_sym} vs {peer_sym} equal_final_selection_score={win_score:.6f}"
        else:
            # Selected has materially lower score without an explained skip — do not lie.
            selection_key = "best_available_after_skip"
            skipped_reason = "unexplained_lower_score_selection"
            why = f"best_available_after_skip: selected {sel_sym} final_selection_score={win_score:.6f}; peer {peer_sym}={peer_score:.6f} not selected"
    else:
        selection_key = "solo_candidate_no_peer"
        why = "solo_candidate_no_peer"
        runner_up_symbol = ""
        runner_up_score = 0.0
        skipped_reason = ""

    return {
        "why_selected": why,
        "selection_key_used": selection_key,
        "winner_symbol": sel_sym,
        "winner_score": round(win_score, 8),
        "runner_up_symbol": runner_up_symbol,
        "runner_up_score": round(float(runner_up_score or 0.0), 8),
        "skipped_reason": skipped_reason,
        "higher_score_skipped": higher_skipped,
        "best_rejected_peer": runner_up_symbol,
        "selected_over_symbol": runner_up_symbol,
        "selected_over_score": round(float(runner_up_score or 0.0), 8),
    }


def assign_v3_selection_ranks(
    candidates: list[Any],
    *,
    open_symbols: set[str] | frozenset[str] | None = None,
    selected: Any | None = None,
    selected_list: list[Any] | None = None,
) -> None:
    """Stamp rank / peer / truthful why on candidates after score-primary sort.

    If ``selected`` / ``selected_list`` is provided (executable fills after
    open/capacity filter), stamp why_selected on each fill — including multi-buy
    extras. Otherwise stamp on score leader (#1) only.
    """
    open_symbols = set(open_symbols or ())
    targets: list[Any] = []
    if selected_list:
        targets = [t for t in selected_list if t is not None]
    elif selected is not None:
        targets = [selected]

    target_ids = {id(t) for t in targets}
    for i, cand in enumerate(candidates):
        dd = dict(getattr(cand, "decision_data", None) or {})
        dd["final_selected_rank"] = i + 1
        # Clear stale why text on non-executable rows.
        if not targets and i != 0:
            dd.pop("why_selected", None)
        elif targets and id(cand) not in target_ids:
            dd.pop("why_selected", None)
            dd.pop("selection_key_used", None)
        setattr(cand, "decision_data", dd)

    if not targets:
        targets = [candidates[0]] if candidates else []
    if not targets:
        return

    for idx, target in enumerate(targets):
        reason = build_truthful_selection_reason(target, candidates, open_symbols=open_symbols)
        dd = dict(getattr(target, "decision_data", None) or {})
        dd.update(reason)
        if idx > 0:
            # Secondary same-bar fills: keep truthful peer compare, mark multi-buy.
            base_why = str(dd.get("why_selected") or "")
            dd["selection_key_used"] = "multi_buy_capacity_fill"
            dd["why_selected"] = f"multi_buy_capacity_fill: {base_why}" if base_why else "multi_buy_capacity_fill"
        setattr(target, "decision_data", dd)


def evaluate_outcome_penalty_for_candidate(decision_data: dict[str, Any], symbol: str) -> dict[str, Any]:
    from backend.services.day_trade_thesis import resolve_setup_identity

    dd = dict(decision_data or {})
    identity = resolve_setup_identity(dd)
    setup = identity["setup_type_canonical"] or str(dd.get("setup_type") or dd.get("entry_thesis") or "")
    regime = identity["day_route_regime"] or str(dd.get("day_route_regime") or dd.get("day_regime") or dd.get("regime") or "neutral")
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
    from backend.services.day_trade_thesis import resolve_setup_identity

    dd = dict(decision_data or {})
    identity = resolve_setup_identity(dd)
    setup = identity["setup_type_canonical"] or str(dd.get("setup_type") or dd.get("entry_thesis") or "")
    regime = identity["day_route_regime"] or str(dd.get("day_route_regime") or dd.get("day_regime") or dd.get("regime") or "neutral")
    dd["setup_type_canonical"] = identity["setup_type_canonical"]
    dd["setup_type_raw"] = identity["setup_type_raw"]
    if identity["setup_type_canonical"]:
        dd["setup_type"] = identity["setup_type_canonical"]
        dd["entry_thesis"] = identity["entry_thesis"] or identity["setup_type_canonical"]

    if db_path is None:
        from backend.database_schema import DATABASE_PATH

        db_path = DATABASE_PATH

    xrp_pen = evaluate_outcome_penalty(symbol, setup, regime, db_path=db_path)
    btc_pen = evaluate_btc_outcome_penalty(symbol, setup, regime, db_path=db_path)
    low_mfe_pen = evaluate_low_mfe_stall_penalty(symbol, setup, regime, db_path=db_path)
    sol_cred = evaluate_sol_outcome_credit(symbol, setup, regime, db_path=db_path)
    eth_cred = evaluate_eth_outcome_credit(symbol, setup, regime, db_path=db_path)

    dd["outcome_churn_penalty_eval"] = xrp_pen
    dd["outcome_btc_penalty_eval"] = btc_pen
    dd["outcome_low_mfe_stall_penalty_eval"] = low_mfe_pen
    dd["outcome_sol_credit_eval"] = sol_cred
    dd["outcome_eth_credit_eval"] = eth_cred
    dd["v3_ranking_fix_applied"] = True

    raw_ev = float(dd.get("selected_net_expected_value_raw") or dd.get("selected_net_expected_value") or dd.get("net_expected_value") or 0.0)
    dd["raw_ev"] = round(raw_ev, 8)
    if dd.get("selected_net_expected_value_raw") is None:
        dd["selected_net_expected_value_raw"] = raw_ev

    dd["raw_rank_score"] = round(float(raw_rank_score), 6)
    # Pre-penalty stamps for explainability / BUY-side audit.
    dd["rank_score_before_outcome_penalty"] = round(float(raw_rank_score), 6)
    dd["selected_net_expected_value_before_outcome_penalty"] = round(raw_ev, 8)

    outcome_rank_delta = 0.0
    final_score_adjustment = 0.0
    ev_mult = 1.0
    size_mult = 1.0
    penalty_reasons: list[str] = []
    penalty_applied = False
    credit_applied = False

    # Soft demotion only — candidate remains eligible for selection.
    dd["outcome_penalty_hard_block"] = False
    dd["hard_block"] = False
    dd["candidate_eligible"] = True
    dd["outcome_churn_penalty_applied"] = False
    dd["outcome_btc_penalty_applied"] = False
    dd["outcome_low_mfe_stall_penalty_applied"] = False

    # P1C: stack low-MFE demotion with symbol-specific penalties (additive).
    if xrp_pen.get("applied"):
        penalty_applied = True
        dd["outcome_churn_penalty_applied"] = True
        outcome_rank_delta += float(xrp_pen.get("rank_delta") or 0.0)
        ev_mult *= float(xrp_pen.get("ev_factor") or 1.0)
        size_mult *= float(xrp_pen.get("size_factor") or 1.0)
        final_score_adjustment += float(xrp_pen.get("final_score_adjustment") or 0.0)
        penalty_reasons.append(str(xrp_pen.get("reason") or "xrp_outcome_penalty"))
        dd["outcome_churn_rank_penalty"] = float(xrp_pen.get("rank_delta") or 0.0)
        dd["outcome_churn_ev_factor"] = float(xrp_pen.get("ev_factor") or 1.0)
        dd["outcome_churn_peer_ev_ceiling"] = float(xrp_pen.get("peer_ev_ceiling") or 0.012)

    if btc_pen.get("applied"):
        penalty_applied = True
        dd["outcome_btc_penalty_applied"] = True
        outcome_rank_delta += float(btc_pen.get("rank_delta") or 0.0)
        ev_mult *= float(btc_pen.get("ev_factor") or 1.0)
        size_mult *= float(btc_pen.get("size_factor") or 1.0)
        final_score_adjustment += float(btc_pen.get("final_score_adjustment") or 0.0)
        penalty_reasons.append(str(btc_pen.get("reason") or "btc_outcome_penalty"))
        dd["outcome_btc_rank_penalty"] = float(btc_pen.get("rank_delta") or 0.0)

    if low_mfe_pen.get("applied"):
        penalty_applied = True
        dd["outcome_low_mfe_stall_penalty_applied"] = True
        outcome_rank_delta += float(low_mfe_pen.get("rank_delta") or 0.0)
        ev_mult *= float(low_mfe_pen.get("ev_factor") or 1.0)
        # size_factor stays 1.0 on low-MFE path by design
        final_score_adjustment += float(low_mfe_pen.get("final_score_adjustment") or 0.0)
        pen_reason = str(
            low_mfe_pen.get("outcome_penalty_reason")
            or low_mfe_pen.get("reason")
            or "repeated_low_mfe_stall_losses"
        )
        penalty_reasons.append(pen_reason)
        dd["outcome_low_mfe_stall_rank_penalty"] = float(low_mfe_pen.get("rank_delta") or 0.0)
        dd["outcome_low_mfe_stall_ev_factor"] = float(low_mfe_pen.get("ev_factor") or 1.0)
        dd["low_mfe_stall_count"] = int(low_mfe_pen.get("low_mfe_stall_count") or 0)
        dd["giveback_weak_count"] = int(low_mfe_pen.get("giveback_weak_count") or 0)
        dd["bucket_net_pnl"] = low_mfe_pen.get("bucket_net_pnl")
        dd["bucket_profit_factor"] = low_mfe_pen.get("bucket_profit_factor")
        dd["outcome_penalty_rank_delta"] = float(low_mfe_pen.get("rank_delta") or 0.0)
        dd["outcome_penalty_ev_factor"] = float(low_mfe_pen.get("ev_factor") or 1.0)
        dd["outcome_penalty_final_score_adjustment"] = float(low_mfe_pen.get("final_score_adjustment") or 0.0)
        dd["penalty_generation"] = str(low_mfe_pen.get("penalty_generation") or P1C_PENALTY_GENERATION)

    if xrp_pen.get("applied"):
        peer_ceiling = float(xrp_pen.get("peer_ev_ceiling") or 0.012)
        cap_mult = float(xrp_pen.get("peer_ev_cap_multiplier") or 0.88)
        scaled_ev = raw_ev * ev_mult
        dd["adjusted_ev"] = round(min(scaled_ev, peer_ceiling * cap_mult), 8)
    else:
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
    dd["final_selection_score_before_outcome_penalty"] = compute_final_selection_score(
        adjusted_ev=float(raw_ev),
        outcome_adjusted_rank=float(raw_rank_score),
        raw_rank_score=float(raw_rank_score),
        buy_margin=bm,
        final_score_adjustment=0.0,
    )
    dd["final_selection_score"] = compute_final_selection_score(
        adjusted_ev=float(dd["adjusted_ev"]),
        outcome_adjusted_rank=float(dd["outcome_adjusted_rank_score"]),
        raw_rank_score=float(raw_rank_score),
        buy_margin=bm,
        final_score_adjustment=final_score_adjustment,
    )
    dd["selection_score"] = dd["final_selection_score"]
    dd["adjusted_rank_used_in_final_selection"] = True
    # Always expose eval + defer telemetry for BUY explain persistence.
    if "low_mfe_stall_count" not in dd:
        dd["low_mfe_stall_count"] = int(low_mfe_pen.get("low_mfe_stall_count") or 0)
    if "bucket_net_pnl" not in dd and low_mfe_pen.get("bucket_net_pnl") is not None:
        dd["bucket_net_pnl"] = low_mfe_pen.get("bucket_net_pnl")
    if "bucket_profit_factor" not in dd and low_mfe_pen.get("bucket_profit_factor") is not None:
        dd["bucket_profit_factor"] = low_mfe_pen.get("bucket_profit_factor")
    dd["low_mfe_stall_fill_deferred"] = bool(should_defer_low_mfe_stall_fill(dd))
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
