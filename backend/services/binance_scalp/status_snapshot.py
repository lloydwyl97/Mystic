"""Read-only Binance.US scalp readiness snapshot — no writes."""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import redis
from backend.services.binance_scalp.calibration_profiles import economics_for_config
from backend.services.binance_scalp.config import ScalpConfig, get_scalp_config
from backend.services.binance_scalp.economics import ScalpEconomics
from backend.services.binance_scalp.market_reader import ScalpMarketReader, symbol_base
from backend.services.binance_scalp.momentum_gross_estimate import (
    compute_momentum_gross_estimate,
)
from backend.services.binance_scalp.momentum_tracker import MomentumDiagnostics, MomentumTracker
from backend.services.binance_scalp.orderbook_book import walk_buy_notional, walk_sell_qty
from backend.services.binance_scalp.protected_preflight import run_scalp_preflight
from backend.services.binance_scalp.redis_keys import scan_key, runner_state_key
from backend.services.binance_scalp.scalp_control import is_entry_armed
from backend.services.binance_scalp.scalp_strategy_router import ScalpStrategyRouter
from backend.services.binance_scalp.strategies.kline_cache import KlineCache
from backend.services.binance_scalp.scalp_candidate_ranking import (
    _min_tradeable_score,
    _min_confident_rank,
    _rank_tie_margin,
)
from backend.services.binance_scalp.near_pass import (
    NEAR_PASS_THRESHOLD,
    _distance_to_pass,
    is_high_quality_near_pass,
    warm_momentum,
)


def _overlay_runner_scan(
    symbol_rows: list[dict[str, Any]],
    rclient: redis.Redis,
    *,
    prefix: str,
    strategy_router: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Merge live runner scan snapshots (momentum/regime) into status rows."""
    regimes: dict[str, str] = {}
    router_symbols = (strategy_router or {}).get("symbols") or {}
    for row in symbol_rows:
        sym = str(row.get("symbol") or "")
        if not sym or row.get("error"):
            continue
        try:
            raw = rclient.get(scan_key(prefix, sym))
            if not raw:
                continue
            scan = json.loads(raw)
        except Exception:
            continue
        regime = str(scan.get("micro_regime") or "")
        if regime:
            regimes[sym] = regime
            row["micro_regime"] = regime
            row["runner_micro_regime"] = regime
        samples = int(scan.get("momentum_sample_count") or 0)
        hist = float(scan.get("momentum_history_sec") or 0.0)
        runner_warmed = samples >= 4 and hist >= 30.0
        if samples > int(row.get("momentum_samples") or 0):
            row["momentum_samples"] = samples
            row["momentum_history_sec"] = hist
            row["momentum_confirmed"] = bool(scan.get("momentum_confirmed"))
            row["runner_momentum_overlay"] = True
        elif runner_warmed:
            row["momentum_samples"] = max(samples, int(row.get("momentum_samples") or 0))
            row["momentum_history_sec"] = max(hist, float(row.get("momentum_history_sec") or 0.0))
            row["momentum_confirmed"] = bool(scan.get("momentum_confirmed"))
            row["runner_momentum_overlay"] = True
        row["runner_scan_age_sec"] = max(0.0, time.time() - float(scan.get("updated_at_epoch") or 0))

        if runner_warmed and row.get("reject_reason") == "MOMENTUM_DATA_INSUFFICIENT":
            row["reject_reason"] = None
            row["momentum_insufficient_cleared_by_runner"] = True
            sym_router = router_symbols.get(sym) or {}
            best = sym_router.get("best_setup") or {}
            router_reject = sym_router.get("hard_block") or sym_router.get("soft_reason")
            if not router_reject and isinstance(best, dict):
                router_reject = best.get("reject_reason")
            rank_score = sym_router.get("rank_score")
            if sym_router.get("router_entry_ready"):
                row["reject_reason"] = None
                row["preflight_pass"] = True
                # Honest naming: PASS only if preflight-ready AND entry is armed.
                armed = bool(is_entry_armed(rclient, prefix=prefix))
                row["would_enter_if_armed"] = True
                row["would_enter"] = bool(armed)
                row["entry_armed"] = armed
                row["decision"] = "PASS" if armed else "READY_TO_WATCH"
                continue
            elif router_reject:
                row["reject_reason"] = router_reject
                row["rank_score"] = rank_score
                row["best_setup_name"] = sym_router.get("best_setup_name")
            elif rank_score is not None:
                row["reject_reason"] = f"RANK_BELOW_MIN:{rank_score}"
                row["rank_score"] = rank_score
            else:
                row["reject_reason"] = "STRATEGY_NO_VALID_SETUP"
            row["decision"] = _symbol_decision(row)
    return regimes


_ENGINE_DECISION_TO_STATUS = {
    "WOULD_ENTER": "PASS",
    "PASS_NOT_ARMED": "READY_TO_WATCH",
    "NO_SIGNAL": "NO_SIGNAL",
    "BLOCKED": "BLOCKED",
}


def _load_last_decision(rclient: redis.Redis, *, prefix: str) -> dict[str, Any] | None:
    """Read the canonical pre-order decision the live paper engine published on
    its last tick (see BinanceScalpPaperEngine._publish_last_decision). This is
    the actual engine truth — never re-derived from a second simulation."""
    try:
        from backend.services.binance_scalp.redis_keys import last_decision_key

        raw = rclient.get(last_decision_key(prefix))
        if not raw:
            return None
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _load_runner_state(rclient: redis.Redis, *, prefix: str) -> dict[str, Any] | None:
    try:
        raw = rclient.get(runner_state_key(prefix))
        if not raw:
            return None
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _derive_operational_summary(
    *,
    runner_state: dict[str, Any] | None,
    open_positions: int,
    config: ScalpConfig,
    runner_active: bool,
) -> dict[str, Any]:
    max_open = int(config.max_open_positions)
    rs = runner_state or {}
    age = max(0.0, time.time() - float(rs.get("updated_at_epoch") or 0.0)) if rs else None
    mode = str(rs.get("operational_mode") or "")
    if not runner_active:
        mode = "runner_dead"
    elif not mode:
        if open_positions >= max_open:
            mode = "max_open_positions_reached"
        elif open_positions > 0:
            mode = "entry_scan_active"
        else:
            mode = "entry_scan_active"
    return {
        "operational_mode": mode,
        "runner_state_age_sec": round(age, 1) if age is not None else None,
        "open_count": open_positions,
        "max_open_positions": max_open,
        "open_symbols": rs.get("open_symbols") or [],
        "products_scanned": rs.get("products_scanned") or list(config.products),
        "entry_blocked_reason": rs.get("entry_blocked_reason"),
        "momentum_warmed": rs.get("momentum_warmed"),
    }


def _read_mem_kb() -> dict[str, int]:
    mem: dict[str, int] = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith(("MemAvailable:", "SwapFree:", "MemTotal:")):
                    k, v = line.split(":")
                    mem[k.strip()] = int(v.split()[0])
    except OSError:
        pass
    return mem


def _redis_orderbook_freshness(
    client: redis.Redis,
    symbols: tuple[str, ...],
    *,
    rest_spreads: dict[str, float | None],
) -> dict[str, Any]:
    """Redis orderbook:* is informational; scalp status uses live REST depth."""
    out: dict[str, Any] = {}
    now_epoch = time.time()
    for sym in symbols:
        base = symbol_base(sym)
        key = f"orderbook:{base}"
        ts_raw = client.hget(key, "timestamp")
        ts_utc = client.hget(key, "ts_utc")
        source = client.hget(key, "source")
        spread_raw = client.hget(key, "bid_ask_spread")
        age_sec: float | None = None
        if ts_utc:
            try:
                age_sec = max(0.0, now_epoch - float(ts_utc))
            except (TypeError, ValueError):
                age_sec = None
        elif ts_raw:
            try:
                ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                age_sec = (datetime.now(timezone.utc) - ts).total_seconds()
            except (TypeError, ValueError):
                age_sec = None
        redis_spread = float(spread_raw) if spread_raw is not None else None
        rest_spread = rest_spreads.get(sym)
        out[sym] = {
            "redis_key": key,
            "timestamp": ts_raw,
            "ts_utc": ts_utc,
            "source": source,
            "age_sec": age_sec,
            "redis_spread_decimal": redis_spread,
            "redis_spread_note": "orderbook service uses (ask-bid)/bid; scalp uses (ask-bid)/mid",
            "rest_spread_decimal": rest_spread,
            "redis_rest_spread_delta": (abs(redis_spread - rest_spread) if redis_spread is not None and rest_spread is not None else None),
        }
    return out


def _shadow_projection(
    snap,
    mom: MomentumDiagnostics,
    econ: ScalpEconomics,
    config: ScalpConfig,
) -> dict[str, Any]:
    """Diagnostics only — computed even when spread gate would reject early."""
    estimate = compute_momentum_gross_estimate(snap, mom, econ)
    notional = config.max_notional_paper
    buy_walk = walk_buy_notional(snap.asks, notional, snap.best_ask)
    sell_qty = buy_walk.filled_qty if buy_walk.filled_qty > 0 else (notional / snap.best_ask if snap.best_ask > 0 else 0.0)
    sell_walk = walk_sell_qty(snap.bids, sell_qty, snap.best_bid)
    required = econ.entry_required_gross_edge_pct(snap.spread_pct, buy_walk.impact_pct, sell_walk.impact_pct)
    projected = estimate.projected_gross_move_pct
    return {
        "projected_gross_pct": projected,
        "required_gross_pct": required,
        "projected_surplus_pct": projected - required,
        "trend_slope_15s": estimate.trend_slope_15s,
        "trend_slope_30s": estimate.trend_slope_30s,
        "trend_slope_60s": estimate.trend_slope_60s,
        "recent_range_pct": estimate.recent_range_pct,
        "realized_volatility_pct": estimate.realized_volatility_pct,
        "breakout_strength_pct": estimate.breakout_strength_pct,
        "breakout_confirmed_shadow": estimate.breakout_confirmed,
        "imbalance_boost_pct": estimate.imbalance_boost_pct,
        "imbalance_raw": estimate.imbalance_raw,
        "momentum_gross_estimate_pct": estimate.momentum_gross_estimate_pct,
        "note": "shadow diagnostics; gate_reject may differ from shadow breakout/momentum",
    }


# Genuine operational/safety preflight failures only (stale/invalid data, excessive
# spread, excessive price impact, insufficient depth, no executable net edge, fee
# model unverified, paper execution disabled). Anything else (no candidate ranked
# this tick, setup/quality scoring below floor, momentum not yet confirmed) is an
# ordinary ranking outcome, not an operational block — see status_snapshot honesty
# repair: "BLOCKED" must be reserved for real safety/operational conditions.
_GENUINE_SAFETY_REJECT_REASONS = frozenset(
    {
        "SPREAD_TOO_WIDE",
        "PRICE_IMPACT_TOO_HIGH",
        "DEPTH_INSUFFICIENT",
        "NET_EDGE_BELOW_MIN",
        "NET_PROFIT_TARGET_NOT_MET",
        "ORDERBOOK_MISSING",
        "FEE_MODEL_UNVERIFIED",
        "SCALP_PAPER_DISABLED",
        "NO_MARKET_DATA",
    }
)


def _is_genuine_safety_block(reject_reason: str | None) -> bool:
    if not reject_reason:
        return False
    code = str(reject_reason).split(":", 1)[0].strip().upper()
    return code in _GENUINE_SAFETY_REJECT_REASONS


def _symbol_decision(row: dict) -> str:
    if row.get("error"):
        return "BLOCKED"
    if row.get("would_enter_if_armed"):
        return "PASS"
    if row.get("would_arm_high_quality_near_pass"):
        return "READY_TO_WATCH"
    dist = float((row.get("distance_to_pass") or {}).get("distance_to_pass_pct") or 999.0)
    if dist <= NEAR_PASS_THRESHOLD:
        return "NEAR_PASS"
    if _is_genuine_safety_block(row.get("reject_reason")):
        return "BLOCKED"
    # No candidate ranked/eligible this tick — an ordinary ranking outcome, not a block.
    return "NO_SIGNAL"


def _overall_decision(rows: list[dict]) -> str:
    decisions = [_symbol_decision(r) for r in rows if not r.get("error")]
    if not decisions:
        return "NO_SIGNAL"
    priority = ("PASS", "READY_TO_WATCH", "NEAR_PASS", "BLOCKED", "NO_SIGNAL")
    for d in priority:
        if d in decisions:
            return d
    return "NO_SIGNAL"


def _top_blocker(rows: list[dict], *, operational: dict[str, Any] | None = None) -> str | None:
    mode = str((operational or {}).get("operational_mode") or "")
    if mode in {"max_open_positions_reached", "exit_watch_active"}:
        blocked = (operational or {}).get("entry_blocked_reason")
        return str(blocked) if blocked else None
    reasons = [r.get("reject_reason") for r in rows if r.get("reject_reason") and not r.get("momentum_insufficient_cleared_by_runner") and r.get("reject_reason") != "MOMENTUM_DATA_INSUFFICIENT"]
    if not reasons:
        return None
    return max(set(reasons), key=reasons.count)


def _evaluate_symbol_status(
    sym: str,
    reader: ScalpMarketReader,
    tracker: MomentumTracker,
    econ: ScalpEconomics,
    config: ScalpConfig,
) -> dict[str, Any]:
    read_started = time.time()
    snap = reader.read(sym)
    if snap is None:
        return {
            "symbol": sym,
            "error": "NO_MARKET_DATA",
            "decision": "BLOCKED",
        }

    now = time.time()
    rest_fetched_at = datetime.fromtimestamp(read_started, tz=timezone.utc).isoformat()
    tracker.record(sym, now, snap.best_bid, snap.mid)
    mom = tracker.diagnostics(sym, now, snap.best_bid, snap.mid)
    pf = run_scalp_preflight(
        snap,
        econ,
        config,
        side="BUY",
        notional_usd=config.max_notional_paper,
        check_paper_enabled=False,
        momentum=mom,
        apply_entry_gate=True,
    )
    reach = pf.reachability or {}
    projected = float(reach.get("projected_gross_move_pct") or 0.0)
    required = float(reach.get("required_gross_move_pct") or 0.0)
    surplus = float(reach.get("projected_surplus_pct") or 0.0)
    dist = _distance_to_pass(
        projected_gross=projected,
        required_gross=required,
        projected_surplus=surplus,
        min_surplus=econ.min_projected_surplus_pct,
    )

    eval_row = {
        "symbol": sym,
        "spread_pct": snap.spread_pct,
        "buy_impact_pct": float(pf.buy_impact_pct),
        "sell_impact_pct": float(pf.sell_impact_pct),
        "projected_gross": projected,
        "required_gross": required,
        "projected_surplus": surplus,
        "momentum_confirmed": mom.momentum_confirmed,
        "breakout_confirmed": bool(reach.get("breakout_confirmed")),
        "reject_reason": pf.reject_reason or None,
        "preflight_pass": pf.passed,
        "distance_to_pass": dist,
    }
    would_arm = is_high_quality_near_pass(eval_row, econ)
    would_enter = pf.passed
    impact_pct = max(float(pf.buy_impact_pct), float(pf.sell_impact_pct))
    shadow = _shadow_projection(snap, mom, econ, config)

    return {
        "symbol": sym,
        "best_bid": snap.best_bid,
        "best_ask": snap.best_ask,
        "spread_pct": snap.spread_pct,
        "spread_cap_pct": econ.spread_cap_for_symbol(sym) if not config.scalp_live and (config.calibration_mode or config.scalp_paper_enabled) else econ.spread_cap_pct,
        "uniform_spread_cap_pct": econ.spread_cap_pct,
        "paper_spread_cap_pct": econ.spread_cap_for_symbol(sym) if econ.paper_spread_caps else None,
        "impact_pct_for_notional": impact_pct,
        "impact_notional_usd": config.max_notional_paper,
        "buy_impact_pct": float(pf.buy_impact_pct),
        "sell_impact_pct": float(pf.sell_impact_pct),
        "impact_cap_pct": econ.impact_cap_pct,
        "projected_gross_pct": projected,
        "required_gross_pct": required,
        "projected_surplus_pct": surplus,
        "min_projected_surplus_pct": econ.min_projected_surplus_pct,
        "momentum_confirmed": mom.momentum_confirmed,
        "breakout_confirmed": bool(reach.get("breakout_confirmed")),
        "reject_reason": pf.reject_reason or None,
        "distance_to_pass": dist,
        "would_arm_high_quality_near_pass": would_arm,
        "would_arm": would_arm,
        "would_enter_if_armed": would_enter,
        "would_enter": would_enter,
        "momentum_samples": mom.sample_count,
        "momentum_history_sec": mom.history_sec,
        "book_source": snap.book_source,
        "rest_depth_fetched_at": rest_fetched_at,
        "spread_formula": "(best_ask - best_bid) / mid",
        "gate_blocks_before_projection": pf.reject_reason == "SPREAD_TOO_WIDE" and not pf.reachability,
        "shadow_projection": shadow,
        "decision": _symbol_decision(
            {
                "would_enter_if_armed": would_enter,
                "would_arm_high_quality_near_pass": would_arm,
                "distance_to_pass": dist,
                "reject_reason": pf.reject_reason or None,
            }
        ),
    }


def _evaluate_strategy_router(
    config: ScalpConfig,
    econ: ScalpEconomics,
    reader: ScalpMarketReader,
    tracker: MomentumTracker,
    *,
    warm_rounds: int,
) -> dict[str, Any]:
    """Multi-strategy router view — complements legacy momentum preflight."""
    klines = KlineCache()
    router = ScalpStrategyRouter(
        config=config,
        econ=econ,
        reader=reader,
        momentum=tracker,
        klines=klines,
    )
    now = time.time()
    per_symbol: dict[str, Any] = {}
    ranked_entries: list[dict[str, Any]] = []

    for sym in config.products:
        best, all_sigs, meta = router.evaluate_symbol(
            sym,
            epoch=now,
            notional_usd=config.max_notional_paper,
        )
        ranked_for_sym = meta.get("ranked") or []
        best_ranked_row = max(ranked_for_sym, key=lambda r: float(r.get("rank_score") or 0.0), default=None)
        sym_row = {
            "best_setup": best.as_dict() if best else None,
            "router_entry_ready": bool(meta.get("entry_eligible")),
            "rank_score": meta.get("best_rank_score"),
            "best_setup_name": meta.get("best_setup"),
            "hard_block": meta.get("hard_block"),
            "soft_reason": meta.get("soft_reason"),
            "regime": meta.get("regime"),
            "strategies": [s.as_dict() for s in all_sigs],
            "ranked": ranked_for_sym,
        }
        per_symbol[sym] = sym_row
        # Enrich with full diagnostics from the best ranked row (or meta)
        best_row = best_ranked_row or {}
        ranked_entries.append(
            {
                "symbol": sym,
                "setup_name": meta.get("best_setup") or (best.setup_name if best else None),
                "score": meta.get("best_rank_score") or (best.score if best else 0.0),
                "rank_score": meta.get("best_rank_score"),
                "spread_pct": best.spread_pct if best else 0.0,
                "reject_reason": meta.get("hard_block") or meta.get("soft_reason"),
                "entry_eligible": bool(meta.get("entry_eligible")),
                "hard_block": meta.get("hard_block"),
                # Additional ranking diagnostics (existing values; null if not applicable)
                "soft_reason": meta.get("soft_reason"),
                "base_score": best_row.get("base_score"),
                "momentum_boost": best_row.get("momentum_boost"),
                "reachability_multiplier": best_row.get("reachability_multiplier"),
                "reachability_surplus_pct": meta.get("reachability_surplus"),
                "expected_move_pct": best_row.get("expected_move_pct"),
                "required_target_pct": best_row.get("required_target_pct"),
                "target_gap_pct": best_row.get("target_gap_pct"),
                "regime": meta.get("regime"),
                "regime_native": (best_row.get("regime_native") if best_row else None),
                "memory_delta": best_row.get("memory_delta"),
                "recent_win_rate": best_row.get("recent_win_rate"),
                "m15": best_row.get("m15"),
                "m30": best_row.get("m30"),
                "m60": best_row.get("m60"),
                "impact_pct": None,  # available via depth check in router ctx if needed later
                "tie_margin": _rank_tie_margin(),
                "rank_floor": _min_tradeable_score(),
                "min_confident_rank": _min_confident_rank(),
                "final_below_floor_reason": (None if meta.get("entry_eligible") else (meta.get("hard_block") or meta.get("soft_reason") or "BELOW_MIN_SCORE")),
            }
        )

    ranked_entries.sort(key=lambda r: (-int(bool(r.get("entry_eligible"))), -float(r.get("rank_score") or r.get("score") or 0), float(r.get("spread_pct") or 0)))
    inventory = router.strategy_inventory()
    eligible = [r for r in ranked_entries if r.get("entry_eligible")]
    best_overall = eligible[0] if eligible else (ranked_entries[0] if ranked_entries else None)
    global_hard_block = None if eligible else (best_overall.get("hard_block") if best_overall else "NO_CANDIDATES")

    return {
        "inventory": inventory,
        "overall_entry_ready": bool(eligible),
        "best_candidate": best_overall,
        "best_global_candidate": best_overall,
        "ranked_candidates": ranked_entries,
        "global_hard_block": global_hard_block,
        "symbols": per_symbol,
        "warm_rounds_used": warm_rounds,
        "note": "ranking engine: soft setup misses score; hard safety blocks trade",
    }


def build_scalp_status(*, warm_rounds: int = 0, warm_interval_sec: float = 5.0) -> dict[str, Any]:
    """Build read-only scalp readiness snapshot."""
    config = get_scalp_config()
    econ = economics_for_config(config)
    reader = ScalpMarketReader(config)
    tracker = MomentumTracker()
    symbols = config.products

    if warm_rounds > 0:
        warm_momentum(reader, tracker, symbols, rounds=warm_rounds, interval_sec=warm_interval_sec)

    symbol_rows = [_evaluate_symbol_status(sym, reader, tracker, econ, config) for sym in symbols]

    db_path = Path(config.database_path)
    open_positions = 0
    if db_path.exists():
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            open_positions = conn.execute("SELECT COUNT(*) FROM scalp_paper_positions WHERE status='OPEN'").fetchone()[0]

    rclient = redis.from_url(config.redis_url, decode_responses=True)
    rest_spreads = {row["symbol"]: row.get("spread_pct") for row in symbol_rows if not row.get("error")}
    freshness = _redis_orderbook_freshness(rclient, symbols, rest_spreads=rest_spreads)

    strategy_router = _evaluate_strategy_router(
        config,
        econ,
        reader,
        tracker,
        warm_rounds=warm_rounds,
    )

    micro_regimes = _overlay_runner_scan(
        symbol_rows,
        rclient,
        prefix=config.redis_key_prefix,
        strategy_router=strategy_router,
    )
    for row in symbol_rows:
        if not row.get("error"):
            row["decision"] = _symbol_decision(row)
    runner_state = _load_runner_state(rclient, prefix=config.redis_key_prefix)
    operational = _derive_operational_summary(
        runner_state=runner_state,
        open_positions=open_positions,
        config=config,
        runner_active=True,
    )

    # Canonical parity: prefer the engine's own last-tick decision over this
    # endpoint's independent preflight simulation. Only fall back to the local
    # simulation when the canonical publish is missing/stale (engine not
    # running this tick cycle) — and say so explicitly via `decision_source`.
    last_decision = _load_last_decision(rclient, prefix=config.redis_key_prefix)
    decision_source = "engine_canonical"
    if last_decision and last_decision.get("decision") in _ENGINE_DECISION_TO_STATUS:
        overall = _ENGINE_DECISION_TO_STATUS[str(last_decision["decision"])]
    else:
        decision_source = "status_simulation_fallback"
        overall = _overall_decision(symbol_rows)
    if operational["operational_mode"] == "max_open_positions_reached":
        overall = "WAITING_FOR_EXIT"
    elif operational["operational_mode"] == "exit_watch_active":
        overall = "EXIT_WATCH"

    top_blocker = _top_blocker(symbol_rows, operational=operational)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall_decision": overall,
        "decision_source": decision_source,
        "canonical_engine_decision": last_decision,
        "top_blocker": top_blocker,
        "operational_summary": operational,
        "runner_state": runner_state,
        "warm_rounds_recommended": 12,
        "warm_rounds_note": ("momentum_confirmed requires ~60s history and 60s trend; warm_rounds=6 (~35s) under-warms 60s checks"),
        "fee_model_verified": econ.is_fee_model_verified(),
        "calibration_mode": config.calibration_mode,
        "calibration_profile": config.calibration_profile if config.calibration_mode else "strict",
        "products": list(config.products),
        "scalp_live": config.scalp_live,
        "scalp_paper_enabled": config.scalp_paper_enabled,
        "entry_armed": is_entry_armed(rclient, prefix=config.redis_key_prefix),
        "open_scalp_positions": open_positions,
        "warm_rounds_used": warm_rounds,
        "micro_regimes": micro_regimes,
        "symbols": {row["symbol"]: row for row in symbol_rows},
        "redis_orderbook_freshness": freshness,
        "memory_kb": _read_mem_kb(),
        "strategy_router": strategy_router,
        "disabled_strategies": sorted(config.disabled_strategies),
    }
