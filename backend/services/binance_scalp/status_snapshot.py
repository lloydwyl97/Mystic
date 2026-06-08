"""Read-only Binance.US scalp readiness snapshot — no writes."""

from __future__ import annotations

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
from backend.services.binance_scalp.momentum_tracker import MomentumDiagnostics, MomentumTracker
from backend.services.binance_scalp.scalp_control import is_entry_armed
from backend.services.binance_scalp.momentum_gross_estimate import (
    compute_momentum_gross_estimate,
)
from backend.services.binance_scalp.orderbook_book import walk_buy_notional, walk_sell_qty
from backend.services.binance_scalp.protected_preflight import run_scalp_preflight
from backend.services.binance_scalp.scalp_strategy_router import ScalpStrategyRouter
from backend.services.binance_scalp.strategies.kline_cache import KlineCache
from scripts.watch_scalp_entry_opportunity import (  # noqa: PLC0415 — shared gate helpers
    NEAR_PASS_THRESHOLD,
    _distance_to_pass,
    is_high_quality_near_pass,
    warm_momentum,
)


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
            "redis_rest_spread_delta": (
                abs(redis_spread - rest_spread)
                if redis_spread is not None and rest_spread is not None
                else None
            ),
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
    sell_qty = buy_walk.filled_qty if buy_walk.filled_qty > 0 else (
        notional / snap.best_ask if snap.best_ask > 0 else 0.0
    )
    sell_walk = walk_sell_qty(snap.bids, sell_qty, snap.best_bid)
    required = econ.entry_required_gross_edge_pct(
        snap.spread_pct, buy_walk.impact_pct, sell_walk.impact_pct
    )
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
    return "BLOCKED"


def _overall_decision(rows: list[dict]) -> str:
    decisions = [_symbol_decision(r) for r in rows if not r.get("error")]
    if not decisions:
        return "BLOCKED"
    priority = ("PASS", "READY_TO_WATCH", "NEAR_PASS", "BLOCKED")
    for d in priority:
        if d in decisions:
            return d
    return "BLOCKED"


def _top_blocker(rows: list[dict]) -> str | None:
    reasons = [r.get("reject_reason") for r in rows if r.get("reject_reason")]
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
        "spread_cap_pct": econ.spread_cap_for_symbol(sym)
        if not config.scalp_live
        and (config.calibration_mode or config.scalp_paper_enabled)
        else econ.spread_cap_pct,
        "uniform_spread_cap_pct": econ.spread_cap_pct,
        "paper_spread_cap_pct": econ.spread_cap_for_symbol(sym)
        if econ.paper_spread_caps
        else None,
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
        best, all_sigs = router.evaluate_symbol(
            sym,
            epoch=now,
            notional_usd=config.max_notional_paper,
        )
        sym_row = {
            "best_setup": best.as_dict() if best else None,
            "router_entry_ready": best is not None and best.passed,
            "strategies": [s.as_dict() for s in all_sigs],
        }
        per_symbol[sym] = sym_row
        if best is not None and best.passed:
            ranked_entries.append(
                {
                    "symbol": sym,
                    "setup_name": best.setup_name,
                    "score": best.score,
                    "spread_pct": best.spread_pct,
                    "reject_reason": best.reject_reason,
                }
            )

    ranked_entries.sort(key=lambda r: (-r["score"], r["spread_pct"]))
    inventory = router.strategy_inventory()
    best_overall = ranked_entries[0] if ranked_entries else None

    return {
        "inventory": inventory,
        "overall_entry_ready": best_overall is not None,
        "best_candidate": best_overall,
        "ranked_candidates": ranked_entries,
        "symbols": per_symbol,
        "warm_rounds_used": warm_rounds,
        "note": "router uses enabled strategies; legacy /status symbols block uses momentum preflight",
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

    symbol_rows = [
        _evaluate_symbol_status(sym, reader, tracker, econ, config) for sym in symbols
    ]

    db_path = Path(config.database_path)
    open_positions = 0
    if db_path.exists():
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            open_positions = conn.execute(
                "SELECT COUNT(*) FROM scalp_paper_positions WHERE status='OPEN'"
            ).fetchone()[0]

    rclient = redis.from_url(config.redis_url, decode_responses=True)
    rest_spreads = {
        row["symbol"]: row.get("spread_pct")
        for row in symbol_rows
        if not row.get("error")
    }
    freshness = _redis_orderbook_freshness(rclient, symbols, rest_spreads=rest_spreads)

    strategy_router = _evaluate_strategy_router(
        config,
        econ,
        reader,
        tracker,
        warm_rounds=warm_rounds,
    )

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall_decision": _overall_decision(symbol_rows),
        "top_blocker": _top_blocker(symbol_rows),
        "warm_rounds_recommended": 12,
        "warm_rounds_note": (
            "momentum_confirmed requires ~60s history and 60s trend; "
            "warm_rounds=6 (~35s) under-warms 60s checks"
        ),
        "fee_model_verified": econ.is_fee_model_verified(),
        "calibration_mode": config.calibration_mode,
        "calibration_profile": config.calibration_profile if config.calibration_mode else "strict",
        "products": list(config.products),
        "scalp_live": config.scalp_live,
        "scalp_paper_enabled": config.scalp_paper_enabled,
        "entry_armed": is_entry_armed(rclient, prefix=config.redis_key_prefix),
        "open_scalp_positions": open_positions,
        "warm_rounds_used": warm_rounds,
        "symbols": {row["symbol"]: row for row in symbol_rows},
        "redis_orderbook_freshness": freshness,
        "memory_kb": _read_mem_kb(),
        "strategy_router": strategy_router,
        "disabled_strategies": sorted(config.disabled_strategies),
    }
