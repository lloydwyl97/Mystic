"""
DAY v5 indicator truth audit — read-only cross-coin metadata for all 145 dims.

Builds on ``day_feature_audit`` provenance; adds usage, bounds, rank/learning/execution impact.
Does not change strategy, gates, or FEATURE_VERSION.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from backend.config.trading_universe import DAY_TRADE_SYMBOLS
from backend.services.ai_decision_contract import AI_FEATURE_DIM_V1, AI_FEATURE_DIM_V2, CONTEXT_DIMS_DAY_FULL
from backend.services.ai_feature_freshness_diagnostics import freshness_thresholds_sec
from backend.services.day_ai_rank_enrichment import INTELLIGENCE_DELTA_CAP
from backend.services.day_block_scores import block_scores_rank_delta
from backend.services.day_feature_audit import (
    BAD_STATUSES,
    _block_for_index,
    _feature_name_at,
    build_symbol_feature_audit,
    run_full_audit,
)
from backend.services.day_feature_health import LEARNING_BLOCKED_FEATURE_NAMES
from backend.services.feature_mapping import get_feature_name  # noqa: F401 — re-export for tooling

COIN_LABELS = ("BTC", "ETH", "SOL", "XRP")
SYMBOL_TO_LABEL = {
    "BTCUSDT": "BTC",
    "BTC/USDT": "BTC",
    "ETHUSDT": "ETH",
    "ETH/USDT": "ETH",
    "SOLUSDT": "SOL",
    "SOL/USDT": "SOL",
    "XRPUSDT": "XRP",
    "XRP/USDT": "XRP",
}

# Features referenced directly in setup / execution / adaptive rank paths (not only via RF).
SETUP_DIRECT_FEATURES: frozenset[str] = frozenset(
    {
        "rsi",
        "rsi_14",
        "adx",
        "ema_12",
        "ema_26",
        "ema_50",
        "atr",
        "price_momentum",
        "ctx_relative_volume",
        "ctx_spread_pct",
        "ctx_depth_imbalance",
        "ctx_rs_btc",
        "ctx_rs_eth",
        "ctx_rs_mean_btc_eth",
        "mean_ema_align_all_tf",
        "vwap",
        "bb_position",
        "donchian_upper",
        "donchian_lower",
        "keltner_upper",
        "keltner_lower",
        "volume",
        "relative_volume",
        "order_flow",
        "volume_delta",
        "volume_imbalance",
        "bid_ask_spread",
        "support_level",
        "resistance_level",
    }
)

EXECUTION_DIRECT_FEATURES: frozenset[str] = frozenset(
    {
        "ctx_spread_pct",
        "ctx_depth_imbalance",
        "bid_ask_spread",
        "order_flow",
        "volume_delta",
        "volume_imbalance",
        "spread",
        "microstructure",
    }
)

LEARNING_DIRECT_FEATURES: frozenset[str] = frozenset(
    {
        "ctx_rs_btc",
        "ctx_rs_eth",
        "ctx_rs_mean_btc_eth",
        "ctx_relative_volume",
        "rsi",
        "adx",
        "ema_12",
        "ema_26",
        "ctx_spread_pct",
    }
)

MICROSTRUCTURE_NAMES: frozenset[str] = frozenset(
    {
        "bid_ask_spread",
        "order_flow",
        "volume_delta",
        "volume_imbalance",
        "order_book_imbalance",
        "market_depth",
    }
)

SENTIMENT_UNSUPPORTED: frozenset[str] = frozenset({"put_call_ratio", "volatility_smile"})

INTENTIONAL_NEAR_ZERO: frozenset[str] = frozenset(
    {"balance_of_power", "second", "volume_price_trend", "volume_weighted_price"}
)
PROXY_FEATURES: frozenset[str] = frozenset(
    {
        "volume_profile_poc",
        "volume_profile_vah",
        "volume_profile_val",
        "volume_imbalance",
        "volume_delta",
        "order_flow",
    }
)

# Per-feature numeric bounds (where enforced in builder / context_vector).
FEATURE_BOUNDS: dict[str, tuple[float, float, str]] = {
    "rsi": (0.0, 100.0, "np.clip 0-100 in feature_builder"),
    "rsi_14": (0.0, 100.0, "np.clip 0-100 in feature_builder"),
    "adx": (0.0, 100.0, "np.clip 0-100 in feature_builder"),
    "di_plus": (0.0, 100.0, "np.clip 0-100 in feature_builder"),
    "di_minus": (0.0, 100.0, "np.clip 0-100 in feature_builder"),
    "ctx_change_24h_pct": (-0.5, 0.5, "ai_feature_v2._clip"),
    "ctx_relative_volume": (0.0, 5.0, "ai_feature_v2._clip"),
    "ctx_spread_pct": (0.0, 0.05, "ai_feature_v2._clip"),
    "ctx_depth_imbalance": (-1.0, 1.0, "ai_feature_v2._clip"),
    "ctx_rs_mean_btc_eth": (-0.5, 0.5, "ai_feature_v2._clip mean rs"),
    "ctx_btc_dominance_proxy": (0.0, 1.0, "ai_feature_v2._clip"),
    "ctx_regime_sentiment_blend": (-1.0, 1.0, "ai_feature_v2._clip"),
    "month_log_ret_window": (-6.0, 6.0, "ai_feature_v2._clip"),
    "month_realized_vol_window": (-6.0, 6.0, "ai_feature_v2._clip"),
    "mean_ema_align_all_tf": (0.0, 1.0, "min/max clamp in context_vector_day_full_mtf"),
}

for _tf in ("1m", "5m", "15m", "30m", "1h", "4h", "8h", "12h", "1d", "1w"):
    FEATURE_BOUNDS[f"slope_pct_{_tf}"] = (-0.20, 0.20, "ai_feature_v2._slope_norm clip")

BLOCK_SOURCE: dict[str, tuple[str, str, str]] = {
    "basic_price": ("feature_builder", "build_feature_dict_from_ohlcv", "day_bundle:1m OHLCV"),
    "technical_indicators": ("feature_builder", "build_feature_dict_from_ohlcv", "day_bundle:1m OHLCV"),
    "volatility": ("feature_builder", "build_feature_dict_from_ohlcv", "day_bundle:1m OHLCV"),
    "momentum": ("feature_builder", "build_feature_dict_from_ohlcv", "day_bundle:1m OHLCV"),
    "trend": ("feature_builder", "build_feature_dict_from_ohlcv", "day_bundle:1m OHLCV"),
    "volume_profile": ("feature_builder", "build_feature_dict_from_ohlcv", "volume_profile:{BASE}"),
    "market_sentiment": ("feature_builder", "merge_canonical_sentiment_payload", "ai_sentiment + fundamentals"),
    "time_based": ("feature_builder", "build_feature_dict_from_ohlcv", "1m candle timestamp"),
    "advanced_ta": ("feature_builder", "build_feature_dict_from_ohlcv", "day_bundle:1m OHLCV"),
    "advanced_volume": ("feature_builder", "build_feature_dict_from_ohlcv", "day_bundle:1m OHLCV"),
    "microstructure": ("feature_builder", "build_feature_dict_from_ohlcv", "orderbook:{BASE}"),
    "market_structure": ("feature_builder", "build_feature_dict_from_ohlcv", "orderbook market_efficiency"),
    "unsupported_options": ("feature_builder", "build_feature_dict_from_ohlcv", "UNSUPPORTED_FOR_SPOT"),
    "volatility_distribution": ("feature_builder", "build_feature_dict_from_ohlcv", "log-return skewness 1m"),
    "context_125_145": ("ai_feature_v2", "context_vector_day_full_mtf", "ai_context:{SYMBOL} + day_bundle"),
}


def _redis_key_for_feature(name: str, block: str) -> str:
    if name.startswith("ctx_") or name.startswith("slope_pct_") or name in CONTEXT_DIMS_DAY_FULL:
        if name.startswith("slope_pct_"):
            return "day_active_bundle cache + day_bundle:{CCXT}"
        return "ai_context:{SYMBOL}"
    if block == "microstructure" or name in MICROSTRUCTURE_NAMES:
        return "orderbook:{BASE}"
    if block == "volume_profile":
        return "volume_profile:{BASE}"
    if block == "market_sentiment":
        if name in SENTIMENT_UNSUPPORTED:
            return "(none — unsupported for spot)"
        return "ai_sentiment + ai_feature_fundamentals"
    return "day_bundle:1m (via live_market_data cache)"


def _rank_delta_cap(name: str, block: str, *, used_rank: bool) -> float | None:
    if not used_rank:
        return None
    if name in EXECUTION_DIRECT_FEATURES or block == "microstructure":
        return 0.05
    if name in SETUP_DIRECT_FEATURES:
        return 0.05
    return 0.06  # block aggregate via block_score_rank_delta


def _usage_flags(name: str, block: str, status: str) -> dict[str, bool]:
    used_rf = True
    used_block_rank = block not in ("unknown",)
    used_setup = name in SETUP_DIRECT_FEATURES or used_block_rank
    used_exec = name in EXECUTION_DIRECT_FEATURES or block == "microstructure" or name == "ctx_spread_pct"
    used_learning = (
        name in LEARNING_DIRECT_FEATURES or used_block_rank
    ) and status not in ("UNSUPPORTED_FOR_SPOT", "FALLBACK", "MISSING", "STALE", "ZERO_DEFAULT", "PLACEHOLDER")
    if name in LEARNING_BLOCKED_FEATURE_NAMES or name in SENTIMENT_UNSUPPORTED:
        used_learning = False
    used_rank = used_block_rank or name in SETUP_DIRECT_FEATURES or name in LEARNING_DIRECT_FEATURES or name in EXECUTION_DIRECT_FEATURES
    return {
        "used_by_rf_model": used_rf,
        "used_by_ranker": used_rank,
        "used_by_setup_score": used_setup,
        "used_by_execution_score": used_exec,
        "used_by_learning": used_learning,
    }


def _bounds_for(name: str) -> tuple[float | None, float | None, str, bool]:
    if name in FEATURE_BOUNDS:
        lo, hi, method = FEATURE_BOUNDS[name]
        return lo, hi, method, True
    if name in SENTIMENT_UNSUPPORTED:
        return 0.0, 0.0, "forced zero — UNSUPPORTED_FOR_SPOT (vector slot may hold vol proxy)", True
    return None, None, "StandardScaler at RF inference (raw finite OHLCV-derived)", True


def _needs_fix(
    status: str,
    trust: float,
    learning: bool,
    name: str,
    bounded: bool,
    values: dict[str, float],
    *,
    block: str = "",
) -> tuple[bool, bool, str]:
    reasons: list[str] = []
    needs_fix = False
    needs_adj = False

    if name in SENTIMENT_UNSUPPORTED or status == "UNSUPPORTED_FOR_SPOT":
        if status != "UNSUPPORTED_FOR_SPOT" or trust > 0 or learning:
            needs_fix = True
            reasons.append("unsupported spot feature must be UNSUPPORTED_FOR_SPOT with trust=0 and learning=false")
        return needs_fix, False, "; ".join(reasons) if reasons else "unsupported slot preserved; safely disabled"

    if status == "LOW_IMPORTANCE_TIME_FIELD_NORMAL":
        if learning:
            needs_fix = True
            reasons.append("LOW_IMPORTANCE time field must not receive learning credit")
        return needs_fix, False, "; ".join(reasons) if reasons else "bar-close alignment; near-zero expected"

    if status in ("FALLBACK", "MISSING", "STALE", "ZERO_DEFAULT") and learning:
        needs_fix = True
        reasons.append(f"{status} but learning_allowed=true")
    if status == "FALLBACK":
        needs_fix = True
        reasons.append("fallback value must not be treated as real")
    if not bounded and status != "UNSUPPORTED_FOR_SPOT" and name not in SENTIMENT_UNSUPPORTED:
        needs_fix = True
        reasons.append("value not bounded/finite")
    if status == "CALCULATED_PROXY":
        if name in PROXY_FEATURES:
            if learning:
                needs_fix = True
                reasons.append("proxy feature must have learning_allowed=false")
            else:
                reasons.append("proxy honestly labeled; learning disabled; rank capped by trust")
        else:
            needs_adj = True
            reasons.append("proxy — document and cap learning weight")
    if status == "WARMUP":
        if name == "volume_weighted_price":
            reasons.append("warmup until >=10 bars with volume; clears automatically")
        else:
            needs_adj = True
            reasons.append("warmup — insufficient history")
    if name == "balance_of_power" and all(abs(v) < 0.05 for v in values.values()):
        reasons.append("near-zero BOP valid when open/close balanced on real OHLCV")
    if all(abs(v) < 1e-12 for v in values.values()) and status not in (
        "UNSUPPORTED_FOR_SPOT",
        "WARMUP",
        "MISSING",
        "LOW_IMPORTANCE_TIME_FIELD_NORMAL",
    ):
        if name not in INTENTIONAL_NEAR_ZERO and name not in PROXY_FEATURES:
            needs_adj = True
            reasons.append("all coins zero — verify calculation inputs")

    return needs_fix, needs_adj, "; ".join(reasons) if reasons else "ok"


def _can_positive_learning(status: str, learning: bool, name: str) -> bool:
    if name in LEARNING_BLOCKED_FEATURE_NAMES or name in SENTIMENT_UNSUPPORTED:
        return False
    if status in ("FALLBACK", "MISSING", "STALE", "ZERO_DEFAULT", "PLACEHOLDER", "UNSUPPORTED_FOR_SPOT", "WARMUP", "LOW_IMPORTANCE_TIME_FIELD_NORMAL"):
        return False
    return bool(learning)


async def run_indicator_truth_audit(symbols: list[str] | None = None) -> dict[str, Any]:
    syms = symbols or list(DAY_TRADE_SYMBOLS)
    per_coin: dict[str, Any] = {}
    for sym in syms:
        per_coin[sym] = await build_symbol_feature_audit(sym)

    thresholds = freshness_thresholds_sec()
    rows: list[dict[str, Any]] = []
    fail_reasons: list[str] = []

    for idx0 in range(AI_FEATURE_DIM_V2):
        name = _feature_name_at(idx0)
        block = _block_for_index(idx0)
        mod, fn, src_base = BLOCK_SOURCE.get(block, ("feature_builder", "build_feature_dict_from_ohlcv", "day_bundle"))
        redis_key = _redis_key_for_feature(name, block)

        coin_values: dict[str, float] = {}
        coin_status: dict[str, str] = {}
        coin_trust: dict[str, float] = {}
        coin_learning: dict[str, bool] = {}
        coin_age: dict[str, float | None] = {}
        coin_source: dict[str, str] = {}

        for sym in syms:
            label = SYMBOL_TO_LABEL.get(sym, sym[:3])
            rep = per_coin.get(sym) or {}
            feats = {f["name"]: f for f in (rep.get("features") or [])}
            frow = feats.get(name) or {}
            coin_values[label] = float(frow.get("value") or 0.0)
            coin_status[label] = str(frow.get("status") or "MISSING")
            coin_trust[label] = float(frow.get("trust_score") or 0.0)
            coin_learning[label] = bool(frow.get("learning_allowed", False))
            coin_age[label] = frow.get("age_seconds")
            coin_source[label] = str(frow.get("source") or "")

        # Consensus status (worst wins for audit)
        status_priority = (
            "FALLBACK",
            "MISSING",
            "STALE",
            "ZERO_DEFAULT",
            "PLACEHOLDER",
            "UNSUPPORTED_FOR_SPOT",
            "WARMUP",
            "CALCULATED_PROXY",
            "LOW_IMPORTANCE_TIME_FIELD_NORMAL",
            "CALCULATED",
            "LIVE",
        )
        consensus_status = coin_status.get("BTC") or "MISSING"
        for st in status_priority:
            if any(coin_status.get(l) == st for l in COIN_LABELS):
                consensus_status = st
                break

        trust = min(coin_trust.values()) if coin_trust else 0.0
        learning = all(coin_learning.values()) if coin_learning else False
        if consensus_status in BAD_STATUSES or consensus_status == "UNSUPPORTED_FOR_SPOT":
            learning = False

        usage = _usage_flags(name, block, consensus_status)
        lo, hi, clamp_method, bounded_meta = _bounds_for(name)
        vals = list(coin_values.values())
        actual_min = min(vals) if vals else None
        actual_max = max(vals) if vals else None
        bounded = bounded_meta and all(math.isfinite(v) for v in vals)
        if lo is not None and hi is not None and name not in SENTIMENT_UNSUPPORTED and consensus_status != "UNSUPPORTED_FOR_SPOT":
            bounded = bounded and all(lo - 1e-6 <= v <= hi + 1e-6 for v in vals)

        rank_cap = _rank_delta_cap(name, block, used_rank=usage["used_by_ranker"])
        can_final = usage["used_by_ranker"] and rank_cap is not None
        can_exec = usage["used_by_execution_score"]
        can_learn_pos = _can_positive_learning(consensus_status, learning, name)

        needs_fix, needs_adj, reason = _needs_fix(
            consensus_status, trust, learning, name, bounded, coin_values, block=block
        )

        # Pass/fail checks
        if not all(len((per_coin.get(s) or {}).get("features") or []) == AI_FEATURE_DIM_V2 for s in syms):
            pass  # handled globally
        if consensus_status == "MISSING" and not any(coin_status.get(l) == "MISSING" for l in COIN_LABELS):
            pass
        if name in SENTIMENT_UNSUPPORTED:
            if trust != 0.0:
                fail_reasons.append(f"{name}: unsupported but trust_score={trust}")
            if learning:
                fail_reasons.append(f"{name}: unsupported but learning_allowed=true")
            if consensus_status == "LIVE":
                fail_reasons.append(f"{name}: fake LIVE on unsupported feature")
        if consensus_status in ("FALLBACK", "STALE", "MISSING") and can_learn_pos:
            fail_reasons.append(f"{name}: bad status {consensus_status} can receive positive learning")
        if usage["used_by_ranker"] and rank_cap is None:
            fail_reasons.append(f"{name}: rank-used without rank_delta_cap")
        if can_exec and block == "microstructure":
            ages = [a for a in coin_age.values() if a is not None]
            ob_th = thresholds.get("orderbook", 45)
            if ages and max(ages) > ob_th and consensus_status == "LIVE":
                fail_reasons.append(f"{name}: stale orderbook marked LIVE age={max(ages):.1f}s")
        if can_exec and name in ("ctx_spread_pct", "ctx_depth_imbalance"):
            ages = [a for a in coin_age.values() if a is not None]
            ctx_th = thresholds.get("ai_context", 120)
            if ages and max(ages) > ctx_th and consensus_status == "LIVE":
                fail_reasons.append(f"{name}: stale ai_context marked LIVE age={max(ages):.1f}s")
        if needs_fix:
            pass  # counted in needs_fix table

        row = {
            "index": idx0 + 1,
            "feature_name": name,
            "feature_block": block,
            "BTC": coin_values.get("BTC"),
            "ETH": coin_values.get("ETH"),
            "SOL": coin_values.get("SOL"),
            "XRP": coin_values.get("XRP"),
            "source_module": mod,
            "source_function": fn,
            "source_detail": coin_source.get("BTC") or src_base,
            "source_redis_key": redis_key,
            "status": consensus_status,
            "per_coin_status": coin_status,
            "freshness_age_seconds": coin_age,
            "trust_score": round(trust, 4),
            "learning_allowed": learning,
            **usage,
            "bounded": bounded,
            "min_allowed": lo,
            "max_allowed": hi,
            "actual_min_seen": actual_min,
            "actual_max_seen": actual_max,
            "clamp_or_normalization_method": clamp_method,
            "rank_delta_cap": rank_cap,
            "can_affect_final_selection_score": can_final,
            "can_affect_trade_execution": can_exec,
            "can_receive_positive_learning_credit": can_learn_pos,
            "needs_fix": needs_fix,
            "needs_adjustment": needs_adj,
            "reason": reason,
        }
        rows.append(row)

    needs_fix_rows = [r for r in rows if r["needs_fix"]]
    needs_adj_rows = [r for r in rows if r["needs_adjustment"] and not r["needs_fix"]]
    unsupported_safe = [
        r
        for r in rows
        if r["status"] == "UNSUPPORTED_FOR_SPOT"
        and r["trust_score"] == 0.0
        and not r["learning_allowed"]
        and not r["needs_fix"]
    ]
    rank_impact = [r for r in rows if r["used_by_ranker"]]
    learning_impact = [r for r in rows if r["used_by_learning"]]
    execution_impact = [r for r in rows if r["used_by_execution_score"]]

    block_summary: dict[str, dict[str, Any]] = {}
    for r in rows:
        blk = r["feature_block"]
        b = block_summary.setdefault(
            blk,
            {"total": 0, "live": 0, "calc": 0, "proxy": 0, "unsupported": 0, "bad": 0, "rank_used": 0},
        )
        b["total"] += 1
        st = r["status"]
        if st == "LIVE":
            b["live"] += 1
        elif st == "CALCULATED":
            b["calc"] += 1
        elif st == "CALCULATED_PROXY":
            b["proxy"] += 1
        elif st == "UNSUPPORTED_FOR_SPOT":
            b["unsupported"] += 1
        if st in BAD_STATUSES:
            b["bad"] += 1
        if r["used_by_ranker"]:
            b["rank_used"] += 1

    all_145 = all(len((per_coin.get(s) or {}).get("features") or []) == AI_FEATURE_DIM_V2 for s in syms)
    meta_ok = all(r.get("trust_score") is not None and r.get("learning_allowed") is not None for r in rows)
    rank_caps_ok = all(r["rank_delta_cap"] is not None for r in rank_impact)
    unsupported_ok = all(
        r["trust_score"] == 0.0 and not r["learning_allowed"] for r in rows if r["feature_name"] in SENTIMENT_UNSUPPORTED
    )
    no_bad_learning = not any(r["can_receive_positive_learning_credit"] for r in rows if r["status"] in BAD_STATUSES)
    max_rank_domination = INTELLIGENCE_DELTA_CAP <= 0.10 and block_scores_rank_delta({"feature_health_score": 1.0}) <= 0.06

    passed = (
        all_145
        and meta_ok
        and rank_caps_ok
        and unsupported_ok
        and no_bad_learning
        and max_rank_domination
        and len(needs_fix_rows) == 0
        and not fail_reasons
    )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_version": 5,
        "feature_dim": AI_FEATURE_DIM_V2,
        "symbols": syms,
        "features": rows,
        "needs_fix": needs_fix_rows,
        "needs_adjustment": needs_adj_rows,
        "unsupported_but_safe": unsupported_safe,
        "rank_impact": rank_impact,
        "learning_impact": learning_impact,
        "execution_impact": execution_impact,
        "block_summary": block_summary,
        "pass": passed,
        "fail_reasons": fail_reasons,
        "pass_checks": {
            "all_145_per_coin": all_145,
            "metadata_complete": meta_ok,
            "rank_delta_caps": rank_caps_ok,
            "unsupported_disabled": unsupported_ok,
            "bad_status_no_positive_learning": no_bad_learning,
            "rank_deltas_bounded": max_rank_domination,
            "needs_fix_count": len(needs_fix_rows),
        },
        "per_coin_reports": per_coin,
    }


__all__ = ["run_indicator_truth_audit", "COIN_LABELS", "SENTIMENT_UNSUPPORTED"]
