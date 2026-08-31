"""SCALP indicator truth audit — metadata for all scalp intelligence features."""

from __future__ import annotations

import math
from typing import Any

from backend.services.scalp_feature_audit import BAD_STATUSES, build_symbol_scalp_audit
from backend.services.scalp_feature_contract import SCALP_FEATURE_DIM, SCALP_FEATURE_NAMES, SCALP_FEATURE_VERSION, _block_for_index

COIN_LABELS = ("BTC", "ETH", "SOL", "XRP")
SYMBOL_TO_LABEL = {
    "BTCUSDT": "BTC",
    "ETHUSDT": "ETH",
    "SOLUSDT": "SOL",
    "XRPUSDT": "XRP",
}

DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")

MODEL_FEATURES = frozenset(SCALP_FEATURE_NAMES)
EXEC_FEATURES = frozenset({"spread_pct", "order_book_imbalance", "orderbook_age_sec", "impact_pct", "depth_sufficient_flag"})
SETUP_FEATURES = frozenset({"mid_change_30s", "kline_volume_ratio", "kline_rsi_proxy", "kline_vwap_distance", "signal_score"})


def _usage(name: str, block: str, status: str) -> dict[str, bool]:
    used_rank = block != "unknown"
    used_learning = status in ("LIVE", "CALCULATED") and name not in EXEC_FEATURES
    return {
        "used_by_scalp_model": name in MODEL_FEATURES,
        "used_by_scalp_ranker": used_rank,
        "used_by_scalp_setup_score": name in SETUP_FEATURES or used_rank,
        "used_by_scalp_execution_score": name in EXEC_FEATURES or block == "microstructure",
        "used_by_scalp_learning": used_learning,
    }


def _needs_fix_adj(status: str, learning: bool, name: str, values: dict[str, float]) -> tuple[bool, bool, str]:
    if status in BAD_STATUSES and learning:
        return True, False, f"{status} with learning_allowed=true"
    if status == "FALLBACK":
        return True, False, "fallback must not receive credit"
    if status == "CALCULATED_PROXY" and learning:
        return True, False, "proxy must have learning_allowed=false"
    if status == "CALCULATED_PROXY":
        return False, False, "proxy honestly labeled; learning disabled"
    if status == "WARMUP":
        return False, True, "warmup until sufficient samples/history"
    if all(abs(v) < 1e-12 for v in values.values()) and status not in ("WARMUP", "MISSING", "UNSUPPORTED_FOR_SPOT"):
        return False, True, "all coins zero — verify inputs"
    return False, False, "ok"


def run_scalp_indicator_truth_audit(symbols: list[str] | None = None) -> dict[str, Any]:
    syms = list(symbols or DEFAULT_SYMBOLS)
    per: dict[str, Any] = {s: build_symbol_scalp_audit(s) for s in syms}
    rows: list[dict[str, Any]] = []
    fail: list[str] = []

    for idx0 in range(SCALP_FEATURE_DIM):
        name = SCALP_FEATURE_NAMES[idx0]
        block = _block_for_index(idx0)
        coin_values: dict[str, float] = {}
        coin_status: dict[str, str] = {}
        coin_trust: dict[str, float] = {}
        coin_learning: dict[str, bool] = {}
        coin_age: dict[str, float | None] = {}

        for sym in syms:
            label = SYMBOL_TO_LABEL.get(sym, sym[:3])
            feats = {f["name"]: f for f in (per[sym].get("features") or [])}
            frow = feats.get(name) or {}
            coin_values[label] = float(frow.get("value") or 0.0)
            coin_status[label] = str(frow.get("status") or "MISSING")
            coin_trust[label] = float(frow.get("trust_score") or 0.0)
            coin_learning[label] = bool(frow.get("learning_allowed", False))
            coin_age[label] = frow.get("age_seconds")

        priority = ("FALLBACK", "MISSING", "STALE", "ZERO_DEFAULT", "PLACEHOLDER", "UNSUPPORTED_FOR_SPOT", "WARMUP", "CALCULATED_PROXY", "CALCULATED", "LIVE")
        consensus = coin_status.get("BTC") or "MISSING"
        for st in priority:
            if any(coin_status.get(label) == st for label in COIN_LABELS):
                consensus = st
                break

        trust = min(coin_trust.values()) if coin_trust else 0.0
        learning = all(coin_learning.values()) if coin_learning else False
        if consensus in BAD_STATUSES:
            learning = False
        usage = _usage(name, block, consensus)
        vals = list(coin_values.values())
        bounded = all(math.isfinite(v) for v in vals)
        needs_fix, needs_adj, reason = _needs_fix_adj(consensus, learning, name, coin_values)
        can_learn = learning and consensus in ("LIVE", "CALCULATED")
        rank_cap = 0.05 if usage["used_by_scalp_ranker"] else None

        if consensus in ("FALLBACK", "STALE", "MISSING") and can_learn:
            fail.append(f"{name}: bad status {consensus} can receive positive learning")
        if needs_fix:
            fail.append(f"{name}: needs_fix — {reason}")

        rows.append(
            {
                "index": idx0 + 1,
                "feature_name": name,
                "feature_block": block,
                **{label: coin_values.get(label) for label in COIN_LABELS},
                "status": consensus,
                "per_coin_status": coin_status,
                "freshness_age_seconds": coin_age,
                "trust_score": round(trust, 4),
                "learning_allowed": learning,
                **usage,
                "bounded": bounded,
                "rank_delta_cap": rank_cap,
                "can_affect_scalp_selection": usage["used_by_scalp_ranker"],
                "can_affect_trade_execution": usage["used_by_scalp_execution_score"],
                "can_receive_positive_learning_credit": can_learn,
                "needs_fix": needs_fix,
                "needs_adjustment": needs_adj,
                "reason": reason,
            }
        )

    return {
        "feature_version": SCALP_FEATURE_VERSION,
        "feature_dim": SCALP_FEATURE_DIM,
        "symbols": syms,
        "features": rows,
        "needs_fix_count": sum(1 for r in rows if r["needs_fix"]),
        "needs_adjustment_count": sum(1 for r in rows if r["needs_adjustment"]),
        "fail_reasons": fail,
        "pass": not fail and all((per[s] or {}).get("pass") for s in syms),
        "all_features_per_coin": all(len((per[s] or {}).get("features") or []) == SCALP_FEATURE_DIM for s in syms),
    }


__all__ = ["run_scalp_indicator_truth_audit"]
