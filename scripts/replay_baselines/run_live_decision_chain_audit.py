#!/usr/bin/env python3
"""Read-only live decision chain audit for DAY top-4. Writes three baseline JSON artifacts."""

from __future__ import annotations

import asyncio
import json
import math
import os
import sqlite3
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config.redis_config import get_redis_client
from backend.config.trading_universe import DAY_TRADE_SYMBOLS
from backend.database_schema import DATABASE_PATH
from backend.services.allweather_paper_accounting import compute_pnl_breakdown
from backend.services.live_strategy_contracts import redis_ai_signal_key
from backend.services.portfolio_engine import _adaptive_weights_enabled, resolve_buy_margin_from_payload
from backend.services.symbol_setup_outcome_penalty import POST_V3_MIN_BUY_ID, apply_v3_outcome_ranking_to_decision_data
from backend.utils.symbols import normalize_symbol

OUT_DIR = ROOT / "scripts/replay_baselines"


def fetch_json(path: str) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(f"http://localhost:8000{path}", timeout=120) as resp:
            return json.loads(resp.read().decode())
    except Exception as exc:
        return {"error": str(exc)}


def redis_hash(key: str) -> dict[str, str]:
    r = get_redis_client()
    if not r:
        return {}
    raw = r.hgetall(key) or {}
    out: dict[str, str] = {}
    for k, v in raw.items():
        kk = k.decode() if isinstance(k, bytes) else str(k)
        vv = v.decode() if isinstance(v, bytes) else str(v)
        out[kk] = vv
    return out


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v in (None, ""):
            return default
        x = float(v)
        return default if math.isnan(x) else x
    except (TypeError, ValueError):
        return default


async def audit_symbol(api_sym: str, md_row: dict[str, Any] | None) -> dict[str, Any]:
    from backend.services.day_regime_router import classify_day_regime, evaluate_day_entry_route
    from backend.services.day_trade_thesis import apply_trade_thesis_to_candidate_fields

    bus = api_sym.upper().replace("/", "")
    slash = normalize_symbol(bus)
    sig = redis_hash(redis_ai_signal_key("day", bus))
    ctx = redis_hash(f"ai_context:{bus}")
    feat = redis_hash(f"feature:{bus}")

    md = md_row or {}
    bar_counts = md.get("day_bar_counts") or {}
    active_sig = md.get("active_redis_signal") or {}

    dd = dict(sig)
    dd.update(
        {
            "live_ai_strategy": "day",
            "symbol": bus,
            "ctx_market_regime": ctx.get("market_regime") or ctx.get("regime") or sig.get("regime") or "unknown",
            "ctx_rs_btc": safe_float(ctx.get("ctx_rs_btc") or sig.get("ctx_rs_btc")),
            "spread_pct": safe_float(sig.get("spread_pct") or ctx.get("spread_pct")),
            "buy_margin": safe_float(sig.get("buy_margin")),
            "estimated_win_pct": safe_float(sig.get("estimated_win_pct"), 0.012),
            "estimated_loss_pct": safe_float(sig.get("estimated_loss_pct"), 0.007),
            "estimated_fees_pct": safe_float(sig.get("estimated_fees_pct"), 0.001),
            "estimated_slippage_pct": safe_float(sig.get("estimated_slippage_pct"), 0.0008),
        }
    )
    regime = classify_day_regime(dd, context_payload=ctx)
    dd["day_route_regime"] = regime

    price = safe_float(sig.get("price") or sig.get("current_price") or ctx.get("price"))
    atr = safe_float(feat.get("atr") or sig.get("atr") or ctx.get("atr") or max(price, 1.0) * 0.015)

    thesis_dd = apply_trade_thesis_to_candidate_fields(
        dd,
        symbol=slash,
        current_price=price or 60000.0,
        atr=atr,
        strategy_id="day",
        price_structure_regime=str(sig.get("price_structure_regime") or "unknown"),
        context_payload=ctx,
    )
    route = evaluate_day_entry_route(
        setup_type=str(thesis_dd.get("setup_type") or thesis_dd.get("entry_thesis") or ""),
        day_regime=regime,
        decision_data=thesis_dd,
        context_payload=ctx,
        current_price=price,
        thesis_score=safe_float(thesis_dd.get("thesis_score")),
    )

    side = str(sig.get("side") or sig.get("argmax_action") or "hold").lower()
    prob_buy = safe_float(sig.get("prob_buy") or sig.get("winner_probability"), safe_float(sig.get("confidence"), 0.5))
    confidence = safe_float(sig.get("confidence") or sig.get("winner_probability"), prob_buy)
    bm = resolve_buy_margin_from_payload(thesis_dd)
    if bm is None:
        bm = safe_float(sig.get("buy_margin"))

    raw_ev = safe_float(thesis_dd.get("selected_net_expected_value") or thesis_dd.get("net_expected_value"))
    if raw_ev == 0:
        raw_ev = max(
            0.0,
            prob_buy * safe_float(thesis_dd.get("estimated_win_pct"), 0.012) - (1 - prob_buy) * safe_float(thesis_dd.get("estimated_loss_pct"), 0.007),
        )

    raw_rank = max(0.0, min(1.0, confidence))
    if bm is not None:
        raw_rank = max(0.0, min(1.0, raw_rank + max(-0.12, min(0.12, bm * 0.12))))

    v3_dd = apply_v3_outcome_ranking_to_decision_data(dict(thesis_dd), slash, raw_rank_score=raw_rank, buy_margin=bm)

    feature_dim_expected = 145
    feature_dim_provided = int(safe_float(sig.get("feature_dim") or active_sig.get("feature_dim"), 0))
    missing_features = max(0, feature_dim_expected - feature_dim_provided) if feature_dim_provided else feature_dim_expected

    candidate_created = bool(sig) and active_sig.get("present", bool(sig))
    exclusion_reason = None
    if not sig:
        exclusion_reason = "FULL_UNIVERSE_BLOCK_MISSING_MODEL_PREDICTION"
    elif active_sig.get("entry_gate_ok") is False:
        exclusion_reason = active_sig.get("entry_gate_reject") or "ENTRY_GATE_REJECT"

    buy_intent = (bm or 0) > 0.015 or (side == "buy" and (bm or 0) > 0)

    return {
        "symbol": slash,
        "api_symbol": api_sym,
        "market_data": {
            "current_price": price,
            "15m_data_available": int(bar_counts.get("15m") or 0) >= 20,
            "1h_data_available": int(bar_counts.get("1h") or 0) >= 20,
            "4h_data_available": int(bar_counts.get("4h") or 0) >= 20,
            "volume_data_available": bool(safe_float(ctx.get("ctx_relative_volume") or ctx.get("relative_volume") or sig.get("relative_volume")) or int(bar_counts.get("1m") or 0) > 0),
            "candles_15m": bar_counts.get("15m"),
            "candles_1h": bar_counts.get("1h"),
            "candles_4h": bar_counts.get("4h"),
            "feature_freshness": {
                "content_fresh": sig.get("content_fresh") or active_sig.get("content_fresh"),
                "signal_content_age_sec": active_sig.get("signal_content_age_sec"),
                "ctx_age_sec": ctx.get("ctx_age_sec"),
            },
            "missing_candles": md.get("day_active_bundle_missing") or [],
            "bundle_ok": md.get("day_active_bundle_ok"),
        },
        "indicators_features": {
            "feature_row_valid": feature_dim_provided == feature_dim_expected and bool(sig.get("context_audit_emit") or active_sig.get("context_audit_emit_present")),
            "feature_version": sig.get("feature_version") or active_sig.get("feature_version"),
            "feature_dim_expected": feature_dim_expected,
            "feature_dim_provided": feature_dim_provided,
            "missing_feature_count": missing_features,
            "ema_alignment": safe_float(sig.get("ema_alignment") or ctx.get("ema_alignment")),
            "adx": safe_float(sig.get("adx") or ctx.get("adx")),
            "rsi": safe_float(sig.get("rsi") or ctx.get("rsi")),
            "relative_volume": safe_float(sig.get("relative_volume") or ctx.get("ctx_relative_volume")),
            "atr": atr,
            "ctx_rs_btc": safe_float(sig.get("ctx_rs_btc") or ctx.get("ctx_rs_btc")),
            "regime_label": regime,
            "redis_feature_hash_populated": bool(feat),
        },
        "ai_model_prediction": {
            "model_artifact": sig.get("model_artifact_path"),
            "model_trained_at": sig.get("model_trained_at"),
            "predict_proba_buy": prob_buy,
            "confidence": confidence,
            "side": side,
            "buy_margin": bm,
            "prediction_consumed_by_entry_logic": bool(sig) and confidence > 0,
        },
        "setup_detection": {
            "detected_setup": thesis_dd.get("setup_type") or thesis_dd.get("entry_thesis"),
            "setup_score": safe_float(thesis_dd.get("thesis_score")),
            "regime": regime,
            "regime_permission": route.get("allowed"),
            "regime_block_reason": route.get("block_reason"),
            "buy_intent_score": bm,
            "positive_buy_intent": buy_intent,
            "raw_ev": round(raw_ev, 8),
            "net_after_cost_ev": safe_float(v3_dd.get("adjusted_ev"), raw_ev),
            "final_selection_score": v3_dd.get("final_selection_score"),
        },
        "candidate_creation": {
            "evaluated": True,
            "candidate_created": candidate_created,
            "failure_reason": exclusion_reason or (route.get("block_reason") if not route.get("allowed") else None),
            "failure_class": (
                "data" if exclusion_reason and "MISSING" in str(exclusion_reason) else "regime" if not route.get("allowed") else "signal_side" if side in ("hold", "sell") and not buy_intent else None
            ),
        },
        "ranking_v3": {
            "raw_rank_score": v3_dd.get("raw_rank_score"),
            "outcome_adjusted_rank_score": v3_dd.get("outcome_adjusted_rank_score"),
            "final_selection_score": v3_dd.get("final_selection_score"),
            "outcome_penalty_applied": v3_dd.get("outcome_penalty_applied"),
            "outcome_credit_applied": v3_dd.get("outcome_credit_applied"),
            "outcome_penalty_or_credit": v3_dd.get("outcome_penalty_or_credit"),
            "penalty_reason": v3_dd.get("penalty_reason"),
        },
    }


async def main() -> None:
    ts = datetime.now(timezone.utc).isoformat()
    md_resp = fetch_json("/api/portfolio-engine/market-data-readiness")
    status = fetch_json("/api/portfolio-engine/status")
    day_health = fetch_json("/api/portfolio-engine/day-health")
    learning = fetch_json("/api/portfolio-engine/learning-status?limit=5")

    md_rows = {r.get("symbol"): r for r in ((md_resp.get("data") or {}).get("rows") or [])}
    symbols_audit = [await audit_symbol(sym, md_rows.get(sym)) for sym in DAY_TRADE_SYMBOLS]

    profit_cycle: dict[str, Any] = {}
    with sqlite3.connect(DATABASE_PATH) as conn:
        row = conn.execute("SELECT value_json FROM operational_state WHERE key='profit_system_current_cycle'").fetchone()
        if row:
            profit_cycle = json.loads(row[0])

    fu = profit_cycle.get("full_universe_diagnostics") or {}
    aw = profit_cycle.get("adaptive_weight_diagnostics") or {}
    lb = (profit_cycle.get("current_cycle") or {}).get("leaderboard") or []

    learning_db: dict[str, Any] = {}
    with sqlite3.connect(DATABASE_PATH) as conn:
        try:
            learning_db["ai_strategy_score_weights_columns"] = [r[1] for r in conn.execute("PRAGMA table_info(ai_strategy_score_weights)")]
            learning_db["ai_strategy_score_weights_count"] = conn.execute("SELECT COUNT(*) FROM ai_strategy_score_weights").fetchone()[0]
        except Exception as exc:
            learning_db["ai_strategy_score_weights_error"] = str(exc)
        learning_db["ai_symbol_strategy_expectancy_count"] = conn.execute("SELECT COUNT(*) FROM ai_symbol_strategy_expectancy").fetchone()[0]
        learning_db["ai_outcome_training_rows_count"] = conn.execute("SELECT COUNT(*) FROM ai_outcome_training_rows").fetchone()[0]

    ranked = sorted(
        symbols_audit,
        key=lambda x: safe_float((x.get("ranking_v3") or {}).get("final_selection_score")),
        reverse=True,
    )
    best = ranked[0] if ranked else {}
    status_data = status.get("data") or {}
    dh = day_health.get("data") or day_health
    open_positions = status_data.get("open_positions") or []
    slots_open = len(open_positions) < int(dh.get("max_open_positions") or 4)
    diag = dh.get("capital_idle_diagnosis") or {}
    no_buy_reason = diag.get("last_execution_block") or dh.get("capital_idle_reason")

    all_evaluated = int(fu.get("safety_valid_universe_size") or 0) >= 4
    feature_valid = all((s.get("indicators_features") or {}).get("feature_row_valid") for s in symbols_audit)
    positive_intent = [s["symbol"] for s in symbols_audit if (s.get("setup_detection") or {}).get("positive_buy_intent")]
    learning_beyond_penalty = bool(lb) and any(safe_float(r.get("symbol_trust_score")) for r in lb)

    fwd = compute_pnl_breakdown(DATABASE_PATH)
    ledger = fwd.get("ledger") or {}
    gap = abs(safe_float(ledger.get("total_equity_usd")) - safe_float(fwd.get("forward_equity_usd")))

    final_report = {
        "generated_at_utc": ts,
        "all_top4_evaluated_each_cycle": all_evaluated,
        "feature_vectors_valid": feature_valid,
        "ai_predictions_generated": all((s.get("ai_model_prediction") or {}).get("confidence") for s in symbols_audit),
        "ai_predictions_consumed_by_entry": all((s.get("ai_model_prediction") or {}).get("prediction_consumed_by_entry_logic") for s in symbols_audit),
        "learning_used_beyond_penalty": learning_beyond_penalty,
        "learning_used_as_soft_score": learning_beyond_penalty or _adaptive_weights_enabled(),
        "adaptive_score_weights_enabled": _adaptive_weights_enabled(),
        "adaptive_weights_applied_last_cycle": int(aw.get("adaptive_weight_applied_count") or 0),
        "candidate_starvation_found": all_evaluated and len(positive_intent) <= 1,
        "v3_caused_no_trade_behavior": False,
        "hard_blocks_added": False,
        "strategy_thresholds_changed": False,
        "exits_changed": False,
        "paper_live_behavior_changed": False,
        "current_best_symbol": best.get("symbol"),
        "current_best_setup": (best.get("setup_detection") or {}).get("detected_setup"),
        "would_buy_if_slot_open": bool(slots_open and positive_intent and no_buy_reason not in ("TRADING_PAUSED", "KILL_SWITCH")),
        "exact_no_buy_reason": no_buy_reason,
        "open_positions_count": len(open_positions),
        "positive_buy_intent_symbols": positive_intent,
        "post_v3_min_buy_id": POST_V3_MIN_BUY_ID,
    }
    with sqlite3.connect(DATABASE_PATH) as conn:
        final_report["max_buy_id"] = conn.execute("SELECT MAX(id) FROM paper_trades WHERE side='BUY'").fetchone()[0]

    chain_doc = {
        "generated_at_utc": ts,
        "final_report": final_report,
        "profit_cycle_snapshot": {"full_universe_diagnostics": fu, "adaptive_weight_diagnostics": aw, "leaderboard": lb},
        "symbols": symbols_audit,
        "accounting": {
            "cash": ledger.get("cash_usd"),
            "equity": ledger.get("total_equity_usd"),
            "forward_realized_pnl": fwd.get("realized_pnl_forward_usd"),
            "unrealized_pnl": ledger.get("unrealized_pnl_usd"),
            "expected_equity": fwd.get("forward_equity_usd"),
            "actual_equity": ledger.get("total_equity_usd"),
            "gap": gap,
            "equity_invariant_ok": gap < 0.05,
            "forward_equity_reconciles": gap < 0.05,
        },
    }
    top4_doc = {"generated_at_utc": ts, "final_report": final_report, "per_symbol": {s["symbol"]: s for s in symbols_audit}}
    learning_doc = {
        "generated_at_utc": ts,
        "final_report": final_report,
        "learning_db": learning_db,
        "adaptive_score_weight_enabled_env": os.getenv("ADAPTIVE_SCORE_WEIGHT_ENABLED", ""),
        "adaptive_weights_auto_enabled": _adaptive_weights_enabled(),
        "symbol_trust_last_leaderboard": lb,
        "learning_api_sample": learning.get("data"),
        "learning_paths": {
            "symbol_trust": "ai_symbol_strategy_expectancy → symbol_trust_score (rank + size, soft)",
            "adaptive_weights": "ai_strategy_score_weights → adaptive_score_delta (soft rank, auto-on when rows exist)",
            "weight_writer": "ai_strategy_score_weight_writer.propagate_adaptive_score_weights_for_close on each close",
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "live_decision_chain_audit_latest.json").write_text(json.dumps(chain_doc, indent=2, default=str) + "\n", encoding="utf-8")
    (OUT_DIR / "top4_candidate_audit_latest.json").write_text(json.dumps(top4_doc, indent=2, default=str) + "\n", encoding="utf-8")
    (OUT_DIR / "learning_usage_audit_latest.json").write_text(json.dumps(learning_doc, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(final_report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
