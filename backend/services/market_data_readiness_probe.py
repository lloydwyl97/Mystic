"""
Live market-data visibility probe — uses **only** Binance.US + active Mystic paths.

Does not synthesize candles; does not fabricate ticks. Intended for diagnostics
(`/api/portfolio-engine/market-data-readiness`).
"""

from __future__ import annotations

import math
import os
import time
from datetime import datetime, timezone
from typing import Any

from backend.config.day_active_timeframes import DAY_ACTIVE_TIMEFRAMES
from backend.config.trading_universe import DAY_TRADE_SYMBOLS
from backend.services.ai_day_htf_features import build_day_htf_feature_vector_145
from backend.services.ai_decision_contract import CONTEXT_DIMS_DAY_FULL, FEATURE_VERSION_DAY_FULL_MTF
from backend.services.ai_entry_context_gate import evaluate_signal_hash_for_entry
from backend.services.day_active_market_bundle import (
    apply_day_bundle_stagger,
    async_fetch_day_active_ohlcv_bundle,
    validate_day_active_bundle,
)
from backend.services.feature_mapping import FEATURE_MAPPING
from backend.services.live_strategy_contracts import LiveStrategyId, redis_ai_signal_key
from backend.utils.canonical_symbol_formatter import CanonicalSymbolFormatter
from backend.utils.symbols import to_ccxt_symbol

# Lightweight visibility across native intervals Binance.US may serve (counts only).
PROBE_TIMEFRAMES: tuple[str, ...] = DAY_ACTIVE_TIMEFRAMES

from backend.config.mystic_api_schedule import MARKET_READINESS_CACHE_SEC

_readiness_result_cache: dict[str, Any] | None = None
_readiness_result_cache_ts: float = 0.0


def api_to_ccxt(api_symbol: str) -> str:
    return to_ccxt_symbol(api_symbol)


ACTIVE_PRIMARY_CLOCK_CCXT_TF: str = __import__("backend.config.ai_primary_clock", fromlist=["DAY_PRIMARY_CCXT_TF"]).DAY_PRIMARY_CCXT_TF


async def market_data_dashboard_meta_async() -> dict[str, Any]:
    """Cheap snapshot for dashboard (Redis heartbeat + universe), no Binance hammering."""
    from backend.config.redis_config import get_shared_redis_async

    last_ts: float | None = None
    redis_ok = False
    try:
        r = await get_shared_redis_async()
        raw = await r.get("market_data:last_update")
        if raw:
            s = raw.decode() if isinstance(raw, bytes) else str(raw)
            last_ts = float(s)
        redis_ok = True
    except Exception:
        pass
    return {
        "universe_api_symbols": list(DAY_TRADE_SYMBOLS),
        "day_active_required_timeframes": list(DAY_ACTIVE_TIMEFRAMES),
        "primary_signal_clock_ccxt": ACTIVE_PRIMARY_CLOCK_CCXT_TF,
        "redis_connected": redis_ok,
        "last_market_data_epoch": last_ts,
        "last_market_data_iso": (datetime.fromtimestamp(last_ts, tz=timezone.utc).isoformat() if last_ts else None),
        "feature_mapping_slots": len(FEATURE_MAPPING),
        "day_model_feature_dim": 145,
        "feature_version_day_live": FEATURE_VERSION_DAY_FULL_MTF,
        "context_slots_day_full": len(CONTEXT_DIMS_DAY_FULL),
        "indicator_registry_named_count": len(FEATURE_MAPPING),
    }


def _decode_redis_hash(raw: dict[Any, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in (raw or {}).items():
        ks = k.decode() if isinstance(k, (bytes, bytearray)) else str(k)
        if isinstance(v, (bytes, bytearray)):
            out[ks] = v.decode(errors="replace")
        elif v is None:
            out[ks] = ""
        else:
            out[ks] = str(v)
    return out


async def _fetch_active_redis_signal_proof(api_sym: str) -> tuple[dict[str, str], dict[str, Any]]:
    """Same Redis hash contract the bar/signal consumer uses (ai_signal:day:<BUS>)."""
    from backend.config.redis_config import get_shared_redis_async

    bus = api_sym.strip().upper().replace("/", "")
    key = redis_ai_signal_key(LiveStrategyId.DAY.value, bus)
    meta: dict[str, Any] = {"redis_key": key, "present": False, "proof_ok": False}
    try:
        r = await get_shared_redis_async()
        raw = await r.hgetall(key)
    except Exception as exc:
        meta["error"] = str(exc)
        return {}, meta

    if not raw:
        meta["missing_fields"] = ["redis_signal_absent"]
        return {}, meta

    dd = _decode_redis_hash(raw)
    meta["present"] = True
    try:
        fv = int(float(dd.get("feature_version") or 0))
    except (TypeError, ValueError):
        fv = 0
    try:
        fd = int(float(dd.get("feature_dim") or 0))
    except (TypeError, ValueError):
        fd = 0
    meta["feature_version"] = fv
    meta["feature_dim"] = fd
    meta["side"] = dd.get("side")
    meta["confidence"] = dd.get("confidence")
    meta["timestamp"] = dd.get("timestamp")
    meta["content_fresh"] = dd.get("content_fresh")
    meta["signal_content_stale"] = dd.get("signal_content_stale")
    meta["content_age_sec"] = dd.get("content_age_sec")
    try:
        sig_ts = float(dd.get("timestamp") or 0)
        sig_age = max(0.0, time.time() - sig_ts) if sig_ts > 0 else None
    except (TypeError, ValueError):
        sig_age = None
    meta["signal_content_age_sec"] = sig_age
    try:
        meta["redis_ttl_sec"] = int(await r.ttl(key))
    except Exception:
        meta["redis_ttl_sec"] = None
    from backend.config.ai_signal_bus import MAX_SIGNAL_AGE_SEC

    meta["content_timestamp_fresh"] = bool(
        sig_age is not None and sig_age <= float(MAX_SIGNAL_AGE_SEC) and (dd.get("signal_content_stale") or "").strip() != "1" and (dd.get("content_fresh") or "1").strip() != "0"
    )
    meta["context_audit_emit_present"] = bool((dd.get("context_audit_emit") or "").strip())

    ok_entry, reject_code, detail = evaluate_signal_hash_for_entry(dd)
    meta["entry_gate_ok"] = ok_entry
    meta["entry_gate_reject"] = reject_code
    meta["entry_gate_detail"] = detail

    proof_ok = ok_entry and fv == FEATURE_VERSION_DAY_FULL_MTF and fd == 145 and meta["context_audit_emit_present"] and meta["content_timestamp_fresh"]
    meta["proof_ok"] = proof_ok
    if not proof_ok:
        missing: list[str] = []
        if fv != FEATURE_VERSION_DAY_FULL_MTF:
            missing.append(f"feature_version_need_{FEATURE_VERSION_DAY_FULL_MTF}_have_{fv}")
        if fd != 145:
            missing.append(f"feature_dim_need_145_have_{fd}")
        if not meta["context_audit_emit_present"]:
            missing.append("context_audit_emit")
        if not meta["content_timestamp_fresh"]:
            missing.append("signal_content_stale_or_age_exceeded")
        if reject_code:
            missing.append(str(reject_code))
        meta["missing_fields"] = missing
    return dd, meta


def _bundle_bar_counts(bundle: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for tf in DAY_ACTIVE_TIMEFRAMES:
        rows = bundle.get(tf) if isinstance(bundle, dict) else None
        counts[tf] = len(rows) if isinstance(rows, list) else 0
    return counts


def _tf_detail_from_bundle(bundle: dict[str, Any], min_probe: int) -> dict[str, dict[str, Any]]:
    detail: dict[str, dict[str, Any]] = {}
    for tf in PROBE_TIMEFRAMES:
        rows = bundle.get(tf) if isinstance(bundle, dict) else None
        n = len(rows) if isinstance(rows, list) else 0
        oldest = newest = None
        if isinstance(rows, list) and n > 0:
            oldest = int(rows[0][0])
            newest = int(rows[-1][0])
        detail[tf] = {
            "rows": n,
            "need_probe_min_for_display_only": min_probe,
            "tf_probe_ok_display": n >= min_probe,
            "oldest_ms": oldest,
            "newest_ms": newest,
            "source": "day_active_bundle_snapshot",
        }
    return detail


def _orderbook_features_from_redis(api_sym: str) -> dict[str, float] | None:
    """Load live orderbook microstructure features from Redis (parity with signal path)."""
    try:
        from backend.config.redis_config import get_shared_redis_sync

        r = get_shared_redis_sync()
        if not r:
            return None
        base = CanonicalSymbolFormatter.to_base(api_sym)
        for key in (f"orderbook:{base}", f"orderbook:{api_sym}"):
            raw = r.hgetall(key) or {}
            if not raw:
                continue
            out: dict[str, float] = {}
            for k, v in raw.items():
                kk = k.decode() if isinstance(k, bytes) else str(k)
                vv = v.decode() if isinstance(v, bytes) else v
                try:
                    out[kk] = float(vv)
                except (TypeError, ValueError):
                    continue
            if out:
                return out
    except Exception:
        return None
    return None


async def probe_market_data_readiness() -> dict[str, Any]:
    global _readiness_result_cache, _readiness_result_cache_ts

    now = time.time()
    if _readiness_result_cache is not None and (now - _readiness_result_cache_ts) < MARKET_READINESS_CACHE_SEC:
        cached = dict(_readiness_result_cache)
        cached["cached"] = True
        cached["cache_age_sec"] = round(now - _readiness_result_cache_ts, 1)
        cached["cache_ttl_sec"] = MARKET_READINESS_CACHE_SEC
        return cached

    await apply_day_bundle_stagger("readiness")

    from backend.services.live_market_data import live_market_data_service

    svc = live_market_data_service
    results: list[dict[str, Any]] = []
    errors: list[str] = []

    if svc is None:
        return {
            "success": False,
            "error": "live_market_data_service_unavailable",
            "rows": [],
            "timeframe_source": {"day_active_contract": list(DAY_ACTIVE_TIMEFRAMES)},
        }

    try:
        from backend.services.history_context_gates import mtf_min_bars_per_timeframe

        min_probe = int(mtf_min_bars_per_timeframe())
    except Exception:
        min_probe = 20

    for api_sym in DAY_TRADE_SYMBOLS:
        ccxt_sym = api_to_ccxt(api_sym)
        missing: list[str] = []
        row: dict[str, Any] = {"symbol": api_sym, "ccxt": ccxt_sym}

        signal_dd, signal_meta = await _fetch_active_redis_signal_proof(api_sym)
        row["active_redis_signal"] = signal_meta
        active_proof_ok = bool(signal_meta.get("proof_ok"))

        # One coherent bundle snapshot drives day gate, bar counts, month context, and feature vector.
        bundle = await async_fetch_day_active_ohlcv_bundle(svc, ccxt_sym)
        bundle_ok, miss_b = validate_day_active_bundle(bundle)
        bar_counts = _bundle_bar_counts(bundle)
        row["day_bar_counts"] = bar_counts
        row["day_active_bundle_ok"] = bundle_ok
        row["day_active_bundle_missing"] = miss_b if not bundle_ok else []
        row["day_gate_ok"] = bundle_ok
        row["day_gate_missing_reasons"] = miss_b if not bundle_ok else []
        if not bundle_ok:
            missing.extend(miss_b if miss_b else ["day_bundle_invalid"])

        row["past_candles_ok"] = bundle_ok
        row["history_context_reason"] = "" if bundle_ok else ";".join(miss_b[:6])

        tf_detail = _tf_detail_from_bundle(bundle, min_probe)
        row["timeframe_probe_detail"] = tf_detail
        row["timeframes_probe_ok_display"] = all(tf_detail[t].get("tf_probe_ok_display") for t in PROBE_TIMEFRAMES)
        if not row["timeframes_probe_ok_display"]:
            for tf in PROBE_TIMEFRAMES:
                if not tf_detail[tf].get("tf_probe_ok_display"):
                    missing.append(f"tf_visible:{tf}_rows<{min_probe}")

        row["month_context_ready"] = bool(bundle_ok and bundle.get("_month_vec"))

        ticker: dict[str, Any] | None = None
        try:
            ticker = await svc.get_ticker(ccxt_sym)
        except Exception as e:
            errors.append(f"{api_sym}:ticker:{e}")
            missing.append("ticker")

        price = float(ticker.get("price", 0) or 0) if isinstance(ticker, dict) else 0.0
        bid = float(ticker.get("bid", 0) or 0) if isinstance(ticker, dict) else 0.0
        ask = float(ticker.get("ask", 0) or 0) if isinstance(ticker, dict) else 0.0
        vol = float(ticker.get("volume_24h", 0) or 0) if isinstance(ticker, dict) else 0.0

        row["price_ok"] = price > 0
        row["volume_ok"] = vol > 0
        row["spread_ok"] = bid > 0 and ask > 0 and ask >= bid
        if not row["price_ok"]:
            missing.append("price")
        if not row["volume_ok"]:
            missing.append("volume_24h")
        if not row["spread_ok"]:
            missing.append("bid_ask_spread")

        vec145: list[float] | None = None
        if bundle_ok:
            try:
                vec145 = build_day_htf_feature_vector_145(
                    symbol_ccxt=ccxt_sym,
                    day_bundle=bundle,
                    volume_profile=None,
                    orderbook=_orderbook_features_from_redis(api_sym),
                    sentiment=None,
                    ai_context={},
                )
            except Exception as e:
                missing.append(f"feature_145:{e}")

        finite = 0
        if vec145 and len(vec145) == 145:
            finite = sum(1 for x in vec145 if isinstance(x, (int, float)) and math.isfinite(float(x)))

        tf_used_buy: dict[str, bool] = {tf: bool(bundle_ok and len(bundle.get(tf) or []) > 5) for tf in DAY_ACTIVE_TIMEFRAMES}
        row["indicator_primary_dim_named"] = len(FEATURE_MAPPING)
        row["feature_vector_dimension"] = 145
        row["feature_version_day_contract"] = FEATURE_VERSION_DAY_FULL_MTF
        row["indicator_ok"] = bool(vec145) and len(vec145) == 145 and finite >= 143
        row["technical_124_consumer_path"] = "feature_builder.build_feature_vector_124@native_1m"
        row["context_21_consumer_path"] = "ai_feature_v2.context_vector_day_full_mtf(slopes × all_TF + month + redis_macro)"
        row["tf_actively_consumed_buy_context"] = tf_used_buy

        refresh_path_px = 0.0
        if isinstance(ticker, dict):
            refresh_path_px = float(ticker.get("price") or ticker.get("last") or 0.0)
        refresh_path_ok = refresh_path_px > 0

        execution_adapter_price_ok = False
        execution_adapter_px: float | None = None
        try:
            from backend.services.execution_adapter import get_execution_adapter

            execution_adapter_px = await get_execution_adapter().get_current_price(api_sym)
            execution_adapter_price_ok = execution_adapter_px is not None and float(execution_adapter_px) > 0
        except Exception:
            execution_adapter_price_ok = False

        pi_prices_ok = False
        try:
            pi = __import__(
                "backend.services.portfolio_engine_integration",
                fromlist=["get_portfolio_integration"],
            ).get_portfolio_integration()
            px = getattr(pi, "current_prices", {}) or {}
            for key in (ccxt_sym, api_sym):
                if not key:
                    continue
                v = px.get(key)
                if v is not None and float(v or 0) > 0:
                    pi_prices_ok = True
                    break
        except Exception:
            pi_prices_ok = False

        row["integration_price_row_ok"] = pi_prices_ok
        row["refresh_path_last_px"] = refresh_path_px
        row["refresh_path_ok"] = refresh_path_ok
        row["execution_adapter_price_ok"] = execution_adapter_price_ok
        if execution_adapter_px is not None:
            row["execution_adapter_last_px"] = float(execution_adapter_px)

        venue_prices_ok = bool(row["price_ok"] and refresh_path_ok and execution_adapter_price_ok)
        if not refresh_path_ok:
            missing.append("integration_refresh_path_last_px")
        if not execution_adapter_price_ok:
            missing.append("execution_adapter.get_current_price")

        bundle_buy_ready = bool(bundle_ok and row["indicator_ok"] and vec145)
        if active_proof_ok:
            row["ai_buy_context_ready"] = True
            row["buy_context_ready"] = True
            row["buy_context_missing"] = []
        else:
            row["ai_buy_context_ready"] = bundle_buy_ready
            row["buy_context_ready"] = bundle_buy_ready
            row["buy_context_missing"] = [] if bundle_buy_ready else list(missing)
            if not bundle_buy_ready:
                missing.append("ai_buy_context_incomplete")

        if active_proof_ok:
            row["ai_hold_context_ready"] = bool(venue_prices_ok)
            row["hold_context_ready"] = row["ai_hold_context_ready"]
            row["ai_sell_context_ready"] = bool(row["ai_hold_context_ready"])
            row["sell_context_ready"] = row["ai_sell_context_ready"]
        else:
            row["ai_hold_context_ready"] = bool(bundle_ok and venue_prices_ok)
            row["hold_context_ready"] = row["ai_hold_context_ready"]
            row["ai_sell_context_ready"] = bool(row["ai_hold_context_ready"] and bundle_ok and not miss_b)
            row["sell_context_ready"] = row["ai_sell_context_ready"]

        tf_used_ai = dict(tf_used_buy)
        tf_used_ai["month_from_daily_derived_only"] = bool(row["month_context_ready"])
        row["tf_used_ai_proof"] = tf_used_ai

        vec_consume = dict.fromkeys(("buy_path", "hold_path", "sell_path"), False)
        if active_proof_ok:
            vec_consume["buy_path"] = True
            vec_consume["hold_path"] = row["ai_hold_context_ready"]
            vec_consume["sell_path"] = row["ai_sell_context_ready"]
        elif row["indicator_ok"]:
            vec_consume["buy_path"] = bundle_buy_ready
            vec_consume["hold_path"] = row["ai_hold_context_ready"]
            vec_consume["sell_path"] = row["ai_sell_context_ready"]
        row["indicator_vector_consumed_proof"] = vec_consume

        row["last_full_bundle_ts_ms"] = max(
            (tf_detail[t].get("newest_ms") or 0 for t in DAY_ACTIVE_TIMEFRAMES if tf_detail[t].get("newest_ms")),
            default=0,
        )

        if signal_meta.get("present") and not active_proof_ok:
            missing.extend([f"active_redis:{x}" for x in signal_meta.get("missing_fields") or []])

        row["missing_fields"] = sorted(set(missing))
        row["informational_notes"] = [
            "Readiness uses one day_active_bundle snapshot per symbol (no per-TF duplicate Binance fetches).",
            "When active Redis ai_signal:day:* proof is fresh (fv=5, dim=145, context_audit_emit), buy/hold/sell context aligns with the live engine consumer.",
            "Bundle gaps remain listed in missing_fields even when active proof is valid.",
        ]
        if signal_dd:
            row["redis_signal_side"] = signal_dd.get("side")
            row["redis_signal_confidence"] = signal_dd.get("confidence")
        results.append(row)

    all_ai_move = all(r.get("sell_context_ready") and r.get("buy_context_ready") for r in results)
    payload = {
        "success": True,
        "timestamp_epoch": time.time(),
        "timestamp_iso": datetime.now(timezone.utc).isoformat(),
        "cached": False,
        "cache_age_sec": 0.0,
        "cache_ttl_sec": MARKET_READINESS_CACHE_SEC,
        "ai_can_act_all_symbols": all_ai_move,
        "ai_cannot_act_reason": None if all_ai_move else "one_or_more_symbols_failed_contract_or_vector",
        "timeframe_source": {
            "day_active_contract_python": "backend/config/day_active_timeframes.py",
            "day_active_bundle_fetch": "backend/services/day_active_market_bundle.py::async_fetch_day_active_ohlcv_bundle",
            "day_bundle_validate": "backend/services/day_active_market_bundle.py::validate_day_active_bundle",
            "day_coverage_gate_python": "backend/services/history_context_gates.py::evaluate_day_active_timeframe_coverage",
            "PROBE_TIMEFRAMES_native_fetch_only": list(PROBE_TIMEFRAMES),
            "PRIMARY_CLOCK_ONLY_FOR_DISPLAY_NOT_ENTRY_GATE": ACTIVE_PRIMARY_CLOCK_CCXT_TF,
        },
        "indicator_registry": {
            "path": "backend/services/feature_mapping.py::FEATURE_MAPPING",
            "calculation_path": "backend/services/feature_builder.py::build_feature_vector_124 (primary = native 1m bundle rows)",
            "day_145_path": "backend/services/ai_day_htf_features.py::build_day_htf_feature_vector_145",
            "named_feature_slots": len(FEATURE_MAPPING),
            "context_dim_names_DAY_V5": list(CONTEXT_DIMS_DAY_FULL),
            "feature_version_day_live": FEATURE_VERSION_DAY_FULL_MTF,
            "contract_dim_v5": 145,
        },
        "rows": results,
        "probe_errors": errors,
    }
    _readiness_result_cache = payload
    _readiness_result_cache_ts = time.time()
    return payload
