"""
DAY v5 full feature audit — 145-dim vector health for top-4 symbols.

Read-only inspection + provenance from ``build_day_htf_feature_vector_145`` path.
Does not change trading gates or FEATURE_VERSION.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import time
from datetime import datetime, timezone
from typing import Any

from backend.config.day_active_timeframes import DAY_ACTIVE_TIMEFRAMES
from backend.config.trading_universe import DAY_TRADE_SYMBOLS
from backend.services.ai_decision_contract import AI_FEATURE_DIM_V1, AI_FEATURE_DIM_V2, CONTEXT_DIMS_DAY_FULL, FEATURE_VERSION_CURRENT
from backend.services.ai_feature_freshness_diagnostics import FEATURE_BLOCKS, freshness_thresholds_sec
from backend.services.ai_feature_fundamentals import merge_canonical_sentiment_payload
from backend.services.day_active_market_bundle import (
    DAY_BUNDLE_CACHE_PREFIX,
    _normalize_ccxt_symbol,
    async_read_cached_day_active_bundle,
    validate_day_active_bundle,
)
from backend.services.feature_builder import FEATURE_TRUST_SCORES, build_feature_dict_from_ohlcv
from backend.services.feature_mapping import FEATURE_MAPPING, get_feature_name

CONTEXT_BLOCK = "context_125_145"

BAD_STATUSES: frozenset[str] = frozenset({"FALLBACK", "MISSING", "STALE", "ZERO_DEFAULT", "UNSUPPORTED_FOR_SPOT"})
PASS_STATUSES: frozenset[str] = frozenset({"LIVE", "CALCULATED", "CALCULATED_PROXY", "WARMUP", "LOW_IMPORTANCE_TIME_FIELD_NORMAL"})


def _ccxt_symbol(bus: str) -> str:
    s = (bus or "").upper().replace("/", "")
    if "/" not in s and s.endswith("USDT"):
        return f"{s[:-4]}/USDT"
    return s


def _redis_hgetall(key: str) -> dict[str, str]:
    try:
        from backend.config.redis_config import get_shared_redis_sync

        r = get_shared_redis_sync()
        if not r:
            return {}
        raw = r.hgetall(key) or {}
        out: dict[str, str] = {}
        for k, v in raw.items():
            kk = k.decode() if isinstance(k, bytes) else str(k)
            vv = v.decode() if isinstance(v, bytes) else str(v)
            out[kk] = vv
        return out
    except Exception:
        return {}


def _redis_get_json(key: str) -> dict[str, Any] | None:
    try:
        from backend.config.redis_config import get_shared_redis_sync

        r = get_shared_redis_sync()
        if not r:
            return None
        raw = r.get(key)
        if not raw:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode()
        payload = json.loads(raw)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _iso_age_sec(iso_str: str | None) -> float | None:
    if not iso_str or not str(iso_str).strip():
        return None
    try:
        dt = datetime.fromisoformat(str(iso_str).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds())
    except (TypeError, ValueError):
        return None


def _block_for_index(idx0: int) -> str:
    if idx0 >= AI_FEATURE_DIM_V1:
        return CONTEXT_BLOCK
    one = idx0 + 1
    if 78 <= one <= 90:
        return "market_sentiment"
    for block, (lo, hi) in FEATURE_BLOCKS.items():
        if lo <= one <= hi:
            return block
    return "unknown"


def _feature_name_at(idx0: int) -> str:
    if idx0 < AI_FEATURE_DIM_V1:
        return get_feature_name(idx0 + 1)
    ctx_i = idx0 - AI_FEATURE_DIM_V1
    if 0 <= ctx_i < len(CONTEXT_DIMS_DAY_FULL):
        return CONTEXT_DIMS_DAY_FULL[ctx_i]
    return f"unknown_{idx0 + 1}"


def _repair_recommendation(status: str, name: str, source: str) -> str:
    if status in ("LIVE", "CALCULATED"):
        return "none"
    if status == "CALCULATED_PROXY":
        return f"acceptable proxy; document as proxy not live tape ({name})"
    if status == "WARMUP":
        return "wait for sufficient OHLCV depth or 1d series; do not treat as signal"
    if status == "UNSUPPORTED_FOR_SPOT":
        return "exclude from learning credit until v6 removes or replaces slot"
    if status == "STALE":
        return f"refresh source: {source}"
    if status == "MISSING":
        return f"wire or refresh: {source}"
    if status == "ZERO_DEFAULT":
        return "verify calculation inputs; zero may be valid or missing data"
    if status == "FALLBACK":
        return "replace price-fallback with real calculation (see feature_builder)"
    return "review"


async def _load_inputs_for_symbol(symbol_bus: str) -> dict[str, Any]:
    ccxt = _ccxt_symbol(symbol_bus)
    base = symbol_bus.replace("USDT", "").replace("/", "")

    bundle = await async_read_cached_day_active_bundle(ccxt)
    bundle_ok, bundle_miss = (False, ["no_cache"])
    bundle_age: float | None = None
    if bundle:
        payload = _redis_get_json(f"{DAY_BUNDLE_CACHE_PREFIX}{ccxt.replace('/', '')}")
        if payload:
            bundle_age = max(0.0, time.time() - float(payload.get("fetched_at") or 0))
        try:
            bundle_ok, bundle_miss = validate_day_active_bundle(bundle)
        except Exception as exc:
            bundle_miss = [str(exc)]

    ctx_h = _redis_hgetall(f"ai_context:{symbol_bus}")
    ctx_age = _iso_age_sec(ctx_h.get("ts_utc") or ctx_h.get("ctx_ts_utc") or ctx_h.get("updated_at_utc"))

    ob_h = _redis_hgetall(f"orderbook:{base}")
    from backend.services.order_book_service import orderbook_age_from_meta, parse_orderbook_redis_hash

    orderbook, ob_age = parse_orderbook_redis_hash(ob_h)
    if ob_age is None and ob_h:
        ob_age = orderbook_age_from_meta(ob_h.get("ts_utc"), ob_h.get("updated_at"))

    needs_live_ob = (not orderbook) or float((orderbook or {}).get("bid_ask_spread") or 0.0) <= 0.0
    if needs_live_ob:
        try:
            from backend.services.orderbook_redis_refresher import get_orderbook_redis_refresher

            refresher = get_orderbook_redis_refresher()
            try:
                from backend.config.redis_config import get_shared_redis_async

                r_async = await get_shared_redis_async()
                await refresher.refresh_symbol(base, r_async)
                ob_h = _redis_hgetall(f"orderbook:{base}")
                orderbook, ob_age = parse_orderbook_redis_hash(ob_h)
                if ob_age is None and ob_h:
                    ob_age = orderbook_age_from_meta(ob_h.get("ts_utc"), ob_h.get("updated_at"))
            except Exception:
                pass
        except Exception:
            pass

    if needs_live_ob and ((not orderbook) or float((orderbook or {}).get("bid_ask_spread") or 0.0) <= 0.0):
        try:
            from backend.services.order_book_service import fetch_order_book_features_live

            live_ob = await fetch_order_book_features_live(ccxt)
            if live_ob and float(live_ob.get("bid_ask_spread") or 0) > 0:
                orderbook = {k: float(v) for k, v in live_ob.items() if k not in ("ts_utc", "source")}
                orderbook["ts_utc"] = time.time()
                ob_age = 0.0
        except Exception:
            if orderbook is None:
                orderbook = None

    vp_h = _redis_hgetall(f"volume_profile:{base}")
    volume_profile: dict[str, Any] | None = None
    if vp_h:
        volume_profile = {}
        for k, v in vp_h.items():
            with contextlib.suppress(Exception):
                volume_profile[k] = float(v)

    sentiment: dict[str, Any] | None = None
    try:
        from backend.config.redis_config import get_shared_redis_async
        from backend.services.ai_decision_contract import REDIS_KEY_AI_SENTIMENT

        r = await get_shared_redis_async()
        raw_s = await r.get(REDIS_KEY_AI_SENTIMENT) if r else None
        if raw_s:
            sdec = raw_s.decode() if isinstance(raw_s, bytes) else raw_s
            sentiment = {"fear_greed_index": float(sdec)}
    except Exception:
        sentiment = None

    rows_1m = (bundle or {}).get("1m") or []
    redis_async = None
    try:
        from backend.config.redis_config import get_shared_redis_async

        redis_async = await get_shared_redis_async()
    except Exception:
        redis_async = None
    sentiment = await merge_canonical_sentiment_payload(
        base_symbol=base,
        pair_symbol=ccxt,
        ctx_for_overlay=ctx_h if isinstance(ctx_h, dict) else None,
        redis_client=redis_async,
        ohlcv_1m=rows_1m if isinstance(rows_1m, list) else [],
        existing=sentiment,
    )

    return {
        "symbol_bus": symbol_bus,
        "ccxt": ccxt,
        "bundle": bundle,
        "bundle_ok": bundle_ok,
        "bundle_miss": bundle_miss,
        "bundle_age_sec": bundle_age,
        "ctx_h": ctx_h,
        "ctx_age_sec": ctx_age,
        "orderbook": orderbook,
        "orderbook_age_sec": ob_age,
        "volume_profile": volume_profile,
        "sentiment": sentiment,
    }


async def build_symbol_feature_audit(symbol_bus: str) -> dict[str, Any]:
    """Build full 145-feature audit rows for one symbol."""
    inp = await _load_inputs_for_symbol(symbol_bus)
    bundle = inp["bundle"]
    if not bundle or not inp["bundle_ok"]:
        return {
            "symbol": symbol_bus,
            "error": "day_bundle_missing_or_invalid",
            "missing": inp.get("bundle_miss"),
            "features": [],
            "pass": False,
        }

    from backend.services.ai_day_htf_features import build_day_htf_feature_vector_145

    provenance: dict[str, dict[str, Any]] = {}
    rows_1m = bundle.get("1m") or []
    rows_1d = bundle.get("1d")
    build_feature_dict_from_ohlcv(
        symbol_ccxt=inp["ccxt"],
        ohlcv=rows_1m,
        volume_profile=inp["volume_profile"],
        orderbook=inp["orderbook"],
        sentiment=inp["sentiment"],
        ohlcv_1d=rows_1d if isinstance(rows_1d, list) else None,
        provenance=provenance,
        orderbook_age_sec=inp.get("orderbook_age_sec"),
    )

    vector = build_day_htf_feature_vector_145(
        symbol_ccxt=inp["ccxt"],
        day_bundle=bundle,
        volume_profile=inp["volume_profile"],
        orderbook=inp["orderbook"],
        sentiment=inp["sentiment"],
        ai_context=inp["ctx_h"],
    )

    ctx_prov = build_context_provenance(inp, bundle)
    features: list[dict[str, Any]] = []

    for idx0 in range(AI_FEATURE_DIM_V2):
        name = _feature_name_at(idx0)
        block = _block_for_index(idx0)
        val = float(vector[idx0]) if idx0 < len(vector) else 0.0
        if idx0 < AI_FEATURE_DIM_V1:
            meta = dict(provenance.get(name) or {})
        else:
            meta = dict(ctx_prov.get(name) or {})

        status = str(meta.get("status") or "MISSING")
        source = str(meta.get("source") or "unknown")
        age = meta.get("age_seconds")
        trust = float(meta.get("trust_score") if meta.get("trust_score") is not None else FEATURE_TRUST_SCORES.get(status, 0.0))
        learning = bool(meta.get("learning_allowed", status in ("LIVE", "CALCULATED")))
        is_real = status in ("LIVE", "CALCULATED", "CALCULATED_PROXY")
        is_placeholder = status in BAD_STATUSES or (status == "WARMUP" and abs(val) < 1e-12)

        features.append(
            {
                "index": idx0 + 1,
                "name": name,
                "block": block,
                "value": round(val, 8),
                "source": source,
                "status": status,
                "age_seconds": age,
                "trust_score": round(trust, 4),
                "learning_allowed": learning,
                "is_real": is_real,
                "is_placeholder": is_placeholder,
                "safe_for_learning": learning and status not in ("UNSUPPORTED_FOR_SPOT", "FALLBACK", "STALE", "MISSING"),
                "repair_recommendation": _repair_recommendation(status, name, source),
            }
        )

    summary = _summarize_features(features)
    summary["health_pct"] = round(
        100.0 * (summary["live_count"] + summary["calculated_count"] + summary.get("calculated_proxy_count", 0)) / max(1, summary["total_features"]),
        2,
    )
    bad = [f for f in features if f["status"] in ("FALLBACK", "MISSING", "STALE", "ZERO_DEFAULT") or (f["status"] == "WARMUP" and f["is_placeholder"])]
    allowed_unsupported = {f["name"] for f in features if f["status"] == "UNSUPPORTED_FOR_SPOT"}
    strict_pass = len(bad) == 0 and summary["total_features"] == AI_FEATURE_DIM_V2 and summary.get("fallback_count", 0) == 0

    return {
        "symbol": symbol_bus,
        "feature_version": FEATURE_VERSION_CURRENT,
        "feature_dim": AI_FEATURE_DIM_V2,
        "bundle_age_sec": inp.get("bundle_age_sec"),
        "ctx_age_sec": inp.get("ctx_age_sec"),
        "orderbook_age_sec": inp.get("orderbook_age_sec"),
        "features": features,
        "summary": summary,
        "bad_features": bad,
        "unsupported_for_spot": sorted(allowed_unsupported),
        "pass": strict_pass,
        "pass_note": "PASS: no FALLBACK/MISSING/STALE/placeholder ZERO/WARMUP-zero; UNSUPPORTED_FOR_SPOT and CALCULATED_PROXY allowed when marked",
    }


def build_context_provenance(inp: dict[str, Any], bundle: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Provenance for dims 125-145."""
    out: dict[str, dict[str, Any]] = {}
    ctx = inp.get("ctx_h") or {}
    ctx_age = inp.get("ctx_age_sec")
    bundle_age = inp.get("bundle_age_sec")
    thresholds = freshness_thresholds_sec()
    ctx_stale = ctx_age is not None and ctx_age > thresholds.get("ai_context", 120)
    bundle_stale = bundle_age is not None and bundle_age > thresholds.get("ohlcv_bundle", 110)

    for tf in DAY_ACTIVE_TIMEFRAMES:
        name = f"slope_pct_{tf}"
        rows = bundle.get(tf)
        n = len(rows) if isinstance(rows, list) else 0
        need = 5
        if n < need or bundle_stale:
            st = "STALE" if bundle_stale else "WARMUP"
            out[name] = {
                "status": st,
                "source": f"day_bundle tf={tf} bars={n}",
                "age_seconds": bundle_age,
                "trust_score": 0.1 if st == "STALE" else 0.15,
                "learning_allowed": False,
            }
        else:
            out[name] = {
                "status": "CALCULATED",
                "source": f"native slope {tf} from bundle",
                "age_seconds": bundle_age,
                "trust_score": 0.92,
                "learning_allowed": True,
            }

    mv = bundle.get("_month_vec")
    for i, cname in enumerate(("month_log_ret_window", "month_realized_vol_window")):
        if isinstance(mv, list) and len(mv) > i:
            out[cname] = {
                "status": "CALCULATED",
                "source": "1d month_vec from bundle",
                "age_seconds": bundle_age,
                "trust_score": 0.9,
                "learning_allowed": True,
            }
        else:
            out[cname] = {
                "status": "MISSING",
                "source": "bundle _month_vec",
                "age_seconds": bundle_age,
                "trust_score": 0.0,
                "learning_allowed": False,
            }

    ctx_fields = {
        "mean_ema_align_all_tf": ("CALCULATED", "mtf ema_align mean from bundle snapshots"),
        "ctx_change_24h_pct": ("LIVE" if ctx.get("ctx_change_24h_pct") else "MISSING", "ai_context hash"),
        "ctx_volume_24h_log": ("LIVE" if ctx.get("ctx_volume_24h_usd") else "MISSING", "ai_context hash"),
        "ctx_relative_volume": ("LIVE" if ctx.get("ctx_relative_volume") else "MISSING", "ai_context hash"),
        "ctx_spread_pct": ("LIVE" if ctx.get("ctx_spread_pct") else "MISSING", "ai_context hash"),
        "ctx_depth_imbalance": ("LIVE" if ctx.get("ctx_depth_imbalance") is not None else "MISSING", "ai_context hash"),
        "ctx_rs_mean_btc_eth": ("CALCULATED", "mean ctx_rs_btc + ctx_rs_eth from ai_context"),
        "ctx_btc_dominance_proxy": ("LIVE" if ctx.get("ctx_btc_dominance_proxy") else "MISSING", "ai_context hash"),
        "ctx_regime_sentiment_blend": ("CALCULATED", "regime + sentiment blend from ai_context"),
    }
    ctx_updated_at = ctx.get("ts_utc") or ctx.get("ctx_ts_utc") or ctx.get("updated_at_utc")
    ctx_fresh_mod = 0.25 if ctx_stale else 1.0
    ctx_fresh_status = "STALE" if ctx_stale else ("FRESH" if ctx_age is not None else "UNKNOWN")
    micro_ctx_fields = frozenset({"ctx_spread_pct", "ctx_depth_imbalance"})

    for fname, (st, src) in ctx_fields.items():
        status = "STALE" if ctx_stale and st in ("LIVE", "CALCULATED") else st
        base_trust = FEATURE_TRUST_SCORES.get(status, 0.5)
        trust = base_trust * (ctx_fresh_mod if fname in micro_ctx_fields else 1.0)
        row: dict[str, Any] = {
            "status": status,
            "source": src,
            "age_seconds": ctx_age,
            "trust_score": round(trust, 4),
            "learning_allowed": status in ("LIVE", "CALCULATED") and not ctx_stale,
        }
        if fname in micro_ctx_fields:
            row["orderbook_updated_at"] = ctx_updated_at
            row["freshness_status"] = ctx_fresh_status
            row["freshness_trust_modifier"] = round(ctx_fresh_mod, 4)
        out[fname] = row
    return out


def _summarize_features(features: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {
        "total_features": len(features),
        "live_count": 0,
        "calculated_count": 0,
        "calculated_proxy_count": 0,
        "fallback_count": 0,
        "missing_count": 0,
        "stale_count": 0,
        "zero_default_count": 0,
        "warmup_count": 0,
        "unsupported_count": 0,
        "placeholder_count": 0,
    }
    by_block: dict[str, dict[str, int]] = {}
    for f in features:
        st = f["status"]
        if st == "LIVE":
            counts["live_count"] += 1
        elif st == "CALCULATED":
            counts["calculated_count"] += 1
        elif st == "CALCULATED_PROXY":
            counts["calculated_proxy_count"] += 1
        elif st == "FALLBACK":
            counts["fallback_count"] += 1
        elif st == "MISSING":
            counts["missing_count"] += 1
        elif st == "STALE":
            counts["stale_count"] += 1
        elif st == "ZERO_DEFAULT":
            counts["zero_default_count"] += 1
        elif st == "WARMUP":
            counts["warmup_count"] += 1
        elif st == "UNSUPPORTED_FOR_SPOT":
            counts["unsupported_count"] += 1
        if f.get("is_placeholder"):
            counts["placeholder_count"] += 1
        blk = f["block"]
        by_block.setdefault(blk, {"total": 0, "bad": 0, "good": 0})
        by_block[blk]["total"] += 1
        if st in BAD_STATUSES:
            by_block[blk]["bad"] += 1
        else:
            by_block[blk]["good"] += 1
    counts["by_block"] = by_block
    return counts


async def run_full_audit(symbols: list[str] | None = None) -> dict[str, Any]:
    syms = symbols or list(DAY_TRADE_SYMBOLS)
    per: dict[str, Any] = {}
    all_bad: list[dict[str, Any]] = []
    for sym in syms:
        rep = await build_symbol_feature_audit(sym)
        per[sym] = rep
        for bf in rep.get("bad_features") or []:
            row = dict(bf)
            row["symbol"] = sym
            all_bad.append(row)

    global_pass = all(per.get(s, {}).get("pass") for s in syms) and not all_bad
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_version": FEATURE_VERSION_CURRENT,
        "feature_dim": AI_FEATURE_DIM_V2,
        "symbols": per,
        "all_bad_features": all_bad,
        "pass": global_pass,
    }


def format_coin_summary(coin_report: dict[str, Any]) -> str:
    if coin_report.get("error"):
        return f"ERROR {coin_report.get('symbol')}: {coin_report.get('error')} {coin_report.get('missing')}"
    s = coin_report.get("summary") or {}
    lines = [
        f"=== {coin_report.get('symbol')} ===",
        f"  total={s.get('total_features')} live={s.get('live_count')} calc={s.get('calculated_count')} "
        f"proxy={s.get('calculated_proxy_count')} fallback={s.get('fallback_count')} missing={s.get('missing_count')} "
        f"stale={s.get('stale_count')} zero={s.get('zero_default_count')} warmup={s.get('warmup_count')} "
        f"unsupported={s.get('unsupported_count')} placeholder={s.get('placeholder_count')} "
        f"health={s.get('health_pct')}% pass={coin_report.get('pass')}",
    ]
    return "\n".join(lines)


__all__ = [
    "BAD_STATUSES",
    "PASS_STATUSES",
    "build_symbol_feature_audit",
    "format_coin_summary",
    "run_full_audit",
]
