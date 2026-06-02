"""
Feature freshness + health diagnostics for the 145-dim DAY AI pipeline.

Read-only observation — does not change trading rules or execution gates.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.config.day_active_timeframes import DAY_ACTIVE_TIMEFRAMES, min_bars_for_day_tf
from backend.config.mystic_api_schedule import (
    AI_CONTEXT_LOOP_SEC,
    BAR_INTERVAL_SEC,
    CTX_FRESH_MAX_AGE_SEC,
    DAY_BUNDLE_CACHE_TTL_SEC,
)
from backend.config.trading_universe import TRADING_SYMBOLS
from backend.services.ai_decision_contract import AI_FEATURE_DIM_V1, CONTEXT_DIMS_DAY_FULL
from backend.services.day_active_market_bundle import DAY_BUNDLE_CACHE_PREFIX, validate_day_active_bundle
from backend.services.feature_mapping import get_feature_name
from backend.services.live_strategy_contracts import per_coin_artifact_file

FEATURE_AGE_BLOCKS: tuple[str, ...] = (
    "ohlcv_bundle",
    "ohlcv_1m",
    "orderbook",
    "ai_context",
    "sentiment",
    "news",
    "reddit",
    "telegram",
    "discord",
    "fear_greed",
    "month_vec",
)

IMPORTANCE_ROLLUP: dict[str, tuple[str, ...]] = {
    "technical": ("basic_price", "technical_indicators", "advanced_ta", "momentum"),
    "volatility": ("volatility",),
    "trend": ("trend",),
    "volume": ("volume_profile", "advanced_volume"),
    "time_features": ("time_based",),
    "sentiment": ("market_sentiment",),
    "orderbook_microstructure": ("microstructure",),
    "context": ("context_day_full",),
}

FEATURE_BLOCKS: dict[str, tuple[int, int]] = {
    "basic_price": (1, 10),
    "technical_indicators": (11, 34),
    "volatility": (35, 44),
    "momentum": (45, 59),
    "trend": (60, 69),
    "volume_profile": (70, 77),
    "market_sentiment": (78, 87),
    "time_based": (88, 97),
    "advanced_ta": (98, 105),
    "advanced_volume": (106, 113),
    "microstructure": (114, 121),
}


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or not str(raw).strip():
        return default
    try:
        return max(1, int(str(raw).split()[0]))
    except (TypeError, ValueError):
        return default


def freshness_thresholds_sec() -> dict[str, float]:
    from backend.services.ai_active_sentiment_collector import sentiment_fetch_intervals

    src = sentiment_fetch_intervals()
    return {
        "ohlcv_bundle": float(DAY_BUNDLE_CACHE_TTL_SEC),
        "ohlcv_1m": float(max(120, BAR_INTERVAL_SEC * 2)),
        "orderbook": float(_env_int("ORDERBOOK_REFRESH_INTERVAL_SEC", 45)),
        "ai_context": float(min(CTX_FRESH_MAX_AGE_SEC, AI_CONTEXT_LOOP_SEC * 2)),
        "sentiment": float(_env_int("SENTIMENT_REDIS_TTL_SEC", 600)),
        "news": float(src["news"]),
        "reddit": float(src["reddit"]),
        "telegram": float(src["telegram"]),
        "discord": float(src["discord"]),
        "fear_greed": float(src["fear_greed"]),
        "month_vec": float(_env_int("DAY_MONTH_VEC_STALE_SEC", 86400)),
    }


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


def _epoch_age_sec(epoch: float | None) -> float | None:
    if epoch is None or float(epoch) <= 0:
        return None
    return max(0.0, time.time() - float(epoch))


def _candle_age_sec(rows: list[list] | None) -> float | None:
    if not rows:
        return None
    last = rows[-1]
    if not isinstance(last, (list, tuple)) or not last:
        return None
    try:
        ts_ms = float(last[0])
        if ts_ms > 1e12:
            ts_ms /= 1000.0
        return max(0.0, time.time() - ts_ms)
    except (TypeError, ValueError, IndexError):
        return None


def _read_bundle_with_meta(symbol_bus: str) -> tuple[dict[str, Any] | None, float | None]:
    try:
        from backend.services.day_active_market_bundle import _normalize_ccxt_symbol

        ccxt = _normalize_ccxt_symbol(symbol_bus)
        key = f"{DAY_BUNDLE_CACHE_PREFIX}{ccxt.replace('/', '')}"
        payload = _redis_get_json(key)
        if not payload:
            return None, None
        fetched_at = float(payload.get("fetched_at") or 0) or None
        bundle_raw = payload.get("bundle")
        if not isinstance(bundle_raw, dict):
            return None, fetched_at
        bundle: dict[str, Any] = {}
        for tf in DAY_ACTIVE_TIMEFRAMES:
            rows = bundle_raw.get(tf)
            bundle[tf] = list(rows) if isinstance(rows, list) else []
        if isinstance(bundle_raw.get("_month_vec"), list):
            bundle["_month_vec"] = list(bundle_raw["_month_vec"])
        validate_day_active_bundle(bundle)  # type: ignore[arg-type]
        return bundle, fetched_at
    except Exception:
        return None, None


def _collector_fetch_epochs() -> dict[str, float]:
    epochs: dict[str, float] = {}
    try:
        status = _redis_hgetall("ai_sentiment:status")
        raw = status.get("source_last_fetch_epoch_json")
        if raw:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                for k, v in parsed.items():
                    try:
                        epochs[str(k)] = float(v)
                    except (TypeError, ValueError):
                        pass
    except Exception:
        pass
    if epochs:
        return epochs
    try:
        from backend.services.ai_active_sentiment_collector import _fetch_cache

        c = _fetch_cache
        return {
            "reddit": float(c.reddit_ts or 0),
            "news": float(c.news_ts or 0),
            "telegram": float(c.telegram_ts or 0),
            "discord": float(c.discord_ts or 0),
            "fear_greed": float(c.fear_greed_ts or 0),
        }
    except Exception:
        return {}


def build_feature_age_by_block(symbol_bus: str) -> dict[str, Any]:
    """Per-symbol ages (seconds) for each upstream feature block."""
    base = symbol_bus.replace("USDT", "") if symbol_bus.endswith("USDT") else symbol_bus
    bundle, bundle_fetched_at = _read_bundle_with_meta(symbol_bus)
    ctx = _redis_hgetall(f"ai_context:{symbol_bus}")
    ob = _redis_hgetall(f"orderbook:{base}") or _redis_hgetall(f"orderbook:{symbol_bus}")
    sent = _redis_hgetall(f"ai_sentiment:{symbol_bus}")
    collector_epochs = _collector_fetch_epochs()

    ob_ts = ob.get("ts_utc")
    try:
        ob_age = _epoch_age_sec(float(ob_ts)) if ob_ts else None
    except (TypeError, ValueError):
        ob_age = None

    ages: dict[str, float | None] = {
        "ohlcv_bundle": _epoch_age_sec(bundle_fetched_at),
        "ohlcv_1m": _candle_age_sec(bundle.get("1m") if bundle else None),
        "orderbook": ob_age,
        "ai_context": _iso_age_sec(ctx.get("ts_utc")),
        "sentiment": _iso_age_sec(sent.get("sentiment_ts_utc")),
        "news": _epoch_age_sec(collector_epochs.get("news")),
        "reddit": _epoch_age_sec(collector_epochs.get("reddit")),
        "telegram": _iso_age_sec(sent.get("telegram_ts_utc")) or _epoch_age_sec(collector_epochs.get("telegram")),
        "discord": _iso_age_sec(sent.get("discord_ts_utc")) or _epoch_age_sec(collector_epochs.get("discord")),
        "fear_greed": _epoch_age_sec(collector_epochs.get("fear_greed")),
        "month_vec": _candle_age_sec(bundle.get("1d") if bundle else None),
    }

    thresholds = freshness_thresholds_sec()
    status: dict[str, str] = {}
    for key in FEATURE_AGE_BLOCKS:
        age = ages.get(key)
        lim = thresholds.get(key, 180.0)
        if age is None:
            status[key] = "missing"
        elif age <= lim:
            status[key] = "fresh"
        else:
            status[key] = "stale"

    return {
        "symbol": symbol_bus,
        "ages_sec": {k: (round(v, 1) if v is not None else None) for k, v in ages.items()},
        "thresholds_sec": thresholds,
        "freshness_status": status,
        "bundle_fetched_at_epoch": bundle_fetched_at,
        "ctx_ts_utc": ctx.get("ts_utc"),
        "sentiment_ts_utc": sent.get("sentiment_ts_utc"),
    }


def _block_for_index(idx0: int) -> str:
    if idx0 >= AI_FEATURE_DIM_V1:
        return "context_day_full"
    one_based = idx0 + 1
    for block, (lo, hi) in FEATURE_BLOCKS.items():
        if lo <= one_based <= hi:
            return block
    return "unknown"


def _feature_names_145() -> list[str]:
    names = [get_feature_name(i) for i in range(1, AI_FEATURE_DIM_V1 + 1)]
    names.extend(list(CONTEXT_DIMS_DAY_FULL))
    return names


def build_feature_health_score(
    *,
    symbol_bus: str,
    age_report: dict[str, Any] | None = None,
    zero_filled_count: int | None = None,
    inactive_slot_count: int = 0,
    optional_inactive_count: int = 0,
    missing_context_dims: list[str] | None = None,
) -> dict[str, Any]:
    """Aggregate freshness + completeness into a 0–100 health score."""
    age_report = age_report or build_feature_age_by_block(symbol_bus)
    freshness = age_report.get("freshness_status") or {}
    fresh_blocks = [k for k, v in freshness.items() if v == "fresh"]
    stale_blocks = [k for k, v in freshness.items() if v == "stale"]
    missing_blocks = [k for k, v in freshness.items() if v == "missing"]
    inactive_slots = max(0, int(inactive_slot_count))
    optional_inactive = max(0, int(optional_inactive_count))
    zero_count = zero_filled_count if zero_filled_count is not None else 0
    missing_ctx = list(missing_context_dims or [])

    block_total = len(FEATURE_AGE_BLOCKS)
    freshness_score = (len(fresh_blocks) / block_total) * 55.0 if block_total else 0.0
    stale_penalty = min(25.0, len(stale_blocks) * 4.0)
    missing_penalty = min(20.0, len(missing_blocks) * 5.0)
    inactive_penalty = min(10.0, inactive_slots * 1.5)
    zero_penalty = min(15.0, (zero_count / 145.0) * 15.0)
    score = max(0.0, min(100.0, freshness_score + 45.0 - stale_penalty - missing_penalty - inactive_penalty - zero_penalty))

    return {
        "symbol": symbol_bus,
        "health_score": round(score, 1),
        "fresh_feature_blocks": fresh_blocks,
        "stale_feature_blocks": stale_blocks,
        "missing_feature_blocks": missing_blocks,
        "inactive_slots": inactive_slots,
        "optional_inactive_slots": optional_inactive,
        "missing_context_dims": missing_ctx[:10],
        "zero_filled_count": zero_count,
        "components": {
            "freshness_score": round(freshness_score, 2),
            "stale_penalty": round(stale_penalty, 2),
            "missing_penalty": round(missing_penalty, 2),
            "inactive_penalty": round(inactive_penalty, 2),
            "zero_penalty": round(zero_penalty, 2),
        },
    }


def build_feature_importance_by_block() -> dict[str, Any]:
    """Roll up per-coin RF importances into user-facing block groups."""
    models_dir = Path("models/active")
    per_coin: dict[str, Any] = {}
    rollup_keys = list(IMPORTANCE_ROLLUP.keys())

    for sym in TRADING_SYMBOLS:
        path = per_coin_artifact_file(models_dir, "day", sym)
        entry: dict[str, Any] = {
            "feature_version": None,
            "feature_dim": None,
            "feature_importance_by_block": dict.fromkeys(rollup_keys, 0.0),
        }
        if not path.exists():
            per_coin[sym] = entry
            continue
        try:
            with path.open("rb") as f:
                art = pickle.load(f)
            model = art.get("model") if isinstance(art, dict) else None
            entry["feature_version"] = int(art.get("feature_version") or 0)
            entry["feature_dim"] = int(art.get("feature_dim") or 0)
            if model is None or not hasattr(model, "feature_importances_"):
                per_coin[sym] = entry
                continue
            imp = list(model.feature_importances_)
            block_totals: dict[str, float] = dict.fromkeys(rollup_keys, 0.0)
            for i, val in enumerate(imp):
                block = _block_for_index(i)
                for rollup, members in IMPORTANCE_ROLLUP.items():
                    if block in members:
                        block_totals[rollup] += float(val)
                        break
            total = sum(block_totals.values()) or 1.0
            entry["feature_importance_by_block"] = {k: round(v / total * 100.0, 2) for k, v in sorted(block_totals.items(), key=lambda x: -x[1])}
        except Exception as exc:
            entry["error"] = str(exc)
        per_coin[sym] = entry

    return {
        "strategy_id": "day",
        "rollup_groups": rollup_keys,
        "models": per_coin,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


async def build_feature_freshness_report() -> dict[str, Any]:
    """Full per-symbol freshness + health report."""
    coins: dict[str, Any] = {}
    for sym in TRADING_SYMBOLS:
        ages = build_feature_age_by_block(sym)
        coins[sym] = {
            "feature_age_by_block": ages,
            "feature_health_score": build_feature_health_score(symbol_bus=sym, age_report=ages),
        }
    return {
        "thresholds_sec": freshness_thresholds_sec(),
        "cadence_env": {
            "DAY_AI_SIGNAL_LOOP_SEC": _env_int("DAY_AI_SIGNAL_LOOP_SEC", 120),
            "DAY_BUNDLE_CACHE_TTL_SEC": DAY_BUNDLE_CACHE_TTL_SEC,
            "AI_CONTEXT_LOOP_SEC": AI_CONTEXT_LOOP_SEC,
            "ORDERBOOK_REFRESH_INTERVAL_SEC": _env_int("ORDERBOOK_REFRESH_INTERVAL_SEC", 45),
            "MAX_SIGNAL_AGE_SEC": _env_int("MAX_SIGNAL_AGE_SEC", 300),
            "CTX_FRESH_MAX_AGE_SEC": CTX_FRESH_MAX_AGE_SEC,
        },
        "symbols": coins,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


__all__ = [
    "FEATURE_AGE_BLOCKS",
    "IMPORTANCE_ROLLUP",
    "build_feature_age_by_block",
    "build_feature_freshness_report",
    "build_feature_health_score",
    "build_feature_importance_by_block",
    "freshness_thresholds_sec",
]
