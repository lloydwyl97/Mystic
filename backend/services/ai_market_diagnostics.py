"""
Read-only AI market understanding diagnostics for the DAY top-4 engine.

Does not change trading rules, thresholds, or execution paths.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import pickle
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.config.day_active_timeframes import DAY_ACTIVE_TIMEFRAMES, min_bars_for_day_tf
from backend.config.trading_universe import TRADING_SYMBOLS
from backend.database_schema import DATABASE_PATH
from backend.services.ai_decision_contract import (
    AI_FEATURE_DIM_V1,
    AI_FEATURE_DIM_V2,
    CONTEXT_DIMS_DAY_FULL,
    EXTERNAL_API_SENTIMENT_FEATURES,
    LEGACY_SENTIMENT_FEATURE_NAMES,
)
from backend.services.day_active_market_bundle import validate_day_active_bundle
from backend.services.feature_mapping import FEATURE_MAPPING, get_feature_name
from backend.services.live_strategy_contracts import per_coin_artifact_file

logger = logging.getLogger(__name__)

MODEL_STALE_HOURS = 72

# Numeric zeros that are valid indicator/time values — not missing-data signals.
LEGITIMATE_ZERO_FEATURE_NAMES: frozenset[str] = frozenset(
    {
        "second",
        "minute",
        "balance_of_power",
        "volume_price_trend",
        "volume_ratio",
        "volatility_ratio",
        "aroon_up",
        "aroon_down",
        "williams_r",
        "price_impact",
    }
)

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

SENTIMENT_SLOT_NAMES = list(LEGACY_SENTIMENT_FEATURE_NAMES)


def _feature_names_145() -> list[str]:
    names = [get_feature_name(i) for i in range(1, AI_FEATURE_DIM_V1 + 1)]
    names.extend(list(CONTEXT_DIMS_DAY_FULL))
    return names


def _block_for_index(idx0: int) -> str:
    if idx0 >= AI_FEATURE_DIM_V1:
        return "context_day_full"
    one_based = idx0 + 1
    for block, (lo, hi) in FEATURE_BLOCKS.items():
        if lo <= one_based <= hi:
            return block
    return "unknown"


def _orderbook_features_from_redis(symbol_bus: str, base_symbol: str) -> dict[str, float] | None:
    """Parse live orderbook hash the same way as ai_signal_generator._assemble_day_live_features."""
    ob_raw = _redis_hgetall(f"orderbook:{base_symbol}") or _redis_hgetall(f"orderbook:{symbol_bus}")
    if not ob_raw:
        return None
    out: dict[str, float] = {}
    for k, v in ob_raw.items():
        with contextlib.suppress(TypeError, ValueError):
            out[str(k)] = float(v)
    return out or None


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


def _read_bundle_cache_sync(symbol_bus: str) -> dict[str, Any] | None:
    try:
        from backend.config.redis_config import get_shared_redis_sync
        from backend.services.day_active_market_bundle import DAY_BUNDLE_CACHE_PREFIX, _normalize_ccxt_symbol

        r = get_shared_redis_sync()
        if not r:
            return None
        ccxt = _normalize_ccxt_symbol(symbol_bus)
        key = f"{DAY_BUNDLE_CACHE_PREFIX}{ccxt.replace('/', '')}"
        raw = r.get(key)
        if not raw:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode()
        payload = json.loads(raw)
        if isinstance(payload, dict) and "bundle" in payload:
            bundle_raw = payload.get("bundle")
        else:
            bundle_raw = payload
        if not isinstance(bundle_raw, dict):
            return None
        bundle: dict[str, Any] = {}
        for tf in DAY_ACTIVE_TIMEFRAMES:
            rows = bundle_raw.get(tf)
            bundle[tf] = list(rows) if isinstance(rows, list) else []
        if isinstance(bundle_raw.get("_month_vec"), list):
            bundle["_month_vec"] = list(bundle_raw["_month_vec"])
        validate_day_active_bundle(bundle)  # type: ignore[arg-type]
        return bundle
    except Exception:
        return None


def _sentiment_env_status() -> dict[str, Any]:
    news_key = bool(os.getenv("NEWS_API_KEY", "").strip())
    reddit_id = bool(os.getenv("REDDIT_CLIENT_ID", "").strip())
    reddit_secret = bool(os.getenv("REDDIT_CLIENT_SECRET", "").strip())
    social_active = reddit_id and reddit_secret
    return {
        "NEWS_API_KEY": {"configured": news_key, "active": news_key},
        "REDDIT_CLIENT_ID": {"configured": reddit_id, "active": social_active},
        "REDDIT_CLIENT_SECRET": {"configured": reddit_secret, "active": social_active},
        "fear_greed_api": {"configured": True, "active": True, "note": "alternative.me via ai_market_context"},
        "inactive_slots_when_keys_missing": [n for n in SENTIMENT_SLOT_NAMES if n in EXTERNAL_API_SENTIMENT_FEATURES or n != "fear_greed_index"],
    }


async def build_feature_completeness_report() -> dict[str, Any]:
    """Per-coin feature pipeline completeness (read-only)."""
    coins: dict[str, Any] = {}
    for sym in TRADING_SYMBOLS:
        ccxt = f"{sym[:-4]}/USDT" if sym.endswith("USDT") else sym
        sig = _redis_hgetall(f"ai_signal:day:{sym}")
        ctx = _redis_hgetall(f"ai_context:{sym}")
        fv = int(float(sig.get("feature_version") or 0))
        fd = int(float(sig.get("feature_dim") or 0))

        bundle = _read_bundle_cache_sync(sym)
        bundle_ok, bundle_missing = (False, ["bundle_cache_miss"])
        mtf_status: dict[str, Any] = {}
        zero_count = None
        zero_features: list[str] = []
        inactive_sentiment: list[str] = []
        missing_technical: list[str] = []
        missing_context: list[str] = []
        orderbook_available = False
        if bundle:
            bundle_ok, bundle_missing = validate_day_active_bundle(dict(bundle))
            if "_month_vec" not in bundle:
                missing_technical.append("day_bundle_missing_month_vec")
            for tf in DAY_ACTIVE_TIMEFRAMES:
                rows = bundle.get(tf) if isinstance(bundle.get(tf), list) else []
                need = min_bars_for_day_tf(tf)
                mtf_status[tf] = {"bars": len(rows), "required": need, "ok": len(rows) >= need}

        optional_inactive: list[str] = []
        canonical_sentiment_paths: dict[str, str] = {}
        sentiment: dict[str, Any] = {}
        try:
            from backend.config.redis_config import get_shared_redis_async
            from backend.services.ai_day_htf_features import build_day_htf_feature_vector_145
            from backend.services.ai_feature_fundamentals import merge_canonical_sentiment_payload
            from backend.services.optional_feature_slots import is_optional_slot
            from backend.utils.canonical_symbol_formatter import CanonicalSymbolFormatter

            if ctx.get("ctx_sentiment_fear_greed"):
                with contextlib.suppress(TypeError, ValueError):
                    sentiment["fear_greed_index"] = float(ctx["ctx_sentiment_fear_greed"])
            sent_hash_pre = _redis_hgetall(f"ai_sentiment:{sym}")
            if not sentiment.get("fear_greed_index") and sent_hash_pre.get("fear_greed_index"):
                with contextlib.suppress(TypeError, ValueError):
                    sentiment["fear_greed_index"] = float(sent_hash_pre["fear_greed_index"])

            redis_async = get_shared_redis_async()
            base = CanonicalSymbolFormatter.to_base(sym)
            sentiment = await merge_canonical_sentiment_payload(
                base_symbol=base,
                pair_symbol=sym,
                ctx_for_overlay=ctx or None,
                redis_client=redis_async,
                ohlcv_1m=bundle.get("1m") if bundle else [],
                existing=sentiment or None,
            )
            try:
                from backend.config.redis_config import get_shared_redis_sync

                r_sync = get_shared_redis_sync()
                if r_sync:
                    status_raw = r_sync.get(f"mystic:canonical_sentiment_status:{base}")
                    if status_raw:
                        if isinstance(status_raw, bytes):
                            status_raw = status_raw.decode()
                        st = json.loads(status_raw)
                        if isinstance(st, dict):
                            canonical_sentiment_paths = {
                                "social_path": str(st.get("social_path") or ""),
                                "news_path": str(st.get("news_path") or ""),
                            }
            except Exception:
                pass

            for slot in SENTIMENT_SLOT_NAMES:
                if slot in sentiment and sentiment[slot] is not None:
                    continue
                if is_optional_slot(slot):
                    optional_inactive.append(slot)
                else:
                    inactive_sentiment.append(slot)

            if bundle and bundle.get("1m"):
                orderbook_for_vec = _orderbook_features_from_redis(sym, base)
                vec = build_day_htf_feature_vector_145(
                    symbol_ccxt=ccxt,
                    day_bundle=bundle,
                    volume_profile=None,
                    orderbook=orderbook_for_vec,
                    sentiment=sentiment or None,
                    ai_context=ctx or None,
                )
                names = _feature_names_145()
                zero_count = 0
                for i, v in enumerate(vec):
                    if abs(float(v)) < 1e-12:
                        nm = names[i] if i < len(names) else f"dim_{i}"
                        if nm in LEGITIMATE_ZERO_FEATURE_NAMES:
                            continue
                        zero_count += 1
                        if len(zero_features) < 40:
                            zero_features.append(nm)
                for i, name in enumerate(CONTEXT_DIMS_DAY_FULL):
                    if abs(float(vec[AI_FEATURE_DIM_V1 + i])) < 1e-12:
                        missing_context.append(name)
        except Exception as exc:
            missing_technical.append(f"vector_build_error:{exc}")

        ob = _redis_hgetall(f"orderbook:{sym}") or _redis_hgetall(f"orderbook:{sym[:-4]}")
        orderbook_available = bool(ob.get("bid_ask_spread") and float(ob.get("bid_ask_spread") or 0) > 0)
        sent_hash = _redis_hgetall(f"ai_sentiment:{sym}")
        month_vec_ok = bool(bundle and bundle.get("_month_vec"))
        sentiment_ok = bool(sent_hash.get("sentiment_ts_utc") and sent_hash.get("sentiment_stale") == "no")
        source_missing_count = len([x for x in (sent_hash.get("sentiment_sources_missing") or "").split(",") if x])

        coins[sym] = {
            "feature_version": fv,
            "feature_dim": fd,
            "zero_filled_count": zero_count,
            "zero_filled_sample": zero_features[:20],
            "inactive_sentiment_slots": inactive_sentiment,
            "optional_inactive_slots": optional_inactive,
            "inactive_slot_count": len(inactive_sentiment),
            "optional_inactive_count": len(optional_inactive),
            "canonical_sentiment_paths": canonical_sentiment_paths,
            "merged_sentiment_preview": {k: round(float(sentiment[k]), 6) if k in sentiment else None for k in ("fear_greed_index", "social_sentiment", "news_sentiment", "market_dominance", "vix")},
            "source_missing_count": source_missing_count,
            "missing_technical_blocks": missing_technical,
            "missing_context_dims": missing_context[:21],
            "mtf_bundle_completeness": {"ok": bundle_ok, "missing": bundle_missing, "timeframes": mtf_status},
            "month_vec_ok": month_vec_ok,
            "orderbook_features_available": orderbook_available,
            "orderbook_ok": orderbook_available,
            "sentiment_ok": sentiment_ok,
            "sentiment_redis": {
                "social_sentiment_score": sent_hash.get("social_sentiment_score"),
                "news_sentiment_score": sent_hash.get("news_sentiment_score"),
                "fear_greed_index": sent_hash.get("fear_greed_index"),
                "sentiment_ts_utc": sent_hash.get("sentiment_ts_utc"),
                "sentiment_stale": sent_hash.get("sentiment_stale"),
                "sources_active": sent_hash.get("sentiment_sources_active"),
                "sources_missing": sent_hash.get("sentiment_sources_missing"),
            },
            "sentiment_news_available": bool(os.getenv("NEWS_API_KEY", "").strip()),
            "social_sentiment_available": bool(os.getenv("REDDIT_CLIENT_ID", "").strip() and os.getenv("REDDIT_CLIENT_SECRET", "").strip()),
            "ai_context_present": bool(ctx),
            "signal_present": bool(sig),
        }
        try:
            from backend.services.ai_feature_freshness_diagnostics import (
                build_feature_age_by_block,
                build_feature_health_score,
            )

            age_report = build_feature_age_by_block(sym)
            coins[sym]["feature_age_by_block"] = age_report
            coins[sym]["feature_health_score"] = build_feature_health_score(
                symbol_bus=sym,
                age_report=age_report,
                zero_filled_count=zero_count or 0,
                inactive_slot_count=len(inactive_sentiment),
                optional_inactive_count=len(optional_inactive),
                missing_context_dims=missing_context,
            )
        except Exception as exc:
            coins[sym]["feature_age_by_block_error"] = str(exc)[:200]

    return {"symbols": coins, "sentiment_status": _sentiment_env_status(), "generated_at": datetime.now(timezone.utc).isoformat()}


def build_feature_importance_report() -> dict[str, Any]:
    """Per-coin RF feature importance from active artifacts."""
    models_dir = Path("models/active")
    names = _feature_names_145()
    per_coin: dict[str, Any] = {}

    for sym in TRADING_SYMBOLS:
        path = per_coin_artifact_file(models_dir, "day", sym)
        entry: dict[str, Any] = {
            "path": str(path),
            "exists": path.exists(),
            "feature_version": None,
            "feature_dim": None,
            "accuracy": None,
            "trained_at": None,
            "model_age_hours": None,
            "top_20": [],
            "zero_importance_count": 0,
            "zero_importance_sample": [],
            "block_importance_totals": {},
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
            entry["accuracy"] = round(float(art.get("accuracy") or 0), 4)
            trained_at = str(art.get("trained_at") or "")
            entry["trained_at"] = trained_at
            if trained_at:
                try:
                    dt = datetime.fromisoformat(trained_at.replace("Z", "+00:00"))
                    entry["model_age_hours"] = round((datetime.now(timezone.utc) - dt).total_seconds() / 3600, 1)
                except (ValueError, TypeError):
                    pass
            if model is not None and hasattr(model, "feature_importances_"):
                imp = list(model.feature_importances_)
                dim = len(imp)
                pairs = []
                block_totals: dict[str, float] = dict.fromkeys(list(FEATURE_BLOCKS) + ["context_day_full"], 0.0)
                for i, val in enumerate(imp):
                    nm = names[i] if i < len(names) else f"dim_{i}"
                    pairs.append((nm, float(val)))
                    block = _block_for_index(i)
                    block_totals[block] = block_totals.get(block, 0.0) + float(val)
                pairs.sort(key=lambda x: x[1], reverse=True)
                entry["top_20"] = [{"feature": n, "importance": round(v, 6)} for n, v in pairs[:20]]
                zeros = [(n, v) for n, v in pairs if v <= 1e-12]
                entry["zero_importance_count"] = len(zeros)
                entry["zero_importance_sample"] = [n for n, _ in zeros[:25]]
                total = sum(block_totals.values()) or 1.0
                entry["block_importance_totals"] = {k: round(v / total * 100, 2) for k, v in sorted(block_totals.items(), key=lambda x: -x[1])}
        except Exception as exc:
            entry["error"] = str(exc)
        per_coin[sym] = entry

    return {"strategy_id": "day", "models": per_coin, "generated_at": datetime.now(timezone.utc).isoformat()}


def build_model_freshness_report(db_path: str = DATABASE_PATH) -> dict[str, Any]:
    """Model age, stale flags, promotion status — no auto-promotion."""
    models_dir = Path("models/active")
    version_dir = Path("models/versions/per_coin")
    stale_threshold_h = MODEL_STALE_HOURS
    coins: dict[str, Any] = {}

    ensure_tables = False
    try:
        from backend.services.ai_canonical_storage import ensure_ai_canonical_tables

        ensure_ai_canonical_tables(db_path)
        ensure_tables = True
    except Exception:
        pass

    for sym in TRADING_SYMBOLS:
        active = per_coin_artifact_file(models_dir, "day", sym)
        meta: dict[str, Any] = {
            "active_path": str(active),
            "exists": active.exists(),
            "trained_at": None,
            "age_hours": None,
            "stale": None,
            "accuracy": None,
            "feature_version": None,
            "feature_dim": None,
            "retrain_candidate_available": False,
            "latest_candidate_path": None,
            "promotion_status": "unknown",
        }
        if active.exists():
            try:
                with active.open("rb") as f:
                    art = pickle.load(f)
                trained_at = str(art.get("trained_at") or "")
                meta["trained_at"] = trained_at
                meta["accuracy"] = round(float(art.get("accuracy") or 0), 4)
                meta["feature_version"] = int(art.get("feature_version") or 0)
                meta["feature_dim"] = int(art.get("feature_dim") or 0)
                if trained_at:
                    dt = datetime.fromisoformat(trained_at.replace("Z", "+00:00"))
                    age_h = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
                    meta["age_hours"] = round(age_h, 1)
                    meta["stale"] = age_h > stale_threshold_h
            except Exception as exc:
                meta["error"] = str(exc)

        pattern = sorted(version_dir.glob(f"day_{sym}_*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
        if pattern:
            meta["retrain_candidate_available"] = True
            meta["latest_candidate_path"] = str(pattern[0])

        if ensure_tables:
            try:
                with sqlite3.connect(db_path) as conn:
                    row = conn.execute(
                        """
                        SELECT event_type, reason, created_at FROM ai_model_promotion_events
                        WHERE strategy_id='day' AND symbol=?
                        ORDER BY id DESC LIMIT 1
                        """,
                        (sym,),
                    ).fetchone()
                    if row:
                        meta["promotion_status"] = f"{row[0]}:{row[1]} @ {row[2]}"
                    elif meta["exists"]:
                        meta["promotion_status"] = "active_deployed"
            except sqlite3.Error:
                pass

        coins[sym] = meta

    return {
        "stale_threshold_hours": stale_threshold_h,
        "auto_promote_disabled": True,
        "models": coins,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _normalize_regime(label: str | None) -> str:
    if not label:
        return "chop"
    s = str(label).strip().lower()
    if s in ("bull", "trend_up", "trending_up", "uptrend", "risk_on"):
        return "trending_up"
    if s in ("bear", "trend_down", "trending_down", "downtrend", "risk_off"):
        return "trending_down"
    if "up" in s:
        return "trending_up"
    if "down" in s:
        return "trending_down"
    return "chop"


def build_regime_performance_report(db_path: str = DATABASE_PATH) -> dict[str, Any]:
    """Outcome stats grouped by symbol and regime."""
    rows_out: list[dict[str, Any]] = []
    reject_rates: dict[str, Any] = {}
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            outcome_rows = conn.execute(
                """
                SELECT symbol, context_json, net_pnl_pct, realized_pct, hold_seconds,
                       good_bad_memory_class, gross_pnl_pct
                FROM ai_outcome_training_rows
                WHERE strategy_id='day'
                ORDER BY id DESC LIMIT 2000
                """
            ).fetchall()

            buckets: dict[tuple[str, str], dict[str, Any]] = {}
            active_symbols = set(TRADING_SYMBOLS) | {s.replace("USDT", "/USDT") for s in TRADING_SYMBOLS}
            for r in outcome_rows:
                sym = str(r["symbol"] or "")
                if sym not in active_symbols and sym.replace("/", "").upper() not in set(TRADING_SYMBOLS):
                    continue
                regime = "chop"
                try:
                    ctx = json.loads(r["context_json"] or "{}")
                    if isinstance(ctx, dict):
                        regime = _normalize_regime(ctx.get("ctx_market_regime") or ctx.get("market_regime"))
                except (json.JSONDecodeError, TypeError):
                    pass
                key = (sym, regime)
                b = buckets.setdefault(
                    key,
                    {"symbol": sym, "regime": regime, "wins": 0, "losses": 0, "net_sum": 0.0, "hold_sum": 0.0, "count": 0},
                )
                net = float(r["net_pnl_pct"] or r["realized_pct"] or r["gross_pnl_pct"] or 0)
                b["count"] += 1
                b["net_sum"] += net
                b["hold_sum"] += float(r["hold_seconds"] or 0)
                mem = str(r["good_bad_memory_class"] or "").upper()
                if net > 0 or mem == "GOOD":
                    b["wins"] += 1
                else:
                    b["losses"] += 1

            for b in buckets.values():
                c = max(1, b["count"])
                rows_out.append(
                    {
                        "symbol": b["symbol"],
                        "regime": b["regime"],
                        "trades": b["count"],
                        "wins": b["wins"],
                        "losses": b["losses"],
                        "avg_net_profit_pct": round(b["net_sum"] / c, 6),
                        "avg_hold_seconds": round(b["hold_sum"] / c, 1),
                        "repair_add_used": None,
                    }
                )

            repair = conn.execute(
                """
                SELECT symbol, COUNT(*) FROM paper_trades
                WHERE side='BUY' AND explainability_json LIKE '%repair_add%'
                GROUP BY symbol
                """
            ).fetchall()
            repair_map = {str(a): int(b) for a, b in repair}
            for row in rows_out:
                row["repair_add_used"] = repair_map.get(row["symbol"], 0) > 0

            rej = conn.execute(
                """
                SELECT symbol, COUNT(*) as cnt FROM portfolio_engine_rejects
                WHERE side='BUY' AND filter_name='PROTECTED_PREFLIGHT'
                GROUP BY symbol
                """
            ).fetchall()
            attempts = conn.execute(
                """
                SELECT symbol, COUNT(*) as cnt FROM portfolio_engine_rejects
                WHERE side='BUY'
                GROUP BY symbol
                """
            ).fetchall()
            att_map = {str(a): int(b) for a, b in attempts}
            for sym, cnt in rej:
                sym_s = str(sym)
                att = att_map.get(sym_s, 0) or 1
                reject_rates[sym_s] = round(cnt / att, 4)
    except Exception as exc:
        return {"error": str(exc), "rows": [], "protected_preflight_reject_rate_by_symbol": {}}

    rows_out.sort(key=lambda x: (x["symbol"], x["regime"]))
    return {
        "rows": rows_out,
        "protected_preflight_reject_rate_by_symbol": reject_rates,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def build_outcome_quality_audit(db_path: str = DATABASE_PATH, limit: int = 50) -> dict[str, Any]:
    """Verify closed-trade learning capture quality."""
    limit = max(1, min(200, int(limit)))
    audits: list[dict[str, Any]] = []
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            sells = conn.execute(
                """
                SELECT trade_id, symbol, timestamp, pnl, pnl_pct, explainability_json,
                       exit_type, entry_timestamp, strategy_id
                FROM paper_trades
                WHERE side='SELL' AND pnl IS NOT NULL
                  AND COALESCE(exit_type,'') NOT IN ('ADMIN_POSITION_CLEAR', 'STALE_PRE_CORRECTION_POSITION_CLEAR')
                ORDER BY timestamp DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()

            for s in sells:
                sym = str(s["symbol"] or "")
                explain = {}
                try:
                    explain = json.loads(s["explainability_json"] or "{}")
                except (json.JSONDecodeError, TypeError):
                    pass
                entry_ts = s["entry_timestamp"]
                oc = conn.execute(
                    """
                    SELECT id, features_json, context_json, good_bad_memory_class, strategy_id
                    FROM ai_outcome_training_rows
                    WHERE symbol=? ORDER BY id DESC LIMIT 3
                    """,
                    (sym,),
                ).fetchall()
                outcome_match = None
                for o in oc:
                    if o["features_json"]:
                        outcome_match = dict(o)
                        break
                tl = conn.execute(
                    """
                    SELECT id, close_reason, extra_json FROM trade_learning_outcomes
                    WHERE symbol=? ORDER BY id DESC LIMIT 1
                    """,
                    (sym.replace("/", "") if False else sym,),
                ).fetchone()

                pnl = float(s["pnl"] or 0)
                good_label = pnl > 0
                gb = (outcome_match or {}).get("good_bad_memory_class")
                label_ok = gb is None or (str(gb).upper() == "GOOD") == good_label or (str(gb).upper() == "BAD") == (not good_label)

                audits.append(
                    {
                        "trade_id": s["trade_id"],
                        "symbol": sym,
                        "exit_type": s["exit_type"],
                        "pnl": pnl,
                        "entry_features_captured": bool(outcome_match and outcome_match.get("features_json")),
                        "entry_context_captured": bool(outcome_match and outcome_match.get("context_json")),
                        "model_version_captured": bool(explain.get("feature_version") or explain.get("artifact_sha256")),
                        "protected_preflight_captured": "protected_preflight" in json.dumps(explain).lower() or "preflight" in json.dumps(explain).lower(),
                        "executable_fill_captured": bool(explain.get("entry_price") or explain.get("fill_price")),
                        "learning_row_written": tl is not None,
                        "outcome_row_written": outcome_match is not None,
                        "good_bad_label_correct": label_ok,
                        "issues": [
                            name
                            for name, ok in [
                                ("missing_entry_features", bool(outcome_match and outcome_match.get("features_json"))),
                                ("missing_entry_context", bool(outcome_match and outcome_match.get("context_json"))),
                                ("missing_model_version", bool(explain.get("feature_version"))),
                                ("missing_learning_row", tl is not None),
                                ("label_mismatch", label_ok),
                            ]
                            if not ok
                        ],
                    }
                )
    except Exception as exc:
        return {"error": str(exc), "audits": []}

    return {
        "audits": audits,
        "summary": {
            "total": len(audits),
            "all_features_ok": sum(1 for a in audits if a["entry_features_captured"]),
            "all_learning_ok": sum(1 for a in audits if a["learning_row_written"]),
            "label_correct": sum(1 for a in audits if a["good_bad_label_correct"]),
            "missing_model_version_count": sum(1 for a in audits if "missing_model_version" in a.get("issues", [])),
            "audit_pass_count": sum(1 for a in audits if not a.get("issues")),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def build_sentiment_slot_status() -> dict[str, Any]:
    """Report external sentiment feed availability — never fabricates values."""
    from backend.services.ai_active_sentiment_collector import REDIS_SENTIMENT_STATUS_KEY, env_source_config
    from backend.services.discord_social_sentiment_live import discord_readiness

    discord_env = discord_readiness()
    redis_status = _redis_hgetall(REDIS_SENTIMENT_STATUS_KEY)
    discord_fetch_ok = (redis_status.get("discord_fetch_ok") or "").lower() == "yes"
    cfg = env_source_config(discord_fetch_ok=discord_fetch_ok)
    status = _sentiment_env_status()
    per_symbol: dict[str, Any] = {}
    for sym in TRADING_SYMBOLS:
        h = _redis_hgetall(f"ai_sentiment:{sym}")
        if h:
            per_symbol[sym] = {
                "social_sentiment_score": h.get("social_sentiment_score"),
                "reddit_sentiment_score": h.get("reddit_sentiment_score"),
                "discord_sentiment_score": h.get("discord_sentiment_score"),
                "discord_message_count": h.get("discord_message_count"),
                "discord_matched_count": h.get("discord_matched_count"),
                "discord_ts_utc": h.get("discord_ts_utc"),
                "discord_stale": h.get("discord_stale"),
                "discord_error": h.get("discord_error"),
                "telegram_sentiment_score": h.get("telegram_sentiment_score"),
                "telegram_message_count": h.get("telegram_message_count"),
                "telegram_matched_count": h.get("telegram_matched_count"),
                "telegram_ts_utc": h.get("telegram_ts_utc"),
                "telegram_error": h.get("telegram_error"),
                "news_sentiment_score": h.get("news_sentiment_score"),
                "fear_greed_index": h.get("fear_greed_index"),
                "sentiment_ts_utc": h.get("sentiment_ts_utc"),
                "sentiment_stale": h.get("sentiment_stale"),
                "sources_active": h.get("sentiment_sources_active"),
                "sources_missing": h.get("sentiment_sources_missing"),
            }
    slots: list[dict[str, Any]] = []
    for name in SENTIMENT_SLOT_NAMES:
        if name == "fear_greed_index":
            state = "active"
            source = "alternative.me / ai_market_context"
        elif name == "news_sentiment":
            state = "active" if cfg["news"]["enabled"] else "inactive_missing_api_key"
            source = "NEWS_API_KEY"
        elif name == "social_sentiment":
            state = "active" if cfg["reddit"]["enabled"] or cfg["telegram"]["enabled"] or cfg["discord"].get("read_active") else "inactive_missing_api_key"
            source = "reddit+telegram+discord collector"
        else:
            state = "inactive_not_wired"
            source = "none"
        slots.append({"slot": name, "status": state, "source": source})
    return {
        "sources": {
            "reddit_active": cfg["reddit"]["enabled"],
            "discord_active": cfg["discord"].get("read_active"),
            "discord_read_ready": cfg["discord"].get("read_ready"),
            "discord_configured": cfg["discord"].get("configured"),
            "discord_bot_configured": cfg["discord"].get("bot_configured"),
            "discord_channel_configured": cfg["discord"].get("channel_configured"),
            "discord_webhook_only": cfg["discord"].get("webhook_only"),
            "telegram_active": cfg["telegram"]["enabled"] and cfg["telegram"]["configured"],
            "news_active": cfg["news"]["enabled"],
            "fear_greed_active": True,
            "twitter_disabled": True,
        },
        "discord_env": {
            "bot_configured": discord_env["bot_configured"],
            "channel_configured": discord_env["channel_configured"],
            "channel_env_configured": discord_env.get("channel_env_configured"),
            "channel_webhook_resolvable": discord_env.get("channel_webhook_resolvable"),
            "read_ready": discord_env["read_ready"],
            "webhook_only": discord_env["webhook_only"],
            "last_error": redis_status.get("discord_last_error") or "",
            "channel_source": redis_status.get("discord_channel_source") or "",
            "fetch_ok": discord_fetch_ok,
            "message_count": redis_status.get("discord_message_count") or "0",
            "ts_utc": redis_status.get("discord_ts_utc") or "",
        },
        "redis_status": redis_status,
        "per_symbol": per_symbol,
        "slots": slots,
        "env": status,
        "never_fabricated": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


async def build_full_ai_diagnostics_report(db_path: str = DATABASE_PATH) -> dict[str, Any]:
    from backend.services.ai_feature_freshness_diagnostics import (
        build_feature_freshness_report,
        build_feature_importance_by_block,
    )
    from backend.services.ai_missed_opportunity_observer import get_missed_opportunity_report
    from backend.services.ai_post_trade_feature_review import get_post_trade_feature_review_report

    return {
        "feature_completeness": await build_feature_completeness_report(),
        "feature_freshness": await build_feature_freshness_report(),
        "feature_importance": build_feature_importance_report(),
        "feature_importance_by_block": build_feature_importance_by_block(),
        "model_freshness": build_model_freshness_report(db_path),
        "regime_performance": build_regime_performance_report(db_path),
        "outcome_quality": build_outcome_quality_audit(db_path),
        "post_trade_feature_reviews": get_post_trade_feature_review_report(db_path=db_path),
        "missed_opportunities": get_missed_opportunity_report(db_path=db_path),
        "sentiment_status": build_sentiment_slot_status(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


__all__ = [
    "build_feature_completeness_report",
    "build_feature_importance_report",
    "build_full_ai_diagnostics_report",
    "build_model_freshness_report",
    "build_outcome_quality_audit",
    "build_regime_performance_report",
    "build_sentiment_slot_status",
]
