"""Decision-time CLOCK-V2 feature capture. Observability only.

Runs after production has already chosen BUY/HOLD. Fail-open.
Never ranks, sizes, authorizes, or exits. Never writes inspected=true.
Never zero-imputes missing market fields.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from backend.config.execution_cost_model import (
    expected_exchange_commission_rt_pct,
    expected_slippage_rt_pct,
    honest_all_in_rt_pct,
)
from backend.services.day_4h_entry_features import COINS, HOLD_SYMBOL
from backend.services.day_path_clock_dataset import in_sealed_lock, load_asof_1m_bars
from backend.services.day_path_clock_features import (
    build_clock_features,
    clip_asof,
    normalize_bars,
    parse_as_of,
    window_quality,
)
from backend.services.day_path_clock_pipeline import FORBIDDEN_OUTCOME_KEYS
from backend.services.day_path_clock_v2 import REQUIRED_CLOCK_V2_FIELDS, SCHEMA_VERSION
from backend.services.day_path_input_validity import MAX_GAP_SEC, MAX_LAST_BAR_AGE_SEC

logger = logging.getLogger(__name__)

TABLE_ARTIFACT = "day_path_clock_v2_candidate_artifact"
FEATURE_CONTRACT_VERSION = "day_path_clock_v2_capture_1"
KLINE_SOURCE = "redis_klines_1m"
FEATURE_OHLCV_SOURCE = "feature_ohlcv"
QUOTE_REASON_NO_QUOTE = "NO_VALID_DECISION_QUOTE"

NOT_COMPUTED_FOR_CANDIDATE = "NOT_COMPUTED_FOR_CANDIDATE"
NOT_PERSISTED = "NOT_PERSISTED"
SOURCE_DATA_GAP = "SOURCE_DATA_GAP"
SOURCE_DATA_STALE = "SOURCE_DATA_STALE"
NO_QUOTE = "NO_QUOTE"
CANDIDATE_INELIGIBLE = "CANDIDATE_INELIGIBLE"
SCHEMA_VERSION_MISSING = "SCHEMA_VERSION_MISSING"
OTHER = "OTHER"

MISSINGNESS_CATEGORIES = (
    NOT_COMPUTED_FOR_CANDIDATE,
    NOT_PERSISTED,
    SOURCE_DATA_GAP,
    SOURCE_DATA_STALE,
    NO_QUOTE,
    CANDIDATE_INELIGIBLE,
    SCHEMA_VERSION_MISSING,
    OTHER,
)

MISSINGNESS_BITS = {name: 1 << i for i, name in enumerate(REQUIRED_CLOCK_V2_FIELDS)}

CLOCK_FIELDS = (
    "ret_5m",
    "ret_15m",
    "ret_30m",
    "realized_vol_10m",
    "btc_rel_ret_5m",
    "rel_volume_15m",
)

SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_ARTIFACT} (
    decision_group_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    created_at TEXT NOT NULL,
    decision_timestamp TEXT NOT NULL,
    feature_schema_version TEXT NOT NULL,
    feature_contract_version TEXT NOT NULL,
    eligible INTEGER NOT NULL,
    eligibility_reason TEXT,
    production_p_buy REAL,
    shadow_candidate_p_buy REAL,
    p_buy_provenance TEXT,
    feature_json TEXT NOT NULL,
    missingness_bitmap INTEGER NOT NULL DEFAULT 0,
    missingness_reasons_json TEXT,
    provenance_json TEXT,
    quote_json TEXT,
    lock_window INTEGER NOT NULL DEFAULT 0,
    inspected INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (decision_group_id, symbol)
);
CREATE INDEX IF NOT EXISTS idx_clock_v2_artifact_created ON {TABLE_ARTIFACT}(created_at);
"""


def capture_enabled() -> bool:
    raw = os.getenv("DAY_CLOCK_V2_CAPTURE", "true")
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def ensure_artifact_schema(db_path: str | Path) -> None:
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _num(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def _api(symbol: str) -> str:
    s = str(symbol or "").strip().upper().replace("/", "").replace("-", "")
    return s


def _redis_client(existing: Any = None) -> Any:
    if existing is not None:
        return existing
    try:
        import redis

        url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
        return redis.from_url(url, decode_responses=True)
    except Exception:
        return None


def read_shadow_p_buy(symbol: str, redis_client: Any = None) -> tuple[float | None, str]:
    """Read frozen signal-hash p_buy. Never used for ranking."""
    if _api(symbol) == HOLD_SYMBOL:
        return None, NOT_COMPUTED_FOR_CANDIDATE
    r = _redis_client(redis_client)
    if r is None:
        return None, OTHER
    try:
        raw = r.hgetall(f"ai_signal:day:{_api(symbol)}") or {}
    except Exception:
        return None, OTHER
    for key in ("prob_buy", "p_buy", "winner_probability"):
        val = _num(raw.get(key))
        if val is not None:
            return val, "redis_signal"
    if raw:
        return None, NOT_COMPUTED_FOR_CANDIDATE
    return None, OTHER


def read_redis_klines_1m(symbol: str, *, as_of: Any, redis_client: Any = None) -> tuple[list[Any], dict[str, Any]]:
    """Authoritative live 1m cache. No REST. No forward-fill."""
    meta = {"source": KLINE_SOURCE, "raw_count": 0, "reason": None}
    if _api(symbol) == HOLD_SYMBOL:
        meta["reason"] = NOT_COMPUTED_FOR_CANDIDATE
        return [], meta
    r = _redis_client(redis_client)
    if r is None:
        meta["reason"] = OTHER
        return [], meta
    try:
        raw = r.get(f"klines:{_api(symbol)}:1m")
    except Exception:
        meta["reason"] = OTHER
        return [], meta
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not raw:
        meta["reason"] = SOURCE_DATA_GAP
        return [], meta
    try:
        rows = json.loads(raw)
    except Exception:
        meta["reason"] = OTHER
        return [], meta
    if not isinstance(rows, list):
        meta["reason"] = OTHER
        return [], meta
    when = parse_as_of(as_of) or datetime.now(timezone.utc)
    bars = clip_asof(normalize_bars(rows), when)
    meta["raw_count"] = len(bars)
    return bars, meta


def classify_clock_source(bars: list[Any], *, as_of: Any, lookback_sec: int) -> str | None:
    when = parse_as_of(as_of)
    if when is None:
        return OTHER
    if not bars:
        return SOURCE_DATA_GAP
    start = _as_utc(when) - timedelta(seconds=int(lookback_sec))
    quality = window_quality(bars, start=start, end=_as_utc(when))
    last_age = quality.get("latest_bar_age_seconds")
    max_gap = quality.get("max_gap_seconds")
    if last_age is None:
        return SOURCE_DATA_GAP
    if last_age > MAX_LAST_BAR_AGE_SEC:
        return SOURCE_DATA_STALE
    if max_gap is not None and max_gap > MAX_GAP_SEC:
        return SOURCE_DATA_GAP
    if not quality.get("valid"):
        return SOURCE_DATA_GAP
    return None


def read_decision_quote(symbol: str, redis_client: Any = None) -> dict[str, Any]:
    """Decision-time bid/ask only. Never substitutes estimated_all_in_cost_bps."""
    empty = {
        "best_bid": None,
        "best_ask": None,
        "mid": None,
        "spread_bps": None,
        "quote_timestamp": None,
        "quote_age_ms": None,
        "quote_source": None,
        "reason": QUOTE_REASON_NO_QUOTE if _api(symbol) != HOLD_SYMBOL else NOT_COMPUTED_FOR_CANDIDATE,
    }
    if _api(symbol) == HOLD_SYMBOL:
        return {
            **empty,
            "spread_bps": 0.0,
            "reason": None,
            "quote_source": "hold_zero",
        }
    try:
        from backend.services.decision_book_tape import snapshot_book

        book = snapshot_book(symbol, redis_client=redis_client)
    except Exception:
        return empty
    bid = _num(book.get("best_bid"))
    ask = _num(book.get("best_ask"))
    mid = _num(book.get("mid"))
    if bid is None or ask is None or bid <= 0 or ask <= 0 or ask < bid:
        return empty
    if mid is None or mid <= 0:
        mid = (bid + ask) / 2.0
    spread_bps = ((ask - bid) / mid) * 1e4 if mid > 0 else None
    age_sec = _num(book.get("book_age_sec"))
    ts = book.get("ts_utc")
    if ts in (None, "") and age_sec is not None:
        ts = datetime.fromtimestamp(datetime.now(timezone.utc).timestamp() - age_sec, tz=timezone.utc).isoformat()
    return {
        "best_bid": bid,
        "best_ask": ask,
        "mid": mid,
        "spread_bps": spread_bps,
        "quote_timestamp": ts or _now_iso(),
        "quote_age_ms": (age_sec * 1000.0) if age_sec is not None else None,
        "quote_source": book.get("book_source") or "snapshot_book",
        "reason": None,
    }


def _fourh_fields(contract: dict[str, Any], symbol: str) -> dict[str, Any]:
    group = contract.get("4h_entry_telemetry") or {}
    tel = dict(group.get(symbol) or {}) if isinstance(group, dict) else {}
    if not tel:
        for row in contract.get("candidates") or []:
            if str(row.get("symbol")) == symbol:
                tel = dict(row.get("4h_entry_telemetry") or {})
                break
    return {
        "production_4h_break_true_at_decision": tel.get("production_4h_break_true_at_decision", tel.get("production_4h_break_true_now")),
        "distance_to_4h_break_bps": tel.get("distance_to_4h_break_bps"),
        "4h_range_position": tel.get("4h_range_position"),
    }


def _candidate_row(contract: dict[str, Any], symbol: str) -> dict[str, Any]:
    for row in contract.get("candidates") or []:
        if str(row.get("symbol")) == symbol:
            return dict(row)
    return {}


def _p_buy_fields(row: dict[str, Any], symbol: str, redis_client: Any = None) -> dict[str, Any]:
    production = _num(row.get("p_buy"))
    shadow, shadow_src = read_shadow_p_buy(symbol, redis_client=redis_client)
    if production is not None:
        provenance = "production_candidate"
    elif shadow is not None:
        provenance = "redis_signal"
    elif symbol == HOLD_SYMBOL:
        provenance = "hold"
    else:
        provenance = "missing"
    return {
        "production_p_buy": production,
        "shadow_candidate_p_buy": shadow,
        "p_buy": production if production is not None else shadow,
        "p_buy_provenance": provenance,
        "shadow_source": shadow_src,
    }


def _clock_bars(
    symbol: str,
    *,
    as_of: Any,
    db_path: str | Path | None,
    redis_client: Any = None,
) -> tuple[list[Any], dict[str, Any]]:
    bars, meta = read_redis_klines_1m(symbol, as_of=as_of, redis_client=redis_client)
    if bars:
        return bars, meta
    if db_path:
        fallback = load_asof_1m_bars(db_path, symbol, as_of)
        parsed = clip_asof(normalize_bars(fallback), parse_as_of(as_of) or datetime.now(timezone.utc))
        if parsed:
            return parsed, {"source": FEATURE_OHLCV_SOURCE, "raw_count": len(parsed), "reason": None}
    return bars, meta


def _source_quality(bars: list[Any], *, as_of: Any) -> dict[str, Any]:
    when = parse_as_of(as_of) or datetime.now(timezone.utc)
    if not bars:
        return {
            "feature_cutoff_ts": when.isoformat(),
            "source_latest_ts": None,
            "source_age_seconds": None,
            "max_gap_seconds": None,
            "observation_count": 0,
            "feature_contract_version": FEATURE_CONTRACT_VERSION,
        }
    stamps = [b.ts for b in bars]
    gaps = [(stamps[i] - stamps[i - 1]).total_seconds() for i in range(1, len(stamps))]
    latest = stamps[-1]
    return {
        "feature_cutoff_ts": when.isoformat(),
        "source_latest_ts": latest.isoformat(),
        "source_age_seconds": (_as_utc(when) - latest).total_seconds(),
        "max_gap_seconds": max(gaps) if gaps else None,
        "observation_count": len(bars),
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
    }


def missingness_bitmap(missing_fields: list[str]) -> int:
    bits = 0
    for name in missing_fields:
        bits |= MISSINGNESS_BITS.get(name, 0)
    return bits


def classify_historical_p_buy(*, eligible: bool, p_buy: Any, exclusion_reason: str | None) -> str | None:
    if p_buy is not None:
        return None
    if not eligible and str(exclusion_reason or "") == "NO_SCORED_CANDIDATE":
        return NOT_PERSISTED
    if not eligible:
        return CANDIDATE_INELIGIBLE
    return NOT_COMPUTED_FOR_CANDIDATE


def classify_historical_clock(*, quality_reasons: list[str], persisted: bool) -> str:
    if not persisted:
        return NOT_PERSISTED
    if "stale_last_bar" in quality_reasons:
        return SOURCE_DATA_STALE
    if "gap_exceeded" in quality_reasons or "row_count" in quality_reasons:
        return SOURCE_DATA_GAP
    return SOURCE_DATA_GAP


def classify_historical_spread(*, contract_spread: Any, clock_spread: Any) -> str | None:
    if clock_spread is not None:
        return None
    if contract_spread is not None:
        return NOT_PERSISTED
    return NO_QUOTE


def build_candidate_artifact(
    contract: dict[str, Any],
    symbol: str,
    *,
    db_path: str | Path | None = None,
    redis_client: Any = None,
    as_of: Any = None,
) -> dict[str, Any]:
    """Pure builder. Does not mutate ``contract`` or any live decision."""
    row = _candidate_row(contract, symbol)
    when = parse_as_of(as_of or contract.get("decision_timestamp") or contract.get("created_at")) or datetime.now(timezone.utc)
    eligible = True if symbol == HOLD_SYMBOL else bool(row.get("eligible"))
    eligibility_reason = None if eligible else (row.get("exclusion_reason") or "INELIGIBLE")
    p_buy = _p_buy_fields(row, symbol, redis_client=redis_client)
    fourh = _fourh_fields(contract, symbol)
    quote = read_decision_quote(symbol, redis_client=redis_client)
    bars, bar_meta = ([], {"source": "hold", "raw_count": 0, "reason": NOT_COMPUTED_FOR_CANDIDATE})
    btc_bars: list[Any] = []
    btc_meta: dict[str, Any] = {}
    if symbol != HOLD_SYMBOL:
        bars, bar_meta = _clock_bars(symbol, as_of=when, db_path=db_path, redis_client=redis_client)
        btc_bars, btc_meta = _clock_bars("BTCUSDT", as_of=when, db_path=db_path, redis_client=redis_client)
    quality = _source_quality(bars, as_of=when)
    feats = build_clock_features(
        bars,
        as_of=when,
        symbol=symbol,
        btc_bars=btc_bars,
        p_buy=p_buy["p_buy"],
        legacy_path_ev=0.0 if symbol == HOLD_SYMBOL else row.get("path_ev", row.get("legacy_path_ev")),
        final_rank_score=0.0 if symbol == HOLD_SYMBOL else row.get("final_rank_score"),
        structure=fourh,
        quote_spread_bps=quote.get("spread_bps") if quote.get("spread_bps") is not None else None,
    )
    # Never substitute the cost model for an actual quote spread.
    if symbol == HOLD_SYMBOL:
        feats["spread_bps"] = 0.0
        feats["estimated_all_in_cost_bps"] = 0.0
    else:
        feats["spread_bps"] = quote.get("spread_bps")
        feats["estimated_all_in_cost_bps"] = honest_all_in_rt_pct(symbol) * 1e4
    feats["p_buy"] = p_buy["p_buy"]
    feats["production_p_buy"] = p_buy["production_p_buy"]
    feats["shadow_candidate_p_buy"] = p_buy["shadow_candidate_p_buy"]
    feats["p_buy_provenance"] = p_buy["p_buy_provenance"]
    for key, value in fourh.items():
        feats[key] = value
    if symbol != HOLD_SYMBOL:
        feats["expected_slippage_bps"] = expected_slippage_rt_pct() * 1e4
        feats["commission_rt_bps"] = expected_exchange_commission_rt_pct() * 1e4
    reasons: dict[str, str] = {}
    if symbol == HOLD_SYMBOL:
        for name in CLOCK_FIELDS:
            if feats.get(name) is None:
                reasons[name] = NOT_COMPUTED_FOR_CANDIDATE
        if p_buy["p_buy"] is None:
            reasons["p_buy"] = NOT_COMPUTED_FOR_CANDIDATE
    else:
        for name in CLOCK_FIELDS:
            if feats.get(name) is None:
                lookback = {
                    "ret_5m": 5 * 60,
                    "ret_15m": 15 * 60,
                    "ret_30m": 30 * 60,
                    "realized_vol_10m": 10 * 60,
                    "btc_rel_ret_5m": 5 * 60,
                    "rel_volume_15m": 15 * 60,
                }[name]
                src = btc_bars if name == "btc_rel_ret_5m" else bars
                reasons[name] = classify_clock_source(src, as_of=when, lookback_sec=lookback) or SOURCE_DATA_GAP
        if feats.get("p_buy") is None:
            reasons["p_buy"] = CANDIDATE_INELIGIBLE if not eligible else NOT_COMPUTED_FOR_CANDIDATE
        if feats.get("spread_bps") is None:
            reasons["spread_bps"] = NO_QUOTE
        if feats.get("legacy_path_ev") is None:
            reasons["legacy_path_ev"] = NOT_COMPUTED_FOR_CANDIDATE if eligible else CANDIDATE_INELIGIBLE
        if feats.get("final_rank_score") is None:
            reasons["final_rank_score"] = NOT_COMPUTED_FOR_CANDIDATE if eligible else CANDIDATE_INELIGIBLE
        for name in (
            "production_4h_break_true_at_decision",
            "distance_to_4h_break_bps",
            "4h_range_position",
            "estimated_all_in_cost_bps",
        ):
            if feats.get(name) is None:
                reasons[name] = OTHER
    missing = [name for name in REQUIRED_CLOCK_V2_FIELDS if feats.get(name) is None and not (symbol == HOLD_SYMBOL and name != "symbol")]
    if symbol == HOLD_SYMBOL:
        missing = [name for name in ("symbol",) if feats.get(name) != HOLD_SYMBOL]
    forbidden = FORBIDDEN_OUTCOME_KEYS.intersection(feats)
    if forbidden:
        raise RuntimeError(f"clock-v2 capture leaked outcome keys: {sorted(forbidden)}")
    artifact = {
        "decision_group_id": str(contract.get("decision_group_id") or ""),
        "symbol": symbol,
        "created_at": str(contract.get("decision_timestamp") or _now_iso()),
        "decision_timestamp": when.isoformat(),
        "feature_schema_version": SCHEMA_VERSION,
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "eligible": eligible,
        "eligibility_reason": eligibility_reason,
        "production_p_buy": p_buy["production_p_buy"],
        "shadow_candidate_p_buy": p_buy["shadow_candidate_p_buy"],
        "p_buy_provenance": p_buy["p_buy_provenance"],
        "features": feats,
        "missing_fields": missing,
        "missingness_bitmap": missingness_bitmap(missing),
        "missingness_reasons": reasons,
        "provenance": {
            **quality,
            "kline_source": bar_meta.get("source"),
            "btc_kline_source": btc_meta.get("source"),
            "kline_reason": bar_meta.get("reason"),
            "shadow_p_buy_source": p_buy["shadow_source"],
        },
        "quote": quote,
        "lock_window": in_sealed_lock(when.isoformat()),
        "inspected": False,
    }
    return artifact


def group_completeness(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    """FEATURE_COMPLETE uses eligible actions only. Ineligible coins stay visible."""
    by_sym = {str(a["symbol"]): a for a in artifacts}
    hold = by_sym.get(HOLD_SYMBOL)
    coins = [by_sym.get(s) for s in COINS]
    feature_ok = bool(hold and hold.get("symbol") == HOLD_SYMBOL)
    for art in coins:
        if art is None:
            feature_ok = False
            continue
        if art.get("eligible"):
            required_missing = [n for n in REQUIRED_CLOCK_V2_FIELDS if (art.get("features") or {}).get(n) is None]
            if required_missing:
                feature_ok = False
        # ineligible: no fake features required
    rectangular = True
    for art in coins:
        if art is None:
            rectangular = False
            continue
        feats = art.get("features") or {}
        if any(feats.get(n) is None for n in REQUIRED_CLOCK_V2_FIELDS):
            rectangular = False
    return {
        "FEATURE_COMPLETE": feature_ok,
        "LABEL_COMPLETE": False,
        "FULLY_COMPARABLE": False,
        "eligible_symbols": [s for s in COINS if by_sym.get(s, {}).get("eligible")] + [HOLD_SYMBOL],
        "ineligible_symbols": [s for s in COINS if by_sym.get(s) and not by_sym[s].get("eligible")],
        "rectangular_feature_complete": rectangular,
        "hold_available": True,
    }


def persist_group_artifacts(db_path: str | Path, artifacts: list[dict[str, Any]]) -> int:
    ensure_artifact_schema(db_path)
    conn = sqlite3.connect(str(db_path), timeout=30)
    written = 0
    try:
        for art in artifacts:
            if art.get("inspected"):
                raise RuntimeError("clock-v2 capture must never set inspected=true")
            feats = art.get("features") or {}
            if FORBIDDEN_OUTCOME_KEYS.intersection(feats) or FORBIDDEN_OUTCOME_KEYS.intersection(art):
                raise RuntimeError("clock-v2 capture must not persist outcomes")
            conn.execute(
                f"""
                INSERT OR REPLACE INTO {TABLE_ARTIFACT}(
                    decision_group_id, symbol, created_at, decision_timestamp,
                    feature_schema_version, feature_contract_version, eligible,
                    eligibility_reason, production_p_buy, shadow_candidate_p_buy,
                    p_buy_provenance, feature_json, missingness_bitmap,
                    missingness_reasons_json, provenance_json, quote_json,
                    lock_window, inspected
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0)
                """,
                (
                    art["decision_group_id"],
                    art["symbol"],
                    art["created_at"],
                    art["decision_timestamp"],
                    art["feature_schema_version"],
                    art["feature_contract_version"],
                    1 if art["eligible"] else 0,
                    art.get("eligibility_reason"),
                    art.get("production_p_buy"),
                    art.get("shadow_candidate_p_buy"),
                    art.get("p_buy_provenance"),
                    json.dumps(art.get("features") or {}, default=str),
                    int(art.get("missingness_bitmap") or 0),
                    json.dumps(art.get("missingness_reasons") or {}, default=str),
                    json.dumps(art.get("provenance") or {}, default=str),
                    json.dumps(art.get("quote") or {}, default=str),
                    1 if art.get("lock_window") else 0,
                ),
            )
            written += 1
        conn.commit()
    finally:
        conn.close()
    return written


def capture_clock_v2_group(
    db_path: str | Path,
    contract: dict[str, Any],
    *,
    redis_client: Any = None,
) -> dict[str, Any] | None:
    """Fail-open persistence. Never mutates ``contract``."""
    if not capture_enabled():
        return None
    if not db_path or not contract.get("decision_group_id"):
        return None
    artifacts = [build_candidate_artifact(contract, symbol, db_path=db_path, redis_client=redis_client) for symbol in (*COINS, HOLD_SYMBOL)]
    persist_group_artifacts(db_path, artifacts)
    completeness = group_completeness(artifacts)
    return {
        "decision_group_id": contract["decision_group_id"],
        "candidates": artifacts,
        "completeness": completeness,
        "inspected": False,
    }


def capture_clock_v2_fail_open(db_path: str | Path, contract: dict[str, Any]) -> None:
    try:
        capture_clock_v2_group(db_path, contract)
    except Exception as exc:
        logger.warning("DAY_CLOCK_V2_CAPTURE failed: %s", exc)


__all__ = [
    "FEATURE_CONTRACT_VERSION",
    "MISSINGNESS_CATEGORIES",
    "TABLE_ARTIFACT",
    "build_candidate_artifact",
    "capture_clock_v2_fail_open",
    "capture_clock_v2_group",
    "capture_enabled",
    "classify_historical_clock",
    "classify_historical_p_buy",
    "classify_historical_spread",
    "group_completeness",
    "read_decision_quote",
    "read_shadow_p_buy",
]
