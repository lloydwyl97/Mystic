"""SCALP feature audit — truth metadata for scalp intelligence vector."""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from typing import Any

from backend.services.binance_scalp.config import get_scalp_config
from backend.services.binance_scalp.market_reader import ScalpMarketReader
from backend.services.binance_scalp.momentum_tracker import MomentumTracker
from backend.services.binance_scalp.scalp_regime_classifier import classify_scalp_regime
from backend.services.binance_scalp.strategies.kline_cache import KlineCache
from backend.services.scalp_feature_contract import (
    SCALP_FEATURE_DIM,
    SCALP_FEATURE_NAMES,
    SCALP_FEATURE_VERSION,
    SCALP_TRUST_SCORES,
    _block_for_index,
    build_scalp_feature_vector,
)

BAD_STATUSES: frozenset[str] = frozenset(
    {"FALLBACK", "MISSING", "STALE", "ZERO_DEFAULT", "UNSUPPORTED_FOR_SPOT", "PLACEHOLDER"}
)
PASS_STATUSES: frozenset[str] = frozenset({"LIVE", "CALCULATED", "CALCULATED_PROXY", "WARMUP"})

ORDERBOOK_FRESH_SEC = 45.0
KLINE_WARMUP_BARS = 5

_SHARED_KLINE_CACHE = KlineCache()
_SHARED_READERS: dict[str, ScalpMarketReader] = {}


def _shared_reader(cfg=None) -> ScalpMarketReader:
    cfg = cfg or get_scalp_config()
    key = str(cfg.redis_url)
    if key not in _SHARED_READERS:
        _SHARED_READERS[key] = ScalpMarketReader(cfg)
    return _SHARED_READERS[key]


def _feature_status(name: str, value: float, *, ob_age: float | None, kline_bars: int) -> tuple[str, str, float, bool]:
    if name in ("best_bid", "best_ask", "mid_price") and value <= 0:
        return "MISSING", "market_reader depth", 0.0, False
    if name == "orderbook_age_sec":
        if ob_age is None:
            return "CALCULATED", "rest depth fetch", 0.85, True
        if ob_age > ORDERBOOK_FRESH_SEC:
            return "STALE", f"orderbook age {ob_age:.1f}s", SCALP_TRUST_SCORES["STALE"], False
        return "LIVE", "orderbook redis/rest", SCALP_TRUST_SCORES["LIVE"], True
    if name.startswith("kline_") and kline_bars < KLINE_WARMUP_BARS:
        return "WARMUP", f"1m bars={kline_bars}", SCALP_TRUST_SCORES["WARMUP"], False
    if name.startswith("kline_"):
        return "CALCULATED", "1m kline cache", SCALP_TRUST_SCORES["CALCULATED"], True
    if name in ("spread_pct", "order_book_imbalance", "redis_spread_pct"):
        if ob_age is not None and ob_age > ORDERBOOK_FRESH_SEC:
            return "STALE", "stale orderbook overlay", SCALP_TRUST_SCORES["STALE"], False
        return "LIVE", "orderbook:{BASE} read-only", SCALP_TRUST_SCORES["LIVE"], True
    if name.startswith("mid_change") or name.startswith("bid_change") or name in (
        "momentum_confirmed",
        "flat_regime",
        "recent_range_pct",
        "realized_volatility_pct",
        "last_n_ticks_up_count",
        "momentum_sample_count",
    ):
        if value == 0.0 and name == "momentum_sample_count":
            return "WARMUP", "momentum tracker warming", SCALP_TRUST_SCORES["WARMUP"], False
        return "CALCULATED", "MomentumTracker in-memory", SCALP_TRUST_SCORES["CALCULATED"], True
    if name in ("projected_gross_pct", "breakout_signal", "surplus_pct", "adx_1h", "atr_1h_pct"):
        return "CALCULATED_PROXY", "gross estimate / 1h regime proxy", SCALP_TRUST_SCORES["CALCULATED_PROXY"], False
    if name == "micro_regime_score":
        return "CALCULATED", "scalp_regime_classifier", SCALP_TRUST_SCORES["CALCULATED"], True
    if name in ("impact_pct", "depth_sufficient_flag", "expected_move_pct", "signal_score", "signal_confidence", "required_target_pct"):
        return "CALCULATED", "strategy signal economics", SCALP_TRUST_SCORES["CALCULATED"], True
    if name == "same_setup_today_count":
        return "CALCULATED_PROXY", "rolling_scalp_market_state redis", SCALP_TRUST_SCORES["CALCULATED_PROXY"], False
    return "CALCULATED", "scalp feature builder", SCALP_TRUST_SCORES["CALCULATED"], True


def build_symbol_scalp_audit(
    symbol_bus: str,
    *,
    snap: Any | None = None,
    mom_diag: Any | None = None,
    klines: KlineCache | None = None,
    reader: ScalpMarketReader | None = None,
) -> dict[str, Any]:
    cfg = get_scalp_config()
    reader = reader or _shared_reader(cfg)
    klines = klines or _SHARED_KLINE_CACHE
    sym = symbol_bus.upper().replace("/", "")
    if snap is None:
        snap = reader.read(sym)
    if snap is None:
        return {"symbol": sym, "error": "market_snapshot_missing", "features": [], "pass": False}

    _, epoch = datetime.now(timezone.utc).isoformat(), time.time()
    if mom_diag is None:
        mom = MomentumTracker()
        mom.record(sym, epoch, snap.best_bid, snap.mid)
        mom_diag = mom.diagnostics(sym, epoch, snap.best_bid, snap.mid)
    bars = klines.get(sym, minutes=30) or []
    bars_1h = klines.get_1h(sym) or []
    regime = "chop"
    gross = {"adx_1h": 25.0, "atr_1h_pct": 0.01}
    if len(bars_1h) >= 31:
        from backend.services.binance_scalp.scalp_regime_classifier import _atr, classify_scalp_regime

        st = classify_scalp_regime(bars_1h, len(bars_1h) - 1)
        if st:
            regime = st.regime
            gross["adx_1h"] = st.adx
            gross["atr_1h_pct"] = st.atr_pct

    vec = build_scalp_feature_vector(
        snap=snap,
        mom=mom_diag,
        bars_1m=bars,
        micro_regime=regime,
        gross=gross,
    )
    ob_age = float(snap.orderbook_age_sec) if snap.orderbook_age_sec is not None else None
    features: list[dict[str, Any]] = []
    for idx0, val in enumerate(vec):
        name = SCALP_FEATURE_NAMES[idx0]
        block = _block_for_index(idx0)
        st, src, trust, learning = _feature_status(name, float(val), ob_age=ob_age, kline_bars=len(bars))
        if st == "STALE":
            trust *= 0.25
        features.append(
            {
                "index": idx0 + 1,
                "name": name,
                "block": block,
                "value": round(float(val), 8),
                "source": src,
                "status": st,
                "age_seconds": ob_age if block == "microstructure" else None,
                "trust_score": round(trust, 4),
                "learning_allowed": learning and st in ("LIVE", "CALCULATED"),
                "is_real": st in ("LIVE", "CALCULATED", "CALCULATED_PROXY"),
            }
        )

    bad = [f for f in features if f["status"] in BAD_STATUSES or (f["status"] == "WARMUP" and abs(f["value"]) < 1e-12)]
    return {
        "symbol": sym,
        "feature_version": SCALP_FEATURE_VERSION,
        "feature_dim": SCALP_FEATURE_DIM,
        "micro_regime": regime,
        "orderbook_age_sec": ob_age,
        "kline_bars_1m": len(bars),
        "features": features,
        "bad_features": bad,
        "pass": len(bad) == 0 and len(features) == SCALP_FEATURE_DIM,
    }


def build_feature_health_sidecar(audit: dict[str, Any]) -> dict[str, Any]:
    feats = list(audit.get("features") or [])
    good = sum(1 for f in feats if f.get("status") in PASS_STATUSES)
    hp = round(100.0 * good / max(1, len(feats)), 2)
    return {
        "pass": audit.get("pass", False),
        "health_pct": hp,
        "feature_version": SCALP_FEATURE_VERSION,
        "feature_dim": SCALP_FEATURE_DIM,
        "features": feats,
    }


async def run_scalp_feature_audit(symbols: list[str] | None = None) -> dict[str, Any]:
    cfg = get_scalp_config()
    syms = symbols or list(cfg.products)
    per: dict[str, Any] = {}
    for s in syms:
        per[s] = build_symbol_scalp_audit(s)
    return {
        "feature_version": SCALP_FEATURE_VERSION,
        "feature_dim": SCALP_FEATURE_DIM,
        "symbols": syms,
        "per_symbol": per,
        "pass": all((per[s] or {}).get("pass") for s in syms),
    }


__all__ = [
    "BAD_STATUSES",
    "build_feature_health_sidecar",
    "build_symbol_scalp_audit",
    "run_scalp_feature_audit",
]
