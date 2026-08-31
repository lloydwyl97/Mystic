"""
Multi-horizon expected value (item p11).

Every EV number computed elsewhere in Mystic (portfolio_engine's
``_estimate_candidate_net_expected_value``, HoldEV) is implicitly tied to a
single, whatever's-currently-configured holding-period assumption. This
module adds a genuinely multi-horizon view: SCALP's realistic exit horizons
run roughly 30s-20m, DAY's run roughly 15m-24h. A candidate that looks great
assuming a 20m SCALP hold may have a much worse (or better) profile if it
actually gets held only 1m, or actually runs past 20m before exit logic acts.

Each horizon bucket's EV is computed from ONLY the historical trades whose
OWN realized ``hold_seconds`` fell inside that bucket (via
``mfe_mae_distribution_learner.hold_time_bucket`` / the same
``market_role_trade_outcomes`` table already powering the MFE/MAE
distribution learner and adaptive targets). No future information about the
*current* (still-open or not-yet-opened) candidate is used, and no horizon's
estimate borrows rows from another horizon's window — this is what "without
label leakage" means here: horizon buckets are historical, disjoint,
same-symbol/same-strategy strata, not a forward simulation of the live
candidate.

The composite score is additional ranking/diagnostic evidence — per the core
architecture rule it never gates entry on its own; it is exposed for
ranking/EV/confidence/sizing consumption exactly like the other p15-p22
additions (ai_context append-only field for DAY, scalp ranking-meta field for
SCALP).
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any

from backend.database_schema import DATABASE_PATH
from backend.services.mfe_mae_distribution_learner import get_expected_mfe_mae

logger = logging.getLogger(__name__)

# Ordered horizon buckets, using the exact same bucket names as
# mfe_mae_distribution_learner.hold_time_bucket() so both modules stay in
# lockstep on stratification. SCALP: ~30s-20m. DAY: ~15m-24h.
SCALP_HORIZON_BUCKETS: tuple[str, ...] = ("hold_lt_1m", "hold_1m_5m", "hold_5m_20m")
DAY_HORIZON_BUCKETS: tuple[str, ...] = ("hold_lt_15m", "hold_15m_4h", "hold_4h_24h")

_BUCKET_RANGES_SEC: dict[str, tuple[float, float]] = {
    "hold_lt_1m": (0.0, 60.0),
    "hold_1m_5m": (60.0, 300.0),
    "hold_5m_20m": (300.0, 1200.0),
    "hold_gt_20m": (1200.0, float("inf")),
    "hold_lt_15m": (0.0, 900.0),
    "hold_15m_4h": (900.0, 14400.0),
    "hold_4h_24h": (14400.0, 86400.0),
    "hold_gt_24h": (86400.0, float("inf")),
}


def _multi_horizon_enabled() -> bool:
    return str(os.getenv("MULTI_HORIZON_EV_ENABLED", "true")).strip().lower() in ("1", "true", "yes", "on")


def _default_weights(buckets: tuple[str, ...]) -> dict[str, float]:
    """Equal weighting by default; overridable via env for a specific bucket,
    e.g. MULTI_HORIZON_EV_WEIGHT_HOLD_1M_5M=0.5. Renormalized after any
    horizon is dropped for insufficient data."""
    out = {}
    for b in buckets:
        env_key = f"MULTI_HORIZON_EV_WEIGHT_{b.upper()}"
        out[b] = float(os.getenv(env_key, "1.0") or "1.0")
    return out


@dataclass(frozen=True)
class HorizonEV:
    bucket: str
    low_sec: float
    high_sec: float
    win_rate: float
    expected_mfe_pct: float
    expected_mae_pct: float
    net_ev_pct: float
    n_obs: int
    confidence_status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "bucket": self.bucket,
            "low_sec": self.low_sec,
            "high_sec": None if self.high_sec == float("inf") else self.high_sec,
            "win_rate": round(self.win_rate, 4),
            "expected_mfe_pct": round(self.expected_mfe_pct, 6),
            "expected_mae_pct": round(self.expected_mae_pct, 6),
            "net_ev_pct": round(self.net_ev_pct, 6),
            "n_obs": self.n_obs,
            "confidence_status": self.confidence_status,
        }


@dataclass(frozen=True)
class MultiHorizonEVResult:
    symbol: str
    strategy: str
    available: bool
    horizons: tuple[HorizonEV, ...] = field(default_factory=tuple)
    composite_ev_pct: float | None = None
    degraded_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "strategy": self.strategy,
            "available": self.available,
            "horizons": [h.to_dict() for h in self.horizons],
            "composite_ev_pct": round(self.composite_ev_pct, 6) if self.composite_ev_pct is not None else None,
            "degraded_reason": self.degraded_reason,
        }


def _win_rate_for_bucket(symbol: str, strategy: str, bucket: str, *, db_path: str, lookback_days: int) -> tuple[float, int]:
    """Real empirical win rate among rows whose OWN hold_seconds falls in
    `bucket`'s range — same disjoint-by-realized-outcome stratification as
    the MFE/MAE distribution learner, just counting wins vs losses instead of
    excursion percentiles."""
    low, high = _BUCKET_RANGES_SEC.get(bucket, (0.0, float("inf")))
    since_epoch = time.time() - lookback_days * 86400
    since_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(since_epoch))
    try:
        from backend.services.ai_canonical_storage import _symbol_variants_for_lookup

        sym_variants = _symbol_variants_for_lookup(symbol)
    except Exception:
        sym_variants = [symbol.upper()]
    try:
        with sqlite3.connect(db_path, timeout=10) as conn:
            placeholders = ", ".join("?" for _ in sym_variants)
            rows = conn.execute(
                f"""
                SELECT realized_pnl_pct, hold_seconds
                FROM market_role_trade_outcomes
                WHERE strategy = ? AND symbol IN ({placeholders}) AND created_at >= ?
                ORDER BY created_at DESC
                LIMIT 2000
                """,
                (strategy.lower(), *sym_variants, since_iso),
            ).fetchall()
    except Exception as exc:
        logger.debug("MULTI_HORIZON_WIN_RATE_FETCH_FAILED symbol=%s bucket=%s: %s", symbol, bucket, exc)
        return 0.0, 0

    in_bucket = [pnl for pnl, hold_s in rows if hold_s is not None and low <= float(hold_s) < high]
    n = len(in_bucket)
    if n == 0:
        return 0.0, 0
    wins = sum(1 for pnl in in_bucket if pnl is not None and float(pnl) > 0)
    return wins / n, n


def compute_multi_horizon_ev(
    symbol: str,
    strategy: str,
    *,
    cost_pct: float = 0.0015,
    db_path: str = DATABASE_PATH,
    lookback_days: int = 45,
) -> MultiHorizonEVResult:
    """Composite, cost-aware EV across several realistic same-strategy
    holding horizons. `cost_pct` is the caller's own estimated fee+slippage+
    spread round-trip cost (kept as an explicit argument rather than
    hardcoded so DAY/SCALP each pass their own live cost estimate)."""
    strategy_l = strategy.lower()
    if not _multi_horizon_enabled():
        return MultiHorizonEVResult(symbol=symbol.upper(), strategy=strategy_l, available=False, degraded_reason="disabled")

    buckets = SCALP_HORIZON_BUCKETS if strategy_l == "scalp" else DAY_HORIZON_BUCKETS
    weights = _default_weights(buckets)

    horizons: list[HorizonEV] = []
    for bucket in buckets:
        win_rate, n_wr = _win_rate_for_bucket(symbol, strategy_l, bucket, db_path=db_path, lookback_days=lookback_days)
        excursion = get_expected_mfe_mae(symbol, strategy_l, hold_bucket_filter=bucket, db_path=db_path)
        n_obs = min(n_wr, excursion.mfe_n_obs, excursion.mae_n_obs) if n_wr else 0
        confidence = "insufficient_data"
        if excursion.mfe_confidence == excursion.mae_confidence:
            confidence = excursion.mfe_confidence
        elif "insufficient_data" in (excursion.mfe_confidence, excursion.mae_confidence):
            confidence = "insufficient_data"
        else:
            confidence = "low_confidence"

        low, high = _BUCKET_RANGES_SEC.get(bucket, (0.0, float("inf")))
        if n_obs == 0 or confidence == "insufficient_data":
            horizons.append(
                HorizonEV(
                    bucket=bucket,
                    low_sec=low,
                    high_sec=high,
                    win_rate=win_rate,
                    expected_mfe_pct=0.0,
                    expected_mae_pct=0.0,
                    net_ev_pct=0.0,
                    n_obs=n_obs,
                    confidence_status="insufficient_data",
                )
            )
            continue

        net_ev = (win_rate * excursion.expected_mfe_p60) - ((1.0 - win_rate) * excursion.expected_mae_p60) - cost_pct
        horizons.append(
            HorizonEV(
                bucket=bucket,
                low_sec=low,
                high_sec=high,
                win_rate=win_rate,
                expected_mfe_pct=excursion.expected_mfe_p60,
                expected_mae_pct=excursion.expected_mae_p60,
                net_ev_pct=net_ev,
                n_obs=n_obs,
                confidence_status=confidence,
            )
        )

    usable = [h for h in horizons if h.confidence_status != "insufficient_data"]
    if not usable:
        return MultiHorizonEVResult(
            symbol=symbol.upper(),
            strategy=strategy_l,
            available=False,
            horizons=tuple(horizons),
            degraded_reason="insufficient_data_all_horizons",
        )

    weight_sum = sum(weights[h.bucket] for h in usable) or 1.0
    composite = sum(h.net_ev_pct * weights[h.bucket] for h in usable) / weight_sum

    return MultiHorizonEVResult(
        symbol=symbol.upper(),
        strategy=strategy_l,
        available=True,
        horizons=tuple(horizons),
        composite_ev_pct=composite,
        degraded_reason=None if len(usable) == len(horizons) else "partial_horizon_coverage",
    )


_SNAPSHOT_CACHE: dict[tuple[str, str], tuple[MultiHorizonEVResult, float]] = {}
_SNAPSHOT_CACHE_TTL_SEC = float(os.getenv("MULTI_HORIZON_EV_CACHE_TTL_SEC", "300") or "300")


def cached_multi_horizon_ev(symbol: str, strategy: str, **kwargs: Any) -> MultiHorizonEVResult:
    """TTL-cached wrapper for hot loops (SCALP evaluates every few seconds;
    this statistic changes on the order of minutes/hours, not seconds).
    DAY's ai_market_context loop calls compute_multi_horizon_ev directly
    since it already runs on a 60s+ cadence."""
    key = (symbol.upper(), strategy.lower())
    now = time.time()
    cached = _SNAPSHOT_CACHE.get(key)
    if cached is not None and cached[1] > now:
        return cached[0]
    result = compute_multi_horizon_ev(symbol, strategy, **kwargs)
    _SNAPSHOT_CACHE[key] = (result, now + _SNAPSHOT_CACHE_TTL_SEC)
    return result


__all__ = [
    "DAY_HORIZON_BUCKETS",
    "SCALP_HORIZON_BUCKETS",
    "HorizonEV",
    "MultiHorizonEVResult",
    "cached_multi_horizon_ev",
    "compute_multi_horizon_ev",
]
