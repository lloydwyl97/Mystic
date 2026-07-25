"""
ai_regime_validation — checks whether a regime label actually predicts forward
returns, and produces a trust scalar used to scale regime-conditioned trading
bonuses instead of trusting the label blindly.

Background: day_controlled_exits.py grants bull-regime positions a wider
trailing stop, a longer max hold, more lenient giveback thresholds, and a
stall-exit suppression — all keyed off `position.day_route_regime_at_entry`
(the DAY router's per-symbol classify_day_regime() output). That label had
never been checked against what actually happens to price afterward.

Two independently-detected regime layers exist in ai_candidate_snapshots:
  - "regime"            -> global Fear&Greed bull/bear/sideways (market_regime.py)
  - "day_route_regime"  -> per-symbol DAY router bull/range/bear/chop/neutral
                            (added by ai_learning_ingestion.py; accumulates
                            going forward, starts empty on older rows)

Both columns already have forward returns computed for every row
(fwd_ret_1h / fwd_ret_4h) via the existing self-supervised labeling pipeline —
no new historical replay engineering needed, just cross-tabulation.

Local dataset check (2026-07-25): global regime="bull" showed win_rate_1h
~48.6% (n=2,741) — essentially no edge over a coin flip for near-term forward
return. That is the concrete evidence this module encodes: until day-router
"bull" specifically proves an edge with real samples, its bonuses are only
partially trusted, not eliminated (paper trading still explores the tuning).
"""

from __future__ import annotations

import logging
import sqlite3
import time
from typing import Any

from backend.database_schema import DATABASE_PATH

logger = logging.getLogger(__name__)

MIN_VALIDATION_SAMPLES = int(__import__("os").getenv("AI_REGIME_VALIDATION_MIN_SAMPLES", "200"))
_CACHE_TTL_SEC = 1800.0
_cache: dict[tuple[str, str], tuple[float, float, dict[str, Any]]] = {}


def _query_regime_forward_stats(column: str, label: str, db_path: str = DATABASE_PATH) -> dict[str, Any]:
    """Return {n, win_rate_1h, avg_ret_1h, avg_ret_4h} for LABELED snapshots matching label."""
    if column not in ("regime", "day_route_regime"):
        raise ValueError(f"unsupported regime column: {column}")
    empty = {"n": 0, "win_rate_1h": None, "avg_ret_1h": None, "avg_ret_4h": None}
    if not label:
        return empty
    try:
        with sqlite3.connect(db_path, timeout=5) as conn:
            row = conn.execute(
                f"""
                SELECT COUNT(*) n,
                       AVG(CASE WHEN fwd_ret_1h > 0 THEN 1.0 ELSE 0.0 END) win_rate_1h,
                       AVG(fwd_ret_1h) avg_ret_1h,
                       AVG(fwd_ret_4h) avg_ret_4h
                FROM ai_candidate_snapshots
                WHERE label_status = 'LABELED' AND {column} = ? AND fwd_ret_1h IS NOT NULL
                """,
                (label,),
            ).fetchone()
    except Exception as e:
        logger.debug("REGIME_VALIDATION_QUERY_FAILED: column=%s label=%s (%s)", column, label, e)
        return empty
    if row is None or row[0] is None:
        return empty
    n, win_rate, avg_1h, avg_4h = row
    return {
        "n": int(n or 0),
        "win_rate_1h": float(win_rate) if win_rate is not None else None,
        "avg_ret_1h": float(avg_1h) if avg_1h is not None else None,
        "avg_ret_4h": float(avg_4h) if avg_4h is not None else None,
    }


def _scalar_from_stats(stats: dict[str, Any], min_samples: int) -> float:
    """Map observed win-rate/avg-return to a trust scalar in [0.35, 1.00].

    Insufficient data -> 1.0 (no change to existing behaviour; a bonus is only
    ever discounted once there is real evidence it isn't earning its keep,
    never boosted beyond the coded default).
    """
    n = stats.get("n") or 0
    if n < min_samples:
        return 1.0
    win_rate = stats.get("win_rate_1h")
    avg_ret = stats.get("avg_ret_1h")
    if win_rate is None or avg_ret is None:
        return 1.0
    if win_rate >= 0.55 and avg_ret > 0:
        return 1.0
    if win_rate >= 0.50 and avg_ret >= 0:
        return 0.70
    if win_rate >= 0.48:
        return 0.55
    return 0.35


def get_regime_validated_scalar(
    day_route_regime_label: str,
    *,
    min_samples: int = MIN_VALIDATION_SAMPLES,
    db_path: str = DATABASE_PATH,
) -> tuple[float, dict[str, Any]]:
    """Validated-edge trust scalar for a day_route_regime bonus (e.g. "bull").

    Prefers the DAY router's own label once it has enough samples; falls back
    to the global Fear&Greed label of the same name as best-available evidence
    while day_route_regime accumulates real samples. Cached in-process for
    _CACHE_TTL_SEC since this is called from the hot exit-check path.
    """
    label = (day_route_regime_label or "").strip().lower()
    if not label:
        return 1.0, {"source": "none"}

    now = time.time()
    cache_key = ("day_route_regime", label)
    cached = _cache.get(cache_key)
    if cached and (now - cached[1]) < _CACHE_TTL_SEC:
        return cached[0], cached[2]

    stats = _query_regime_forward_stats("day_route_regime", label, db_path=db_path)
    source = "day_route_regime"
    if stats["n"] < min_samples:
        fb_stats = _query_regime_forward_stats("regime", label, db_path=db_path)
        if fb_stats["n"] >= stats["n"]:
            stats = fb_stats
            source = "global_regime_fallback"

    scalar = _scalar_from_stats(stats, min_samples=min_samples)
    detail = {"source": source, **stats, "scalar": scalar}
    _cache[cache_key] = (scalar, now, detail)
    logger.info(
        "REGIME_VALIDATED_SCALAR: label=%s source=%s n=%d win_rate_1h=%s avg_ret_1h=%s -> scalar=%.2f",
        label,
        source,
        stats["n"],
        stats.get("win_rate_1h"),
        stats.get("avg_ret_1h"),
        scalar,
    )
    return scalar, detail


def blend_by_scalar(base: float, bonus_value: float, scalar: float) -> float:
    """Linear-interpolate between 'no bonus' (base, scalar=0) and 'full bonus' (bonus_value, scalar=1)."""
    s = max(0.0, min(1.0, float(scalar)))
    return float(base) + (float(bonus_value) - float(base)) * s


__all__ = [
    "MIN_VALIDATION_SAMPLES",
    "blend_by_scalar",
    "get_regime_validated_scalar",
]
