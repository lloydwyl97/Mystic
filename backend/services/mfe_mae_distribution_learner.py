"""Cross-engine MFE/MAE distribution learning (item p5 of the institutional upgrade).

Both DAY (`portfolio_engine.py`) and SCALP (`binance_scalp/paper_engine.py`)
already write every closed trade's MFE (`mfe_pct`), MAE (`mae_pct`),
`realized_pnl_pct`, `hold_seconds`, `market_regime`, `volatility_score`, and
`momentum_score` into the shared `market_role_trade_outcomes` table
(`market_role_outcome_learner.record_trade_outcome`). This module adds real
percentile-distribution learning on top of that existing, already-populated
table — no new write path, no new ingestion plumbing.

Stratification cascades from most to least specific until a stratum has
enough samples (mirrors the sample-size-backoff pattern already used by
`day_adaptive_trail.py` and `market_role_outcome_learner.get_learning_stats`):

    (strategy, symbol, vol_bucket, momentum_bucket)
      -> (strategy, symbol, vol_bucket)
      -> (strategy, symbol)
      -> (strategy)                      # cross-symbol fallback

MFE percentiles are computed over WINNERS only (realized_pnl_pct > 0) since
losers never built a meaningful favorable excursion. MAE percentiles are
computed over LOSERS only (realized_pnl_pct <= 0) for the mirror-image
reason — winners' adverse excursion is not representative of "how far
against you can this arm go before it's a real loss."

This module never gates a trade. It is a pure statistics provider consumed
by adaptive targets (day_controlled_exits / scalp exit tuning), adaptive
MAE/loss handling, and the HoldEV continuous exit signal.
"""

from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any

from backend.database_schema import DATABASE_PATH

MIN_OBS: int = int(os.getenv("MFE_MAE_DIST_MIN_OBS", "8"))
LOOKBACK_DAYS: int = int(os.getenv("MFE_MAE_DIST_LOOKBACK_DAYS", "45"))
_CACHE_TTL_SEC: float = 120.0
_cache: dict[str, tuple[float, DistributionResult]] = {}

_PERCENTILES: tuple[float, ...] = (0.10, 0.25, 0.50, 0.60, 0.75, 0.80, 0.90)


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    p = max(0.0, min(1.0, p))
    idx = p * (len(sorted_vals) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def vol_bucket(volatility_score: float | None) -> str | None:
    """Tertile-style bucket. volatility_score is the same 0..1-ish scale
    already written into market_role_trade_outcomes.volatility_score."""
    if volatility_score is None:
        return None
    v = float(volatility_score)
    lo = float(os.getenv("MFE_MAE_VOL_LOW_MAX", "0.25"))
    hi = float(os.getenv("MFE_MAE_VOL_HIGH_MIN", "0.55"))
    if v < lo:
        return "low_vol"
    if v > hi:
        return "high_vol"
    return "mid_vol"


def momentum_bucket(momentum_score: float | None) -> str | None:
    """momentum_score is centered at 0.5 (neutral) in the shared schema."""
    if momentum_score is None:
        return None
    m = float(momentum_score)
    if m < 0.40:
        return "momentum_down"
    if m > 0.60:
        return "momentum_up"
    return "momentum_flat"


def hold_time_bucket(hold_seconds: float | None, strategy: str) -> str | None:
    if hold_seconds is None:
        return None
    h = float(hold_seconds)
    if strategy.lower() == "scalp":
        if h < 60:
            return "hold_lt_1m"
        if h < 300:
            return "hold_1m_5m"
        if h < 1200:
            return "hold_5m_20m"
        return "hold_gt_20m"
    if h < 900:
        return "hold_lt_15m"
    if h < 14400:
        return "hold_15m_4h"
    if h < 86400:
        return "hold_4h_24h"
    return "hold_gt_24h"


@dataclass(frozen=True)
class DistributionResult:
    n_obs: int
    percentiles: dict[str, float]  # "p10".."p90" -> value
    confidence_status: str  # insufficient_data / low_confidence / confident
    stratum_used: str
    fallback_from: str = ""


def _confidence_status(n: int) -> str:
    full = max(MIN_OBS * 4, MIN_OBS + 1)
    if n < MIN_OBS:
        return "insufficient_data"
    if n < full:
        return "low_confidence"
    return "confident"


def _build_result(vals: list[float], *, stratum_used: str, fallback_from: str = "") -> DistributionResult:
    n = len(vals)
    sv = sorted(vals)
    pct = {f"p{int(p * 100)}": round(_percentile(sv, p), 6) for p in _PERCENTILES}
    return DistributionResult(
        n_obs=n,
        percentiles=pct,
        confidence_status=_confidence_status(n),
        stratum_used=stratum_used,
        fallback_from=fallback_from,
    )


def _empty_result(stratum_used: str = "none") -> DistributionResult:
    return DistributionResult(
        n_obs=0,
        percentiles={f"p{int(p * 100)}": 0.0 for p in _PERCENTILES},
        confidence_status="insufficient_data",
        stratum_used=stratum_used,
    )


def _fetch_rows(
    *,
    strategy: str,
    symbol: str | None,
    winners_only: bool,
    losers_only: bool,
    db_path: str,
    lookback_days: int,
) -> list[tuple[float, float, float]]:
    """Return (mfe_pct, mae_pct, volatility_score/momentum via caller filter already applied) rows.

    Filtering by vol_bucket/momentum_bucket happens in Python (small result
    sets — outcome tables are pruned to OUTCOME_RETENTION_DAYS=60 already)
    rather than SQL, since the bucket boundaries are logic, not stored columns.
    """
    since_iso_epoch = time.time() - lookback_days * 86400
    since_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(since_iso_epoch))
    clauses = ["strategy = ?", "created_at >= ?"]
    params: list[Any] = [strategy.lower(), since_iso]
    if symbol:
        try:
            from backend.services.ai_canonical_storage import _symbol_variants_for_lookup

            sym_variants = _symbol_variants_for_lookup(symbol)
        except Exception:
            sym_variants = [symbol.upper()]
        clauses.append("symbol IN (" + ", ".join("?" for _ in sym_variants) + ")")
        params.extend(sym_variants)
    if winners_only:
        clauses.append("realized_pnl_pct > 0")
    elif losers_only:
        clauses.append("realized_pnl_pct <= 0")
    where = " AND ".join(clauses)
    try:
        with sqlite3.connect(db_path, timeout=10) as conn:
            rows = conn.execute(
                f"""
                SELECT mfe_pct, mae_pct, volatility_score, momentum_score, hold_seconds
                FROM market_role_trade_outcomes
                WHERE {where}
                ORDER BY created_at DESC
                LIMIT 2000
                """,
                params,
            ).fetchall()
    except Exception:
        return []
    return rows or []


def _apply_bucket_filters(
    rows: list[tuple],
    *,
    vol_b: str | None,
    mom_b: str | None,
    hold_b: str | None,
    strategy: str,
) -> list[tuple]:
    out = []
    for r in rows:
        _mfe, _mae, vscore, mscore, hold_s = r
        if vol_b is not None and vol_bucket(vscore) != vol_b:
            continue
        if mom_b is not None and momentum_bucket(mscore) != mom_b:
            continue
        if hold_b is not None and hold_time_bucket(hold_s, strategy) != hold_b:
            continue
        out.append(r)
    return out


def _cache_key(*parts: Any) -> str:
    return "|".join(str(p) for p in parts)


def get_mfe_distribution(
    symbol: str,
    strategy: str,
    *,
    vol_bucket_filter: str | None = None,
    momentum_bucket_filter: str | None = None,
    hold_bucket_filter: str | None = None,
    db_path: str = DATABASE_PATH,
    lookback_days: int = LOOKBACK_DAYS,
) -> DistributionResult:
    """Percentile distribution of MFE among WINNING trades, cascading from
    the most specific requested stratification down to (strategy) alone."""
    return _get_distribution(
        symbol,
        strategy,
        field_idx=0,
        winners_only=True,
        losers_only=False,
        vol_bucket_filter=vol_bucket_filter,
        momentum_bucket_filter=momentum_bucket_filter,
        hold_bucket_filter=hold_bucket_filter,
        db_path=db_path,
        lookback_days=lookback_days,
    )


def get_mae_distribution(
    symbol: str,
    strategy: str,
    *,
    vol_bucket_filter: str | None = None,
    momentum_bucket_filter: str | None = None,
    hold_bucket_filter: str | None = None,
    db_path: str = DATABASE_PATH,
    lookback_days: int = LOOKBACK_DAYS,
) -> DistributionResult:
    """Percentile distribution of MAE among LOSING trades (same cascade)."""
    return _get_distribution(
        symbol,
        strategy,
        field_idx=1,
        winners_only=False,
        losers_only=True,
        vol_bucket_filter=vol_bucket_filter,
        momentum_bucket_filter=momentum_bucket_filter,
        hold_bucket_filter=hold_bucket_filter,
        db_path=db_path,
        lookback_days=lookback_days,
    )


def _get_distribution(
    symbol: str,
    strategy: str,
    *,
    field_idx: int,
    winners_only: bool,
    losers_only: bool,
    vol_bucket_filter: str | None,
    momentum_bucket_filter: str | None,
    hold_bucket_filter: str | None,
    db_path: str,
    lookback_days: int,
) -> DistributionResult:
    if not symbol or not str(symbol).strip():
        # An empty/missing symbol must never silently fall through to an
        # unfiltered (all-symbols) query — that would look like a
        # symbol-specific stratum while actually being cross-symbol data.
        return _empty_result(stratum_used="no_symbol")

    ck = _cache_key(
        "mfe" if field_idx == 0 else "mae",
        strategy,
        symbol,
        vol_bucket_filter,
        momentum_bucket_filter,
        hold_bucket_filter,
    )
    now = time.time()
    cached = _cache.get(ck)
    if cached and (now - cached[0]) < _CACHE_TTL_SEC:
        return cached[1]

    all_symbol_rows = _fetch_rows(
        strategy=strategy,
        symbol=symbol,
        winners_only=winners_only,
        losers_only=losers_only,
        db_path=db_path,
        lookback_days=lookback_days,
    )

    # Cascade: (symbol, vol, mom, hold) -> (symbol, vol, mom) -> (symbol, vol) -> (symbol) -> (strategy-only)
    cascades: list[tuple[str, dict[str, str | None]]] = []
    if vol_bucket_filter or momentum_bucket_filter or hold_bucket_filter:
        cascades.append(("symbol+vol+mom+hold", {"vol_b": vol_bucket_filter, "mom_b": momentum_bucket_filter, "hold_b": hold_bucket_filter}))
    if vol_bucket_filter or momentum_bucket_filter:
        cascades.append(("symbol+vol+mom", {"vol_b": vol_bucket_filter, "mom_b": momentum_bucket_filter, "hold_b": None}))
    if vol_bucket_filter:
        cascades.append(("symbol+vol", {"vol_b": vol_bucket_filter, "mom_b": None, "hold_b": None}))
    cascades.append(("symbol", {"vol_b": None, "mom_b": None, "hold_b": None}))

    result: DistributionResult | None = None
    fallback_from = ""
    for stratum_name, filt in cascades:
        rows = _apply_bucket_filters(all_symbol_rows, strategy=strategy, **filt)
        vals = [float(r[field_idx]) for r in rows if r[field_idx] is not None]
        if len(vals) >= MIN_OBS:
            result = _build_result(vals, stratum_used=stratum_name, fallback_from=fallback_from)
            break
        fallback_from = stratum_name

    if result is None:
        # Final fallback: strategy-only, cross-symbol.
        cross_rows = _fetch_rows(
            strategy=strategy,
            symbol=None,
            winners_only=winners_only,
            losers_only=losers_only,
            db_path=db_path,
            lookback_days=lookback_days,
        )
        vals = [float(r[field_idx]) for r in cross_rows if r[field_idx] is not None]
        if len(vals) >= MIN_OBS:
            result = _build_result(vals, stratum_used="strategy_cross_symbol", fallback_from=fallback_from)
        else:
            result = _empty_result(stratum_used="insufficient_everywhere")

    _cache[ck] = (now, result)
    return result


@dataclass(frozen=True)
class ExpectedExcursion:
    symbol: str
    strategy: str
    expected_mfe_p60: float
    expected_mae_p60: float
    mfe_confidence: str
    mae_confidence: str
    mfe_stratum: str
    mae_stratum: str
    mfe_n_obs: int
    mae_n_obs: int


def get_expected_mfe_mae(
    symbol: str,
    strategy: str,
    *,
    vol_bucket_filter: str | None = None,
    momentum_bucket_filter: str | None = None,
    hold_bucket_filter: str | None = None,
    db_path: str = DATABASE_PATH,
) -> ExpectedExcursion:
    """Convenience combined accessor for adaptive targets / HoldEV / adaptive
    MAE handling — the single call those subsystems should use."""
    mfe = get_mfe_distribution(
        symbol,
        strategy,
        vol_bucket_filter=vol_bucket_filter,
        momentum_bucket_filter=momentum_bucket_filter,
        hold_bucket_filter=hold_bucket_filter,
        db_path=db_path,
    )
    mae = get_mae_distribution(
        symbol,
        strategy,
        vol_bucket_filter=vol_bucket_filter,
        momentum_bucket_filter=momentum_bucket_filter,
        hold_bucket_filter=hold_bucket_filter,
        db_path=db_path,
    )
    return ExpectedExcursion(
        symbol=symbol.upper(),
        strategy=strategy.lower(),
        expected_mfe_p60=mfe.percentiles.get("p60", 0.0),
        expected_mae_p60=mae.percentiles.get("p60", 0.0),
        mfe_confidence=mfe.confidence_status,
        mae_confidence=mae.confidence_status,
        mfe_stratum=mfe.stratum_used,
        mae_stratum=mae.stratum_used,
        mfe_n_obs=mfe.n_obs,
        mae_n_obs=mae.n_obs,
    )


__all__ = [
    "DistributionResult",
    "ExpectedExcursion",
    "get_expected_mfe_mae",
    "get_mae_distribution",
    "get_mfe_distribution",
    "hold_time_bucket",
    "momentum_bucket",
    "vol_bucket",
]
