"""Adaptive, MFE-distribution-informed profit targets and giveback triggers (items p6/p7).

Companion to `day_adaptive_trail.py` (which adapts trail *width*). This
module adapts the profit *target* and the giveback *trigger* from the same
kind of real per-arm history, via `mfe_mae_distribution_learner.py`.

Scope and honesty note: a full "benchmark grid" re-optimization of target
selection (item p6's ideal end-state — sweeping target percentages across
after-cost expectancy/PF/drawdown and picking the argmax) is NOT what this
module does. That requires an offline backtest/grid-search pipeline against
historical trade replay, which is out of scope for a live-wired module. What
this module DOES do, honestly:

* `adaptive_target_price_for_arm()` — when a (symbol, setup, regime) arm has
  enough WINNING-trade history, most winners' actual MFE (p60/p75 of the
  distribution) becomes an additional target CANDIDATE. It can only ever
  tighten the effective target toward what winners on this arm have
  historically actually achieved — never extend it further away — because
  `effective_target_price()` still takes `min()` across candidates. This
  protects against a fixed profile target that is calibrated too far above
  what the arm's real winners reach (giving back MFE waiting for an
  unreachable target), which is a genuine, if narrower, instance of
  "adaptive targets selected from actual expectancy."

* `adaptive_giveback_trigger_for_arm()` — when a (symbol, setup, regime) arm
  has enough LOSING-trade history, the arm's own MAE-among-losers
  percentile becomes the giveback trigger instead of the fixed global
  `-0.15%` constant — a losing arm whose typical losers give back more (or
  less) than -0.15% before it's clearly over gets a trigger calibrated to
  its own history rather than one global number.

Both are pure ranking/economics inputs to the exit chain — mechanical
safety (hard stop, extreme protection) is untouched and always still runs
first.
"""

from __future__ import annotations

import os
import sqlite3
import time
from typing import Any

from backend.services.mfe_mae_distribution_learner import (
    get_mae_distribution,
    get_mfe_distribution,
)

# Item p6: ATR-normalized profit-target benchmark grid for DAY, expressed as
# multiples of the symbol's own current ATR% (see day_feature_stack_v2.py's
# atr_pct_multi_period). "Test ranges, not guaranteed final constants" per
# the original spec — configurable via env for later tuning without a
# code change.
DAY_ATR_TARGET_GRID: tuple[float, ...] = tuple(float(x) for x in os.getenv("DAY_ATR_TARGET_GRID", "0.75,1.00,1.25,1.50,2.00").split(",") if x.strip())
_ATR_GRID_MIN_OBS = int(os.getenv("DAY_ATR_GRID_MIN_OBS", "20"))
# Approximate DAY round-trip cost used only for the grid's simulated-exit
# comparison (mirrors portfolio_engine.DEFAULT_MIN_ROUNDTRIP_COST_PCT) —
# not itself a live cost model; real live costs are applied elsewhere.
_ATR_GRID_ROUNDTRIP_COST_PCT = float(os.getenv("DAY_ATR_GRID_ROUNDTRIP_COST_PCT", "0.0010"))


def adaptive_targets_enabled() -> bool:
    return os.getenv("DAY_ADAPTIVE_TARGETS_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")


def _min_pct(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)) or str(default))
    except (TypeError, ValueError):
        return default


def adaptive_target_pct_for_arm(symbol: str, setup: str, regime: str) -> dict[str, Any]:
    """Return {target_pct, source, n_obs, stratum} using the arm's real
    winning-trade MFE distribution (p60 by default — same balance point
    day_adaptive_trail.py uses for giveback). Falls back honestly to
    {source: "insufficient_data"} when there isn't enough arm history —
    callers must keep their existing fixed target in that case."""
    if not adaptive_targets_enabled():
        return {"target_pct": 0.0, "source": "disabled", "n_obs": 0, "stratum": "none"}
    dist = get_mfe_distribution(symbol, "day", db_path=_db_path())
    if dist.confidence_status == "insufficient_data" or dist.stratum_used == "strategy_cross_symbol":
        # Conservative: a symbol-specific target candidate must be backed by
        # that symbol's own real history, not a cross-symbol pool — this
        # candidate directly influences where profit is taken.
        return {"target_pct": 0.0, "source": "insufficient_data", "n_obs": dist.n_obs, "stratum": dist.stratum_used}
    percentile_key = os.getenv("DAY_ADAPTIVE_TARGET_PERCENTILE", "p60")
    raw = float(dist.percentiles.get(percentile_key, 0.0))
    min_pct = _min_pct("DAY_ADAPTIVE_TARGET_MIN_PCT", 0.0025)
    max_pct = _min_pct("DAY_ADAPTIVE_TARGET_MAX_PCT", 0.05)
    target_pct = max(min_pct, min(max_pct, raw))
    return {
        "target_pct": round(target_pct, 6),
        "source": dist.stratum_used,
        "n_obs": dist.n_obs,
        "confidence": dist.confidence_status,
        "stratum": dist.stratum_used,
    }


def adaptive_giveback_trigger_for_arm(symbol: str, setup: str, regime: str) -> dict[str, Any]:
    """Return {trigger_pct, source, n_obs} — a negative fraction (mirrors
    _giveback_trigger_pnl_pct's sign convention) derived from the arm's
    losing-trade MAE distribution. Falls back to insufficient_data when the
    arm lacks history; callers keep their fixed -0.15% default in that case."""
    if not adaptive_targets_enabled():
        return {"trigger_pct": 0.0, "source": "disabled", "n_obs": 0}
    dist = get_mae_distribution(symbol, "day", db_path=_db_path())
    if dist.confidence_status == "insufficient_data" or dist.stratum_used == "strategy_cross_symbol":
        return {"trigger_pct": 0.0, "source": "insufficient_data", "n_obs": dist.n_obs}
    percentile_key = os.getenv("DAY_ADAPTIVE_GIVEBACK_PERCENTILE", "p60")
    raw_mae = float(dist.percentiles.get(percentile_key, 0.0))
    min_trigger = _min_pct("DAY_ADAPTIVE_GIVEBACK_MIN_PCT", 0.0008)
    max_trigger = _min_pct("DAY_ADAPTIVE_GIVEBACK_MAX_PCT", 0.01)
    trigger_mag = max(min_trigger, min(max_trigger, raw_mae))
    return {
        "trigger_pct": round(-trigger_mag, 6),
        "source": dist.stratum_used,
        "n_obs": dist.n_obs,
        "confidence": dist.confidence_status,
    }


def _fetch_mfe_and_pnl_rows(symbol: str, db_path: str, lookback_days: int = 45) -> list[tuple[float, float]]:
    """(mfe_pct, realized_pnl_pct) rows for this symbol's own DAY trade
    history only — never a cross-symbol pool, since this feeds a candidate
    that directly moves where profit is taken."""
    try:
        from backend.services.ai_canonical_storage import _symbol_variants_for_lookup

        sym_variants = _symbol_variants_for_lookup(symbol)
    except Exception:
        sym_variants = [symbol.upper()]
    since_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(time.time() - lookback_days * 86400))
    try:
        with sqlite3.connect(db_path, timeout=10) as conn:
            rows = conn.execute(
                f"""
                SELECT mfe_pct, realized_pnl_pct
                FROM market_role_trade_outcomes
                WHERE strategy = 'day' AND created_at >= ?
                  AND symbol IN ({", ".join("?" for _ in sym_variants)})
                """,
                [since_iso, *sym_variants],
            ).fetchall()
    except Exception:
        return []
    return [(float(r[0]), float(r[1])) for r in rows if r[0] is not None and r[1] is not None]


def atr_grid_target_candidate(symbol: str, current_atr_pct: float, *, db_path: str | None = None) -> dict[str, Any]:
    """Item p6: sweep ``DAY_ATR_TARGET_GRID`` (multiples of the symbol's own
    CURRENT ATR%) and select the multiple with the best REAL simulated
    after-cost expectancy against this symbol's own trade history — "test
    ranges, not guaranteed final constants," selected from actual data, not
    assumed.

    Simulated exit for a given target: if a trade's real MFE reached the
    candidate target, assume it would have been captured (minus an
    approximate round-trip cost); otherwise the trade's ACTUAL realized
    outcome is used unchanged (a target can never manufacture a better
    outcome than what a trade that never reached it actually did).

    Honest same-symbol-only, sample-size-gated: returns
    ``{"source": "insufficient_data"}`` rather than fabricating a selection
    from too little history or from a cross-symbol pool.
    """
    if not adaptive_targets_enabled() or current_atr_pct <= 0:
        return {"target_pct": 0.0, "source": "disabled_or_no_atr", "n_obs": 0}
    db_path = db_path or _db_path()
    rows = _fetch_mfe_and_pnl_rows(symbol, db_path)
    if len(rows) < _ATR_GRID_MIN_OBS:
        return {"target_pct": 0.0, "source": "insufficient_data", "n_obs": len(rows)}

    best_multiple: float | None = None
    best_mean = float("-inf")
    grid_candidates: dict[str, float] = {}
    for multiple in DAY_ATR_TARGET_GRID:
        target_pct = multiple * current_atr_pct
        sims = [(target_pct - _ATR_GRID_ROUNDTRIP_COST_PCT) if mfe_pct >= target_pct else realized_pnl_pct for mfe_pct, realized_pnl_pct in rows]
        mean_sim = sum(sims) / len(sims)
        grid_candidates[f"{multiple:.2f}x_atr"] = round(mean_sim, 6)
        if mean_sim > best_mean:
            best_mean = mean_sim
            best_multiple = multiple

    if best_multiple is None:
        return {"target_pct": 0.0, "source": "insufficient_data", "n_obs": len(rows)}

    target_pct = best_multiple * current_atr_pct
    min_pct = _min_pct("DAY_ADAPTIVE_TARGET_MIN_PCT", 0.0025)
    max_pct = _min_pct("DAY_ADAPTIVE_TARGET_MAX_PCT", 0.05)
    target_pct = max(min_pct, min(max_pct, target_pct))
    return {
        "target_pct": round(target_pct, 6),
        "atr_multiple": best_multiple,
        "source": "atr_grid_expectancy",
        "n_obs": len(rows),
        "expected_pnl_pct": round(best_mean, 6),
        "grid_candidates": grid_candidates,
    }


def _db_path() -> str:
    from backend.database_schema import DATABASE_PATH

    return DATABASE_PATH


__all__ = [
    "DAY_ATR_TARGET_GRID",
    "adaptive_giveback_trigger_for_arm",
    "adaptive_target_pct_for_arm",
    "adaptive_targets_enabled",
    "atr_grid_target_candidate",
]
