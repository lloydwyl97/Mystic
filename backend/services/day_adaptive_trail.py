"""Adaptive trail-stop width from per-arm MFE-giveback history.

Instead of a fixed 5% trail (or a fixed `trail_pct` from thesis), this module
returns a suggested trail width based on the actual MFE-giveback distribution
of past trades on the same `(symbol, setup, regime)` arm. If BTC HTF trend
pullbacks historically gave back 40% of their MFE before failing, the trail
should tighten to 40% of MFE — no more, no less. Learned from the DB, not
hardcoded.

Public API:
* `adaptive_trail_pct_for_arm(symbol, setup, regime)` — returns trail pct
  in [MIN_TRAIL_PCT, MAX_TRAIL_PCT] derived from p60 giveback percentile.
  Falls back to `DAY_DEFAULT_TRAIL_PCT` when insufficient history.

Kill switch: DAY_ADAPTIVE_TRAIL_ENABLED (default true).
Env floors: DAY_ADAPTIVE_TRAIL_MIN_PCT (default 0.003 = 0.3%),
            DAY_ADAPTIVE_TRAIL_MAX_PCT (default 0.03 = 3%),
            DAY_ADAPTIVE_TRAIL_MIN_OBS (default 4),
            DAY_DEFAULT_TRAIL_PCT (default 0.008 = 0.8%).

The module is BUY-side only for now. Trail is applied to open positions;
this helper doesn't mutate — the caller decides when to tighten.
"""

from __future__ import annotations

import os
import sqlite3
import time
from typing import Any

from backend.database_schema import DATABASE_PATH

# In-process cache: computing the p60 percentile on every exit-check tick
# would be wasteful. Keyed by (symbol, setup, regime); expires after TTL.
_TRAIL_CACHE_TTL_SEC = 300.0  # 5 min — arm stats change slowly
_trail_cache: dict[str, tuple[float, float, int]] = {}


def adaptive_trail_enabled() -> bool:
    return os.getenv("DAY_ADAPTIVE_TRAIL_ENABLED", "true").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _floor_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)) or str(default))
    except (TypeError, ValueError):
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)) or str(default))
    except (TypeError, ValueError):
        return default


def _normalize_arm_key(symbol: str, setup: str, regime: str) -> str:
    s = str(symbol or "").strip().upper()
    if "/" not in s and s.endswith("USDT") and len(s) > 4:
        s = f"{s[:-4]}/USDT"
    return f"{s}|{str(setup or '').strip().upper()}|{str(regime or '').strip().lower()}"


def _percentile(sorted_vals: list[float], p: float) -> float:
    """Linear-interpolation percentile. sorted_vals must be sorted ascending."""
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


def _query_giveback_samples(
    symbol: str,
    setup: str,
    regime: str,
    *,
    db_path: str = DATABASE_PATH,
    lookback_days: int = 30,
) -> list[float]:
    """Fetch MFE-giveback values in [0, 1] for the arm over lookback window.

    We pull from ai_outcome_training_rows which persists mfe_giveback_pct
    per closed trade. Only WIN trades are used — losers didn't build MFE
    so their giveback distribution is not representative.
    """
    since_epoch = time.time() - (lookback_days * 86400)
    since_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(since_epoch))
    ss = str(symbol or "").upper().strip()
    if "/" not in ss and ss.endswith("USDT") and len(ss) > 4:
        ss = f"{ss[:-4]}/USDT"

    setup_u = str(setup or "").strip().upper()
    regime_l = str(regime or "").strip().lower()

    out: list[float] = []
    try:
        with sqlite3.connect(db_path, timeout=10) as conn:
            # Join attribution to filter by setup/regime.
            # ai_outcome_training_rows carries symbol + closed_at_utc + mfe_giveback_pct.
            # day_outcome_attribution maps trade → setup_thesis + regime.
            rows = conn.execute(
                """
                SELECT
                    otr.mfe_giveback_pct,
                    otr.outcome_label,
                    doa.setup_thesis,
                    doa.regime
                FROM ai_outcome_training_rows otr
                LEFT JOIN day_outcome_attribution doa ON otr.symbol = doa.symbol
                    AND otr.closed_at_utc = doa.closed_at_utc
                WHERE otr.symbol = ?
                  AND otr.closed_at_utc >= ?
                  AND otr.mfe_giveback_pct IS NOT NULL
                  AND otr.outcome_label = 1
                ORDER BY otr.closed_at_utc DESC
                LIMIT 500
                """,
                (ss, since_iso),
            ).fetchall()
    except sqlite3.OperationalError:
        # Table may not be joinable in some deployments — fall back to symbol-only.
        try:
            with sqlite3.connect(db_path, timeout=10) as conn:
                rows = conn.execute(
                    """
                    SELECT mfe_giveback_pct, outcome_label, NULL, NULL
                    FROM ai_outcome_training_rows
                    WHERE symbol = ? AND closed_at_utc >= ?
                      AND mfe_giveback_pct IS NOT NULL AND outcome_label = 1
                    ORDER BY closed_at_utc DESC LIMIT 500
                    """,
                    (ss, since_iso),
                ).fetchall()
        except Exception:
            return out

    for r in rows:
        try:
            gb = float(r[0] or 0.0)
            row_setup = str(r[2] or "").strip().upper() if r[2] is not None else ""
            row_regime = str(r[3] or "").strip().lower() if r[3] is not None else ""
            # Setup+regime filter is best-effort; if attribution is missing, accept.
            if setup_u and row_setup and row_setup != setup_u:
                continue
            if regime_l and row_regime and row_regime != regime_l:
                continue
            if 0.0 <= gb <= 1.0:
                out.append(gb)
        except (TypeError, ValueError):
            continue
    return out


def adaptive_trail_pct_for_arm(
    symbol: str,
    setup: str,
    regime: str,
    *,
    db_path: str = DATABASE_PATH,
) -> dict[str, Any]:
    """Return {trail_pct, source, n_obs, giveback_p60, giveback_p80, capped}.

    trail_pct is clamped to [DAY_ADAPTIVE_TRAIL_MIN_PCT, DAY_ADAPTIVE_TRAIL_MAX_PCT].
    Fallback when insufficient history: DAY_DEFAULT_TRAIL_PCT.
    """
    min_pct = _floor_env("DAY_ADAPTIVE_TRAIL_MIN_PCT", 0.003)
    max_pct = _floor_env("DAY_ADAPTIVE_TRAIL_MAX_PCT", 0.03)
    default_pct = _floor_env("DAY_DEFAULT_TRAIL_PCT", 0.008)
    min_obs = _int_env("DAY_ADAPTIVE_TRAIL_MIN_OBS", 4)

    if not adaptive_trail_enabled():
        return {
            "trail_pct": max(min_pct, min(max_pct, default_pct)),
            "source": "disabled",
            "n_obs": 0,
            "giveback_p60": 0.0,
            "giveback_p80": 0.0,
            "capped": False,
        }

    key = _normalize_arm_key(symbol, setup, regime)
    now = time.time()
    cached = _trail_cache.get(key)
    if cached and (now - cached[2]) < _TRAIL_CACHE_TTL_SEC:
        trail_pct_cached, _p60_cached, _ts = cached
        return {
            "trail_pct": trail_pct_cached,
            "source": "cache",
            "n_obs": -1,
            "giveback_p60": _p60_cached,
            "giveback_p80": 0.0,
            "capped": trail_pct_cached in (min_pct, max_pct),
        }

    samples = _query_giveback_samples(symbol, setup, regime, db_path=db_path)
    n = len(samples)
    if n < min_obs:
        # Fall back — but still cache the fallback so we don't hammer DB
        _trail_cache[key] = (default_pct, 0.0, now)
        return {
            "trail_pct": max(min_pct, min(max_pct, default_pct)),
            "source": f"insufficient_history_n={n}",
            "n_obs": n,
            "giveback_p60": 0.0,
            "giveback_p80": 0.0,
            "capped": False,
        }

    sorted_samples = sorted(samples)
    p60 = _percentile(sorted_samples, 0.60)
    p80 = _percentile(sorted_samples, 0.80)
    # Interpretation: p60 giveback = "60% of winners gave back at least this
    # much of their MFE before exiting." Trail wider than p60 catches more
    # winners but eats more MFE; narrower cuts winners early. p60 is a
    # reasonable balance.
    # Convert giveback (fraction of MFE lost) to a trail pct (fraction of
    # entry). Empirical fit: trail_pct ~= p60_giveback * 0.4, since typical
    # winning MFE is around 0.6–1.2% and 40% giveback = 0.24–0.48% trail.
    proposed = p60 * 0.40
    capped = False
    if proposed < min_pct:
        proposed = min_pct
        capped = True
    elif proposed > max_pct:
        proposed = max_pct
        capped = True

    _trail_cache[key] = (proposed, p60, now)
    return {
        "trail_pct": round(proposed, 6),
        "source": "arm_history",
        "n_obs": n,
        "giveback_p60": round(p60, 5),
        "giveback_p80": round(p80, 5),
        "capped": capped,
    }


__all__ = [
    "adaptive_trail_pct_for_arm",
    "adaptive_trail_enabled",
]
