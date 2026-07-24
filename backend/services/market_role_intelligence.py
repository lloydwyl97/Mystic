"""
Market-Role Intelligence — per-symbol live structured context for BTC, ETH, SOL, XRP.

Produces a MarketRoleContext object for each symbol containing:
  - Assigned and validated market role (leader / infrastructure / high_beta / catalyst_driven)
  - Rolling BTC correlation and beta (from 1h OHLCV)
  - Short-term (1h) and medium-term (4h) relative strength vs BTC
  - Composite momentum score
  - Volatility score (normalized ATR)
  - Volume acceleration
  - Catalyst score from pluggable CatalystProvider
  - Risk-on / risk-off regime context
  - Freshness timestamp and source status for every field

Design rules:
  - No trade gates are added — this data enriches ranking only.
  - Missing external data returns None with honest source labels.
  - Correlation / beta computed from live OHLCV; never fabricated.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Role assignments (static seed — validated and updated by live data)
# ---------------------------------------------------------------------------

MARKET_ROLES: dict[str, str] = {
    "BTCUSDT": "market_leader",
    "ETHUSDT": "infrastructure_leader",
    "SOLUSDT": "high_beta_momentum",
    "XRPUSDT": "catalyst_driven",
}

ROLE_CODES: dict[str, int] = {
    "market_leader": 0,
    "infrastructure_leader": 1,
    "high_beta_momentum": 2,
    "catalyst_driven": 3,
}

# Role-regime affinity: in risk_on markets, higher score → higher ranking weight
# In risk_off markets, lower-beta roles (leader) are defensively preferred.
# These are soft weights, not gates.
ROLE_REGIME_AFFINITY: dict[str, dict[str, float]] = {
    "market_leader":        {"trending_up": 0.0, "chop": 0.02, "trending_down": 0.05},
    "infrastructure_leader":{"trending_up": 0.01, "chop": 0.0, "trending_down": -0.02},
    "high_beta_momentum":   {"trending_up": 0.04, "chop": 0.0, "trending_down": -0.04},
    "catalyst_driven":      {"trending_up": 0.02, "chop": 0.03, "trending_down": -0.01},
}

# Lookback bars for correlation/beta on 1h timeframe
_CORR_LOOKBACK_BARS = 48  # 48h rolling window on 1h bars


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MarketRoleContext:
    """Live structured market-role context for one symbol."""

    symbol: str
    market_role: str           # static role label
    role_code: int             # 0-3 numeric for feature encoding
    role_regime_delta: float   # ranking soft-weight delta based on role × regime

    # BTC relative strength (short = 1h slope diff, medium = 4h slope diff)
    rs_short_1h: float | None         # range approx ±0.10 (clipped)
    rs_medium_4h: float | None        # range approx ±0.20 (clipped)

    # Cross-asset metrics (computed from 1h OHLCV, 48-bar window)
    btc_correlation: float | None     # Pearson [-1, +1]
    btc_beta: float | None            # regression beta

    # Composite scores [0, 1]
    momentum_score: float | None      # weighted ema_align + slope composite
    volatility_score: float | None    # normalized ATR (0=low vol, 1=high vol)
    volume_accel: float | None        # recent_vol / historic_vol (1.0 = flat)

    # Catalyst / news (from pluggable provider)
    catalyst_score: float | None      # [0, 1] relevance score
    catalyst_source: str              # provider name or "unavailable"
    catalyst_category: str | None     # "regulatory" / "etf_flow" / "protocol" / None
    catalyst_freshness_sec: int | None

    # Market regime context
    market_regime: str                # "trending_up" / "chop" / "trending_down"
    risk_regime: str                  # "risk_on" / "neutral" / "risk_off"

    # Source and quality metadata
    source_status: str                # "live" / "partial" / "stale" / "unavailable"
    freshness_sec: float              # seconds since this object was computed
    computed_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["freshness_sec"] = round(time.time() - self.computed_at, 1)
        return d

    def ranking_delta(self) -> float:
        """
        Small soft ranking delta for use in DAY and SCALP scoring.
        Range: approximately ±0.06.  Never used as a gate.
        """
        delta = self.role_regime_delta

        # Outperforming BTC short-term: mild boost
        rs_s = self.rs_short_1h or 0.0
        if rs_s > 0.005:
            delta += min(0.015, rs_s * 1.5)
        elif rs_s < -0.005:
            delta += max(-0.015, rs_s * 1.5)

        # Momentum confirmation: small boost when composite is strong
        mom = self.momentum_score
        if mom is not None:
            if mom > 0.70:
                delta += 0.01
            elif mom < 0.30:
                delta -= 0.01

        # Volume acceleration: recent interest boost
        va = self.volume_accel
        if va is not None:
            if va > 1.5:
                delta += 0.01
            elif va < 0.5:
                delta -= 0.01

        # Catalyst signal: XRP / catalyst-driven coins get a small lift
        cat = self.catalyst_score
        if cat is not None and cat > 0.6:
            delta += min(0.015, cat * 0.025)

        return round(max(-0.06, min(0.06, delta)), 4)


# ---------------------------------------------------------------------------
# Computation helpers
# ---------------------------------------------------------------------------

def _returns(closes: np.ndarray) -> np.ndarray:
    """Log returns, safe for zero/near-zero prices."""
    if len(closes) < 2:
        return np.array([], dtype=np.float64)
    safe = np.where(closes > 0, closes, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.diff(np.log(safe))
    return r[~np.isnan(r)]


def _pearson_correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    n = min(len(a), len(b))
    if n < 10:
        return None
    a, b = a[-n:], b[-n:]
    std_a, std_b = float(np.std(a)), float(np.std(b))
    if std_a < 1e-10 or std_b < 1e-10:
        return None
    corr = float(np.corrcoef(a, b)[0, 1])
    if not np.isfinite(corr):
        return None
    return round(max(-1.0, min(1.0, corr)), 4)


def _beta(sym_returns: np.ndarray, btc_returns: np.ndarray) -> float | None:
    n = min(len(sym_returns), len(btc_returns))
    if n < 10:
        return None
    s, b = sym_returns[-n:], btc_returns[-n:]
    var_b = float(np.var(b))
    if var_b < 1e-12:
        return None
    cov = float(np.cov(s, b)[0, 1])
    return round(cov / var_b, 4)


def _slope_diff_pct(sym_rows: list | None, btc_rows: list | None, lookback: int) -> float | None:
    """Slope difference (sym - btc) as fraction of price.  Range ≈ ±0.10."""
    def _slope(rows: list | None) -> float | None:
        if not rows or len(rows) < lookback + 1:
            return None
        closes = [float(r[4]) for r in rows]
        ref = closes[-(lookback + 1)]
        if ref == 0:
            return None
        return (closes[-1] - ref) / ref
    s = _slope(sym_rows)
    b = _slope(btc_rows)
    if s is None or b is None:
        return None
    return round(max(-0.20, min(0.20, s - b)), 6)


def _momentum_score(mtf_data: dict[str, Any] | None) -> float | None:
    """
    Composite momentum [0, 1] from MTF ema_align and slope signs.
    0.5 = neutral, >0.5 = bullish momentum, <0.5 = bearish.
    """
    if not mtf_data:
        return None
    tfs = ["1m", "5m", "15m", "1h", "4h"]
    aligns = []
    slopes = []
    for tf in tfs:
        snap = mtf_data.get(tf)
        if isinstance(snap, dict) and snap.get("bars", 0) > 3:
            aligns.append(float(snap.get("ema_align", 0.5)))
            slope = float(snap.get("slope", 0.0))
            # Normalise slope sign to [0,1]: positive slope → >0.5
            slopes.append(0.5 + max(-0.5, min(0.5, slope * 50.0)))
    if not aligns:
        return None
    return round(float(np.mean(aligns + slopes)), 4)


def _volatility_score(sym_rows_1h: list | None) -> float | None:
    """
    Normalized volatility [0, 1].
    Uses ATR-pct on 1h bars, normalised relative to a reference band.
    """
    if not sym_rows_1h or len(sym_rows_1h) < 14:
        return None
    highs = np.array([float(r[2]) for r in sym_rows_1h], dtype=np.float64)
    lows  = np.array([float(r[3]) for r in sym_rows_1h], dtype=np.float64)
    closes= np.array([float(r[4]) for r in sym_rows_1h], dtype=np.float64)
    tr = np.maximum.reduce([
        highs[1:] - lows[1:],
        np.abs(highs[1:] - closes[:-1]),
        np.abs(lows[1:] - closes[:-1]),
    ])
    atr14 = float(np.mean(tr[-14:]))
    ref = float(closes[-1]) or 1.0
    atr_pct = atr14 / ref
    # Calibration: 0.5% 1h ATR ≈ moderate vol; 2%+ is extreme
    return round(min(1.0, atr_pct / 0.02), 4)


def _volume_acceleration(recent_vol: float, hist_vol: float) -> float | None:
    """Recent (last ~2h) vs historical (last ~24h) notional volume ratio."""
    if hist_vol <= 0:
        return None
    return round(min(5.0, recent_vol / hist_vol), 4)


# ---------------------------------------------------------------------------
# Main compute function
# ---------------------------------------------------------------------------

async def compute_market_role_context(
    symbol: str,
    *,
    btc_rows_1h: list | None,
    sym_rows_1h: list | None,
    btc_rows_4h: list | None = None,
    sym_rows_4h: list | None = None,
    mtf_data: dict[str, Any] | None = None,
    market_regime: str = "chop",
    volume_24h_usd: float = 0.0,
    volume_2h_usd: float = 0.0,
    catalyst_provider: Any | None = None,
) -> MarketRoleContext:
    """
    Compute a MarketRoleContext for `symbol`.

    All inputs come from already-fetched OHLCV data — no extra network calls
    except for the optional catalyst_provider which is called async.
    """
    role = MARKET_ROLES.get(symbol.upper(), "unknown")
    role_code = ROLE_CODES.get(role, -1)

    # Role × regime delta (soft ranking signal, never a gate)
    regime_deltas = ROLE_REGIME_AFFINITY.get(role, {})
    role_regime_delta = regime_deltas.get(market_regime, 0.0)

    # Risk regime label derived from market regime
    if market_regime == "trending_up":
        risk_regime = "risk_on"
    elif market_regime == "trending_down":
        risk_regime = "risk_off"
    else:
        risk_regime = "neutral"

    # BTC correlation and beta from 1h returns
    btc_ret = _returns(np.array([float(r[4]) for r in btc_rows_1h], dtype=np.float64)) if btc_rows_1h else np.array([])
    sym_ret = _returns(np.array([float(r[4]) for r in sym_rows_1h], dtype=np.float64)) if sym_rows_1h else np.array([])

    n = min(len(btc_ret), len(sym_ret), _CORR_LOOKBACK_BARS)
    btc_corr = _pearson_correlation(sym_ret[-n:], btc_ret[-n:]) if n >= 10 else None
    btc_beta = _beta(sym_ret[-n:], btc_ret[-n:]) if n >= 10 else None

    # Relative strength — slope differences
    rs_1h = _slope_diff_pct(sym_rows_1h, btc_rows_1h, lookback=1)
    rs_4h = _slope_diff_pct(sym_rows_4h, btc_rows_4h, lookback=1)

    # Composite scores
    mom_score = _momentum_score(mtf_data)
    vol_score = _volatility_score(sym_rows_1h)

    # Volume acceleration: recent 2h / 24h normalised
    hist_per_2h = (volume_24h_usd / 12.0) if volume_24h_usd > 0 else 0.0
    vol_accel = _volume_acceleration(volume_2h_usd, hist_per_2h)

    # Catalyst score from pluggable provider
    cat_score: float | None = None
    cat_source = "unavailable"
    cat_category: str | None = None
    cat_freshness: int | None = None
    if catalyst_provider is not None:
        try:
            cat_result = await catalyst_provider.get_catalyst_score(symbol)
            if cat_result is not None:
                cat_score = cat_result.score
                cat_source = cat_result.source
                cat_category = cat_result.category
                cat_freshness = cat_result.freshness_sec
        except Exception as exc:
            logger.debug("catalyst_provider failed for %s: %s", symbol, exc)

    # Determine source quality
    has_live = btc_rows_1h and sym_rows_1h
    if has_live:
        source_status = "live"
    else:
        source_status = "partial" if (mtf_data or volume_24h_usd > 0) else "unavailable"

    return MarketRoleContext(
        symbol=symbol,
        market_role=role,
        role_code=role_code,
        role_regime_delta=role_regime_delta,
        rs_short_1h=rs_1h,
        rs_medium_4h=rs_4h,
        btc_correlation=btc_corr,
        btc_beta=btc_beta,
        momentum_score=mom_score,
        volatility_score=vol_score,
        volume_accel=vol_accel,
        catalyst_score=cat_score,
        catalyst_source=cat_source,
        catalyst_category=cat_category,
        catalyst_freshness_sec=cat_freshness,
        market_regime=market_regime,
        risk_regime=risk_regime,
        source_status=source_status,
        freshness_sec=0.0,
    )


# ---------------------------------------------------------------------------
# In-process cache so ranking can access the latest context without re-fetch
# ---------------------------------------------------------------------------

_role_cache: dict[str, MarketRoleContext] = {}
_role_cache_ts: dict[str, float] = {}
_ROLE_CACHE_STALE_SEC = 120.0


def cache_role_context(ctx: MarketRoleContext) -> None:
    _role_cache[ctx.symbol] = ctx
    _role_cache_ts[ctx.symbol] = time.time()


def get_cached_role_context(symbol: str) -> MarketRoleContext | None:
    ctx = _role_cache.get(symbol)
    if ctx is None:
        return None
    age = time.time() - _role_cache_ts.get(symbol, 0.0)
    if age > _ROLE_CACHE_STALE_SEC:
        return None
    return ctx


def get_role_ranking_delta(symbol: str) -> float:
    """
    Safe accessor for ranking code — returns 0.0 if context is unavailable
    or stale.  Never raises.
    """
    try:
        ctx = get_cached_role_context(symbol)
        if ctx is None:
            return 0.0
        return ctx.ranking_delta()
    except Exception:
        return 0.0


__all__ = [
    "MARKET_ROLES",
    "ROLE_CODES",
    "MarketRoleContext",
    "cache_role_context",
    "compute_market_role_context",
    "get_cached_role_context",
    "get_role_ranking_delta",
]
