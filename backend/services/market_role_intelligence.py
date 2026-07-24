"""
Market-Role Intelligence — per-symbol live structured context for BTC, ETH, SOL, XRP.

Design rules (all must hold):
  - No permanent directional points awarded for role labels alone.
  - Ranking adjustments come only from live measured data and learned outcomes.
  - BTC self-comparison forces correlation=1.0, beta=1.0, rs=0.0.
  - Candle arrays are timestamp-aligned before computing returns/RS.
  - Missing, duplicate, or insufficient data returns null — never fabricated.
  - Learned adjustments require MIN_OUTCOME_SAMPLES before activation.
  - Total influence of live + learned context on ranking is capped at ±0.08.
  - Nothing here blocks a trade.
"""

from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Role assignments — descriptive metadata only, no ranking points
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

# Lookback bars for correlation/beta on 1h timeframe
_CORR_LOOKBACK_BARS = 48  # 48h rolling window on 1h bars

# Minimum feature weight to avoid noise amplification
_RS_MIN_SIGNAL = 0.003   # 0.3% slope difference before RS contributes
_MOM_NEUTRAL = 0.5       # momentum score == neutral
_VOL_NEUTRAL = 0.35      # volatility normalised mid-point
_VA_NEUTRAL = 1.0        # volume acceleration == flat
_CORR_NEUTRAL = 0.5      # correlation normalised mid-point (raw 0.0 = neutral)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MarketRoleContext:
    """Live structured market-role context for one symbol."""

    symbol: str
    market_role: str           # descriptive label only — no ranking points from label alone
    role_code: int             # 0-3 numeric for feature encoding

    # BTC relative strength (short = 1h slope diff, medium = 4h slope diff)
    rs_short_1h: float | None         # range approx ±0.20 (clipped)
    rs_medium_4h: float | None        # range approx ±0.20 (clipped)

    # Cross-asset metrics (computed from timestamp-aligned 1h OHLCV, 48-bar window)
    btc_correlation: float | None     # Pearson [-1, +1]
    btc_beta: float | None            # regression beta

    # Composite scores
    momentum_score: float | None      # weighted ema_align + slope composite [0,1]
    volatility_score: float | None    # normalized ATR [0=low, 1=high]
    volume_accel: float | None        # recent_vol / historic_vol (1.0 = flat)

    # Catalyst / news (from pluggable provider)
    catalyst_score: float | None      # [0, 1]
    catalyst_source: str              # provider name or "unavailable"
    catalyst_category: str | None
    catalyst_freshness_sec: int | None

    # Market regime context
    market_regime: str                # "trending_up" / "chop" / "trending_down"
    risk_regime: str                  # "risk_on" / "neutral" / "risk_off"

    # Live context adjustment (data-driven, no static role points)
    live_context_adjustment: float = 0.0

    # Source and quality metadata
    source_status: str = "unavailable"
    freshness_sec: float = 0.0
    computed_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["freshness_sec"] = round(time.time() - self.computed_at, 1)
        return d

    def live_ranking_delta(self) -> float:
        """
        Live data-driven ranking adjustment.
        All components come from measured values — no points for role labels.
        Range: ±0.06.  Never a gate.
        """
        delta = 0.0

        # BTC-relative strength: symbol outperforming BTC → mild boost
        rs = self.rs_short_1h
        if rs is not None and abs(rs) >= _RS_MIN_SIGNAL:
            delta += max(-0.015, min(0.015, rs * 1.5))

        # Medium-term RS confirmation
        rs4 = self.rs_medium_4h
        if rs4 is not None and abs(rs4) >= _RS_MIN_SIGNAL:
            delta += max(-0.008, min(0.008, rs4 * 0.8))

        # Momentum: above-neutral is constructive, below-neutral is cautionary
        mom = self.momentum_score
        if mom is not None:
            mom_deviation = mom - _MOM_NEUTRAL           # positive = bullish
            delta += max(-0.012, min(0.012, mom_deviation * 0.12))

        # Volume acceleration: above-flat inflow is constructive
        va = self.volume_accel
        if va is not None:
            va_deviation = va - _VA_NEUTRAL              # positive = accelerating
            delta += max(-0.008, min(0.008, va_deviation * 0.008))

        # BTC correlation: higher correlation during risk-off → tighter BTC linkage
        # No fixed sign — whether that helps or hurts depends on BTC direction,
        # which is captured by rs_short_1h already.
        # Correlation itself contributes zero here; kept for learning attribution.

        # Catalyst: positive score → small constructive lift
        cat = self.catalyst_score
        if cat is not None and cat > 0.55:
            delta += min(0.012, (cat - 0.55) * 0.025)

        return round(max(-0.06, min(0.06, delta)), 4)

    def full_ranking_delta(self, learned_adjustment: float = 0.0) -> float:
        """
        live_ranking_delta() + learned_adjustment, total capped ±0.08.
        learned_adjustment comes from market_role_outcome_learner.
        """
        total = self.live_ranking_delta() + learned_adjustment
        return round(max(-0.08, min(0.08, total)), 4)


# ---------------------------------------------------------------------------
# Timestamp alignment helpers
# ---------------------------------------------------------------------------

def _ts(row: list | tuple) -> int:
    """Extract integer millisecond timestamp from CCXT OHLCV row."""
    try:
        return int(row[0])
    except (TypeError, IndexError, ValueError):
        return 0


def _align_candles(
    sym_rows: list | None,
    btc_rows: list | None,
) -> tuple[list, list]:
    """
    Return two lists of rows sharing the same timestamps, in ascending order.
    Drops rows present in only one series (gaps, exchange outages, etc.).
    Handles duplicates by keeping the last row per timestamp.
    Returns ([], []) when either input is None or empty.
    """
    if not sym_rows or not btc_rows:
        return [], []

    sym_by_ts: dict[int, list] = {}
    for r in sym_rows:
        t = _ts(r)
        if t:
            sym_by_ts[t] = r

    btc_by_ts: dict[int, list] = {}
    for r in btc_rows:
        t = _ts(r)
        if t:
            btc_by_ts[t] = r

    common = sorted(sym_by_ts.keys() & btc_by_ts.keys())
    if not common:
        return [], []

    return [sym_by_ts[t] for t in common], [btc_by_ts[t] for t in common]


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


def _slope_diff_pct(
    sym_rows: list | None,
    btc_rows: list | None,
    lookback: int,
) -> float | None:
    """
    Slope difference (sym - btc) as fraction of price, timestamp-aligned.
    Range ≈ ±0.20.
    """
    sym_aligned, btc_aligned = _align_candles(sym_rows, btc_rows)
    if len(sym_aligned) < lookback + 1 or len(btc_aligned) < lookback + 1:
        return None

    def _slope(rows: list) -> float | None:
        ref = float(rows[-(lookback + 1)][4])
        if ref == 0:
            return None
        return (float(rows[-1][4]) - ref) / ref

    s = _slope(sym_aligned)
    b = _slope(btc_aligned)
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
            slopes.append(0.5 + max(-0.5, min(0.5, slope * 50.0)))
    if not aligns:
        return None
    return round(float(np.mean(aligns + slopes)), 4)


def _volatility_score(sym_rows_1h: list | None) -> float | None:
    """
    Normalized volatility [0, 1].
    0.5% 1h ATR ≈ 0.25; 2%+ 1h ATR ≈ 1.0.
    """
    if not sym_rows_1h or len(sym_rows_1h) < 14:
        return None
    highs  = np.array([float(r[2]) for r in sym_rows_1h], dtype=np.float64)
    lows   = np.array([float(r[3]) for r in sym_rows_1h], dtype=np.float64)
    closes = np.array([float(r[4]) for r in sym_rows_1h], dtype=np.float64)
    tr = np.maximum.reduce([
        highs[1:] - lows[1:],
        np.abs(highs[1:] - closes[:-1]),
        np.abs(lows[1:] - closes[:-1]),
    ])
    atr14 = float(np.mean(tr[-14:]))
    ref = float(closes[-1]) or 1.0
    atr_pct = atr14 / ref
    return round(min(1.0, atr_pct / 0.02), 4)


def _volume_acceleration(recent_vol: float, hist_vol: float) -> float | None:
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

    BTC self-comparison:
      When symbol == BTCUSDT, correlation and beta are forced to 1.0 and rs to 0.0.
      This is the correct mathematical result; forcing it avoids floating-point
      edge cases when the BTC array itself is None or too short.

    Timestamp alignment:
      sym and btc candle arrays are aligned by timestamp before computing
      returns, correlation, beta, or relative strength.
    """
    sym_upper = symbol.upper().replace("/", "")
    role = MARKET_ROLES.get(sym_upper, "unknown")
    role_code = ROLE_CODES.get(role, -1)

    # Risk regime label (used for display/attribution only, not for ranking points)
    if market_regime == "trending_up":
        risk_regime = "risk_on"
    elif market_regime == "trending_down":
        risk_regime = "risk_off"
    else:
        risk_regime = "neutral"

    # ------------------------------------------------------------------
    # BTC self-comparison: force known-correct values, skip computation
    # ------------------------------------------------------------------
    is_btc = sym_upper == "BTCUSDT"

    if is_btc:
        btc_corr: float | None = 1.0
        btc_beta_val: float | None = 1.0
        rs_1h: float | None = 0.0
        rs_4h: float | None = 0.0
    else:
        # Timestamp-align before any computation
        sym_1h_al, btc_1h_al = _align_candles(sym_rows_1h, btc_rows_1h)
        n = min(len(sym_1h_al), len(btc_1h_al), _CORR_LOOKBACK_BARS)

        if n >= 10:
            sym_closes = np.array([float(r[4]) for r in sym_1h_al[-n:]], dtype=np.float64)
            btc_closes = np.array([float(r[4]) for r in btc_1h_al[-n:]], dtype=np.float64)
            sym_ret = _returns(sym_closes)
            btc_ret = _returns(btc_closes)
            nr = min(len(sym_ret), len(btc_ret))
            btc_corr = _pearson_correlation(sym_ret[-nr:], btc_ret[-nr:]) if nr >= 10 else None
            btc_beta_val = _beta(sym_ret[-nr:], btc_ret[-nr:]) if nr >= 10 else None
        else:
            btc_corr = None
            btc_beta_val = None

        # RS using aligned candles
        rs_1h = _slope_diff_pct(sym_rows_1h, btc_rows_1h, lookback=1)
        rs_4h = _slope_diff_pct(sym_rows_4h, btc_rows_4h, lookback=1)

    # Composite scores (from own symbol MTF data, independent of BTC)
    mom_score = _momentum_score(mtf_data)
    vol_score = _volatility_score(sym_rows_1h if not is_btc else btc_rows_1h)

    # Volume acceleration
    hist_per_2h = (volume_24h_usd / 12.0) if volume_24h_usd > 0 else 0.0
    vol_accel = _volume_acceleration(volume_2h_usd, hist_per_2h)

    # Catalyst score from pluggable provider
    cat_score: float | None = None
    cat_source = "unavailable"
    cat_category: str | None = None
    cat_freshness: int | None = None
    if catalyst_provider is not None:
        with contextlib.suppress(Exception):
            cat_result = await catalyst_provider.get_catalyst_score(sym_upper)
            if cat_result is not None:
                cat_score = cat_result.score
                cat_source = cat_result.source
                cat_category = cat_result.category
                cat_freshness = cat_result.freshness_sec

    # Source quality
    has_live = (is_btc and btc_rows_1h) or (not is_btc and sym_rows_1h and btc_rows_1h)
    source_status = "live" if has_live else ("partial" if (mtf_data or volume_24h_usd > 0) else "unavailable")

    ctx = MarketRoleContext(
        symbol=sym_upper,
        market_role=role,
        role_code=role_code,
        rs_short_1h=rs_1h,
        rs_medium_4h=rs_4h,
        btc_correlation=btc_corr,
        btc_beta=btc_beta_val,
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
    ctx.live_context_adjustment = ctx.live_ranking_delta()
    return ctx


# ---------------------------------------------------------------------------
# In-process cache (AI context loop → ranking reads without extra Redis round-trip)
# ---------------------------------------------------------------------------

_role_cache: dict[str, MarketRoleContext] = {}
_role_cache_ts: dict[str, float] = {}
_ROLE_CACHE_STALE_SEC = 120.0


def cache_role_context(ctx: MarketRoleContext) -> None:
    _role_cache[ctx.symbol] = ctx
    _role_cache_ts[ctx.symbol] = time.time()


def get_cached_role_context(symbol: str) -> MarketRoleContext | None:
    sym = symbol.upper().replace("/", "")
    ctx = _role_cache.get(sym)
    if ctx is None:
        return None
    age = time.time() - _role_cache_ts.get(sym, 0.0)
    if age > _ROLE_CACHE_STALE_SEC:
        return None
    return ctx


def get_role_ranking_delta(symbol: str, learned_adjustment: float = 0.0) -> float:
    """
    Safe accessor for ranking code — returns 0.0 if context is unavailable or stale.
    Optionally applies a learned_adjustment on top.
    Never raises.
    """
    try:
        ctx = get_cached_role_context(symbol)
        if ctx is None:
            return 0.0
        return ctx.full_ranking_delta(learned_adjustment)
    except Exception:
        return 0.0


__all__ = [
    "MARKET_ROLES",
    "ROLE_CODES",
    "MarketRoleContext",
    "_align_candles",
    "cache_role_context",
    "compute_market_role_context",
    "get_cached_role_context",
    "get_role_ranking_delta",
]
