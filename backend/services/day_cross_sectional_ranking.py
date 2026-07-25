"""
day_cross_sectional_ranking — compares each DAY-traded symbol's current setup
quality against its BTC/ETH/SOL/XRP peers each cycle, instead of scoring each
of the 4 symbols in complete isolation the way every other ranking input in
day_ai_rank_enrichment.py does.

Mechanism: each symbol publishes a single combined attractiveness score (model
confidence blended with setup quality — the same inputs already computed this
cycle, no new model or feature) to a small shared Redis hash with a short TTL,
then reads its 3 peers' most recent scores back. A bounded ±0.04 nudge (same
cap convention as day_regime_transition.py / day_chart_pattern_detector.py)
rewards a symbol that is genuinely standing out from its peers right now and
is neutral when there isn't enough live peer data or peers all look similar —
this never gates a trade, only nudges relative ranking among the 4 DAY slots.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

logger = logging.getLogger(__name__)

RANK_DELTA_CAP = 0.04
_REDIS_KEY = "day_cross_sectional_scores"
_ENTRY_TTL_SEC = 180
_TOP4_BASES = ("BTC", "ETH", "SOL", "XRP")


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _base(symbol: str) -> str:
    s = (symbol or "").upper().replace("/", "")
    if s.endswith("USDT") and len(s) > 4:
        s = s[:-4]
    return s


def combined_attractiveness_score(decision_data: dict[str, Any]) -> float:
    """Single [0,1] yardstick blending model confidence with setup quality —
    the same two inputs day_ai_rank_enrichment.py already has this cycle for
    this candidate, just combined into one number peers can be compared on."""
    dd = decision_data or {}
    conf = _safe_float(dd.get("confidence"), 0.5)
    setup = _safe_float(dd.get("setup_score"), 0.5)
    return max(0.0, min(1.0, 0.6 * conf + 0.4 * setup))


def publish_cross_sectional_score(symbol: str, score: float) -> None:
    """Best-effort sync write of this symbol's current score for its peers to
    read this cycle. Never blocks or raises — same non-gating contract as every
    other bounded ranking nudge this session."""
    try:
        from backend.config.redis_config import get_shared_redis_sync

        r = get_shared_redis_sync()
        if not r:
            return
        base = _base(symbol)
        if base not in _TOP4_BASES:
            return
        r.hset(_REDIS_KEY, base, f"{time.time():.3f}:{float(score):.6f}")
        r.expire(_REDIS_KEY, _ENTRY_TTL_SEC * 4)
    except Exception as exc:
        logger.debug("CROSS_SECTIONAL_PUBLISH_FAILED %s: %s", symbol, exc)


def read_peer_scores(symbol: str) -> dict[str, float]:
    """Read the other top-4 symbols' most recently published scores. Skips this
    symbol and any peer entry older than _ENTRY_TTL_SEC — stale peer data would
    compare this cycle's read against a peer's outdated snapshot, which is worse
    than treating that peer as simply unavailable."""
    own_base = _base(symbol)
    peers: dict[str, float] = {}
    try:
        from backend.config.redis_config import get_shared_redis_sync

        r = get_shared_redis_sync()
        if not r:
            return peers
        raw = r.hgetall(_REDIS_KEY) or {}
        now = time.time()
        for k, v in raw.items():
            base = k.decode() if isinstance(k, bytes) else str(k)
            if base == own_base or base not in _TOP4_BASES:
                continue
            val = v.decode() if isinstance(v, bytes) else str(v)
            try:
                ts_str, score_str = val.split(":", 1)
                if (now - float(ts_str)) > _ENTRY_TTL_SEC:
                    continue
                peers[base] = float(score_str)
            except (ValueError, IndexError):
                continue
    except Exception as exc:
        logger.debug("CROSS_SECTIONAL_READ_FAILED %s: %s", symbol, exc)
    return peers


def cross_sectional_rank_delta(own_score: float, peer_scores: dict[str, float]) -> float:
    """Bounded (±RANK_DELTA_CAP) nudge from this symbol's z-score-like relative
    attractiveness vs its live top-4 peers this cycle. Requires at least 2 peers
    with live data and real spread between them (std > 1e-6) — with fewer peers,
    or when everyone currently looks about the same, there is nothing real to
    compare against, so this returns 0.0 (neutral, never a fabricated edge)."""
    values = list(peer_scores.values())
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std = math.sqrt(variance)
    if std < 1e-6:
        return 0.0
    z = max(-2.0, min(2.0, (own_score - mean) / std))
    return round(max(-RANK_DELTA_CAP, min(RANK_DELTA_CAP, (z / 2.0) * RANK_DELTA_CAP)), 4)


__all__ = [
    "RANK_DELTA_CAP",
    "combined_attractiveness_score",
    "cross_sectional_rank_delta",
    "publish_cross_sectional_score",
    "read_peer_scores",
]
