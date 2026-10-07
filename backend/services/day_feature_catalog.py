"""Causal catalog of the stored DAY 145-dim vector (``ai_inference_log.features_json``).

Each dim has a name, a family, a research transform and one class:

* ``A`` causal and available: computed from data closed at the vector's
  ``ts_utc`` and persisted on every row.
* ``B`` causal but not historically persisted (none inside the vector; listed
  in ``EXTERNAL_SOURCES`` for streams kept elsewhere or not at all).
* ``C`` non-causal / future-derived (none inside the vector; leakage enters
  only through a join that reads a vector stamped after the decision).
* ``D`` dead: constant in the stored history (zeroed proxies, missing feeds).
* ``E`` duplicate / redundant: identical to, or an exact transform of, a kept
  dim, or a per-symbol identity constant.

Price levels are made comparable across coins by ``x / price - 1`` and price
scales by ``x / price``; volume-like magnitudes by signed ``log1p``. The
transform reads only the same row, so it cannot leak.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from backend.services.ai_decision_contract import CONTEXT_DIMS_DAY_FULL
from backend.services.feature_mapping import FEATURE_MAPPING

DAY_FEATURE_CATALOG_VERSION = "DAY_FEATURE_CATALOG_V1"
DAY_VECTOR_DIM = 145
DAY_VECTOR_VERSION = 5

DAY_FEATURE_NAMES: tuple[str, ...] = tuple(n for n, _ in sorted(FEATURE_MAPPING.items(), key=lambda kv: kv[1])) + tuple(CONTEXT_DIMS_DAY_FULL)

PRICE_LEVEL_DIMS = frozenset((1, 2, 3, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 28, 29, 30, 40, 41, 42, 43, 44, 66, 67, 68, 69, 70, 79, *range(100, 113)))
PRICE_SCALE_DIMS = frozenset((8, 25, 26, 27, 38, 46, 48, 52))
VOLUME_DIMS = frozenset((4, 33, 34, 72, 73, 74, 118))
CALENDAR_DIMS = frozenset(range(90, 100))
SENTIMENT_DIMS = frozenset((80, 81, 82, 83, 84, 144))
BOOK_DIMS = frozenset((116, 117, 118, 119, 120, 121, 140, 141))
HTF_DIMS = frozenset((5, 6, 7, *range(124, 137)))

# Constant over the stored history: zeroed OHLCV flow proxies (113-115, see
# day_feature_health), no put/call or smile feed, whole-minute clock, and the
# 4h slope that the HTF bundle never fills.
DEAD_DIMS: dict[int, str] = {
    83: "put_call_ratio: no feed, always 0",
    98: "second: vectors are stamped on the minute, always 0",
    113: "volume_imbalance: OHLCV proxy, zeroed for learning",
    114: "volume_delta: OHLCV proxy, zeroed for learning",
    115: "order_flow: OHLCV proxy, zeroed for learning",
    122: "volatility_smile: no options feed, always 0",
    129: "slope_pct_4h: never filled by the HTF bundle, always 0",
}

# dim -> kept dim it duplicates (after the research transform).
DUPLICATE_DIMS: dict[int, int] = {
    20: 19,  # rsi_14 == rsi
    29: 12,  # bb_middle == ma_20
    112: 13,  # twap == ma_50
    70: 44,  # psar == parabolic_sar
    58: 47,  # kst == roc
    48: 47,  # momentum / price == roc
    38: 39,  # atr / price == natr
    37: 32,  # volatility == bb_width (corr 1.0)
    46: 32,  # price_volatility / price tracks bb_width (corr >= 0.9999)
    101: 100,  # fib 38.2 is affine in the same high/low range as 23.6
    71: 62,  # trend_strength == adx (corr 1.0)
    94: 91,  # iso_weekday == day_of_week + 1
    103: 9,  # pivot_point == typical_price
    143: 89,  # ctx_btc_dominance_proxy == market_dominance
}

# Per-symbol constants: they encode the coin, not its state.
IDENTITY_DIMS: dict[int, str] = {
    85: "market_cap: supply x price, redundant with price and coin identity",
    86: "supply: per-symbol constant",
    87: "circulating_supply: per-symbol constant",
    88: "max_supply: per-symbol constant",
}

# Causal information that the vector does not carry, with where it lives.
EXTERNAL_SOURCES: dict[str, dict[str, str]] = {
    "day_order_flow_bars": {"class": "A", "note": "signed 15m tape bars since 2026-09-02; join on the decision bar only once closed"},
    "ai_position_heartbeats.flow_*": {"class": "A", "note": "intra-hold flow; position-time, not entry-time"},
    "microstructure_feature_snapshots": {"class": "B", "note": "5 s derived book state, 3-day retention; history before that is gone"},
    "decision_book_tape": {"class": "B", "note": "DAY top-5 book at decision; captured 2026-09-23..24 only"},
    "adaptive_candidate_markouts.features_json (DAY)": {"class": "B", "note": "decision-time DAY state; was dropped on write, persisted from this version"},
    "raw L2 levels + update ids": {"class": "B", "note": "received every 100 ms, never persisted before book_queue_chunks"},
}


def classify(dim: int) -> str:
    if dim in DEAD_DIMS:
        return "D"
    if dim in DUPLICATE_DIMS or dim in IDENTITY_DIMS:
        return "E"
    return "A"


def family(dim: int) -> str:
    if dim == 0:
        return "price"
    for name, dims in (
        ("price_level", PRICE_LEVEL_DIMS),
        ("price_scale", PRICE_SCALE_DIMS),
        ("volume", VOLUME_DIMS),
        ("calendar", CALENDAR_DIMS),
        ("sentiment", SENTIMENT_DIMS),
        ("book", BOOK_DIMS),
        ("htf", HTF_DIMS),
    ):
        if dim in dims:
            return name
    if dim in IDENTITY_DIMS:
        return "identity"
    return "indicator"


CAUSAL_DIMS: tuple[int, ...] = tuple(d for d in range(DAY_VECTOR_DIM) if classify(d) == "A")


def catalog() -> list[dict[str, Any]]:
    out = []
    for d, name in enumerate(DAY_FEATURE_NAMES):
        cls = classify(d)
        reason = DEAD_DIMS.get(d) or IDENTITY_DIMS.get(d) or (f"duplicate of {DUPLICATE_DIMS[d]} {DAY_FEATURE_NAMES[DUPLICATE_DIMS[d]]}" if d in DUPLICATE_DIMS else "")
        out.append({"dim": d, "name": name, "family": family(d), "class": cls, "reason": reason})
    return out


def transform_value(dim: int, value: float, price: float) -> float:
    """Research transform of one dim; reads only the same row."""
    x = float(value)
    if not math.isfinite(x):
        return 0.0
    if dim == 0:
        return math.log(x) if x > 0 else 0.0
    if dim in PRICE_LEVEL_DIMS:
        return x / price - 1.0 if price > 0 else 0.0
    if dim in PRICE_SCALE_DIMS:
        return x / price if price > 0 else 0.0
    if dim in VOLUME_DIMS:
        return math.copysign(math.log1p(abs(x)), x)
    return x


def causal_features(vector: Sequence[float]) -> dict[str, float]:
    """Named, transformed class-A features of one stored vector."""
    if len(vector) != DAY_VECTOR_DIM:
        raise ValueError(f"expected {DAY_VECTOR_DIM} dims, got {len(vector)}")
    price = float(vector[0])
    return {DAY_FEATURE_NAMES[d]: transform_value(d, vector[d], price) for d in CAUSAL_DIMS}


def profile(by_symbol: Mapping[str, Sequence[Sequence[float]]], *, corr_floor: float = 0.9999) -> dict[str, Any]:
    """Empirical check of the static classes on stored vectors.

    Returns dims constant in every symbol and transformed pairs whose
    per-symbol correlation is at least ``corr_floor`` in every symbol.
    """
    import numpy as np

    mats = {s: np.array([[transform_value(d, v[d], float(v[0])) for d in range(DAY_VECTOR_DIM)] for v in rows], dtype=float) for s, rows in by_symbol.items() if len(rows) >= 3}
    if not mats:
        return {"constant": [], "duplicates": []}
    constant = [d for d in range(DAY_VECTOR_DIM) if all(float(m[:, d].std()) == 0.0 for m in mats.values())]
    live = [d for d in range(DAY_VECTOR_DIM) if d not in constant]
    corr = {s: np.nan_to_num(np.corrcoef(m[:, live].T)) for s, m in mats.items()}
    dups = []
    for i, a in enumerate(live):
        for j in range(i + 1, len(live)):
            if all(abs(float(c[i, j])) >= corr_floor for c in corr.values()):
                dups.append((a, live[j]))
    return {"constant": constant, "duplicates": dups}


__all__ = [
    "CAUSAL_DIMS",
    "DAY_FEATURE_CATALOG_VERSION",
    "DAY_FEATURE_NAMES",
    "DAY_VECTOR_DIM",
    "DAY_VECTOR_VERSION",
    "DEAD_DIMS",
    "DUPLICATE_DIMS",
    "EXTERNAL_SOURCES",
    "IDENTITY_DIMS",
    "catalog",
    "causal_features",
    "classify",
    "family",
    "profile",
    "transform_value",
]
