"""Real order-book / trade-tape microstructure engine.

Computes, per top-4 symbol, from the live Binance.US partial-depth snapshot
stream (``{sym}usdt@depth20@100ms``) and aggregated-trade stream
(``{sym}usdt@aggTrade``):

  * Multi-depth order-book imbalance: L1 / L3 / L5 / L10 / L20
  * Microprice (size-weighted best quote) and microprice pressure vs mid
  * Snapshot-based order-flow imbalance (Cont/Kukanov/Stoikov top-of-book OFI),
    accumulated over short windows
  * Aggressive buy/sell volume from the trade tape (``isBuyerMaker`` semantics)
  * Bid/ask queue dynamics inferred from level-to-level size deltas between
    consecutive snapshots (additions / cancellations / replenishment /
    depletion), netted against real trade volume in the same interval
  * Imbalance slope, persistence and reversal frequency across short windows
    (0.25s / 0.5s / 1s / 2s / 5s / 10s / 30s)

HONESTY NOTE: Binance's public retail feeds do not expose a literal add/cancel
event log, and Mystic subscribes to the *partial-depth snapshot* stream (not
the incremental diff-depth stream). "Additions / cancellations / replenishment
/ depletion" here are *inferred* from snapshot-to-snapshot size deltas at each
price level (100ms cadence), netted against real trade volume pulled from the
aggTrade stream. This is a standard, defensible retail-tier approximation. It
is documented as *inferred*, never represented as a literal exchange event
log, and callers must not treat it as such.

ARCHITECTURE RULE: everything this module produces is a ranking / EV / sizing
/ exit input. Nothing here can reject or block a trade. Safety-only hard
blocks live elsewhere (stale data, spread, net-edge, exposure, kill switch).
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import sqlite3
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from backend.database_schema import DATABASE_PATH

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEPTH_LEVELS: tuple[int, ...] = (1, 3, 5, 10, 20)
WINDOWS_SEC: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0)
QUEUE_WINDOWS_SEC: tuple[float, ...] = (1.0, 5.0, 30.0)
MAX_WINDOW_SEC = max(WINDOWS_SEC)

_SNAPSHOT_MAXLEN = 1200  # ~120s at 100ms cadence, generous vs 30s max window
_TRADE_MAXLEN = 8000

_DB_PERSIST_INTERVAL_SEC = float(os.getenv("MICROSTRUCTURE_DB_PERSIST_INTERVAL_SEC", "5.0"))
_RANKING_DELTA_CAP = float(os.getenv("MICROSTRUCTURE_RANKING_DELTA_CAP", "0.03"))
_PRICE_LEVEL_TOL_REL = 1e-7  # relative tolerance for matching a price level across snapshots


def _now() -> float:
    return time.time()


@dataclass
class _DepthSample:
    ts: float
    bid_px: float
    bid_sz: float
    ask_px: float
    ask_sz: float
    obi: dict[int, float]  # depth -> imbalance in [-1, 1]
    bids_top10: tuple[tuple[float, float], ...]
    asks_top10: tuple[tuple[float, float], ...]


@dataclass
class _SymbolState:
    depth_hist: deque[_DepthSample] = field(default_factory=lambda: deque(maxlen=_SNAPSHOT_MAXLEN))
    trade_hist: deque[tuple[float, float, bool]] = field(default_factory=lambda: deque(maxlen=_TRADE_MAXLEN))
    # Cumulative OFI increments (ts, increment) so window sums are cheap.
    ofi_hist: deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=_SNAPSHOT_MAXLEN))
    last_db_persist_ts: float = 0.0


_STATE: dict[str, _SymbolState] = {}
_TABLE_READY = False


def _state_for(symbol: str) -> _SymbolState:
    s = symbol.upper().replace("/", "").replace("USDT", "") or symbol.upper()
    st = _STATE.get(s)
    if st is None:
        st = _SymbolState()
        _STATE[s] = st
    return st


def _base(symbol: str) -> str:
    return symbol.upper().replace("/", "").replace("USDT", "") or symbol.upper()


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------


def imbalance_at_depth(bids: list[tuple[float, float]], asks: list[tuple[float, float]], depth: int) -> float:
    """Order-book imbalance over top ``depth`` levels, in [-1, +1] (+ = bid-heavy)."""
    bid_vol = sum(sz for _, sz in bids[:depth])
    ask_vol = sum(sz for _, sz in asks[:depth])
    total = bid_vol + ask_vol
    if total <= 0:
        return 0.0
    return max(-1.0, min(1.0, (bid_vol - ask_vol) / total))


def microprice(bid_px: float, bid_sz: float, ask_px: float, ask_sz: float) -> float:
    """Size-weighted price between best bid/ask (Stoikov microprice).

    Weighted toward the side with LESS size (that side is more likely to move
    the price first), i.e. microprice = bid*ask_sz/(bid_sz+ask_sz) + ask*bid_sz/(bid_sz+ask_sz).
    """
    denom = bid_sz + ask_sz
    if denom <= 0 or bid_px <= 0 or ask_px <= 0:
        return (bid_px + ask_px) / 2.0 if (bid_px > 0 and ask_px > 0) else 0.0
    return (bid_px * ask_sz + ask_px * bid_sz) / denom


def _ofi_increment(prev: _DepthSample, curr: _DepthSample) -> float:
    """Top-of-book OFI increment (Cont, Kukanov, Stoikov 2014), snapshot form.

    e_n = 1[bid_n >= bid_{n-1}] * bidsize_n - 1[bid_n <= bid_{n-1}] * bidsize_{n-1}
        - 1[ask_n <= ask_{n-1}] * asksize_n + 1[ask_n >= ask_{n-1}] * asksize_{n-1}

    Positive = net buy-side pressure at the top of book.
    """
    bid_term = 0.0
    if curr.bid_px >= prev.bid_px:
        bid_term += curr.bid_sz
    if curr.bid_px <= prev.bid_px:
        bid_term -= prev.bid_sz
    ask_term = 0.0
    if curr.ask_px <= prev.ask_px:
        ask_term -= curr.ask_sz
    if curr.ask_px >= prev.ask_px:
        ask_term += prev.ask_sz
    return bid_term + ask_term


def _levels_match(a: float, b: float) -> bool:
    if a <= 0 or b <= 0:
        return False
    return abs(a - b) <= max(a, b) * _PRICE_LEVEL_TOL_REL


def _queue_deltas(
    prev_levels: tuple[tuple[float, float], ...],
    curr_levels: tuple[tuple[float, float], ...],
) -> tuple[float, float]:
    """Infer gross size (added, removed) across index-aligned top-10 levels.

    Levels are compared by (rank position, price tolerance). When a level's
    price shifts between snapshots (consumed/inserted), the displaced size is
    counted fully as removed/added respectively. This is the standard
    lightweight approximation for queue dynamics from partial-depth snapshots
    — see module docstring.
    """
    added = 0.0
    removed = 0.0
    n = max(len(prev_levels), len(curr_levels))
    for i in range(n):
        p_px, p_sz = prev_levels[i] if i < len(prev_levels) else (0.0, 0.0)
        c_px, c_sz = curr_levels[i] if i < len(curr_levels) else (0.0, 0.0)
        if p_px <= 0 and c_px <= 0:
            continue
        if p_px <= 0:
            added += c_sz
            continue
        if c_px <= 0:
            removed += p_sz
            continue
        if _levels_match(p_px, c_px):
            delta = c_sz - p_sz
            if delta > 0:
                added += delta
            elif delta < 0:
                removed += -delta
        else:
            # Level shifted: old level fully consumed/cancelled, new level fully added.
            removed += p_sz
            added += c_sz
    return added, removed


def _linreg_slope(points: list[tuple[float, float]]) -> float:
    """Simple OLS slope (y per second) for (ts, value) pairs."""
    n = len(points)
    if n < 2:
        return 0.0
    t0 = points[0][0]
    xs = [p[0] - t0 for p in points]
    ys = [p[1] for p in points]
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=False))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den <= 1e-12:
        return 0.0
    return num / den


def _persistence_and_reversals(values: list[float]) -> tuple[float, float]:
    """(fraction of samples sharing the current sign, reversals per second)."""
    if not values:
        return 0.0, 0.0
    current_sign = 1 if values[-1] > 0 else (-1 if values[-1] < 0 else 0)
    if current_sign == 0:
        same = sum(1 for v in values if v == 0)
    else:
        same = sum(1 for v in values if (v > 0) == (current_sign > 0) and v != 0)
    persistence = same / len(values)
    reversals = 0
    prev_sign = 0
    for v in values:
        sign = 1 if v > 0 else (-1 if v < 0 else 0)
        if sign != 0 and prev_sign != 0 and sign != prev_sign:
            reversals += 1
        if sign != 0:
            prev_sign = sign
    return persistence, reversals


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


def record_snapshot(
    symbol: str,
    bids: list[list[float]] | list[tuple[float, float]],
    asks: list[list[float]] | list[tuple[float, float]],
    ts: float | None = None,
) -> None:
    """Feed one partial-depth snapshot (already-parsed floats, best-to-worst)."""
    if not bids or not asks:
        return
    try:
        b = [(float(p), float(q)) for p, q in bids]
        a = [(float(p), float(q)) for p, q in asks]
    except (TypeError, ValueError):
        return
    if b[0][0] <= 0 or a[0][0] <= 0:
        return

    t = float(ts if ts is not None else _now())
    obi = {d: imbalance_at_depth(b, a, d) for d in DEPTH_LEVELS}
    sample = _DepthSample(
        ts=t,
        bid_px=b[0][0],
        bid_sz=b[0][1],
        ask_px=a[0][0],
        ask_sz=a[0][1],
        obi=obi,
        bids_top10=tuple(b[:10]),
        asks_top10=tuple(a[:10]),
    )

    st = _state_for(symbol)
    prev = st.depth_hist[-1] if st.depth_hist else None
    if prev is not None:
        ofi_inc = _ofi_increment(prev, sample)
        st.ofi_hist.append((t, ofi_inc))
    st.depth_hist.append(sample)
    _evict_old(st.depth_hist, t)
    _evict_old(st.ofi_hist, t)

    _maybe_persist(symbol, st, t)


def record_agg_trade(symbol: str, qty: float, is_buyer_maker: bool, ts: float | None = None) -> None:
    """Feed one aggTrade print.

    Binance semantics: ``isBuyerMaker=True`` means the buy side was resting
    (passive) and the sell side was the aggressor -> aggressive SELL.
    ``isBuyerMaker=False`` -> aggressive BUY.
    """
    try:
        q = float(qty)
    except (TypeError, ValueError):
        return
    if q <= 0:
        return
    t = float(ts if ts is not None else _now())
    st = _state_for(symbol)
    st.trade_hist.append((t, q, bool(is_buyer_maker)))
    _evict_old(st.trade_hist, t)


def _item_ts(item: Any) -> float:
    return item.ts if isinstance(item, _DepthSample) else item[0]


def _evict_old(dq: deque, now_ts: float) -> None:
    cutoff = now_ts - MAX_WINDOW_SEC - 5.0
    while dq and _item_ts(dq[0]) < cutoff:
        dq.popleft()


def _window(dq_iterable, now_ts: float, window_sec: float):
    cutoff = now_ts - window_sec
    return [item for item in dq_iterable if _item_ts(item) >= cutoff]


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------


def compute_features(symbol: str) -> dict[str, Any]:
    """Full microstructure feature dict for one symbol. Empty dict if no data."""
    st = _STATE.get(_base(symbol))
    if st is None or not st.depth_hist:
        return {}

    now_ts = st.depth_hist[-1].ts
    latest = st.depth_hist[-1]
    mid = (latest.bid_px + latest.ask_px) / 2.0
    mp = microprice(latest.bid_px, latest.bid_sz, latest.ask_px, latest.ask_sz)

    out: dict[str, Any] = {
        "symbol": _base(symbol),
        "ts": now_ts,
        "data_age_sec": max(0.0, _now() - now_ts),
        "best_bid": latest.bid_px,
        "best_ask": latest.ask_px,
        "mid": mid,
        "microprice": mp,
        "microprice_pressure": ((mp - mid) / mid) if mid > 0 else 0.0,
        "spread_pct": ((latest.ask_px - latest.bid_px) / mid) if mid > 0 else 0.0,
        "sample_count": len(st.depth_hist),
    }
    for d in DEPTH_LEVELS:
        out[f"obi_l{d}"] = round(latest.obi.get(d, 0.0), 6)

    depth_samples = list(st.depth_hist)
    ofi_samples = list(st.ofi_hist)
    trade_samples = list(st.trade_hist)

    for w in WINDOWS_SEC:
        tag = _window_tag(w)
        win_depth = _window(depth_samples, now_ts, w)

        l1_series = [(s.ts, s.obi.get(1, 0.0)) for s in win_depth]
        l10_series = [(s.ts, s.obi.get(10, 0.0)) for s in win_depth]

        slope_l1 = _linreg_slope(l1_series)
        slope_l10 = _linreg_slope(l10_series)
        persist_l1, rev_l1 = _persistence_and_reversals([v for _, v in l1_series])
        persist_l10, rev_l10 = _persistence_and_reversals([v for _, v in l10_series])
        out[f"obi_l1_slope_{tag}"] = round(slope_l1, 6)
        out[f"obi_l10_slope_{tag}"] = round(slope_l10, 6)
        out[f"obi_l1_persistence_{tag}"] = round(persist_l1, 4)
        out[f"obi_l10_persistence_{tag}"] = round(persist_l10, 4)
        out[f"obi_l1_reversal_freq_{tag}"] = round(rev_l1 / w, 4)
        out[f"obi_l10_reversal_freq_{tag}"] = round(rev_l10 / w, 4)

        win_ofi = _window(ofi_samples, now_ts, w)
        out[f"ofi_{tag}"] = round(sum(v for _, v in win_ofi), 6)

        win_trades = _window(trade_samples, now_ts, w)
        buy_vol = sum(q for _, q, ibm in win_trades if not ibm)
        sell_vol = sum(q for _, q, ibm in win_trades if ibm)
        total_vol = buy_vol + sell_vol
        out[f"agg_buy_vol_{tag}"] = round(buy_vol, 8)
        out[f"agg_sell_vol_{tag}"] = round(sell_vol, 8)
        out[f"agg_flow_imbalance_{tag}"] = round((buy_vol - sell_vol) / total_vol, 6) if total_vol > 0 else 0.0

    for w in QUEUE_WINDOWS_SEC:
        tag = _window_tag(w)
        win_depth = _window(depth_samples, now_ts, w)
        bid_added = bid_removed = ask_added = ask_removed = 0.0
        for i in range(1, len(win_depth)):
            b_add, b_rem = _queue_deltas(win_depth[i - 1].bids_top10, win_depth[i].bids_top10)
            a_add, a_rem = _queue_deltas(win_depth[i - 1].asks_top10, win_depth[i].asks_top10)
            bid_added += b_add
            bid_removed += b_rem
            ask_added += a_add
            ask_removed += a_rem

        win_trades = _window(trade_samples, now_ts, w)
        # Aggressive sells consume bid depth; aggressive buys consume ask depth.
        sell_vol = sum(q for _, q, ibm in win_trades if ibm)
        buy_vol = sum(q for _, q, ibm in win_trades if not ibm)
        bid_cancelled = max(0.0, bid_removed - sell_vol)
        ask_cancelled = max(0.0, ask_removed - buy_vol)

        out[f"bid_depth_added_{tag}"] = round(bid_added, 8)
        out[f"bid_depth_removed_{tag}"] = round(bid_removed, 8)
        out[f"bid_cancelled_{tag}"] = round(bid_cancelled, 8)
        out[f"bid_replenished_{tag}"] = round(bid_added, 8)
        out[f"ask_depth_added_{tag}"] = round(ask_added, 8)
        out[f"ask_depth_removed_{tag}"] = round(ask_removed, 8)
        out[f"ask_cancelled_{tag}"] = round(ask_cancelled, 8)
        out[f"ask_replenished_{tag}"] = round(ask_added, 8)

    return out


def _window_tag(w: float) -> str:
    if w < 1.0:
        return f"{int(w * 1000)}ms"
    return f"{int(w)}s"


# ---------------------------------------------------------------------------
# Bounded ranking delta (never a gate — see module docstring)
# ---------------------------------------------------------------------------


def get_microstructure_ranking_delta(symbol: str) -> float:
    """Bounded soft ranking nudge in [-cap, +cap] combining OFI direction,
    imbalance persistence and microprice pressure. Never raises; returns 0.0
    on any error or insufficient data. This is an EV/ranking input only.
    """
    try:
        feats = compute_features(symbol)
        if not feats or feats.get("data_age_sec", 999) > 10.0:
            return 0.0
        ofi_5s = float(feats.get("ofi_5s", 0.0))
        mp_pressure = float(feats.get("microprice_pressure", 0.0))
        agg_flow_5s = float(feats.get("agg_flow_imbalance_5s", 0.0))

        # OFI has no fixed scale (depends on symbol's typical top-of-book size),
        # so squash with tanh using a fixed soft scale of 5 base-asset units.
        ofi_signed = math.tanh(ofi_5s / 5.0) if abs(ofi_5s) > 1e-9 else 0.0

        # Robust, monotonic composite: OFI direction + real aggressor flow +
        # microprice pressure (size-weighted quote pulling away from mid).
        signal = (0.40 * ofi_signed) + (0.30 * agg_flow_5s) + (0.30 * math.tanh(mp_pressure * 500.0))
        delta = max(-_RANKING_DELTA_CAP, min(_RANKING_DELTA_CAP, signal * _RANKING_DELTA_CAP))
        return round(delta, 6)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Persistence (throttled, best-effort — never blocks the hot path)
# ---------------------------------------------------------------------------


def _ensure_table() -> None:
    global _TABLE_READY
    if _TABLE_READY:
        return
    try:
        with sqlite3.connect(DATABASE_PATH, timeout=5) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS microstructure_feature_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    ts_utc REAL NOT NULL,
                    obi_l1 REAL, obi_l3 REAL, obi_l5 REAL, obi_l10 REAL, obi_l20 REAL,
                    microprice REAL, microprice_pressure REAL, spread_pct REAL,
                    ofi_1s REAL, ofi_5s REAL, ofi_30s REAL,
                    obi_l10_persistence_5s REAL, obi_l10_reversal_freq_5s REAL,
                    agg_flow_imbalance_5s REAL,
                    bid_cancelled_5s REAL, ask_cancelled_5s REAL,
                    ranking_delta REAL,
                    features_json TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_microstructure_symbol_ts ON microstructure_feature_snapshots(symbol, ts_utc)")
            conn.commit()
        _TABLE_READY = True
    except Exception as exc:
        logger.debug("microstructure_engine: table init failed: %s", exc)


def _maybe_persist(symbol: str, st: _SymbolState, now_ts: float) -> None:
    if (now_ts - st.last_db_persist_ts) < _DB_PERSIST_INTERVAL_SEC:
        return
    st.last_db_persist_ts = now_ts
    with contextlib.suppress(Exception):
        feats = compute_features(symbol)
        if not feats:
            return
        _persist_row(symbol, feats)


def _persist_row(symbol: str, feats: dict[str, Any]) -> None:
    _ensure_table()
    import json as _json

    try:
        with sqlite3.connect(DATABASE_PATH, timeout=5) as conn:
            conn.execute(
                """
                INSERT INTO microstructure_feature_snapshots (
                    symbol, ts_utc, obi_l1, obi_l3, obi_l5, obi_l10, obi_l20,
                    microprice, microprice_pressure, spread_pct,
                    ofi_1s, ofi_5s, ofi_30s,
                    obi_l10_persistence_5s, obi_l10_reversal_freq_5s,
                    agg_flow_imbalance_5s, bid_cancelled_5s, ask_cancelled_5s,
                    ranking_delta, features_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _base(symbol),
                    feats.get("ts", time.time()),
                    feats.get("obi_l1", 0.0),
                    feats.get("obi_l3", 0.0),
                    feats.get("obi_l5", 0.0),
                    feats.get("obi_l10", 0.0),
                    feats.get("obi_l20", 0.0),
                    feats.get("microprice", 0.0),
                    feats.get("microprice_pressure", 0.0),
                    feats.get("spread_pct", 0.0),
                    feats.get("ofi_1s", 0.0),
                    feats.get("ofi_5s", 0.0),
                    feats.get("ofi_30s", 0.0),
                    feats.get("obi_l10_persistence_5s", 0.0),
                    feats.get("obi_l10_reversal_freq_5s", 0.0),
                    feats.get("agg_flow_imbalance_5s", 0.0),
                    feats.get("bid_cancelled_5s", 0.0),
                    feats.get("ask_cancelled_5s", 0.0),
                    get_microstructure_ranking_delta(symbol),
                    _json.dumps(feats, default=str),
                ),
            )
            conn.commit()
    except Exception as exc:
        logger.debug("microstructure_engine: persist_row failed for %s: %s", symbol, exc)


# ---------------------------------------------------------------------------
# Redis publish (mirrors ai_market_context ctx_role_intel_json append pattern)
# ---------------------------------------------------------------------------


async def publish_to_redis_async(symbol: str, redis_client: Any, *, ttl_sec: int = 30) -> bool:
    """Write the full feature set to ``microstructure:{BASE}`` and a compact
    subset merged into ``orderbook:{BASE}`` for existing consumers. Best-effort.
    """
    feats = compute_features(symbol)
    if not feats or redis_client is None:
        return False
    import json as _json

    base = _base(symbol)
    try:
        full_mapping = {k: str(v) for k, v in feats.items() if k != "symbol"}
        full_mapping["ranking_delta"] = str(get_microstructure_ranking_delta(symbol))
        pipe = redis_client.pipeline(transaction=True)
        pipe.hset(f"microstructure:{base}", mapping=full_mapping)
        pipe.expire(f"microstructure:{base}", ttl_sec)
        # Compact append onto orderbook:{BASE} so existing DAY consumers benefit
        # without needing to know about the new hash.
        compact = {
            "obi_l1": str(feats.get("obi_l1", 0.0)),
            "obi_l5": str(feats.get("obi_l5", 0.0)),
            "obi_l10": str(feats.get("obi_l10", 0.0)),
            "microprice": str(feats.get("microprice", 0.0)),
            "microprice_pressure": str(feats.get("microprice_pressure", 0.0)),
            "ofi_5s": str(feats.get("ofi_5s", 0.0)),
            "agg_flow_imbalance_5s": str(feats.get("agg_flow_imbalance_5s", 0.0)),
            "microstructure_ranking_delta": str(get_microstructure_ranking_delta(symbol)),
            "microstructure_json": _json.dumps(feats, default=str)[:4000],
        }
        pipe.hset(f"orderbook:{base}", mapping=compact)
        await pipe.execute()
        return True
    except Exception as exc:
        logger.debug("microstructure_engine: publish_to_redis_async failed for %s: %s", symbol, exc)
        return False


def get_stats() -> dict[str, Any]:
    return {
        "symbols_tracked": list(_STATE.keys()),
        "depth_samples": {s: len(st.depth_hist) for s, st in _STATE.items()},
        "trade_samples": {s: len(st.trade_hist) for s, st in _STATE.items()},
    }


__all__ = [
    "DEPTH_LEVELS",
    "QUEUE_WINDOWS_SEC",
    "WINDOWS_SEC",
    "compute_features",
    "get_microstructure_ranking_delta",
    "get_stats",
    "imbalance_at_depth",
    "microprice",
    "publish_to_redis_async",
    "record_agg_trade",
    "record_snapshot",
]
