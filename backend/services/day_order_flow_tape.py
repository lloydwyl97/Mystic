"""Real order flow for DAY, derived from the Binance.US aggTrade tape.

The 145-dim feature vector carries `volume_delta`, `order_flow`, and
`volume_imbalance` at indices 113-115, but those are OHLCV proxies
(`volume * sign(close - open)`) and are correctly zeroed by
`day_feature_health.zero_learning_blocked_feature_dims`. A sign-of-candle proxy
is not order flow: it cannot tell a dip absorbed by buyers from a dip sold into.

`agg_trade_collector` already persists every print to the Redis stream
`scalp:tape:{SYMBOL}` with the `isBuyerMaker` aggressor flag, so genuine signed
volume is available without a new subscription. `microstructure_engine` evicts
tape older than 35 seconds, which is too short for a 15-minute decision clock;
this module reads the retained stream directly instead.

Nothing here feeds the live model or any exit decision. It is capture only, so
that a future entry artifact can be trained on real flow rather than a proxy.

Binance.US top-4 prints are sparse. Every window therefore reports
`trade_count` and `stale_sec` alongside the imbalance so a genuinely balanced
tape stays distinguishable from an absent one — the exact conflation that made
the OHLCV proxy untrustworthy in the first place.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from typing import Any

from backend.services.binance_scalp.structural_tape import TAPE_STREAM_PREFIX

logger = logging.getLogger(__name__)

FLOW_VERSION = 1
BAR_SEC = 900
DEFAULT_WINDOWS_SEC: tuple[int, ...] = (60, 300, 900, 3600)
MAX_TAPE_READ = 12000


@dataclass(frozen=True)
class TapePrint:
    trade_ts: float
    price: float
    qty: float
    is_buyer_maker: bool


@dataclass(frozen=True)
class FlowWindow:
    """Signed-volume summary over one lookback window.

    `imbalance` is 0.0 when no prints landed in the window. Read it together
    with `trade_count`: zero flow and zero data are different states.
    """

    window_sec: int
    trade_count: int
    buy_qty: float
    sell_qty: float
    buy_notional: float
    sell_notional: float

    @property
    def cvd_qty(self) -> float:
        return self.buy_qty - self.sell_qty

    @property
    def cvd_notional(self) -> float:
        return self.buy_notional - self.sell_notional

    @property
    def imbalance(self) -> float:
        total = self.buy_qty + self.sell_qty
        return 0.0 if total <= 0 else (self.buy_qty - self.sell_qty) / total

    @property
    def notional_imbalance(self) -> float:
        total = self.buy_notional + self.sell_notional
        return 0.0 if total <= 0 else (self.buy_notional - self.sell_notional) / total

    @property
    def has_data(self) -> bool:
        return self.trade_count > 0


def _redis(client: Any | None = None) -> Any | None:
    if client is not None:
        return client
    try:
        from backend.config.redis_config import get_shared_redis_sync

        return get_shared_redis_sync()
    except Exception:
        logger.debug("order-flow tape: redis unavailable", exc_info=True)
        return None


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def bus_symbol(symbol: str) -> str:
    """Normalize to the bus form the tape stream is keyed on.

    Callers arrive in both forms: the portfolio engine holds positions as
    ETH/USDT while the collector writes scalp:tape:ETHUSDT. Mirrors
    structural_tape.parse_agg_payload so both sides agree on one key.
    """
    s = str(symbol or "").upper().replace("/", "").replace("-", "").replace(":", "")
    if s and not s.endswith("USDT"):
        s = f"{s}USDT"
    return s


def read_tape(
    symbol: str,
    *,
    since_ts: float,
    until_ts: float | None = None,
    client: Any | None = None,
) -> list[TapePrint]:
    """Prints for `symbol` with trade_ts in [since_ts, until_ts].

    Stream ids are recv-time milliseconds, so the range query is widened by a
    minute on each side and the exact bound is applied to the `T` trade
    timestamp carried in the entry.
    """
    r = _redis(client)
    if r is None:
        return []
    key = f"{TAPE_STREAM_PREFIX}{bus_symbol(symbol)}"
    lo_ms = max(0, int((since_ts - 60.0) * 1000))
    hi = "+" if until_ts is None else f"{int((until_ts + 60.0) * 1000)}"
    try:
        entries = r.xrange(key, f"{lo_ms}", hi, count=MAX_TAPE_READ)
    except Exception:
        logger.debug("order-flow tape: xrange failed for %s", symbol, exc_info=True)
        return []

    out: list[TapePrint] = []
    for _sid, fields in entries or []:
        try:
            ts = _as_float(fields.get("T")) / 1000.0
            if ts <= 0 or ts < since_ts:
                continue
            if until_ts is not None and ts > until_ts:
                continue
            qty = _as_float(fields.get("q"))
            price = _as_float(fields.get("p"))
            if qty <= 0 or price <= 0:
                continue
            out.append(
                TapePrint(
                    trade_ts=ts,
                    price=price,
                    qty=qty,
                    is_buyer_maker=str(fields.get("m", "0")) in ("1", "True", "true"),
                )
            )
        except (AttributeError, TypeError, ValueError):
            continue
    out.sort(key=lambda p: p.trade_ts)
    return out


def summarize(prints: list[TapePrint], window_sec: int) -> FlowWindow:
    buy_qty = sell_qty = buy_notional = sell_notional = 0.0
    for p in prints:
        notional = p.price * p.qty
        # is_buyer_maker=True means the aggressor sold into resting bids.
        if p.is_buyer_maker:
            sell_qty += p.qty
            sell_notional += notional
        else:
            buy_qty += p.qty
            buy_notional += notional
    return FlowWindow(
        window_sec=window_sec,
        trade_count=len(prints),
        buy_qty=buy_qty,
        sell_qty=sell_qty,
        buy_notional=buy_notional,
        sell_notional=sell_notional,
    )


def flow_windows(
    symbol: str,
    *,
    now: float | None = None,
    windows_sec: tuple[int, ...] = DEFAULT_WINDOWS_SEC,
    client: Any | None = None,
) -> dict[int, FlowWindow]:
    """Signed-volume summary per window, all from one tape read."""
    now = float(now if now is not None else time.time())
    longest = max(windows_sec) if windows_sec else BAR_SEC
    prints = read_tape(symbol, since_ts=now - longest, until_ts=now, client=client)
    return {w: summarize([p for p in prints if p.trade_ts >= now - w], w) for w in sorted(windows_sec)}


def heartbeat_flow_fields(
    symbol: str,
    *,
    now: float | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    """Flat, DB-ready intra-hold flow snapshot for `ai_position_heartbeats`."""
    now = float(now if now is not None else time.time())
    try:
        wins = flow_windows(symbol, now=now, client=client)
    except Exception:
        logger.debug("order-flow tape: window build failed for %s", symbol, exc_info=True)
        return {}
    if not wins:
        return {}
    newest = 0.0
    for w in wins.values():
        if w.has_data:
            newest = max(newest, 1.0)
    w5, w15, w60 = wins.get(300), wins.get(900), wins.get(3600)
    stale = _tape_stale_sec(symbol, now=now, client=client)
    return {
        "flow_trades_15m": int(w15.trade_count) if w15 else 0,
        "flow_cvd_qty_15m": float(w15.cvd_qty) if w15 else 0.0,
        "flow_imbalance_5m": float(w5.imbalance) if w5 else 0.0,
        "flow_imbalance_15m": float(w15.imbalance) if w15 else 0.0,
        "flow_imbalance_60m": float(w60.imbalance) if w60 else 0.0,
        "flow_notional_imbalance_15m": float(w15.notional_imbalance) if w15 else 0.0,
        "flow_tape_stale_sec": stale,
        "flow_version": FLOW_VERSION if newest > 0 else 0,
    }


def _tape_stale_sec(symbol: str, *, now: float, client: Any | None = None) -> float:
    """Age of the newest print. Large values mean the imbalance is uninformative."""
    r = _redis(client)
    if r is None:
        return -1.0
    try:
        entries = r.xrevrange(f"{TAPE_STREAM_PREFIX}{bus_symbol(symbol)}", "+", "-", count=1)
    except Exception:
        return -1.0
    if not entries:
        return -1.0
    try:
        ts = _as_float(entries[0][1].get("T")) / 1000.0
    except (AttributeError, IndexError, TypeError, ValueError):
        return -1.0
    return max(0.0, now - ts) if ts > 0 else -1.0


def bar_flow_rows(
    symbol: str,
    *,
    since_ts: float,
    until_ts: float | None = None,
    bar_sec: int = BAR_SEC,
    client: Any | None = None,
) -> list[dict[str, Any]]:
    """Bar-aligned flow rows for `day_order_flow_bars`.

    Bars are keyed by their open epoch on the same 15-minute grid the DAY
    decision clock uses, so rows join to `ai_inference_log` on symbol and time.
    Only bars that actually carry prints are returned; an absent row means no
    tape, which is a different fact from a balanced one.
    """
    until_ts = float(until_ts if until_ts is not None else time.time())
    prints = read_tape(symbol, since_ts=since_ts, until_ts=until_ts, client=client)
    if not prints:
        return []

    buckets: dict[int, list[TapePrint]] = {}
    for p in prints:
        buckets.setdefault(int(p.trade_ts // bar_sec) * bar_sec, []).append(p)

    rows: list[dict[str, Any]] = []
    for bar_open, group in sorted(buckets.items()):
        w = summarize(group, bar_sec)
        total_qty = w.buy_qty + w.sell_qty
        total_notional = w.buy_notional + w.sell_notional
        rows.append(
            {
                "symbol": bus_symbol(symbol),
                "bar_open_epoch": int(bar_open),
                "bar_sec": int(bar_sec),
                "trade_count": w.trade_count,
                "buy_qty": w.buy_qty,
                "sell_qty": w.sell_qty,
                "buy_notional": w.buy_notional,
                "sell_notional": w.sell_notional,
                "cvd_qty": w.cvd_qty,
                "cvd_notional": w.cvd_notional,
                "imbalance": w.imbalance,
                "notional_imbalance": w.notional_imbalance,
                "vwap": (total_notional / total_qty) if total_qty > 0 else 0.0,
                "first_price": group[0].price,
                "last_price": group[-1].price,
                "first_print_epoch": group[0].trade_ts,
                "last_print_epoch": group[-1].trade_ts,
                "coverage_sec": group[-1].trade_ts - group[0].trade_ts,
                "flow_version": FLOW_VERSION,
            }
        )
    return rows


def window_dict(win: FlowWindow) -> dict[str, Any]:
    d = asdict(win)
    d.update(
        {
            "cvd_qty": win.cvd_qty,
            "cvd_notional": win.cvd_notional,
            "imbalance": win.imbalance,
            "notional_imbalance": win.notional_imbalance,
        }
    )
    return d
