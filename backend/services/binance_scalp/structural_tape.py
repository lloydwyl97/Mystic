"""Binance.US aggTrade tape for structural LP. Redis-bridged from uvicorn collector.

Eligible sell-side print: is_buyer_maker=True (aggressor sold into bids).
Eligible buy-side print: is_buyer_maker=False (aggressor bought into asks).
Cancellations are never treated as queue consumption.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

FILL_MODEL_VERSION = "structural_event_queue_v1"
TAPE_STREAM_PREFIX = "scalp:tape:"
TAPE_FRESH_KEY_PREFIX = "scalp:tape:fresh:"
TAPE_MAXLEN = 8000
DEFAULT_STALE_SEC = 3.0


@dataclass(frozen=True)
class TradeEvent:
    symbol: str
    agg_id: int
    price: float
    qty: float
    is_buyer_maker: bool
    trade_ts: float
    recv_ts: float
    source: str = "binance_us_aggtrade"

    @property
    def aggressor_sold(self) -> bool:
        return bool(self.is_buyer_maker)

    @property
    def aggressor_bought(self) -> bool:
        return not bool(self.is_buyer_maker)

    def key(self) -> str:
        return f"{self.symbol}:{self.agg_id}"


def parse_agg_payload(payload: dict[str, Any], *, recv_ts: float | None = None) -> TradeEvent | None:
    symbol = str(payload.get("s") or payload.get("symbol") or "").upper().replace("/", "")
    if symbol and not symbol.endswith("USDT"):
        symbol = f"{symbol}USDT"
    try:
        agg_id = int(payload.get("a") or payload.get("agg_id") or 0)
        price = float(payload.get("p") or payload.get("price") or 0.0)
        qty = float(payload.get("q") or payload.get("qty") or 0.0)
    except (TypeError, ValueError):
        return None
    if not symbol or agg_id <= 0 or price <= 0 or qty <= 0:
        return None
    raw_m = payload.get("m", payload.get("is_buyer_maker"))
    is_buyer_maker = bool(raw_m)
    trade_ts_ms = payload.get("T") or payload.get("trade_ts_ms")
    try:
        trade_ts = float(trade_ts_ms) / 1000.0 if trade_ts_ms and float(trade_ts_ms) > 1e12 else float(payload.get("trade_ts") or payload.get("T") or 0.0)
    except (TypeError, ValueError):
        trade_ts = 0.0
    if trade_ts <= 0:
        return None
    return TradeEvent(
        symbol=symbol,
        agg_id=agg_id,
        price=price,
        qty=qty,
        is_buyer_maker=is_buyer_maker,
        trade_ts=trade_ts,
        recv_ts=float(recv_ts if recv_ts is not None else time.time()),
        source=str(payload.get("source") or "binance_us_aggtrade"),
    )


def event_eligible_for_quote(event: TradeEvent, *, side: str, price: float, posted_ts: float, last_agg_id: int) -> bool:
    """Post-placement, in-order, sided, at-or-through prints only."""
    if event.agg_id <= int(last_agg_id):
        return False
    if event.trade_ts <= float(posted_ts):
        return False
    if event.price <= 0 or event.qty <= 0:
        return False
    side_u = str(side or "").upper()
    if side_u == "BID":
        return event.aggressor_sold and event.price <= float(price) + 1e-12
    if side_u == "ASK":
        return event.aggressor_bought and event.price >= float(price) - 1e-12
    return False


def apply_queue(
    *,
    queue_ahead: float,
    queue_consumed: float,
    remaining_qty: float,
    event_qty: float,
) -> tuple[float, float, float]:
    """Eligible volume eats queue-ahead first. Cancellations are not passed in."""
    ahead_left = max(0.0, float(queue_ahead) - float(queue_consumed))
    qty = max(0.0, float(event_qty))
    if qty <= 0 or remaining_qty <= 0:
        return float(queue_consumed), 0.0, float(remaining_qty)
    if ahead_left > 0:
        take_ahead = min(ahead_left, qty)
        queue_consumed = float(queue_consumed) + take_ahead
        qty -= take_ahead
    fill = min(float(remaining_qty), qty)
    return float(queue_consumed), float(fill), float(remaining_qty) - fill


def publish_trade_event(redis_client: Any, event: TradeEvent) -> None:
    if redis_client is None:
        return
    fields = {
        "s": event.symbol,
        "a": str(event.agg_id),
        "p": str(event.price),
        "q": str(event.qty),
        "m": "1" if event.is_buyer_maker else "0",
        "T": str(int(event.trade_ts * 1000)),
        "recv": str(event.recv_ts),
        "source": event.source,
    }
    redis_client.xadd(f"{TAPE_STREAM_PREFIX}{event.symbol}", fields, maxlen=TAPE_MAXLEN, approximate=True)
    redis_client.set(
        f"{TAPE_FRESH_KEY_PREFIX}{event.symbol}",
        json.dumps({"trade_ts": event.trade_ts, "recv_ts": event.recv_ts, "agg_id": event.agg_id}),
        ex=30,
    )


def consume_trade_events(
    redis_client: Any,
    symbol: str,
    *,
    last_stream_id: str,
    count: int = 200,
) -> tuple[list[TradeEvent], str]:
    if redis_client is None:
        return [], last_stream_id or "0-0"
    start = last_stream_id or "0-0"
    rows = redis_client.xread({f"{TAPE_STREAM_PREFIX}{symbol}": start}, count=count, block=None)
    events: list[TradeEvent] = []
    new_id = start
    if not rows:
        return events, new_id
    for _key, items in rows:
        for sid, fields in items:
            new_id = str(sid)
            payload = {
                "s": fields.get("s") or symbol,
                "a": fields.get("a"),
                "p": fields.get("p"),
                "q": fields.get("q"),
                "m": fields.get("m") in {"1", "true", "True"},
                "T": fields.get("T"),
                "source": fields.get("source") or "binance_us_aggtrade",
            }
            ev = parse_agg_payload(payload, recv_ts=float(fields.get("recv") or 0) or None)
            if ev is not None:
                events.append(ev)
    return events, new_id


def tape_freshness(redis_client: Any, symbol: str, *, now: float, stale_sec: float = DEFAULT_STALE_SEC) -> dict[str, Any]:
    out = {
        "symbol": symbol,
        "fresh": False,
        "age_sec": None,
        "last_trade_ts": None,
        "stale_sec": float(stale_sec),
        "source": "binance_us_aggtrade",
    }
    if redis_client is None:
        return out
    raw = redis_client.get(f"{TAPE_FRESH_KEY_PREFIX}{symbol}")
    if not raw:
        return out
    try:
        payload = json.loads(raw)
        last = float(payload.get("trade_ts") or 0.0)
    except (TypeError, ValueError, json.JSONDecodeError):
        return out
    age = float(now) - last
    out["last_trade_ts"] = last
    out["age_sec"] = age
    out["fresh"] = last > 0 and 0 <= age <= float(stale_sec)
    return out
