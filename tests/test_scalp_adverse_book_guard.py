"""Adverse stop uses only a current executable bid. DAY is not involved."""

from __future__ import annotations

import json
import time

from backend.services.binance_scalp.market_reader import (
    _read_ws_depth,
    _update_id_is_newer,
    book_behind_recent_tape,
    publish_ws_depth,
)
from backend.services.scalp_v2.exit_evaluator import SCALP_V2_EXIT_ADVERSE, SCALP_V2_EXIT_CATASTROPHIC, evaluate_scalp_v2_exit


class _MemRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.streams: dict[str, list[tuple[str, dict[str, str]]]] = {}

    def get(self, key: str) -> str | None:
        return self.store.get(key)

    def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.store[key] = value

    def xrevrange(self, key: str, *args: object, count: int | None = None, **kwargs: object) -> list:
        rows = list(reversed(self.streams.get(key, [])))
        return rows[: int(count)] if count else rows


def _pos():
    class P:
        engine_id = "SCALP_V2"
        cost_basis = 117.94
        entry_price = 117.94
        highest_price = 117.94
        lowest_price = 117.94
        adaptive_decision: dict = {}
        symbol = "SOL/USDT"

    return P()


def _tape(mem: _MemRedis, price: float, trade_ts: float) -> None:
    ms = int(trade_ts * 1000)
    mem.streams["scalp:tape:SOLUSDT"] = [
        ("1-0", {"s": "SOLUSDT", "a": "1", "p": str(price), "q": "0.424", "m": "0", "T": str(ms)}),
    ]


def test_crossed_book_is_not_a_top_of_book(monkeypatch) -> None:
    from backend.services.binance_scalp.market_reader import ScalpMarketReader

    reader = ScalpMarketReader.__new__(ScalpMarketReader)
    reader._redis = object()
    monkeypatch.setattr(
        "backend.services.binance_scalp.market_reader._read_ws_depth",
        lambda _r, _s: ([[117.80, 1.0]], [[117.70, 1.0]], 0.1),
    )
    assert reader.read_top_of_book("SOLUSDT") is None


def test_stale_book_is_not_a_quote() -> None:
    mem = _MemRedis()
    mem.store["scalp:ws_depth:SOLUSDT"] = json.dumps({"fetched_at": time.time() - 5, "bids": [[117.75, 1.0]], "asks": [[117.76, 1.0]]})
    assert _read_ws_depth(mem, "SOLUSDT") is None  # type: ignore[arg-type]


def test_book_behind_the_tape_is_not_current() -> None:
    """SOL stop at 18:53:40Z: sale at 117.94, displayed ask 117.76."""
    mem = _MemRedis()
    now = 1790967220.0
    _tape(mem, 117.94, 1790967218.155)
    assert book_behind_recent_tape(mem, "SOLUSDT", 117.76, now=now) is True  # type: ignore[arg-type]


def test_older_sale_does_not_reject_a_tracked_book() -> None:
    """SOL stop at 18:56:32Z: last sale was ~45s earlier. The book may be used."""
    mem = _MemRedis()
    now = 1790967392.0
    _tape(mem, 117.89, 1790967347.331)
    assert book_behind_recent_tape(mem, "SOLUSDT", 117.85, now=now) is False  # type: ignore[arg-type]


def test_current_bid_still_triggers_adverse_stop() -> None:
    result = evaluate_scalp_v2_exit(
        position=_pos(),
        current_price=117.75,
        net_pnl_pct=-0.0022,
        hold_minutes=0.5,
        bar_low=117.90,
        symbol="SOL/USDT",
        allow_adverse_stop=True,
    )
    assert result.get("action") == "sell"
    assert result.get("reason") == SCALP_V2_EXIT_ADVERSE


def test_non_current_book_does_not_trigger_adverse_stop() -> None:
    result = evaluate_scalp_v2_exit(
        position=_pos(),
        current_price=117.75,
        net_pnl_pct=-0.0022,
        hold_minutes=0.5,
        bar_low=117.90,
        symbol="SOL/USDT",
        allow_adverse_stop=False,
    )
    assert result.get("action") == "hold"


def test_catastrophic_still_fires_without_an_adverse_quote() -> None:
    result = evaluate_scalp_v2_exit(
        position=_pos(),
        current_price=116.0,
        net_pnl_pct=-0.02,
        hold_minutes=0.5,
        bar_low=116.0,
        symbol="SOL/USDT",
        allow_adverse_stop=False,
    )
    assert result.get("reason") == SCALP_V2_EXIT_CATASTROPHIC


def test_older_depth_update_does_not_replace_the_book(monkeypatch) -> None:
    mem = _MemRedis()
    monkeypatch.setattr("backend.services.binance_scalp.market_reader.redis.from_url", lambda *_a, **_k: mem)
    monkeypatch.setattr("backend.services.binance_scalp.market_reader._WS_DEPTH_REDIS", None)
    publish_ws_depth("SOLUSDT", [[117.88, 1.0]], [[117.89, 1.0]], last_update_id=200)
    publish_ws_depth("SOLUSDT", [[117.75, 1.0]], [[117.76, 1.0]], last_update_id=100)
    stored = json.loads(mem.store["scalp:ws_depth:SOLUSDT"])
    assert stored["bids"][0][0] == 117.88
    assert stored["last_update_id"] == 200
    assert _update_id_is_newer(mem, "scalp:ws_depth:SOLUSDT", 100) is False  # type: ignore[arg-type]
    assert _update_id_is_newer(mem, "scalp:ws_depth:SOLUSDT", 201) is True  # type: ignore[arg-type]
