import time

from backend.config.day_entry_execution import BOOK_STALE_SEC
from backend.services.day_liquidity_gate import apply_liquidity_gate_to_decision_data
from backend.services.spread_book_telemetry import (
    book_redis_key,
    orderbook_redis_key,
    read_market_book,
    shadow_liquidity_compare,
)


class _FakeRedis:
    def __init__(self, hashes: dict[str, dict[str, str]]) -> None:
        self._hashes = hashes

    def hgetall(self, key: str) -> dict[str, str]:
        return dict(self._hashes.get(key) or {})


def _book(bid: float, ask: float, age: float, source: str, ts_field: str) -> dict[str, str]:
    return {"bid": str(bid), "ask": str(ask), ts_field: str(time.time() - age), "source": source}


def test_read_market_book_prefers_fresher_websocket_over_stale_canonical_mark():
    sym = "XRP/USDT"
    redis = _FakeRedis(
        {
            orderbook_redis_key(sym): _book(1.4661, 1.4662, 0.5, "websocket", "updated_at"),
            book_redis_key(sym): _book(1.4667, 1.4671, 18.0, "canonical_mark", "timestamp"),
        }
    )
    book = read_market_book(redis, sym)
    assert book is not None
    assert book["source"] == "websocket"
    assert book["ask"] == 1.4662
    assert book["freshness_sec"] < 2.0


def test_read_market_book_falls_back_to_canonical_mark_when_websocket_is_older():
    sym = "BTC/USDT"
    redis = _FakeRedis(
        {
            orderbook_redis_key(sym): _book(79000.0, 79001.0, 45.0, "websocket", "updated_at"),
            book_redis_key(sym): _book(79100.0, 79102.0, 3.0, "canonical_mark", "timestamp"),
        }
    )
    book = read_market_book(redis, sym)
    assert book is not None
    assert book["source"] == "canonical_mark"
    assert book["ask"] == 79102.0


def test_read_market_book_uses_ts_utc_when_updated_at_missing():
    sym = "SOL/USDT"
    raw = _book(104.11, 104.12, 1.0, "websocket", "ts_utc")
    book = read_market_book(_FakeRedis({orderbook_redis_key(sym): raw}), sym)
    assert book is not None
    assert book["source"] == "websocket"


def test_read_market_book_rejects_untimestamped_and_inverted_books():
    sym = "ETH/USDT"
    no_ts = {"bid": "2570.0", "ask": "2571.0", "source": "websocket"}
    assert read_market_book(_FakeRedis({orderbook_redis_key(sym): no_ts}), sym) is None
    inverted = _book(2571.0, 2570.0, 0.5, "websocket", "updated_at")
    assert read_market_book(_FakeRedis({orderbook_redis_key(sym): inverted}), sym) is None
    assert read_market_book(_FakeRedis({}), sym) is None


def test_book_payload_freshness_uses_single_book_stale_threshold():
    sym = "BTC/USDT"
    inside = read_market_book(_FakeRedis({orderbook_redis_key(sym): _book(1.0, 1.1, BOOK_STALE_SEC - 5.0, "websocket", "updated_at")}), sym)
    outside = read_market_book(_FakeRedis({orderbook_redis_key(sym): _book(1.0, 1.1, BOOK_STALE_SEC + 5.0, "websocket", "updated_at")}), sym)
    assert inside is not None and inside["fresh"] is True
    assert outside is not None and outside["fresh"] is False


def test_shadow_does_not_change_live_liquidity_factor():
    dd = {"spread_pct": 0.0004}
    live = apply_liquidity_gate_to_decision_data(dict(dd), "BTC/USDT")
    before = live["liquidity_quality_size_factor"]
    shadow = shadow_liquidity_compare(
        symbol="BTCUSDT",
        current_decision_data=dd,
        real_spread_bps=0.59,
        current_notional_usd=4000.0,
    )
    after = apply_liquidity_gate_to_decision_data(dict(dd), "BTC/USDT")
    assert after["liquidity_quality_size_factor"] == before
    assert shadow["live_sizing_unchanged"] is True
    assert shadow["proposed_real_spread_liquidity_credit"] >= shadow["current_fallback_liquidity_credit"]
    assert shadow["difference_usd"] != 0.0 or shadow["proposed_position_size_usd"] == 4000.0
