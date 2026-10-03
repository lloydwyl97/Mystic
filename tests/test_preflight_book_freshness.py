"""Fill preflight dates the book by its source, not by local processing."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, patch

import pytest

from backend.config.protected_execution import ORDERBOOK_MAX_AGE_SEC, ORDERBOOK_MISSING, ORDERBOOK_STALE
from backend.services import protected_limit_execution as ple
from backend.services.protected_limit_execution import (
    FRESHNESS_FRESH,
    FRESHNESS_MISSING_UPDATE_ID,
    FRESHNESS_OUT_OF_ORDER,
    FRESHNESS_STALE,
    ExecutableBook,
    assess_book_freshness,
    execution_latency_fields,
    reset_book_identity,
    run_protected_preflight,
)
from backend.utils.symbols import normalize_symbol

BIDS = [[117.88, 5.0], [117.87, 5.0]]
ASKS = [[117.89, 5.0], [117.90, 5.0]]


@pytest.fixture(autouse=True)
def _clean_identity():
    reset_book_identity()
    yield
    reset_book_identity()


def _book(update_id, sent, received=None, bids=None, asks=None):
    return ExecutableBook(
        bids=BIDS if bids is None else bids,
        asks=ASKS if asks is None else asks,
        last_update_id=update_id,
        source_ts=sent,
        receive_ts=sent if received is None else received,
    )


async def _preflight(book, side="BUY"):
    with patch("backend.services.protected_limit_execution._fetch_order_book", AsyncMock(return_value=book)):
        return await run_protected_preflight(
            symbol="SOL/USDT",
            side=side,
            quantity=0.1,
            reference_price=117.885,
            live_capable=True,
        )


@pytest.mark.asyncio
async def test_fresh_book_passes_and_age_is_measured():
    sent = time.time() - 0.4
    pf = await _preflight(_book(100, sent, sent + 0.3))
    assert pf.passed is True
    assert pf.book_freshness["freshness_result"] == FRESHNESS_FRESH
    assert pf.book_age_sec is not None and 0.3 < pf.book_age_sec < 2.0
    audit = pf.to_audit_dict()["book_freshness"]
    for key in (
        "source_book_timestamp",
        "local_receive_timestamp",
        "processing_timestamp",
        "book_age_ms",
        "last_update_id",
        "best_bid",
        "best_ask",
        "spread",
        "freshness_result",
    ):
        assert key in audit
    assert audit["last_update_id"] == 100
    assert audit["best_bid"] == 117.88 and audit["best_ask"] == 117.89


@pytest.mark.asyncio
async def test_stale_book_fails():
    sent = time.time() - (ORDERBOOK_MAX_AGE_SEC + 2.0)
    pf = await _preflight(_book(100, sent))
    assert pf.passed is False
    assert pf.reject_reason == ORDERBOOK_STALE
    assert pf.book_freshness["freshness_result"] == FRESHNESS_STALE


@pytest.mark.asyncio
async def test_backlog_processed_late_stays_stale():
    """Sent long ago, received and processed just now: still old."""
    now = time.time()
    pf = await _preflight(_book(100, now - (ORDERBOOK_MAX_AGE_SEC + 3.0), received=now))
    assert pf.passed is False
    assert pf.reject_reason == ORDERBOOK_STALE
    assert pf.book_age_sec > ORDERBOOK_MAX_AGE_SEC


def test_older_update_id_cannot_replace_newer():
    now = time.time()
    assert assess_book_freshness("SOLUSDT", _book(200, now), now=now)["freshness_result"] == FRESHNESS_FRESH
    late = assess_book_freshness("SOLUSDT", _book(150, now + 0.5), now=now + 0.6)
    assert late["freshness_result"] == FRESHNESS_OUT_OF_ORDER
    assert ple._ACCEPTED_BOOKS[normalize_symbol("SOLUSDT")].update_id == 200


def test_duplicate_update_id_does_not_refresh_age():
    t0 = 1_000_000.0
    first = assess_book_freshness("SOLUSDT", _book(300, t0), now=t0 + 0.2)
    assert first["freshness_result"] == FRESHNESS_FRESH
    again = assess_book_freshness("SOLUSDT", _book(300, t0 + 4.0), now=t0 + ORDERBOOK_MAX_AGE_SEC + 1.0)
    assert again["duplicate_update_id"] is True
    assert again["source_book_timestamp"] == t0
    assert again["freshness_result"] == FRESHNESS_STALE


def test_newer_update_id_is_accepted():
    t0 = 1_000_000.0
    assess_book_freshness("SOLUSDT", _book(300, t0), now=t0 + 0.2)
    nxt = assess_book_freshness("SOLUSDT", _book(301, t0 + 3.0), now=t0 + 3.4)
    assert nxt["freshness_result"] == FRESHNESS_FRESH
    assert ple._ACCEPTED_BOOKS[normalize_symbol("SOLUSDT")].update_id == 301


@pytest.mark.asyncio
async def test_out_of_order_book_fails_preflight():
    now = time.time()
    assert (await _preflight(_book(500, now))).passed is True
    pf = await _preflight(_book(499, time.time()))
    assert pf.passed is False
    assert pf.reject_reason == ORDERBOOK_STALE
    assert pf.book_freshness["freshness_result"] == FRESHNESS_OUT_OF_ORDER


@pytest.mark.asyncio
async def test_missing_update_id_fails_closed():
    pf = await _preflight(_book(None, time.time()))
    assert pf.passed is False
    assert pf.reject_reason == ORDERBOOK_STALE
    assert pf.book_freshness["freshness_result"] == FRESHNESS_MISSING_UPDATE_ID


@pytest.mark.asyncio
async def test_crossed_book_fails():
    pf = await _preflight(_book(100, time.time(), bids=[[117.90, 5.0]], asks=[[117.80, 5.0]]))
    assert pf.passed is False
    assert pf.reject_reason == ORDERBOOK_MISSING


@pytest.mark.asyncio
async def test_zero_spread_fails():
    pf = await _preflight(_book(100, time.time(), bids=[[117.88, 5.0]], asks=[[117.88, 5.0]]))
    assert pf.passed is False
    assert pf.reject_reason == ORDERBOOK_MISSING


@pytest.mark.asyncio
@pytest.mark.parametrize("bids,asks", [([], ASKS), (BIDS, [])])
async def test_missing_side_fails(bids, asks):
    pf = await _preflight(_book(100, time.time(), bids=bids, asks=asks))
    assert pf.passed is False
    assert pf.reject_reason == ORDERBOOK_MISSING


@pytest.mark.asyncio
async def test_current_book_passes_for_sell():
    pf = await _preflight(_book(100, time.time() - 0.2), side="SELL")
    assert pf.passed is True
    assert pf.expected_avg_fill == pytest.approx(117.88)


@pytest.mark.asyncio
async def test_live_service_order_book_carries_identity_and_timing(monkeypatch):
    from backend.services import live_market_data as lmd

    monkeypatch.setattr(
        "backend.utils.binance_credentials.get_binance_us_credentials",
        lambda: ("key", "secret"),
    )
    service = lmd.LiveMarketDataService()

    class _Client:
        def fetch_order_book(self, *_a, **_k):
            self.lastRestRequestTimestamp = int(time.time() * 1000)
            time.sleep(0.05)
            return {"bids": BIDS, "asks": ASKS, "nonce": 777, "timestamp": None}

    monkeypatch.setattr(lmd.ccxt, "binanceus", lambda *_a, **_k: _Client())
    ob = await service.get_order_book("SOL/USDT", limit=5)
    assert ob["last_update_id"] == 777
    assert ob["request_sent_ts"] is not None and ob["received_ts"] >= ob["request_sent_ts"]
    assert ob["received_ts"] - ob["request_sent_ts"] >= 0.04


@pytest.mark.asyncio
async def test_book_age_is_not_hardcoded_zero(monkeypatch):
    sent = time.time() - 1.25

    class _Svc:
        async def get_order_book(self, *_a, **_k):
            return {"bids": BIDS, "asks": ASKS, "last_update_id": 9, "request_sent_ts": sent, "received_ts": sent + 0.5}

    monkeypatch.setattr("backend.services.live_market_data.live_market_data_service", _Svc())
    book = await ple._fetch_order_book("SOL/USDT")
    assert book is not None and book.source_ts == sent
    with patch("backend.services.protected_limit_execution._fetch_order_book", AsyncMock(return_value=book)):
        pf = await run_protected_preflight(symbol="SOL/USDT", side="BUY", quantity=0.1, reference_price=117.885, live_capable=True)
    assert pf.book_age_sec is not None
    assert pf.book_age_sec >= 1.2


def test_execution_latency_fields():
    order = {
        "_mystic_latency": {"order_submit_timestamp": 1000.0, "order_response_timestamp": 1000.4},
        "info": {"transactTime": 1000250},
        "trades": [{"timestamp": 1000300}],
    }
    lat = execution_latency_fields(order)
    assert lat["exchange_ack_timestamp"] == pytest.approx(1000.25)
    assert lat["fill_timestamp"] == pytest.approx(1000.3)
    assert lat["submit_to_ack_ms"] == pytest.approx(250.0)
    assert lat["submit_to_fill_ms"] == pytest.approx(300.0)


def test_scalp_stop_guard_unchanged():
    import inspect

    from backend.services.binance_scalp.market_reader import book_behind_recent_tape
    from backend.services.scalp_v2.exit_calibration import scalp_v2_max_adverse_net_pct
    from backend.services.scalp_v2.exit_evaluator import evaluate_scalp_v2_exit

    assert "allow_adverse_stop" in inspect.signature(evaluate_scalp_v2_exit).parameters
    assert "scalp_v2_max_adverse_net_pct" in inspect.getsource(book_behind_recent_tape)
    assert scalp_v2_max_adverse_net_pct("SOLUSDT") == pytest.approx(0.0015)


def test_day_setups_and_slots_unchanged():
    from backend.services.day_v2.live_signal import ENABLED_SETUPS

    assert frozenset({"HTF_TREND_PULLBACK", "BREAKOUT_CONTINUATION", "RANGE_BOUNCE", "VWAP_REVERSION", "EXHAUSTION_MR"}) == ENABLED_SETUPS
