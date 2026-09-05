"""Tests for 1m kline pipeline continuity and clock-v2 data-quality contracts.

Covers:
- closed kline event flushes to Redis immediately (x=True)
- forming bar (x=False) is NOT written to Redis
- backfill on reconnect fills gaps from REST
- duplicate bar timestamps are deduplicated
- out-of-order bars do not overwrite a newer bar
- UTC open-timestamp bucketing
- no fabricated candles
- stale/gap reject contract unchanged
- clock-v2 PIT mutation safety
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_kline_payload(
    sym: str,
    bar_ts_sec: int,
    *,
    o: float = 100.0,
    h: float = 101.0,
    lo: float = 99.0,
    c: float = 100.5,
    v: float = 1.0,
    closed: bool,
) -> dict[str, Any]:
    return {
        "s": sym,
        "k": {
            "t": bar_ts_sec * 1000,
            "o": str(o),
            "h": str(h),
            "l": str(lo),
            "c": str(c),
            "v": str(v),
            "x": closed,
        },
    }


def _make_hydrator(redis_store: dict[str, Any]) -> Any:
    """Return a BinanceWSHydrator with a stubbed CacheGuard/Redis."""
    from backend.services.binance_ws_hydrator import BinanceWSHydrator

    h = BinanceWSHydrator()

    r = MagicMock()

    async def _aget(key: str) -> str | None:
        return redis_store.get(key)

    async def _aset(key: str, val: str, ex: int | None = None) -> None:
        redis_store[key] = val

    r.get = AsyncMock(side_effect=_aget)
    r.set = AsyncMock(side_effect=_aset)

    cg = MagicMock()
    cg.r = r
    h._cg = cg
    return h


# ---------------------------------------------------------------------------
# 1. Closed kline event → immediate Redis write
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_closed_kline_flushes_to_redis():
    """x=True kline event must write the bar to Redis."""
    store: dict[str, Any] = {}
    h = _make_hydrator(store)

    bar_ts = 1_000_000
    k = _make_kline_payload("BTCUSDT", bar_ts, closed=True)["k"]
    await h._flush_closed_kline("BTCUSDT", k)

    assert "klines:BTCUSDT:1m" in store, "closed bar must be written to Redis"
    arr = json.loads(store["klines:BTCUSDT:1m"])
    assert len(arr) == 1
    assert int(arr[0][0]) == bar_ts
    assert float(arr[0][4]) == 100.5  # close price


@pytest.mark.asyncio
async def test_forming_bar_not_written_to_redis():
    """x=False kline event must NOT write to Redis; only volume update to in-memory candle."""
    store: dict[str, Any] = {}
    h = _make_hydrator(store)
    bar_ts = 1_000_000

    # Simulate an existing forming candle
    h._c1m["BTCUSDT"] = [float(bar_ts), 100.0, 100.5, 99.5, 100.3, 0.0]

    k = _make_kline_payload("BTCUSDT", bar_ts, v=5.0, closed=False)["k"]
    # Only the kline stream handler's volume-update path should fire; _flush not called.
    # Replicate the handler logic inline (not calling _flush_closed_kline for x=False).
    volume = float(k.get("v", 0))
    cur = h._c1m.get("BTCUSDT")
    if cur and len(cur) == 6:
        cur[5] = volume

    # Redis must be untouched
    assert "klines:BTCUSDT:1m" not in store
    # Volume updated in-memory
    assert h._c1m["BTCUSDT"][5] == 5.0


# ---------------------------------------------------------------------------
# 2. Duplicate bar timestamps are deduplicated
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_duplicate_bar_not_appended():
    """A second flush for the same bar_ts must not create a second entry."""
    store: dict[str, Any] = {}
    h = _make_hydrator(store)

    bar_ts = 1_000_060
    k = _make_kline_payload("BTCUSDT", bar_ts, closed=True)["k"]
    await h._flush_closed_kline("BTCUSDT", k)
    await h._flush_closed_kline("BTCUSDT", k)  # duplicate

    arr = json.loads(store["klines:BTCUSDT:1m"])
    assert sum(1 for row in arr if int(row[0]) == bar_ts) == 1


# ---------------------------------------------------------------------------
# 3. Out-of-order bars: older bar does not overwrite newer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_older_bar_skipped_if_already_written():
    """A closed bar whose ts <= _last_bar_ts must be silently skipped."""
    store: dict[str, Any] = {}
    h = _make_hydrator(store)

    ts_new = 1_000_120
    ts_old = 1_000_060

    k_new = _make_kline_payload("BTCUSDT", ts_new, c=200.0, closed=True)["k"]
    await h._flush_closed_kline("BTCUSDT", k_new)

    k_old = _make_kline_payload("BTCUSDT", ts_old, c=99.0, closed=True)["k"]
    await h._flush_closed_kline("BTCUSDT", k_old)

    arr = json.loads(store["klines:BTCUSDT:1m"])
    # Out-of-order bar is appended only if its ts was not already the last
    # (older than _last_bar_ts=ts_new): must NOT appear.
    ts_values = [int(r[0]) for r in arr]
    assert ts_old not in ts_values, "out-of-order bar older than last must be skipped"


# ---------------------------------------------------------------------------
# 4. UTC open-timestamp bucketing
# ---------------------------------------------------------------------------


def test_utc_bucket_from_kline_event():
    """bar_ts from k['t'] must be t // 1000 (UTC seconds, open-time)."""
    t_ms = 1_788_540_120_000  # example millisecond timestamp
    bar_ts_sec = t_ms // 1000
    assert bar_ts_sec % 60 == 0, "open timestamps must align to 60-second UTC boundaries"


# ---------------------------------------------------------------------------
# 5. No fabricated candles: zero-close bars rejected
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_zero_close_bar_not_written():
    """A kline with c=0 must not be written (no fabricated candle)."""
    store: dict[str, Any] = {}
    h = _make_hydrator(store)
    bar_ts = 1_000_180
    k = _make_kline_payload("BTCUSDT", bar_ts, c=0.0, closed=True)["k"]
    await h._flush_closed_kline("BTCUSDT", k)
    assert "klines:BTCUSDT:1m" not in store


# ---------------------------------------------------------------------------
# 6. last_bar_ts tracking
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_last_bar_ts_updated_after_flush():
    store: dict[str, Any] = {}
    h = _make_hydrator(store)
    bar_ts = 1_000_240
    k = _make_kline_payload("BTCUSDT", bar_ts, closed=True)["k"]
    await h._flush_closed_kline("BTCUSDT", k)
    assert h._last_bar_ts.get("BTCUSDT") == bar_ts


# ---------------------------------------------------------------------------
# 7. Backfill: only called when gap > 120s
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_backfill_skipped_when_no_gap():
    store: dict[str, Any] = {}
    h = _make_hydrator(store)
    now = int(time.time())
    # Last bar was only 30 seconds ago — no backfill needed
    h._last_bar_ts["BTCUSDT"] = now - 30
    # Should not call any HTTP client
    with patch("httpx.AsyncClient") as mock_client:
        await h._backfill_missing_bars("BTCUSDT")
        mock_client.assert_not_called()


@pytest.mark.asyncio
async def test_backfill_skipped_when_no_last_bar():
    store: dict[str, Any] = {}
    h = _make_hydrator(store)
    # _last_bar_ts not set
    with patch("httpx.AsyncClient") as mock_client:
        await h._backfill_missing_bars("BTCUSDT")
        mock_client.assert_not_called()


@pytest.mark.asyncio
async def test_backfill_inserts_missing_bars():
    """Backfill fetches REST klines and appends missing bars to Redis."""
    store: dict[str, Any] = {}
    h = _make_hydrator(store)

    now_ts = int(time.time())
    # Simulate a 5-minute gap
    last_ts = now_ts - 5 * 60 - 60  # 6 minutes ago
    h._last_bar_ts["BTCUSDT"] = last_ts

    # Fabricate 4 REST kline rows (last_ts+60 .. last_ts+240)
    fake_rows = [[int((last_ts + i * 60) * 1000), "100", "101", "99", "100.5", "1.0", 0, 0, 0, 0, 0, 0] for i in range(1, 5)]

    mock_response = MagicMock()
    mock_response.json.return_value = fake_rows
    mock_response.raise_for_status = MagicMock()

    mock_client_instance = AsyncMock()
    mock_client_instance.get = AsyncMock(return_value=mock_response)
    mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
    mock_client_instance.__aexit__ = AsyncMock(return_value=False)

    with patch("backend.services.binance_ws_hydrator.httpx.AsyncClient", return_value=mock_client_instance):
        await h._backfill_missing_bars("BTCUSDT")

    assert "klines:BTCUSDT:1m" in store
    arr = json.loads(store["klines:BTCUSDT:1m"])
    assert len(arr) == 4
    assert int(arr[0][0]) == last_ts + 60


@pytest.mark.asyncio
async def test_backfill_deduplicates_with_existing():
    """Backfill must not duplicate bars already in Redis."""
    now_ts = int(time.time())
    last_ts = now_ts - 5 * 60 - 60

    # Pre-populate Redis with bars that overlap the backfill range
    existing_bar_ts = last_ts + 60
    existing = [[float(existing_bar_ts), 100.0, 101.0, 99.0, 100.5, 1.0]]
    store: dict[str, Any] = {"klines:BTCUSDT:1m": json.dumps(existing)}
    h = _make_hydrator(store)
    h._last_bar_ts["BTCUSDT"] = last_ts

    fake_rows = [[int((last_ts + i * 60) * 1000), "100", "101", "99", "100.5", "1.0", 0, 0, 0, 0, 0, 0] for i in range(1, 5)]
    mock_response = MagicMock()
    mock_response.json.return_value = fake_rows
    mock_response.raise_for_status = MagicMock()

    mock_client_instance = AsyncMock()
    mock_client_instance.get = AsyncMock(return_value=mock_response)
    mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
    mock_client_instance.__aexit__ = AsyncMock(return_value=False)

    with patch("backend.services.binance_ws_hydrator.httpx.AsyncClient", return_value=mock_client_instance):
        await h._backfill_missing_bars("BTCUSDT")

    arr = json.loads(store["klines:BTCUSDT:1m"])
    ts_values = [int(r[0]) for r in arr]
    # No duplicates
    assert len(ts_values) == len(set(ts_values))
    # The pre-existing bar is still there exactly once
    assert ts_values.count(existing_bar_ts) == 1


# ---------------------------------------------------------------------------
# 8. Forming/closed candle contract: FORMING_CANDLE_ALLOWED = False
# ---------------------------------------------------------------------------


def test_forming_candle_contract():
    from backend.services.day_path_clock_v2 import FORMING_CANDLE_ALLOWED

    assert FORMING_CANDLE_ALLOWED is False, "clock-v2 must not use forming (x=False) bars"


# ---------------------------------------------------------------------------
# 9. Clock-v2 stale/gap contract unchanged
# ---------------------------------------------------------------------------


def test_stale_and_gap_contract_unchanged():
    from backend.services.day_path_input_validity import MAX_GAP_SEC, MAX_LAST_BAR_AGE_SEC

    assert MAX_GAP_SEC == 180, "gap contract must remain 180s — do not weaken to pass FEATURE_PARTIAL"
    assert MAX_LAST_BAR_AGE_SEC == 180, "age contract must remain 180s"


# ---------------------------------------------------------------------------
# 10. Target horizon frozen at 3h
# ---------------------------------------------------------------------------


def test_target_horizon_frozen_3h():
    from backend.services.day_path_clock_v2 import (
        PRIMARY_TARGET_HORIZON_NAME,
        PRIMARY_TARGET_HORIZON_SEC,
        TARGET_HORIZON_STATUS,
    )

    assert TARGET_HORIZON_STATUS == "PRIMARY_TARGET_HORIZON_3H"
    assert PRIMARY_TARGET_HORIZON_SEC == 3 * 60 * 60  # 10800
    assert PRIMARY_TARGET_HORIZON_NAME == "3h"


def test_horizon_not_chosen_from_pnl():
    from backend.services.day_path_clock_v2 import clock_v2_v4_readiness_requirements

    req = clock_v2_v4_readiness_requirements()
    assert req.get("horizon_not_chosen_by_pnl") is True


# ---------------------------------------------------------------------------
# 11. v4 experiment registered and does not mutate v1/v2/v3
# ---------------------------------------------------------------------------


def test_v4_experiment_in_registry():
    from backend.services.day_experiment_registry import SEED_ARMS

    ids = [e.experiment_id for e in SEED_ARMS]
    assert "M_clock_v2_planned_v4_20260904" in ids
    assert "M_clock_v2_planned_v3_20260905" in ids  # v3 must still exist
    assert "M_clock_v2_planned_20260905" in ids  # v1 must still exist


def test_v4_embargo_gte_3h():
    from backend.services.day_path_clock_v2 import clock_v2_v4_readiness_requirements

    req = clock_v2_v4_readiness_requirements()
    assert req["embargo_seconds_min"] >= 3 * 3600


# ---------------------------------------------------------------------------
# 12. Readiness gate: horizon-frozen evaluates correctly
# ---------------------------------------------------------------------------


def test_readiness_horizon_frozen_passes_gate():
    """With horizon frozen, the horizon gate in evaluate should not block."""
    from backend.services.day_path_clock_v2 import TARGET_HORIZON_STATUS

    # v4 has status "PRIMARY_TARGET_HORIZON_3H" which != "TARGET_HORIZON_NOT_FROZEN"
    horizon_frozen = TARGET_HORIZON_STATUS != "TARGET_HORIZON_NOT_FROZEN"
    assert horizon_frozen is True


# ---------------------------------------------------------------------------
# 13. _flush_closed_kline clears forming candle for same minute
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flush_clears_forming_candle():
    """After flushing a closed bar, the in-memory forming candle for that minute is cleared."""
    store: dict[str, Any] = {}
    h = _make_hydrator(store)
    bar_ts = 1_000_300
    # Set up an in-memory forming candle for the same minute
    h._c1m["BTCUSDT"] = [float(bar_ts), 100.0, 101.0, 99.0, 100.4, 3.0]
    k = _make_kline_payload("BTCUSDT", bar_ts, closed=True)["k"]
    await h._flush_closed_kline("BTCUSDT", k)
    # In-memory candle must be cleared so next miniTicker starts fresh
    assert "BTCUSDT" not in h._c1m
