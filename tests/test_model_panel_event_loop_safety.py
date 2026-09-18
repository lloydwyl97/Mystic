"""Regression guards for the 2026-09-18 API outage.

The dashboard polls /api/portfolio-engine/model-panel on every background cycle.
That handler used to run a multi-minute sklearn + SQLite build directly on the
event loop, so a single poll froze every endpoint on the process, and repeated
polls stacked more copies of the same build. Two invariants keep that fixed:

1. ensure_ai_canonical_tables() is memoized per database path, so the hot
   diagnostics paths stop taking a write lock on a multi-gigabyte database.
2. get_model_panel() offloads to a worker thread and serves a single-flight
   TTL cache, so the event loop stays responsive and concurrent pollers share
   one build.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
import time

import pytest

from backend.endpoints import portfolio_engine_endpoints as pee
from backend.services import ai_canonical_storage as acs


@pytest.fixture(autouse=True)
def _reset_state():
    acs._ENSURED_PATHS.clear()
    pee._model_panel_cache = None
    yield
    acs._ENSURED_PATHS.clear()
    pee._model_panel_cache = None


def test_ensure_ai_canonical_tables_runs_once_per_path(tmp_path, monkeypatch):
    db = tmp_path / "canon.db"
    calls: list[str] = []
    real_connect = sqlite3.connect

    def counting_connect(path, *a, **kw):
        calls.append(str(path))
        return real_connect(path, *a, **kw)

    monkeypatch.setattr(acs.sqlite3, "connect", counting_connect)

    acs.ensure_ai_canonical_tables(db)
    first = len(calls)
    assert first >= 1, "first call must actually create the schema"

    for _ in range(25):
        acs.ensure_ai_canonical_tables(db)
    assert len(calls) == first, "repeat calls must not reopen a write connection"

    acs.ensure_ai_canonical_tables(db, force=True)
    assert len(calls) > first, "force=True must re-apply the schema"


def test_ensure_ai_canonical_tables_still_creates_tables(tmp_path):
    db = tmp_path / "canon.db"
    acs.ensure_ai_canonical_tables(db)
    with sqlite3.connect(db) as conn:
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "ai_inference_log" in names


def test_feature_dim_backfill_is_bounded():
    assert isinstance(acs._BACKFILL_ROW_LIMIT, int)
    assert 0 < acs._BACKFILL_ROW_LIMIT <= 100_000


def test_model_panel_does_not_block_event_loop(monkeypatch):
    """A slow build must not stop the loop from servicing other coroutines."""
    build_started = threading.Event()

    def slow_build() -> dict[str, object]:
        build_started.set()
        time.sleep(0.75)
        return {"success": True, "data": {"per_symbol": []}}

    monkeypatch.setattr(pee, "_build_model_panel_sync", slow_build)

    async def scenario() -> int:
        ticks = 0
        panel = asyncio.create_task(pee.get_model_panel())
        while not panel.done():
            await asyncio.sleep(0.01)
            ticks += 1
        await panel
        return ticks

    ticks = asyncio.run(scenario())
    assert build_started.is_set()
    # A blocking call on the loop would let through ~0 ticks.
    assert ticks > 10, f"event loop was starved during build (only {ticks} ticks)"


def test_model_panel_is_single_flight_and_cached(monkeypatch):
    """Concurrent pollers share one build; later polls hit the cache."""
    builds = 0
    lock = threading.Lock()

    def counting_build() -> dict[str, object]:
        nonlocal builds
        with lock:
            builds += 1
        time.sleep(0.2)
        return {"success": True, "data": {"per_symbol": []}}

    monkeypatch.setattr(pee, "_build_model_panel_sync", counting_build)

    async def scenario():
        first = await asyncio.gather(*(pee.get_model_panel() for _ in range(6)))
        second = await pee.get_model_panel()
        return first, second

    results, cached = asyncio.run(scenario())

    assert builds == 1, f"expected one shared build, got {builds}"
    assert all(r is results[0] for r in results)
    assert cached is results[0], "subsequent poll must be served from cache"
