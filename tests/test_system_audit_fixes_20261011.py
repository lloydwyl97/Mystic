"""Regressions for the 2026-10-11 system audit: order idempotency, kill-switch
read failure, cross-engine buy guard, DAY executable bid, operator pause."""

from __future__ import annotations

import ast
import inspect
import sqlite3
import textwrap
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.services import protected_limit_execution as ple
from backend.services.protected_limit_execution import execute_protected_limit_live


async def _no_sleep(_s):
    return None


class _Venue:
    def __init__(self, *, place_results, probe):
        self.place_calls: list[dict] = []
        self._place_results = list(place_results)
        self.fetch_order = AsyncMock(return_value=probe)
        self.cancel_order = AsyncMock(return_value={"status": "success"})
        self.binance = None

    async def _ensure_initialized(self):
        return None

    async def place_order(self, **kwargs):
        self.place_calls.append(kwargs)
        out = self._place_results.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    monkeypatch.setattr(ple.asyncio, "sleep", _no_sleep)

    async def _enrich(_svc, order, _sym):
        return order

    monkeypatch.setattr(ple, "_enrich_live_order_fills", _enrich)


@pytest.mark.asyncio
async def test_ioc_error_that_reached_the_venue_is_adopted_not_resubmitted():
    filled = {"id": "77", "filled": 0.01, "amount": 0.01, "average": 100.0, "status": "closed"}
    venue = _Venue(place_results=[TimeoutError("read timed out")], probe={"status": "success", "order": filled})

    out = await execute_protected_limit_live(venue, symbol="BTCUSDT", side="buy", quantity=0.01, limit_price=100.0, client_order_id="abc123")

    assert out is not None and out["id"] == "77"
    assert len(venue.place_calls) == 1
    venue.fetch_order.assert_awaited_once()
    assert venue.fetch_order.await_args.args[1] == venue.place_calls[0]["client_order_id"]


@pytest.mark.asyncio
async def test_unconfirmed_ioc_never_sends_a_second_order():
    venue = _Venue(place_results=[{"status": "error", "message": "timeout"}], probe={"status": "error", "message": "network", "code": ""})

    out = await execute_protected_limit_live(venue, symbol="ETHUSDT", side="sell", quantity=0.1, limit_price=2500.0)

    assert out is None
    assert len(venue.place_calls) == 1


@pytest.mark.asyncio
async def test_gtc_only_after_venue_says_the_ioc_does_not_exist():
    gtc = {"status": "success", "order": {"id": "88", "filled": 1.0, "amount": 1.0, "average": 1.4, "status": "closed"}}
    venue = _Venue(place_results=[{"status": "error", "message": "-1013"}, gtc], probe={"status": "error", "code": "-2013"})

    out = await execute_protected_limit_live(venue, symbol="XRPUSDT", side="buy", quantity=1.0, limit_price=1.4, client_order_id="0123456789abcdef0123456789abcdef")

    assert out is not None and out["id"] == "88"
    ids = [c["client_order_id"] for c in venue.place_calls]
    assert len(ids) == 2 and ids[0] != ids[1]
    assert all(cid and len(cid) <= 36 for cid in ids)


@pytest.mark.asyncio
async def test_every_live_order_carries_a_client_order_id():
    ok = {"status": "success", "order": {"id": "1", "filled": 0.5, "amount": 0.5, "average": 110.0, "status": "closed"}}
    venue = _Venue(place_results=[ok], probe={"status": "error"})

    await execute_protected_limit_live(venue, symbol="SOLUSDT", side="buy", quantity=0.5, limit_price=110.0)

    assert venue.place_calls[0]["client_order_id"]


def test_entry_control_read_failure_is_not_resume(tmp_path):
    from backend.services.circuit_breaker_service import read_persisted_entry_control

    db = tmp_path / "broken.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE portfolio_engine_ledger (id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()
    db.write_bytes(b"not a database" * 100)

    out = read_persisted_entry_control(str(db))

    assert out["read_failed"] is True
    assert out["entry_permitted"] is False


def test_unreadable_entry_control_keeps_a_memory_pause(monkeypatch, tmp_path):
    from backend.services import circuit_breaker_service as cbs
    from backend.services.portfolio_engine import KillSwitchMode, PortfolioEngine

    engine = PortfolioEngine.__new__(PortfolioEngine)
    engine.db_path = str(tmp_path / "x.db")
    engine._kill_switch_mode = KillSwitchMode.PAUSE_BUYS
    engine._kill_switch_reason = "operator"
    engine._canonical_failsafe_decision = lambda: {}
    monkeypatch.setattr(cbs, "read_persisted_entry_control", lambda _p: {"requested_kill_mode": "RESUME", "read_failed": True})

    ctrl = engine._effective_entry_control()

    assert ctrl["effective_entry_permitted"] is False
    assert engine._kill_switch_mode is KillSwitchMode.PAUSE_BUYS
    assert "entry_control_unreadable" in str(ctrl["blocking_reason"])


def test_readable_resume_still_releases_a_stale_memory_pause(monkeypatch, tmp_path):
    from backend.services import circuit_breaker_service as cbs
    from backend.services.portfolio_engine import KillSwitchMode, PortfolioEngine

    engine = PortfolioEngine.__new__(PortfolioEngine)
    engine.db_path = str(tmp_path / "x.db")
    engine._kill_switch_mode = KillSwitchMode.PAUSE_BUYS
    engine._kill_switch_reason = "old"
    engine._canonical_failsafe_decision = lambda: {}
    monkeypatch.setattr(cbs, "read_persisted_entry_control", lambda _p: {"requested_kill_mode": "RESUME"})

    assert engine._effective_entry_control()["effective_entry_permitted"] is True


def _positions_db(path):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE portfolio_engine_positions (symbol TEXT, quantity REAL, trade_id TEXT, entry_price REAL, status TEXT, engine_id TEXT, PRIMARY KEY (engine_id, symbol))")
    conn.execute("INSERT INTO portfolio_engine_positions VALUES ('BTC/USDT', 0.002, 'scalp_v2_BTCUSDT_1', 83000.0, 'ACTIVE', 'SCALP_V2')")
    conn.commit()
    conn.close()


def test_scalp_lot_is_not_a_day_duplicate_in_the_authoritative_read(tmp_path):
    from backend.services.portfolio_engine import PortfolioEngine

    db = tmp_path / "pos.db"
    _positions_db(db)
    engine = PortfolioEngine.__new__(PortfolioEngine)
    engine.db_path = str(db)

    assert engine._fetch_authoritative_open_position_sync("BTC/USDT", "DAY_V2") is None
    assert engine._fetch_authoritative_open_position_sync("BTC/USDT", "SCALP_V2") is not None


def test_day_buy_duplicate_guard_is_engine_scoped():
    from backend.services.portfolio_engine import PortfolioEngine

    src = textwrap.dedent(inspect.getsource(PortfolioEngine._execute_buy_fifo_locked))
    tree = ast.parse(src)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and any(isinstance(a, ast.Attribute) and a.attr == "_fetch_authoritative_open_position_sync" for a in n.args)]
    assert calls, "duplicate guard must read the authoritative row"
    assert all(len(c.args) >= 3 for c in calls), "the authoritative read must pass the buying engine"
    assert "self.open_positions.get(normalized_symbol)" not in src


def test_live_buy_without_a_venue_order_is_refused_and_a_venue_fill_is_never_dropped():
    from backend.services.portfolio_engine import PortfolioEngine

    src = inspect.getsource(PortfolioEngine._execute_buy_fifo_locked)
    refuse = src.index('if buy_mode == "live" and not live_order_buy:')
    invariant = src.index("if total_cost > free_cash and live_order_buy:")
    commit = src.index("_commit_atomic_day_open_sync")
    assert src.index("live_order_buy = await execute_protected_limit_live(") < refuse < invariant < commit
    keep = src[invariant : src.index("elif total_cost > free_cash:")]
    assert "return None" not in keep


def test_day_continuation_decides_on_the_executable_bid():
    from backend.services.portfolio_engine import PortfolioEngine

    src = inspect.getsource(PortfolioEngine._check_exit_conditions)
    assert "executable_book(symbol)" in src
    assert "live_day_decision(self.db_path, position, _decision_price" in src


def test_executable_book_is_fresh_or_absent(monkeypatch):
    from backend.services import exact_policy_capture as epc
    from backend.services import spread_book_telemetry as sbt

    monkeypatch.setattr(sbt, "read_market_book", lambda _r, _s: {"bid": 1.3979, "ask": 1.3981, "freshness_sec": 1.0})
    monkeypatch.setattr("backend.config.redis_config.get_redis_client", object)
    assert epc.executable_book("XRPUSDT") == (1.3979, 1.3981)

    monkeypatch.setattr(sbt, "read_market_book", lambda _r, _s: {"bid": 1.3979, "ask": 1.3981, "freshness_sec": 999.0})
    assert epc.executable_book("XRPUSDT") == (None, None)


@pytest.mark.asyncio
async def test_system_pause_endpoint_persists_the_kill_switch(monkeypatch):
    from backend import app_factory

    calls: list[tuple[str, str]] = []

    async def _set(mode, reason=""):
        calls.append((mode, reason))
        return {"success": True}

    monkeypatch.setattr("backend.services.portfolio_engine.get_portfolio_engine", lambda: SimpleNamespace(set_kill_switch=_set))
    app = app_factory.create_app()
    routes = {getattr(r, "path", ""): r for r in app.routes}

    paused = await routes["/api/system/pause-trading"].endpoint(None)
    resumed = await routes["/api/system/resume-trading"].endpoint(None)

    assert calls == [("PAUSE_BUYS", "api_system_pause_trading"), ("RESUME", "api_system_resume_trading")]
    assert paused["status"] == "paused" and resumed["status"] == "resumed"
    app_factory.resume_trading()
