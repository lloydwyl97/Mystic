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
async def test_unconfirmed_gtc_cancel_pauses_entries(monkeypatch):
    resting = {"id": "55", "filled": 0.0, "amount": 1.0, "status": "open"}
    venue = _Venue(place_results=[{"status": "error", "message": "x"}, {"status": "success", "order": dict(resting)}], probe={"status": "error", "code": "-2013"})
    venue.fetch_order = AsyncMock(side_effect=[{"status": "error", "code": "-2013"}] + [{"status": "success", "order": dict(resting)}] * 40)
    paused: list[tuple[str, str]] = []

    async def _pause(mode, reason=""):
        paused.append((mode, reason))
        return {"success": True}

    monkeypatch.setattr(ple, "PROTECTED_LIMIT_ORDER_TIMEOUT_SEC", 0.0)
    monkeypatch.setattr("backend.services.portfolio_engine.get_portfolio_engine", lambda: SimpleNamespace(set_kill_switch=_pause))

    out = await execute_protected_limit_live(venue, symbol="XRPUSDT", side="buy", quantity=1.0, limit_price=1.4)

    assert out is None
    assert paused and paused[0][0] == "PAUSE_BUYS" and "CANCEL_UNCONFIRMED" in paused[0][1]


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


def test_live_entry_needs_a_usable_account_valuation(monkeypatch, tmp_path):
    from backend.services import circuit_breaker_service as cbs
    from backend.services.portfolio_engine import KillSwitchMode, PortfolioEngine

    engine = PortfolioEngine.__new__(PortfolioEngine)
    engine.db_path = str(tmp_path / "x.db")
    engine.principal = 1000.0
    engine._kill_switch_mode = KillSwitchMode.RESUME
    engine._kill_switch_reason = ""
    engine._canonical_failsafe_decision = lambda: {"tripped": False, "usable": False, "reason": "stale_nle_cannot_compare"}
    monkeypatch.setattr(cbs, "read_persisted_entry_control", lambda _p: {"requested_kill_mode": "RESUME"})

    engine._live_execution_enabled = True
    ok, reason = engine._check_kill_switch_buy()
    assert not ok and reason.startswith("ACCOUNT_VALUATION_UNPROVEN")

    engine._live_execution_enabled = False
    assert engine._check_kill_switch_buy() == (True, "")


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
    defer = src.index("DAY_V2_LEARNED_EXIT_DEFERRED_NO_BOOK")
    assert src.index("if not _book_bid and _day_v2_dec") < defer < src.index('if _day_v2_dec and str(_day_v2_dec.get("action") or "") == "sell"')


def test_executable_book_is_fresh_or_absent(monkeypatch):
    from backend.services import exact_policy_capture as epc
    from backend.services import spread_book_telemetry as sbt

    monkeypatch.setattr(sbt, "read_market_book", lambda _r, _s: {"bid": 1.3979, "ask": 1.3981, "freshness_sec": 1.0})
    monkeypatch.setattr("backend.config.redis_config.get_redis_client", object)
    assert epc.executable_book("XRPUSDT") == (1.3979, 1.3981)

    monkeypatch.setattr(sbt, "read_market_book", lambda _r, _s: {"bid": 1.3979, "ask": 1.3981, "freshness_sec": 999.0})
    assert epc.executable_book("XRPUSDT") == (None, None)


def test_orphan_restore_never_turns_a_scalp_buy_into_a_day_lot(tmp_path):
    from backend.services.atomic_execution_book import find_orphaned_day_buys

    db = tmp_path / "orph.db"
    c = sqlite3.connect(db)
    c.execute(
        "create table paper_trades(id integer primary key, trade_id text, symbol text, side text, quantity real, remaining_position real, price real,"
        " timestamp text, diagnostics_json text, explainability_json text, context_snapshot_json text, engine_id text)"
    )
    c.execute("create table portfolio_engine_positions(symbol text, quantity real, trade_id text, engine_id text)")
    c.execute("create table position_close_ledger(detail text)")
    c.execute("create table operational_state(key text, value_json text)")
    c.execute("insert into paper_trades values (1,'mystic_BTC/USDT_1','BTC/USDT','BUY',0.001,0.001,83000,'t','','','','DAY_V2')")
    c.execute("insert into paper_trades values (2,'scalp_v2_ETHUSDT_1','ETH/USDT','BUY',0.1,0.1,2500,'t','','','','SCALP_V2')")
    c.commit()
    c.close()

    found = find_orphaned_day_buys(str(db))

    assert [o["trade_id"] for o in found] == ["mystic_BTC/USDT_1"]


def test_bnb_commission_is_charged_at_the_model_rate_without_debiting_usdt():
    from backend.services.live_fill_economics import apply_live_buy_economics, apply_live_sell_economics, extract_live_commission

    order = {"info": {"fills": [{"price": "100", "qty": "1", "commission": "0.0001", "commissionAsset": "BNB"}]}}
    comm = extract_live_commission(order, symbol="SOL/USDT", fill_price=100.0)
    assert comm.unpriced_assets == ("BNB",)

    received, fee, cash = apply_live_buy_economics(filled_qty=1.0, fill_price=100.0, modeled_fee=0.02, commission=comm)
    assert (received, fee, cash) == (1.0, 0.02, 100.0)

    sell_fee, proceeds = apply_live_sell_economics(quantity=1.0, fill_price=101.0, modeled_fee=0.0202, commission=comm)
    assert sell_fee == 0.0202 and proceeds == 101.0


def test_quote_and_base_commissions_are_unchanged():
    from backend.services.live_fill_economics import apply_live_buy_economics, apply_live_sell_economics, extract_live_commission

    base = extract_live_commission({"info": {"fills": [{"price": "100", "qty": "1", "commission": "0.0002", "commissionAsset": "SOL"}]}}, symbol="SOL/USDT", fill_price=100.0)
    received, fee, cash = apply_live_buy_economics(filled_qty=1.0, fill_price=100.0, modeled_fee=0.0, commission=base)
    assert received == pytest.approx(0.9998) and fee == pytest.approx(0.02) and cash == pytest.approx(100.0)

    quote = extract_live_commission({"info": {"fills": [{"price": "100", "qty": "1", "commission": "0.02", "commissionAsset": "USDT"}]}}, symbol="SOL/USDT", fill_price=100.0)
    sell_fee, proceeds = apply_live_sell_economics(quantity=1.0, fill_price=100.0, modeled_fee=0.0, commission=quote)
    assert sell_fee == pytest.approx(0.02) and proceeds == pytest.approx(99.98)


def test_tests_cannot_reach_a_real_exchange_order():
    import ccxt

    for cls in (ccxt.binance, ccxt.binanceus):
        with pytest.raises(RuntimeError, match="must not send exchange orders"):
            cls().create_order("BTC/USDT", "limit", "buy", 0.001, 1.0)


def test_scalp_buy_honours_the_account_pause():
    from backend.services.portfolio_engine import PortfolioEngine

    src = inspect.getsource(PortfolioEngine.execute_scalp_v2_buy_live)
    pause = src.index('getattr(self, "_trading_paused", False)')
    assert src.index("self._check_kill_switch_buy()") < pause < src.index("execute_protected_limit_live(")
    assert "TRADING_PAUSED" in src


def _dual_lot_engine(monkeypatch):
    from backend.services import live_exchange_equity
    from backend.services.portfolio_engine import PortfolioEngine

    engine = PortfolioEngine.__new__(PortfolioEngine)
    engine.db_path = ":memory:"
    engine._trading_paused = False
    engine._pause_reason = ""
    engine._metrics_reconciliation_adjustments = 0
    engine._position_mark_prices = {"BTC/USDT": 80000.0}

    async def _noop(*_a, **_k):
        return False

    engine._enforce_lot_ownership = _noop
    engine._shrink_unproven_dust_bucket = _noop
    engine._sync_protected_remainder = _noop
    engine._held_engine_dust_qty = lambda _s: 0.0
    engine._floor_to_step = lambda q, _step: q
    monkeypatch.setattr(live_exchange_equity, "stamp_protected_preexisting_dust", lambda *_a, **_k: None)
    lots = [SimpleNamespace(quantity=0.001, entry_price=80000.0, engine_id="DAY_V2"), SimpleNamespace(quantity=0.001, entry_price=80000.0, engine_id="SCALP_V2")]
    return engine, lots


@pytest.mark.asyncio
async def test_material_dual_lot_shortfall_pauses_entries_until_healthy(monkeypatch):
    engine, lots = _dual_lot_engine(monkeypatch)

    await engine._reconcile_dual_engine_lots(symbol="BTC/USDT", lots=lots, exchange_qty=0.0015, qty_step=0.00001, source="test")
    assert engine._trading_paused and engine._pause_reason == "ENGINE_QTY_SHORTFALL:BTC/USDT"
    assert [lot.quantity for lot in lots] == [0.001, 0.001]

    await engine._reconcile_dual_engine_lots(symbol="BTC/USDT", lots=lots, exchange_qty=0.002, qty_step=0.00001, source="test")
    assert not engine._trading_paused and engine._pause_reason == ""


@pytest.mark.asyncio
async def test_dust_sized_shortfall_does_not_pause(monkeypatch):
    engine, lots = _dual_lot_engine(monkeypatch)

    await engine._reconcile_dual_engine_lots(symbol="BTC/USDT", lots=lots, exchange_qty=0.002 - 0.000005, qty_step=0.000001, source="test")

    assert not engine._trading_paused


@pytest.mark.asyncio
async def test_shortfall_never_clears_an_unrelated_pause(monkeypatch):
    engine, lots = _dual_lot_engine(monkeypatch)
    engine._trading_paused = True
    engine._pause_reason = "auto_deleverage_disabled_invariant_violation"

    await engine._reconcile_dual_engine_lots(symbol="BTC/USDT", lots=lots, exchange_qty=0.002, qty_step=0.00001, source="test")

    assert engine._trading_paused and engine._pause_reason == "auto_deleverage_disabled_invariant_violation"


def test_single_lot_with_phantom_held_dust_pauses_and_cuts_nothing(monkeypatch):
    engine, lots = _dual_lot_engine(monkeypatch)
    lot = lots[0]
    engine._held_engine_dust_qty = lambda _s: 0.0005

    engine._check_single_lot_shortfall("BTC/USDT", lot, 0.001)
    assert engine._trading_paused and lot.quantity == 0.001

    engine._held_engine_dust_qty = lambda _s: 0.0
    engine._check_single_lot_shortfall("BTC/USDT", lot, 0.001)
    assert not engine._trading_paused


def test_an_open_never_overwrites_another_trades_live_lot():
    from backend.services.portfolio_engine import refuse_active_lot_overwrite

    conn = sqlite3.connect(":memory:")
    conn.execute("create table portfolio_engine_positions(symbol text, quantity real, trade_id text, status text, engine_id text, primary key (engine_id, symbol))")
    conn.execute("insert into portfolio_engine_positions values ('BTC/USDT', 0.002, 'lot-1', 'ACTIVE', 'DAY_V2')")
    conn.execute("insert into portfolio_engine_positions values ('ETH/USDT', 0.0001, 'lot-2', 'DUST_PENDING', 'DAY_V2')")

    with pytest.raises(sqlite3.IntegrityError, match="ACTIVE_LOT_EXISTS"):
        refuse_active_lot_overwrite(conn, engine_id="DAY_V2", symbol="BTC/USDT", new_trade_id="lot-9")
    refuse_active_lot_overwrite(conn, engine_id="DAY_V2", symbol="BTC/USDT", new_trade_id="lot-1")
    refuse_active_lot_overwrite(conn, engine_id="SCALP_V2", symbol="BTC/USDT", new_trade_id="lot-9")
    refuse_active_lot_overwrite(conn, engine_id="DAY_V2", symbol="ETH/USDT", new_trade_id="lot-9")


def test_engine_less_read_refuses_to_pick_between_engines(tmp_path):
    from backend.services.portfolio_engine import PortfolioEngine

    db = tmp_path / "amb.db"
    _positions_db(db)
    c = sqlite3.connect(db)
    c.execute("INSERT INTO portfolio_engine_positions VALUES ('BTC/USDT', 0.003, 'mystic_BTC/USDT_1', 83000.0, 'ACTIVE', 'DAY_V2')")
    c.commit()
    c.close()
    engine = PortfolioEngine.__new__(PortfolioEngine)
    engine.db_path = str(db)

    assert engine._fetch_authoritative_open_position_sync("BTC/USDT") is None
    assert engine._fetch_authoritative_open_position_sync("BTC/USDT", "DAY_V2")[1] == "mystic_BTC/USDT_1"


def test_episode_without_a_disposition_is_closed_visibly(tmp_path, monkeypatch):
    import time

    from backend.services import exact_policy_capture as epc

    db = str(tmp_path / "ep.db")
    conn = sqlite3.connect(db)
    epc.ensure_capture_schema(conn)
    old = time.time() - 2 * epc.DISPOSITION_GRACE_SEC
    for cid, decided in ((1, old), (2, time.time())):
        conn.execute(
            "INSERT INTO policy_episodes (candidate_id, kind, engine_id, symbol, setup, regime, decided_at, entry_ask, roundtrip_cost, snapshot_json, status, capture_version, funded)"
            " VALUES (?, 'COUNTERFACTUAL_POLICY_OUTCOME', 'DAY_V2', 'BTCUSDT', 'S', 'r', ?, 100.0, 0.0006, '{}', 'OPEN', ?, NULL)",
            (cid, decided, epc.CAPTURE),
        )
    conn.commit()
    conn.close()
    monkeypatch.setattr(epc, "_book", lambda _s: (None, None))

    stats = epc.advance_exact_episodes(db)

    assert stats["disposition_missing"] == 1
    conn = sqlite3.connect(db)
    rows = dict(conn.execute("SELECT candidate_id, status || ':' || COALESCE(exit_reason, '') FROM policy_episodes").fetchall())
    conn.close()
    assert rows == {1: "CLOSED:DISPOSITION_MISSING", 2: "OPEN:"}


def test_churn_guard_is_telemetry_not_an_entry_gate():
    from backend.services.portfolio_engine import PortfolioEngine, RegimeState

    engine = PortfolioEngine.__new__(PortfolioEngine)
    engine._regime_state = RegimeState()
    engine._regime_state.churn_guard_active = True
    engine._regime_state.churn_ratio = 9.0

    assert engine._check_regime_guards("BTC/USDT", 1) == (True, "")


def _equity_engine(tmp_path, *, positions=(), dust=(), protected=(), signed, cash, observed=None):
    import json
    import time

    observed = time.time() if observed is None else observed

    from backend.services.portfolio_engine import PortfolioEngine

    db = tmp_path / "eq.db"
    c = sqlite3.connect(db)
    c.execute("create table portfolio_engine_positions(symbol text, quantity real, engine_id text, status text)")
    c.execute("create table engine_strategy_dust(engine_id text, symbol text, quantity real, source_trade_id text, status text)")
    c.execute("create table protected_external_inventory(symbol text, quantity real)")
    c.execute("create table operational_state(key text primary key, value_json text)")
    c.executemany("insert into portfolio_engine_positions values (?,?,?,?)", positions)
    c.executemany("insert into engine_strategy_dust values (?,?,?,?,'HELD')", dust)
    c.executemany("insert into protected_external_inventory values (?,?)", protected)
    if signed is not None:
        c.execute("insert into operational_state values ('signed_balance_snapshot', ?)", (json.dumps({"balances": signed, "observed_at": observed}),))
    c.commit()
    c.close()
    engine = PortfolioEngine.__new__(PortfolioEngine)
    engine.db_path = str(db)
    engine._live_execution_enabled = True
    engine.cash_balance = cash
    engine._position_mark_prices = {"BTC/USDT": 83000.0}
    owned = sum(p[1] for p in positions) + sum(x[2] for x in dust) + sum(p[1] for p in protected)
    return engine, cash + owned * 83000.0


_LOT = ("BTC/USDT", 0.001, "DAY_V2", "ACTIVE")
_TEN = 10.0 / 83000.0


@pytest.mark.parametrize(
    ("case", "kwargs", "expected"),
    [
        ("consistent", {"positions": [_LOT], "signed": {"USDT": 1000.0, "BTC": 0.001}}, True),
        ("phantom $10", {"positions": [("BTC/USDT", 0.001 + _TEN, "DAY_V2", "ACTIVE")], "signed": {"USDT": 1000.0, "BTC": 0.001}}, False),
        ("missing $10", {"positions": [_LOT], "signed": {"USDT": 1000.0, "BTC": 0.001 + _TEN}}, False),
        ("missing $10 with unowned BNB", {"positions": [_LOT], "signed": {"USDT": 1000.0, "BTC": 0.001 + _TEN, "BNB": 0.01}}, False),
        ("cash $5 off", {"positions": [_LOT], "signed": {"USDT": 1005.0, "BTC": 0.001}}, False),
        (
            "duplicate dust",
            {
                "positions": [("BTC/USDT", 0.00015, "DAY_V2", "DUST_PENDING")],
                "dust": [("DAY_V2", "BTC/USDT", 0.00015, "mystic_BTC/USDT_1")],
                "signed": {"USDT": 1000.0, "BTC": 0.00015},
            },
            False,
        ),
        ("stale protected", {"positions": [_LOT], "protected": [("BTC/USDT", 0.00015)], "signed": {"USDT": 1000.0, "BTC": 0.001}}, False),
    ],
)
def test_signed_equity_invariant_fails_every_material_defect(tmp_path, case, kwargs, expected):
    engine, equity = _equity_engine(tmp_path, cash=1000.0, **kwargs)
    assert engine._signed_equity_matches(equity) is expected, case


def test_live_invariant_needs_a_fresh_signed_snapshot(tmp_path):
    missing, equity = _equity_engine(tmp_path, positions=[_LOT], signed=None, cash=1000.0)
    assert missing._signed_equity_matches(equity) is False

    stale_dir = tmp_path / "stale"
    stale_dir.mkdir()
    stale, equity = _equity_engine(stale_dir, positions=[_LOT], signed={"USDT": 1000.0, "BTC": 0.001}, cash=1000.0, observed=1.0)
    assert stale._signed_equity_matches(equity) is False

    stale._live_execution_enabled = False
    assert stale._signed_equity_matches(equity) is True


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
