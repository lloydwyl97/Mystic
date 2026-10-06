"""SCALP adverse stop on the executable bid at SCALP cadence; dust is not a position."""

from __future__ import annotations

import inspect
import logging
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.config.trading_economics import ESTIMATED_ROUNDTRIP_COST, ORDERBOOK_HALF_SPREAD_ESTIMATE
from backend.services.portfolio_engine import make_position_key
from backend.services.scalp_v2 import exit_evaluator as ev
from backend.services.scalp_v2.exit_evaluator import scalp_v2_net_pnl_at_bid_pct
from tests.test_two_engine_architecture import DAY, SCALP, _engine, _lot


def _monitor(tmp_path, monkeypatch, *, ws=(99.80, 99.85)):
    import backend.services.day_high_water as hw

    monkeypatch.setattr(hw, "load_feature_1m_candles", lambda *_a, **_k: [])
    eng = _engine(tmp_path)
    eng.run_trading_circuit_breaker_check = AsyncMock()
    eng._resolve_exit_monitor_mark = AsyncMock(return_value={"mark_used": 100.0, "bid": 99.95, "price_source_stale": False})
    eng._build_exit_check_telemetry = lambda *_a, **_k: {}
    eng._log_exit_check_telemetry = lambda *_a, **_k: None
    eng._persist_position_to_sqlite = AsyncMock()
    eng._emit_day_health_telemetry = AsyncMock()
    eng._learning_heartbeat_last = {"BTC/USDT": time.time()}
    eng._scalp_book_reader = SimpleNamespace(read_top_of_book=lambda _s: ws)
    check = AsyncMock(return_value=None)
    eng._check_exit_conditions = check
    eng.open_positions[make_position_key(SCALP, "BTC/USDT")] = _lot("BTC/USDT", SCALP, price=100.0)
    eng.open_positions[make_position_key(DAY, "BTC/USDT")] = _lot("BTC/USDT", DAY, price=100.0)
    return eng, check


# ── trigger units ─────────────────────────────────────────────────────────────


def test_net_at_bid_is_in_realized_units():
    remaining = ESTIMATED_ROUNDTRIP_COST - ORDERBOOK_HALF_SPREAD_ESTIMATE
    assert scalp_v2_net_pnl_at_bid_pct(100.0, 99.9) == pytest.approx(-0.001 - remaining)
    # A fill at the trigger bid realizes the trigger net (same entry basis, same cost).
    trigger_bid = 100.0 * (1 - 0.0015 + remaining)
    assert scalp_v2_net_pnl_at_bid_pct(100.0, trigger_bid) == pytest.approx(-0.0015)


def test_adverse_contract_is_still_15_bp():
    assert ev.scalp_v2_max_adverse_net_pct("ETHUSDT") == pytest.approx(0.0015)
    assert pytest.approx(0.015) == ev.SCALP_V2_CATASTROPHIC_PCT


# ── monitor reads the executable book ─────────────────────────────────────────


async def test_scalp_exit_check_gets_ws_best_bid_day_keeps_rest_mark(tmp_path, monkeypatch):
    eng, check = _monitor(tmp_path, monkeypatch)
    await eng.monitor_all_positions({}, 0)
    calls = {c.args[0].engine_id: c for c in check.await_args_list}
    assert calls[SCALP].kwargs["executable_bid"] == 99.80
    assert calls[SCALP].args[1] == 99.85
    assert calls[DAY].kwargs["executable_bid"] is None
    assert eng._resolve_exit_monitor_mark.await_count == 1  # DAY only


async def test_fast_pass_is_scalp_only_and_never_prices_without_a_fresh_book(tmp_path, monkeypatch):
    eng, check = _monitor(tmp_path, monkeypatch, ws=None)
    await eng.monitor_all_positions({}, 0, engine_ids=frozenset({SCALP}), executable_quote_only=True)
    assert check.await_count == 0
    assert eng._resolve_exit_monitor_mark.await_count == 0
    eng.run_trading_circuit_breaker_check.assert_not_awaited()


async def test_full_pass_falls_back_to_rest_bid_only_when_fresh(tmp_path, monkeypatch):
    eng, check = _monitor(tmp_path, monkeypatch, ws=None)
    await eng.monitor_all_positions({}, 0, engine_ids=frozenset({SCALP}))
    assert check.await_args_list[0].kwargs["executable_bid"] == 99.95
    check.reset_mock()
    eng._resolve_exit_monitor_mark = AsyncMock(return_value={"mark_used": 100.0, "bid": 99.95, "price_source_stale": True})
    await eng.monitor_all_positions({}, 0, engine_ids=frozenset({SCALP}))
    assert check.await_args_list[0].kwargs["executable_bid"] is None


async def test_old_adverse_distance_holds_and_a_worse_terminal_sells_the_bid(tmp_path, caplog, monkeypatch):
    eng = _engine(tmp_path)
    eng.execute_sell_fifo = AsyncMock(return_value={"status": "sold"})
    lot = _lot("ETH/USDT", SCALP, price=100.0)
    lot.adaptive_decision = {"risk_estimate": 0.003, "setup": "VWAP_EMA_RECLAIM", "regime": "neutral"}
    lot.entry_time = time.time() - 60
    lot.highest_price = lot.lowest_price = 100.0
    mid = 99.95
    bid = 99.86
    terms = [None]

    def _terminal(*_args, **_kwargs):
        return terms[0]

    monkeypatch.setattr("backend.services.portfolio_engine._open_expected_terminal", _terminal)
    await eng._check_exit_conditions(lot, mid, 0, executable_bid=bid)
    eng.execute_sell_fifo.assert_not_awaited()
    terms[0] = -0.05
    await eng._check_exit_conditions(lot, mid, 0, executable_bid=bid)
    eng.execute_sell_fifo.assert_awaited_once()
    args = eng.execute_sell_fifo.await_args
    assert args.args[2] == bid
    assert args.args[4] == "SCALP_V2_LEARNED_CONTINUATION"

    terms[0] = None
    eng.execute_sell_fifo.reset_mock()
    with caplog.at_level(logging.WARNING):
        await eng._check_exit_conditions(lot, mid, 0, executable_bid=None)
    assert "SCALP_V2_EXIT_NO_EXECUTABLE_BID" in caplog.text
    eng.execute_sell_fifo.assert_not_awaited()


async def test_scalp_fast_exit_wait_runs_scalp_passes_between_full_passes(monkeypatch):
    from backend.services.portfolio_engine_integration import PortfolioEngineIntegration

    monkeypatch.setattr(ev, "SCALP_V2_EXIT_MONITOR_INTERVAL_SEC", 0.5)
    integ = object.__new__(PortfolioEngineIntegration)
    integ.is_running = True
    integ.engine = SimpleNamespace(open_positions={"k": _lot("ETH/USDT", SCALP)})
    integ._monitor_positions_once = AsyncMock(return_value=[])
    await integ._scalp_fast_exit_wait(1.2)
    assert integ._monitor_positions_once.await_count == 2
    assert all(c.kwargs == {"refresh_market_data": False, "engine_ids": frozenset({SCALP})} for c in integ._monitor_positions_once.await_args_list)

    integ.engine.open_positions = {"k": _lot("ETH/USDT", SCALP, status="DUST_PENDING"), "d": _lot("ETH/USDT", DAY)}
    integ._monitor_positions_once.reset_mock()
    await integ._scalp_fast_exit_wait(0.6)
    assert integ._monitor_positions_once.await_count == 0


def test_monitor_loop_has_no_new_process_and_day_path_is_unscoped():
    from backend.services import portfolio_engine_integration as pei

    loop_src = inspect.getsource(pei.PortfolioEngineIntegration._position_monitor_loop)
    assert "_scalp_fast_exit_wait" in loop_src
    assert "create_subprocess" not in loop_src and "Process(" not in loop_src
    once = inspect.getsource(pei.PortfolioEngineIntegration._monitor_positions_once)
    assert "if engine_ids is None:\n            await self._refresh_prices()" in once


# ── dust ──────────────────────────────────────────────────────────────────────


def _dust_reconcile_engine(tmp_path, held: float):
    eng = _engine(tmp_path)
    eng._enforce_lot_ownership = AsyncMock()
    eng._held_engine_dust_qty = lambda _s: held
    eng.open_positions[make_position_key(SCALP, "ETH/USDT")] = _lot("ETH/USDT", SCALP, qty=9.636e-05, status="DUST_PENDING")
    eng.open_positions[make_position_key(DAY, "ETH/USDT")] = _lot("ETH/USDT", DAY, qty=9.536e-05, status="DUST_PENDING")
    return eng


async def test_sub_step_dust_matching_exchange_exactly_is_healthy(tmp_path, caplog):
    held = 9.51e-05 + 0.00029138
    eng = _dust_reconcile_engine(tmp_path, held)
    lots = list(eng.open_positions.values())
    with caplog.at_level(logging.INFO):
        await eng._reconcile_dual_engine_lots(symbol="ETH/USDT", lots=lots, exchange_qty=0.0005782, qty_step=0.0001, source="test")
    assert "ENGINE_QTY_SHORTFALL" not in caplog.text
    assert "DUAL_LOT_HEALTHY" in caplog.text
    assert [lot.quantity for lot in lots] == [9.636e-05, 9.536e-05]


async def test_real_shortfall_is_still_reported(tmp_path, caplog):
    eng = _dust_reconcile_engine(tmp_path, 0.0)
    lots = list(eng.open_positions.values())
    with caplog.at_level(logging.INFO):
        await eng._reconcile_dual_engine_lots(symbol="ETH/USDT", lots=lots, exchange_qty=0.00005, qty_step=0.00001, source="test")
    assert "ENGINE_QTY_SHORTFALL" in caplog.text


async def test_dust_is_not_active_takes_no_slot_and_is_not_monitored(tmp_path, monkeypatch):
    eng, check = _monitor(tmp_path, monkeypatch)
    eng.open_positions.clear()
    eng.open_positions[make_position_key(SCALP, "ETH/USDT")] = _lot("ETH/USDT", SCALP, qty=9.636e-05, status="DUST_PENDING")
    assert eng._engine_held_count(SCALP) == 0
    assert eng._combined_held_count() == 0
    await eng.monitor_all_positions({}, 0)
    await eng.monitor_all_positions({}, 0, engine_ids=frozenset({SCALP}), executable_quote_only=True)
    assert check.await_count == 0
