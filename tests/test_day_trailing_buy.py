"""Deterministic DAY trailing-buy lifecycle. No parameter sweeps."""

from __future__ import annotations

import inspect
import time
from unittest.mock import AsyncMock

import pytest

from backend.config.day_entry_execution import (
    ENTRY_AUTHORITY_TRAILING_BUY,
    trailing_buy_mode_status,
)
from backend.services.day_trailing_buy import (
    DAY_TRADE_SYMBOLS,
    ENTRY_AUTHORITY,
    available_economic_slots,
    formulas_for_symbol,
    honest_round_trip_cost_bps,
    min_dip_bps,
    observe_book,
    rebound_bps_from_spread,
    recover_submitting_intent,
    remaining_watch_notional_cap,
    required_improvement_bps,
    select_ranked_arm_stream,
    sync_book_redis,
)
from backend.services.day_trailing_buy_store import (
    CANCELED,
    EXPIRED,
    SUBMITTING,
    TRAIL_LOW,
    WAIT_DIP,
    claim_submitting,
    create_intent,
    load_active_intents,
    load_intent,
    mark_order_accepted,
    mark_terminal,
    release_submitting_for_retry,
)
from backend.services.portfolio_engine import PortfolioEngine


def _intent(**overrides):
    now = 1_000_000.0
    base = {
        "status": WAIT_DIP,
        "arm_ask": 100.0,
        "min_dip_bps": 14.0,
        "rebound_bps": 4.0,
        "required_improvement_bps": 10.0,
        "lowest_ask": 0.0,
        "lowest_ask_ts": 0.0,
        "expires_at": now + 900,
    }
    base.update(overrides)
    return base


def test_formulas_match_authoritative_costs():
    btc = formulas_for_symbol("BTCUSDT", arm_spread_bps=1.0)
    eth = formulas_for_symbol("ETHUSDT", arm_spread_bps=1.0)
    sol = formulas_for_symbol("SOLUSDT", arm_spread_bps=2.19)
    xrp = formulas_for_symbol("XRPUSDT", arm_spread_bps=1.0)
    assert honest_round_trip_cost_bps("BTCUSDT") == pytest.approx(6.0359, abs=0.02)
    assert honest_round_trip_cost_bps("ETHUSDT") == pytest.approx(6.1697, abs=0.02)
    assert honest_round_trip_cost_bps("SOLUSDT") == pytest.approx(7.6317, abs=0.02)
    assert honest_round_trip_cost_bps("XRPUSDT") == pytest.approx(7.4463, abs=0.02)
    assert btc["required_improvement_bps"] == pytest.approx(10.0)
    assert eth["required_improvement_bps"] == pytest.approx(10.0)
    assert sol["required_improvement_bps"] == pytest.approx(10.6317, abs=0.02)
    assert xrp["required_improvement_bps"] == pytest.approx(10.4463, abs=0.02)
    assert rebound_bps_from_spread(1.0) == 4.0
    assert rebound_bps_from_spread(2.19) == pytest.approx(4.38)
    assert min_dip_bps("BTCUSDT", 1.0) == pytest.approx(14.0)
    assert required_improvement_bps("SOLUSDT") == pytest.approx(honest_round_trip_cost_bps("SOLUSDT") + 3.0)


def test_all_four_symbols_remain_eligible():
    assert DAY_TRADE_SYMBOLS == ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
    for sym in DAY_TRADE_SYMBOLS:
        assert _to_engine_universe(sym)


def _to_engine_universe(symbol: str) -> bool:
    from backend.services.portfolio_engine import DAY_TRADE_SYMBOLS as ENGINE_SYMS

    return symbol in ENGINE_SYMS


def test_insufficient_dip_never_submits():
    d = observe_book(_intent(), ask=99.90, now=1_000_010.0, book_fresh=True)
    assert d.action == "watch"
    assert d.status == WAIT_DIP


def test_falling_ask_updates_low_without_buying():
    armed = _intent(status=TRAIL_LOW, lowest_ask=99.80, lowest_ask_ts=1_000_010.0)
    d1 = observe_book(armed, ask=99.70, now=1_000_020.0, book_fresh=True)
    assert d1.action == "new_low"
    assert d1.lowest_ask == pytest.approx(99.70)
    d2 = observe_book({**armed, "lowest_ask": 99.70}, ask=99.60, now=1_000_030.0, book_fresh=True)
    assert d2.action == "new_low"
    assert d2.status == TRAIL_LOW


def test_exact_rebound_triggers_once():
    armed = _intent(status=TRAIL_LOW, lowest_ask=99.80, lowest_ask_ts=1_000_010.0)
    trigger = 99.80 * (1.0 + 4.0 / 10000.0)
    d = observe_book(armed, ask=trigger, now=1_000_040.0, book_fresh=True)
    assert d.action == "submit"


def test_rebound_that_loses_improvement_does_not_chase():
    armed = _intent(status=TRAIL_LOW, lowest_ask=99.80, lowest_ask_ts=1_000_010.0)
    d = observe_book(armed, ask=99.95, now=1_000_040.0, book_fresh=True)
    assert d.action == "watch"
    assert d.reason == "REBOUND_ABOVE_IMPROVEMENT"
    lost = observe_book(armed, ask=100.0, now=1_000_041.0, book_fresh=True)
    assert lost.action == "expire"
    assert lost.reason == "IMPROVEMENT_LOST"


def test_timeout_expires_without_buying():
    d = observe_book(_intent(), ask=99.50, now=1_000_901.0, book_fresh=True)
    assert d.action == "expire"
    assert d.status == EXPIRED
    assert d.reason == "TIMEOUT"


def test_stale_data_cancels():
    d = observe_book(_intent(), ask=99.50, now=1_000_010.0, book_fresh=False)
    assert d.action == "cancel"
    assert d.reason == "STALE_MARKET_BOOK"


def test_4h_invalidation_cancels():
    d = observe_book(_intent(), ask=99.50, now=1_000_010.0, book_fresh=True, thesis_invalid=True)
    assert d.action == "cancel"
    assert d.reason == "THESIS_4H_INVALID"


def test_invalid_configuration_fails_closed(monkeypatch):
    monkeypatch.delenv("DAY_ENTRY_EXECUTION_MODE", raising=False)
    ok, err, mode = trailing_buy_mode_status()
    assert ok is False
    assert err == "DAY_ENTRY_EXECUTION_MODE_MISSING"
    assert mode == ""
    monkeypatch.setenv("DAY_ENTRY_EXECUTION_MODE", "legacy")
    ok, err, mode = trailing_buy_mode_status()
    assert ok is False
    assert "INVALID" in err
    monkeypatch.setenv("DAY_ENTRY_EXECUTION_MODE", "trailing_buy")
    ok, err, mode = trailing_buy_mode_status()
    assert ok is True
    assert mode == "trailing_buy"


def test_create_intent_does_not_submit(tmp_path):
    db = tmp_path / "tb.db"
    ok, reason, row = create_intent(
        db,
        fields={
            "decision_id": "d1",
            "symbol": "BTC/USDT",
            "arm_ask": 100.0,
            "arm_bid": 99.9,
            "arm_midpoint": 99.95,
            "round_trip_cost_bps": 6.04,
            "spread_bps": 1.0,
            "required_improvement_bps": 10.0,
            "rebound_bps": 4.0,
            "min_dip_bps": 14.0,
            "expires_at": time.time() + 900,
            "quantity": 0.001,
        },
    )
    assert ok
    assert reason == "OK"
    assert row["status"] == WAIT_DIP
    assert row["order_id"] == ""
    assert row["client_order_id"].startswith("tb")


def test_only_one_active_intent_per_symbol(tmp_path):
    db = tmp_path / "tb.db"
    create_intent(
        db,
        fields={
            "decision_id": "d1",
            "symbol": "ETH/USDT",
            "arm_ask": 10.0,
            "arm_bid": 9.9,
            "arm_midpoint": 9.95,
            "round_trip_cost_bps": 6.17,
            "spread_bps": 1.0,
            "required_improvement_bps": 10.0,
            "rebound_bps": 4.0,
            "min_dip_bps": 14.0,
            "expires_at": time.time() + 900,
        },
    )
    ok, _reason, row = create_intent(
        db,
        fields={
            "decision_id": "d2",
            "symbol": "ETH/USDT",
            "arm_ask": 10.1,
            "arm_bid": 10.0,
            "arm_midpoint": 10.05,
            "round_trip_cost_bps": 6.17,
            "spread_bps": 1.0,
            "required_improvement_bps": 10.0,
            "rebound_bps": 4.0,
            "min_dip_bps": 14.0,
            "expires_at": time.time() + 900,
        },
    )
    assert ok
    assert _reason == "PRESERVED_EXISTING"
    assert load_intent(db, row["intent_id"])["decision_id"] == "d1"
    active = load_active_intents(db)
    assert len(active) == 1
    assert active[0]["decision_id"] == "d1"
    assert active[0]["arm_ask"] == pytest.approx(10.0)


def test_conflicting_decision_does_not_replace_submitting(tmp_path):
    db = tmp_path / "tb.db"
    ok, _, first = create_intent(
        db,
        fields={
            "decision_id": "d1",
            "symbol": "SOL/USDT",
            "arm_ask": 20.0,
            "arm_bid": 19.9,
            "arm_midpoint": 19.95,
            "round_trip_cost_bps": 7.63,
            "spread_bps": 2.19,
            "required_improvement_bps": 10.63,
            "rebound_bps": 4.38,
            "min_dip_bps": 15.01,
            "expires_at": time.time() + 900,
        },
    )
    assert ok
    update = __import__("backend.services.day_trailing_buy_store", fromlist=["update_watch"]).update_watch
    update(db, first["intent_id"], status=TRAIL_LOW, lowest_ask=19.9, lowest_ask_ts=time.time())
    claimed, _ = claim_submitting(db, first["intent_id"])
    assert claimed
    ok2, reason, _ = create_intent(
        db,
        fields={
            "decision_id": "d2",
            "symbol": "SOL/USDT",
            "arm_ask": 20.1,
            "arm_bid": 20.0,
            "arm_midpoint": 20.05,
            "round_trip_cost_bps": 7.63,
            "spread_bps": 2.19,
            "required_improvement_bps": 10.63,
            "rebound_bps": 4.38,
            "min_dip_bps": 15.01,
            "expires_at": time.time() + 900,
        },
    )
    assert ok2 is False
    assert reason == "SYMBOL_SUBMITTING"
    assert load_active_intents(db)[0]["decision_id"] == "d1"


def test_duplicate_workers_cannot_duplicate_orders(tmp_path):
    db = tmp_path / "tb.db"
    _, _, row = create_intent(
        db,
        fields={
            "decision_id": "d1",
            "symbol": "XRP/USDT",
            "arm_ask": 0.6,
            "arm_bid": 0.599,
            "arm_midpoint": 0.5995,
            "round_trip_cost_bps": 7.45,
            "spread_bps": 1.0,
            "required_improvement_bps": 10.45,
            "rebound_bps": 4.0,
            "min_dip_bps": 14.45,
            "expires_at": time.time() + 900,
        },
    )
    __import__("backend.services.day_trailing_buy_store", fromlist=["update_watch"]).update_watch(db, row["intent_id"], status=TRAIL_LOW, lowest_ask=0.59, lowest_ask_ts=time.time())
    wins = [claim_submitting(db, row["intent_id"])[0], claim_submitting(db, row["intent_id"])[0]]
    assert wins.count(True) == 1
    assert load_intent(db, row["intent_id"])["status"] == SUBMITTING


@pytest.mark.asyncio
async def test_restart_recovery_adopts_existing_fill(tmp_path):
    db = tmp_path / "tb.db"
    _, _, row = create_intent(
        db,
        fields={
            "decision_id": "dec-fill",
            "symbol": "BTC/USDT",
            "arm_ask": 100.0,
            "arm_bid": 99.9,
            "arm_midpoint": 99.95,
            "round_trip_cost_bps": 6.04,
            "spread_bps": 1.0,
            "required_improvement_bps": 10.0,
            "rebound_bps": 4.0,
            "min_dip_bps": 14.0,
            "expires_at": time.time() + 900,
        },
    )
    __import__("backend.services.day_trailing_buy_store", fromlist=["update_watch"]).update_watch(db, row["intent_id"], status=TRAIL_LOW, lowest_ask=99.8, lowest_ask_ts=time.time())
    claim_submitting(db, row["intent_id"])
    import sqlite3

    conn = sqlite3.connect(str(db))
    conn.execute(
        """
        CREATE TABLE paper_trades (
            trade_id TEXT, decision_id TEXT, symbol TEXT, side TEXT, price REAL, explainability_json TEXT
        )
        """
    )
    conn.execute("INSERT INTO paper_trades VALUES ('t1','dec-fill','BTC/USDT','BUY',99.82,'{}')")
    conn.commit()
    conn.close()

    class _Eng:
        db_path = str(db)
        _live_service = None

        def _release_entry_reservation(self, *a, **k):
            return None

    await recover_submitting_intent(_Eng(), load_intent(db, row["intent_id"]))
    recovered = load_intent(db, row["intent_id"])
    assert recovered["status"] == "FILLED"
    assert recovered["trade_id"] == "t1"


@pytest.mark.asyncio
async def test_restart_recovery_does_not_guess_resubmit(tmp_path):
    db = tmp_path / "tb.db"
    _, _, row = create_intent(
        db,
        fields={
            "decision_id": "dec-open",
            "symbol": "ETH/USDT",
            "arm_ask": 10.0,
            "arm_bid": 9.99,
            "arm_midpoint": 9.995,
            "round_trip_cost_bps": 6.17,
            "spread_bps": 1.0,
            "required_improvement_bps": 10.0,
            "rebound_bps": 4.0,
            "min_dip_bps": 14.0,
            "expires_at": time.time() + 900,
        },
    )
    __import__("backend.services.day_trailing_buy_store", fromlist=["update_watch"]).update_watch(db, row["intent_id"], status=TRAIL_LOW, lowest_ask=9.9, lowest_ask_ts=time.time())
    claim_submitting(db, row["intent_id"])
    mark_order_accepted(db, row["intent_id"], order_id="ex-1")

    class _Eng:
        db_path = str(db)
        _live_service = None

        def _release_entry_reservation(self, *a, **k):
            return None

    await recover_submitting_intent(_Eng(), load_intent(db, row["intent_id"]))
    recovered = load_intent(db, row["intent_id"])
    assert recovered["status"] == SUBMITTING
    assert recovered["order_accepted"] is True
    assert release_submitting_for_retry(db, row["intent_id"]) is False


def test_process_bar_does_not_call_execute_buy_fifo():
    src = inspect.getsource(PortfolioEngine.process_bar_candidates)
    assert "await self.execute_buy_fifo" not in src
    assert "_arm_trailing_buy_ranked_stream" in src


def test_path_ev_hold_is_not_authoritative_in_trailing_buy():
    src = inspect.getsource(PortfolioEngine.process_bar_candidates)
    hold_idx = src.find("DAY_PATH_EV_HOLD")
    arm_idx = src.find("_arm_trailing_buy_ranked_stream")
    assert hold_idx != -1
    assert arm_idx != -1
    assert arm_idx > hold_idx
    assert "path_ev_authoritative" in src
    assert "TRAILING_BUY_STREAM_AUTHORITY" in inspect.getsource(PortfolioEngine._arm_trailing_buy_ranked_stream)


def test_additional_bar_buys_cannot_submit():
    src = inspect.getsource(PortfolioEngine._execute_additional_bar_buys)
    assert "execute_buy_fifo" not in src
    assert "MULTI_BUY_SKIP_OLD_RANK" in src


def test_no_legacy_fallback_in_process_bar():
    src = inspect.getsource(PortfolioEngine.process_bar_candidates)
    assert "legacy" not in src.lower() or "LEGACY" not in src
    assert "DAY_ENTRY_EXECUTION_MODE" not in src or "trailing_buy" in src
    assert "execute_buy_fifo(" not in src


@pytest.mark.asyncio
async def test_execute_buy_fifo_blocks_legacy_when_trailing_mode(monkeypatch):
    monkeypatch.setenv("DAY_ENTRY_EXECUTION_MODE", "trailing_buy")
    engine = PortfolioEngine.__new__(PortfolioEngine)
    engine._buy_execution_locks = {}
    engine._execute_buy_fifo_locked = AsyncMock(return_value={"trade_id": "x"})
    out = await PortfolioEngine.execute_buy_fifo(
        engine,
        "BTC/USDT",
        1.0,
        100.0,
        95.0,
        1.0,
        0.7,
        0,
        None,
        decision_id="d1",
    )
    assert out is None
    engine._execute_buy_fifo_locked.assert_not_called()
    out2 = await PortfolioEngine.execute_buy_fifo(
        engine,
        "BTC/USDT",
        1.0,
        100.0,
        95.0,
        1.0,
        0.7,
        0,
        None,
        decision_id="d1",
        entry_authority=ENTRY_AUTHORITY_TRAILING_BUY,
    )
    assert out2 == {"trade_id": "x"}


def test_entry_authority_constant():
    assert ENTRY_AUTHORITY == "DAY_TRAILING_BUY_CONFIRMED"


def test_capability_fail_closed_on_missing_mode(monkeypatch):
    monkeypatch.delenv("DAY_ENTRY_EXECUTION_MODE", raising=False)
    ok, err, _ = trailing_buy_mode_status()
    assert not ok
    assert err == "DAY_ENTRY_EXECUTION_MODE_MISSING"


@pytest.mark.asyncio
async def test_pause_kill_prevents_submission():
    from backend.services.day_trailing_buy import _pre_submit_safety

    class _Eng:
        _trading_paused = True
        _pause_reason = "deploy"
        open_positions = {}

        def _check_kill_switch_buy(self):
            return False, "KILL_SWITCH_PAUSE_BUYS: operator"

        async def _can_open_position(self, *a, **k):
            return True, ""

        def _pending_buy_order_symbols(self):
            return set()

    ok, reason = await _pre_submit_safety(_Eng(), {"symbol": "BTC/USDT", "notional_usd": 10, "decision_id": "d"}, 99.0)
    assert ok is False
    assert "KILL" in reason or "PAUSE" in reason


@pytest.mark.asyncio
async def test_occupied_symbol_prevents_submission():
    from backend.services.day_trailing_buy import _pre_submit_safety

    class _Pos:
        pass

    class _Eng:
        _trading_paused = False
        open_positions = {"BTC/USDT": _Pos()}

        def _check_kill_switch_buy(self):
            return True, ""

        async def _can_open_position(self, *a, **k):
            return False, "POSITION_ALREADY_OPEN"

        def _pending_buy_order_symbols(self):
            return set()

    ok, reason = await _pre_submit_safety(_Eng(), {"symbol": "BTC/USDT", "notional_usd": 10, "decision_id": "d"}, 99.0)
    assert ok is False
    assert reason == "POSITION_ALREADY_OPEN"


@pytest.mark.asyncio
async def test_economic_slots_cannot_overcommit():
    from backend.services.day_trailing_buy import _pre_submit_safety

    class _Eng:
        _trading_paused = False
        open_positions = {}

        def _check_kill_switch_buy(self):
            return True, ""

        async def _can_open_position(self, *a, **k):
            return False, "MAX_POSITIONS_REACHED"

        def _pending_buy_order_symbols(self):
            return set()

    ok, reason = await _pre_submit_safety(_Eng(), {"symbol": "ETH/USDT", "notional_usd": 50, "decision_id": "d"}, 10.0)
    assert ok is False
    assert reason == "MAX_POSITIONS_REACHED"


def test_existing_exits_unchanged():
    src = inspect.getsource(PortfolioEngine)
    assert "async def monitor_all_positions" in src
    assert "async def execute_sell_fifo" in src
    assert "_check_exit_conditions" in src
    buy = inspect.getsource(PortfolioEngine.process_bar_candidates)
    assert "execute_sell_fifo" not in buy


def test_cancel_reason_persists(tmp_path):
    db = tmp_path / "tb.db"
    _, _, row = create_intent(
        db,
        fields={
            "decision_id": "d-kill",
            "symbol": "BTC/USDT",
            "arm_ask": 100.0,
            "arm_bid": 99.9,
            "arm_midpoint": 99.95,
            "round_trip_cost_bps": 6.04,
            "spread_bps": 1.0,
            "required_improvement_bps": 10.0,
            "rebound_bps": 4.0,
            "min_dip_bps": 14.0,
            "expires_at": time.time() + 900,
        },
    )
    mark_terminal(db, row["intent_id"], CANCELED, reason="KILL_OR_PAUSE")
    done = load_intent(db, row["intent_id"])
    assert done["status"] == CANCELED
    assert done["cancel_reason"] == "KILL_OR_PAUSE"
    assert load_active_intents(db) == []


class _StreamCand:
    def __init__(self, symbol, score=0.1, decision_id=""):
        self.symbol = symbol
        self.confidence = 0.6
        self.trend_score = 0.5
        self.chop_score = 0.4
        self.coin_edge_score = 0.5
        self.atr = 1.0
        self.current_price = 100.0
        self.decision_data = {"final_selection_score": score, "live_ai_strategy": "day"}
        self.decision_id = decision_id or f"d-{symbol}"
        self.sleeve = "ACTIVE"
        self.price_structure_regime = "unknown"
        self.composite_score = 0.6


def test_select_ranked_arm_stream_keeps_all_four():
    ranked = [_StreamCand("BTCUSDT", 0.4), _StreamCand("ETHUSDT", 0.3)]
    stream = [
        _StreamCand("BTCUSDT", 0.4),
        _StreamCand("ETHUSDT", 0.3),
        _StreamCand("SOLUSDT", 0.2),
        _StreamCand("XRPUSDT", 0.1),
    ]
    out = select_ranked_arm_stream(ranked, stream)
    assert [_api_sym(c.symbol) for c in out] == ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]


def _api_sym(symbol: str) -> str:
    return str(symbol or "").replace("/", "").upper()


def test_select_ranked_arm_stream_uses_snapshot_when_rank_empty():
    stream = [_StreamCand(s, i) for i, s in enumerate(("XRPUSDT", "SOLUSDT", "ETHUSDT", "BTCUSDT"), start=1)]
    out = select_ranked_arm_stream([], stream)
    assert {c.symbol for c in out} == {"BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"}
    assert len(out) == 4


def test_available_slots_four_when_flat():
    assert available_economic_slots(held=0, pending_orders=0, max_positions=4) == 4
    assert available_economic_slots(held=1, pending_orders=0, max_positions=4) == 3
    assert available_economic_slots(held=0, pending_orders=1, max_positions=4) == 3


def test_remaining_slot_cap_lets_fourth_coin_arm():
    assert remaining_watch_notional_cap(free_cash=227.12, remaining_new_slots=4) == pytest.approx(56.78)
    assert remaining_watch_notional_cap(free_cash=40.03, remaining_new_slots=1) == pytest.approx(40.03)
    assert remaining_watch_notional_cap(free_cash=40.03, remaining_new_slots=0) == 0.0


def test_sync_book_redis_ignores_async_client(monkeypatch):
    class _Async:
        def hgetall(self, _key):
            return None

    monkeypatch.setattr("backend.config.redis_config.get_redis_client", lambda: "SYNC")
    assert sync_book_redis(_Async()) == "SYNC"
    assert sync_book_redis(None) == "SYNC"


def test_later_cycle_does_not_reset_arm_or_low(tmp_path):
    db = tmp_path / "tb.db"
    _, _, first = create_intent(
        db,
        fields={
            "decision_id": "d1",
            "symbol": "BTC/USDT",
            "arm_ask": 100.0,
            "arm_bid": 99.9,
            "arm_midpoint": 99.95,
            "round_trip_cost_bps": 6.04,
            "spread_bps": 1.0,
            "required_improvement_bps": 10.0,
            "rebound_bps": 4.0,
            "min_dip_bps": 14.0,
            "expires_at": time.time() + 900,
        },
    )
    update = __import__("backend.services.day_trailing_buy_store", fromlist=["update_watch"]).update_watch
    update(db, first["intent_id"], status=TRAIL_LOW, lowest_ask=99.70, lowest_ask_ts=time.time())
    ok, reason, row = create_intent(
        db,
        fields={
            "decision_id": "d-later",
            "symbol": "BTC/USDT",
            "arm_ask": 101.0,
            "arm_bid": 100.9,
            "arm_midpoint": 100.95,
            "round_trip_cost_bps": 6.04,
            "spread_bps": 1.0,
            "required_improvement_bps": 10.0,
            "rebound_bps": 4.0,
            "min_dip_bps": 14.0,
            "expires_at": time.time() + 900,
        },
    )
    assert ok
    assert reason == "PRESERVED_EXISTING"
    assert row["intent_id"] == first["intent_id"]
    assert row["arm_ask"] == pytest.approx(100.0)
    assert row["lowest_ask"] == pytest.approx(99.70)
    assert row["status"] == TRAIL_LOW


@pytest.mark.asyncio
async def test_ranked_stream_arms_four_on_path_ev_hold(tmp_path, monkeypatch):
    monkeypatch.setenv("DAY_ENTRY_EXECUTION_MODE", "trailing_buy")
    monkeypatch.setenv("MAX_OPEN_POSITIONS", "4")
    created: list[str] = []

    async def _fake_arm(_engine, **kwargs):
        created.append(str(kwargs["symbol"]))
        return {
            "trailing_buy_armed": True,
            "intent": {
                "intent_id": f"i-{kwargs['symbol']}",
                "symbol": kwargs["symbol"],
                "arm_ask": 1.0,
                "status": WAIT_DIP,
                "decision_id": kwargs["decision_id"],
            },
        }

    monkeypatch.setattr("backend.services.day_trailing_buy.arm_selected_candidate", _fake_arm)
    monkeypatch.setattr("backend.config.redis_config.get_redis_client", lambda: None)
    monkeypatch.setattr(
        "backend.services.day_trailing_buy_store.load_active_intents",
        lambda _db: [{"symbol": s, "status": WAIT_DIP} for s in created],
    )
    engine = PortfolioEngine.__new__(PortfolioEngine)
    engine.db_path = str(tmp_path / "tb.db")
    engine.open_positions = {}
    engine.coin_performance = {}
    engine._total_equity = 227.0
    engine._available_balance = 227.0
    engine._pending_buy_notional = lambda: 0.0
    engine._day_entry_held_count = lambda: 0
    engine._pending_buy_order_symbols = set
    engine._check_kill_switch_buy = lambda: (True, "")
    engine._day_path_ev_entry_block_reason = lambda _symbol, _max: None
    engine._entry_ensure_constraints = AsyncMock()
    engine.calculate_position_size = lambda *_a, **_k: (0.01, 90.0, 1.0)
    engine._calculate_total_open_risk = lambda: 0.0
    stream = [_StreamCand(s) for s in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")]
    out = await PortfolioEngine._arm_trailing_buy_ranked_stream(
        engine,
        ranked_candidates=stream,
        stream_candidates=stream,
        bar_timestamp=1_000,
        path_ev_decision={"path_ev_winner": "HOLD", "selected_action": "HOLD"},
    )
    assert created == ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
    assert out["trailing_buy_armed"] is True
    assert out["active_intent_count"] == 4
    assert {row["state"] for row in out["intents"]} == {WAIT_DIP}
