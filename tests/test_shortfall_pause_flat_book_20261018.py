"""Regression tests for clearing resolved ENGINE_QTY_SHORTFALL pauses on a flat book."""

from __future__ import annotations

import logging
import sqlite3
from types import SimpleNamespace

import pytest

from backend.services.engine_strategy_dust import ensure_schema as ensure_dust_schema
from backend.services.filled_asset_custody import QUARANTINED_FILLED_ASSET, mark, remember
from backend.services.portfolio_engine import AccountStatus, KillSwitchMode, OpenPosition, PortfolioEngine


def _init_db(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS portfolio_engine_ledger (
                id INTEGER PRIMARY KEY,
                principal REAL, cash_balance REAL, positions_value REAL,
                realized_pnl REAL, unrealized_pnl REAL, total_equity REAL,
                account_status TEXT, trading_paused INTEGER, pause_reason TEXT,
                last_updated TEXT, version INTEGER,
                kill_switch_mode TEXT DEFAULT 'RESUME', kill_switch_reason TEXT DEFAULT ''
            )
            """
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO portfolio_engine_ledger
            (id, principal, cash_balance, positions_value, realized_pnl, unrealized_pnl, total_equity,
             account_status, trading_paused, pause_reason, last_updated, version, kill_switch_mode, kill_switch_reason)
            VALUES (1, 1084.94, 1084.94, 0.0, 0.0, 0.0, 1084.94, 'HEALTHY', 0, '', '2026-10-18T22:00:00Z', 1, 'RESUME', '')
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS portfolio_engine_positions (
                symbol TEXT, quantity REAL, entry_price REAL, entry_time REAL,
                trade_id TEXT, stop_price REAL, take_profit_1_price REAL, take_profit_2_price REAL,
                trailing_stop_price REAL, tp1_hit INTEGER, highest_price REAL, atr_at_entry REAL,
                entry_bar_timestamp REAL, confidence_at_entry REAL, last_updated TEXT, entry_fee REAL,
                sleeve TEXT, entry_strategy_id TEXT, repair_add_count INTEGER, last_repair_add_ts REAL,
                repair_add_trade_ids TEXT, average_entry_after_repair REAL, original_position_cost REAL,
                thesis_json TEXT, ai_exit_confidence REAL, ai_exit_recommended INTEGER, ai_exit_signal_time REAL,
                entry_decision_id TEXT, lowest_price REAL, status TEXT, dust_detected_at REAL,
                dust_qty_canonical REAL, entry_intent_id TEXT, entry_reservation_id TEXT,
                entry_order_id TEXT, entry_client_order_id TEXT, entry_fill_ids_json TEXT,
                engine_id TEXT, scalp_opportunity_id TEXT, reconcile_reason TEXT,
                PRIMARY KEY (engine_id, symbol)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT, symbol TEXT, side TEXT, quantity REAL, price REAL,
                remaining_position REAL DEFAULT 0, engine_id TEXT, is_synthetic INTEGER DEFAULT 0
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT, cash_balance REAL, principal REAL, positions_value REAL,
                total_equity REAL, total_open_risk REAL, open_positions_count INTEGER
            )
            """
        )
        ensure_dust_schema(conn)
        conn.commit()
    finally:
        conn.close()


def _make_engine(tmp_path, monkeypatch) -> PortfolioEngine:
    from backend.services import live_exchange_equity

    monkeypatch.setenv("DAY_ENTRY_EXECUTION_MODE", "trailing_buy")
    db_path = str(tmp_path / "shortfall.db")
    _init_db(db_path)
    monkeypatch.setattr(live_exchange_equity, "stamp_protected_preexisting_dust", lambda *_a, **_k: None)

    eng = PortfolioEngine.__new__(PortfolioEngine)
    eng.db_path = db_path
    eng.principal = 1084.94
    eng.cash_balance = 1084.94
    eng._available_balance = 1084.94
    eng._positions_value = 0.0
    eng._realized_pnl = 0.0
    eng._unrealized_pnl = 0.0
    eng._total_equity = 1084.94
    eng._total_open_risk = 0.0
    eng._account_status = AccountStatus.HEALTHY
    eng._trading_paused = False
    eng._pause_reason = ""
    eng._kill_switch_mode = KillSwitchMode.RESUME
    eng._kill_switch_reason = ""
    eng._live_execution_enabled = True
    eng._live_service = SimpleNamespace()
    eng._positions_initialized = True
    eng._last_snapshot_time = 0.0
    eng._snapshot_interval_sec = 3600.0
    eng._snapshot_retention_days = 7
    eng._metrics_reconciliation_adjustments = 0
    eng._position_mark_prices = {"BTC/USDT": 81800.0, "ETH/USDT": 2395.0, "SOL/USDT": 103.9, "XRP/USDT": 1.30}
    eng._symbol_constraints = {
        "BTC/USDT": {"qty_step": 0.00001, "min_notional": 5.0},
        "ETH/USDT": {"qty_step": 0.0001, "min_notional": 5.0},
        "SOL/USDT": {"qty_step": 0.001, "min_notional": 5.0},
        "XRP/USDT": {"qty_step": 0.1, "min_notional": 5.0},
    }
    eng.open_positions = {}
    eng._deletion_lock = __import__("asyncio").Lock()
    eng._sell_execution_locks = {}

    async def _noop(*_a, **_k):
        return False

    eng._import_missing_exchange_positions = _noop
    eng._enforce_lot_ownership = _noop
    eng._shrink_unproven_dust_bucket = _noop
    eng._sync_protected_remainder = _noop
    eng._ensure_symbol_constraints = _noop
    eng._recompute_positions_values = _noop
    eng._compute_total_open_risk = lambda: None
    eng._write_reconcile_state_file = lambda *_a, **_k: None
    eng._floor_to_step = lambda q, _step: q
    eng._canonical_failsafe_decision = lambda: {"usable": True, "tripped": False, "nle": 1084.94}
    return eng


def _insert_held_dust(db_path: str, symbol: str, engine_id: str, trade_id: str, qty: float, price: float, entry_time: float = 1791300000.0) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO engine_strategy_dust
            (engine_id, symbol, source_trade_id, quantity, quantity_exact, entry_price, provenance_json, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'HELD', '2026-10-10T00:00:00Z')
            """,
            (engine_id, symbol, trade_id, qty, f"{qty:.8f}", price, f'{{"entry_time": {entry_time}}}'),
        )
        conn.commit()
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_exact_production_shortfall_retires_dust_and_clears_on_flat_book(tmp_path, monkeypatch, caplog):
    eng = _make_engine(tmp_path, monkeypatch)
    _insert_held_dust(eng.db_path, "BTC/USDT", "DAY_V2", "mystic_BTC_dust_1", 0.00074211, 81800.0)
    lot_day = OpenPosition(
        symbol="BTC/USDT",
        quantity=0.00000934,
        entry_price=81800.0,
        entry_time=1791565273.0,
        trade_id="mystic_BTC_lot",
        stop_price=0.0,
        take_profit_1_price=0.0,
        take_profit_2_price=0.0,
        engine_id="DAY_V2",
        status="DUST_PENDING",
    )
    lot_scalp = OpenPosition(
        symbol="BTC/USDT",
        quantity=0.00000965,
        entry_price=81800.0,
        entry_time=1792064311.0,
        trade_id="scalp_BTC_lot",
        stop_price=0.0,
        take_profit_1_price=0.0,
        take_profit_2_price=0.0,
        engine_id="SCALP_V2",
        status="DUST_PENDING",
    )
    eng.open_positions = {"DAY_V2::BTC/USDT": lot_day, "SCALP_V2::BTC/USDT": lot_scalp}

    # A-D: signed BTC < engine-owned BTC raises ENGINE_QTY_SHORTFALL:BTC/USDT and blocks entries.
    await eng._reconcile_dual_engine_lots(symbol="BTC/USDT", lots=[lot_day, lot_scalp], exchange_qty=0.00000583, qty_step=0.00001, source="periodic_reconcile")
    await eng._persist_ledger_to_sqlite()
    assert eng._trading_paused is True
    assert eng._pause_reason == "ENGINE_QTY_SHORTFALL:BTC/USDT"
    cap = eng.get_trading_capability_status()
    assert cap["effective_entry_permitted"] is False
    assert "ENGINE_QTY_SHORTFALL:BTC/USDT" in str(cap["blocking_reason"])

    # E-G: external Binance dust conversion (venue ref 2472343419) retires lots and leaves 0 open positions.
    async def _dust_logs():
        return [
            {
                "operateTime": 1792362770000,
                "userAssetDribbletDetails": [
                    {
                        "fromAsset": "BTC",
                        "toAsset": "USDT",
                        "amount": "0.00000583",
                        "transferredAmount": "0.45162518",
                        "serviceChargeAmount": "0.02376975",
                        "operateTime": 1792362770000,
                        "tranId": 2472343419,
                    }
                ],
            }
        ]

    eng._venue_dust_conversions = _dust_logs
    eng._fill_owned_qty = lambda _pos: 0.00001

    async def _del_pos(_sym, _eid):
        return None

    eng._delete_position_from_sqlite = _del_pos

    # First reconcile at ex_qty=0 vanishes the two DUST_PENDING lots AND retires held dust, then clears pause.
    with caplog.at_level(logging.WARNING):
        await eng.run_live_reconcile({"USDT": 1084.94, "BTC": 0.0}, free_balances={"USDT": 1084.94, "BTC": 0.0})

    assert eng.open_positions == {}
    assert eng._held_engine_dust_qty("BTC/USDT") == 0.0
    assert eng._trading_paused is False
    assert eng._pause_reason == ""
    assert "ENGINE_QTY_SHORTFALL_CLEARED symbol=BTC/USDT" in caplog.text
    after = eng.get_trading_capability_status()
    assert after["effective_entry_permitted"] is True
    assert after["kill_switch_mode"] == "RESUME"

    # Also prove that if the pause was already latched when open_positions was empty, flat-book reconcile clears it.
    eng._trading_paused = True
    eng._pause_reason = "ENGINE_QTY_SHORTFALL:BTC/USDT"
    await eng._persist_ledger_to_sqlite()
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        await eng.run_live_reconcile({"USDT": 1084.94}, free_balances={"USDT": 1084.94})
    assert eng._trading_paused is False and eng._pause_reason == ""
    assert "ENGINE_QTY_SHORTFALL_CLEARED symbol=BTC/USDT" in caplog.text
    with sqlite3.connect(eng.db_path) as conn:
        paused, reason = conn.execute("SELECT trading_paused, pause_reason FROM portfolio_engine_ledger WHERE id=1").fetchone()
    assert paused == 0 and reason == ""

    # 11. Idempotency over 12 repeated flat-book reconciliations.
    caplog.clear()
    for _ in range(12):
        await eng.run_live_reconcile({"USDT": 1084.94}, free_balances={"USDT": 1084.94})
    assert eng._trading_paused is False and eng._pause_reason == ""
    assert "ENGINE_QTY_SHORTFALL_CLEARED" not in caplog.text
    assert eng.cash_balance == pytest.approx(1084.94)
    assert eng.get_trading_capability_status()["effective_entry_permitted"] is True


@pytest.mark.asyncio
async def test_genuine_shortfall_on_flat_book_does_not_clear(tmp_path, monkeypatch):
    eng = _make_engine(tmp_path, monkeypatch)
    # 0.0005 BTC held dust (~$40.90) with 0 open positions and 0 signed BTC, no conversion log.
    _insert_held_dust(eng.db_path, "BTC/USDT", "DAY_V2", "unconverted_btc_dust", 0.0005, 81800.0)
    eng._venue_dust_conversions = lambda: __import__("asyncio").sleep(0, result=[])
    eng._trading_paused = True
    eng._pause_reason = "ENGINE_QTY_SHORTFALL:BTC/USDT"

    await eng.run_live_reconcile({"USDT": 1084.94, "BTC": 0.0}, free_balances={"USDT": 1084.94})

    assert eng._trading_paused is True
    assert eng._pause_reason == "ENGINE_QTY_SHORTFALL:BTC/USDT"
    assert eng.get_trading_capability_status()["effective_entry_permitted"] is False


@pytest.mark.asyncio
async def test_operator_hold_and_kill_switch_survive_reconciliation(tmp_path, monkeypatch):
    eng = _make_engine(tmp_path, monkeypatch)
    eng._kill_switch_mode = KillSwitchMode.PAUSE_BUYS
    eng._kill_switch_reason = "OPERATOR_HOLD"
    with sqlite3.connect(eng.db_path) as conn:
        conn.execute("UPDATE portfolio_engine_ledger SET kill_switch_mode='PAUSE_BUYS', kill_switch_reason='OPERATOR_HOLD' WHERE id=1")
        conn.commit()

    for _ in range(5):
        await eng.run_live_reconcile({"USDT": 1084.94}, free_balances={"USDT": 1084.94})

    cap = eng.get_trading_capability_status()
    assert cap["effective_entry_permitted"] is False
    assert cap["kill_switch_mode"] == "PAUSE_BUYS"
    assert cap["kill_switch_reason"] == "OPERATOR_HOLD"


@pytest.mark.asyncio
async def test_other_safety_blockers_are_never_cleared_by_shortfall_cleanup(tmp_path, monkeypatch):
    from backend.services import circuit_breaker_service

    eng = _make_engine(tmp_path, monkeypatch)

    # 1. Non-shortfall trading_paused reason (e.g. UNBOOKED_LIVE_FILL / invariant).
    eng._trading_paused = True
    eng._pause_reason = "UNBOOKED_LIVE_FILL:BTC/USDT:999"
    await eng.run_live_reconcile({"USDT": 1084.94}, free_balances={"USDT": 1084.94})
    assert eng._trading_paused is True and eng._pause_reason == "UNBOOKED_LIVE_FILL:BTC/USDT:999"

    # 2. Quarantined filled asset custody blocks clearing even if pause_reason is shortfall.
    eng._pause_reason = "ENGINE_QTY_SHORTFALL:BTC/USDT"
    remember(eng.db_path, engine_id="DAY_V2", symbol="BTC/USDT", trade_id="q_trade", quantity=0.001, fill_price=80000.0)
    mark(eng.db_path, engine_id="DAY_V2", trade_id="q_trade", status=QUARANTINED_FILLED_ASSET)
    await eng.run_live_reconcile({"USDT": 1084.94}, free_balances={"USDT": 1084.94})
    assert eng._trading_paused is True and eng._pause_reason == "ENGINE_QTY_SHORTFALL:BTC/USDT"

    # Resolve custody, now test active equity circuit breaker / daily-loss freeze.
    with sqlite3.connect(eng.db_path) as conn:
        conn.execute("DELETE FROM filled_asset_custody")
        conn.commit()
    monkeypatch.setattr(
        circuit_breaker_service,
        "read_persisted_entry_control",
        lambda _db: {
            "requested_kill_mode": "RESUME",
            "requested_kill_reason": "",
            "equity_circuit_breaker_active": True,
            "daily_loss_freeze_active": True,
            "read_failed": False,
        },
    )
    await eng.run_live_reconcile({"USDT": 1084.94}, free_balances={"USDT": 1084.94})
    # Shortfall itself cleared, but breaker and freeze still block entries.
    assert eng._trading_paused is False
    cap = eng.get_trading_capability_status()
    assert cap["effective_entry_permitted"] is False
    assert cap["equity_circuit_breaker_active"] is True
    assert cap["daily_loss_freeze_active"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("symbol,asset,price,qty", [("ETH/USDT", "ETH", 2395.0, 0.02), ("SOL/USDT", "SOL", 103.9, 0.5), ("XRP/USDT", "XRP", 1.30, 50.0)])
async def test_multi_asset_shortfall_pause_clears_only_when_that_asset_reconciles(tmp_path, monkeypatch, symbol, asset, price, qty):
    eng = _make_engine(tmp_path, monkeypatch)
    _insert_held_dust(eng.db_path, symbol, "SCALP_V2", f"dust_{asset}", qty, price)
    eng._venue_dust_conversions = lambda: __import__("asyncio").sleep(0, result=[])
    eng._trading_paused = True
    eng._pause_reason = f"ENGINE_QTY_SHORTFALL:{symbol}"

    # Shortfall still present (exchange has 0 of asset): stays paused.
    await eng.run_live_reconcile({"USDT": 1084.94, asset: 0.0}, free_balances={"USDT": 1084.94})
    assert eng._trading_paused is True and eng._pause_reason == f"ENGINE_QTY_SHORTFALL:{symbol}"

    # Physical balance matches engine-owned quantity: clears automatically.
    await eng.run_live_reconcile({"USDT": 1084.94, asset: qty}, free_balances={"USDT": 1084.94})
    assert eng._trading_paused is False and eng._pause_reason == ""
    assert eng.get_trading_capability_status()["effective_entry_permitted"] is True
