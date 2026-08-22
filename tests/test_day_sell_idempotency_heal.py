"""DAY sell identity: TP1 leftover must close; heal must not resurrect a full close."""

from __future__ import annotations

import json
import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.services.portfolio_engine import ExitType, OpenPosition, PortfolioEngine, Sleeve


def _init_test_db(db_path: Path, *, cash: float, principal: float = 25_000.0) -> None:
    from backend.database_schema import initialize_paper_trading_schema

    engine = PortfolioEngine(db_path=str(db_path), principal=principal, test_mode=True)
    engine._ensure_db_schema()
    initialize_paper_trading_schema(str(db_path))
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO portfolio_engine_ledger (
                id, principal, cash_balance, positions_value,
                realized_pnl, unrealized_pnl, total_equity,
                account_status, trading_paused, pause_reason, last_updated, version
            ) VALUES (1, ?, ?, 0, 0, 0, ?, 'HEALTHY', 0, NULL, datetime('now'), 1)
            """,
            (principal, cash, cash),
        )
        conn.commit()


def _seed_position(
    db_path: Path,
    *,
    symbol: str,
    trade_id: str,
    qty: float,
    entry: float,
    cash_after_buy: float,
    highest: float | None = None,
    tp1_hit: int = 0,
) -> None:
    now = time.time()
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            """
            INSERT INTO portfolio_engine_positions (
                symbol, quantity, entry_price, entry_time, trade_id,
                stop_price, take_profit_1_price, take_profit_2_price,
                highest_price, lowest_price, tp1_hit, atr_at_entry, entry_bar_timestamp,
                confidence_at_entry, last_updated, sleeve
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), ?)
            """,
            (
                symbol,
                qty,
                entry,
                now,
                trade_id,
                entry * 0.97,
                entry * 1.03,
                0.0,
                highest if highest is not None else entry,
                entry * 0.99,
                tp1_hit,
                entry * 0.01,
                0,
                0.5,
                Sleeve.ACTIVE.value,
            ),
        )
        conn.execute(
            """
            INSERT INTO paper_trades (
                trade_id, paper_run_id, mode, symbol, side, quantity, price,
                remaining_position, timestamp, status, strategy_id
            ) VALUES (?, 'test-run', 'paper', ?, 'BUY', ?, ?, ?, datetime('now'), 'executed', 'day')
            """,
            (trade_id, symbol, qty, entry, qty),
        )
        conn.execute(
            """
            UPDATE portfolio_engine_ledger SET cash_balance = ?, total_equity = ?, positions_value = ?
            WHERE id = 1
            """,
            (cash_after_buy, cash_after_buy + qty * entry, qty * entry),
        )
        conn.commit()


def _allowed_sell_eval(symbol: str = "XRP/USDT", **overrides):
    base = {
        "symbol": symbol,
        "reason": "NET_PROFIT_EXIT",
        "exit_branch": "profit",
        "avg_entry_price": 1.378,
        "mark_price": 1.45,
        "mark_source": "test",
        "mark_age_seconds": 0.0,
        "gross_pct": 0.05,
        "effective_exit_cost_pct": 0.001,
        "effective_roundtrip_cost_pct": 0.002,
        "net_exit_pct": 0.049,
        "roundtrip_net_pct": 0.049,
        "required_profit_buffer_pct": 0.001,
        "allowed": True,
        "block_reason": "",
        "emergency_flag": False,
        "measured_cost_sample_count": 0,
        "measured_cost_p75": 0.0,
    }
    base.update(overrides)
    return base


def _wire_sell_engine(engine: PortfolioEngine, symbol: str, qty: float, sell_price: float) -> None:
    engine._live_execution_enabled = False
    engine._exit_in_progress = set()
    engine._paper_service = MagicMock(paper_run_id="test-run")
    engine._paper_service.redis_client = None
    engine._check_kill_switch_sell = MagicMock(return_value=(True, ""))
    engine._record_reject = AsyncMock()
    engine._delete_position_from_sqlite = AsyncMock()
    engine._clear_all_quarantines = AsyncMock()
    engine._update_coin_performance = AsyncMock()
    engine._record_thesis_regime_outcome = MagicMock()
    engine._record_day_bucket_outcome = MagicMock()
    engine._record_position_close_ledger = AsyncMock()
    engine._write_closed_lot_tombstone = AsyncMock()
    engine._persist_quality_cooldowns = AsyncMock()
    engine.record_sell_cooldown = MagicMock()
    engine._resolve_learning_close_reason = MagicMock(return_value="NET_PROFIT_EXIT")
    engine._get_loss_hold_until = AsyncMock(return_value=None)
    engine._set_loss_hold_until = AsyncMock()
    engine.get_rolling_24h_risk_metrics = AsyncMock(return_value=(0.0, 0))
    engine._validate_invariants = AsyncMock(return_value=True)
    engine._record_audit = AsyncMock()
    engine._entry_ensure_constraints = AsyncMock()
    engine._ensure_symbol_constraints = AsyncMock()
    engine._symbol_constraints[symbol] = {"qty_step": 0.00001, "min_qty": 0.00001, "min_notional": 1.0}
    engine._normalize_order_amount = MagicMock(return_value=(qty, "ok", qty))
    engine._dust_check = MagicMock(return_value=(False, qty, "", qty * sell_price))
    engine._floor_to_step = lambda q, _s: q
    engine._evaluate_sell_profitability = AsyncMock(
        return_value=_allowed_sell_eval(symbol=symbol, mark_price=sell_price, avg_entry_price=engine.open_positions[symbol].entry_price)
    )


async def _sell(engine: PortfolioEngine, symbol: str, qty: float, price: float, reason: str = "NET_PROFIT_EXIT"):
    mock_pf = MagicMock(passed=True, expected_avg_fill=price)
    mock_pf.to_audit_dict = MagicMock(return_value={"passed": True})
    with patch("backend.services.protected_limit_execution.run_protected_preflight", AsyncMock(return_value=mock_pf)):
        with patch("backend.services.protected_limit_execution.USE_PROTECTED_LIMIT_EXECUTION", True):
            with patch(
                "backend.services.paper_trading_service.get_paper_trading_service",
                return_value=engine._paper_service,
            ):
                return await engine.execute_sell_fifo(
                    symbol,
                    qty,
                    price,
                    ExitType.TAKE_PROFIT_1,
                    reason,
                    force_sell=True,
                )


def test_idempotency_allows_same_qty_reason_while_remainder_open():
    engine = PortfolioEngine(principal=10_000.0, test_mode=True)
    assert (
        engine._sell_idempotency_duplicate_sync(
            "XRP/USDT",
            "mystic_XRP/USDT_lot1",
            594.48313,
            "NET_PROFIT_EXIT",
            available_qty=594.48313,
        )
        is False
    )


def test_idempotency_blocks_when_lot_qty_already_consumed():
    engine = PortfolioEngine(principal=10_000.0, test_mode=True)
    assert (
        engine._sell_idempotency_duplicate_sync(
            "XRP/USDT",
            "mystic_XRP/USDT_lot1",
            594.48313,
            "NET_PROFIT_EXIT",
            available_qty=0.0,
        )
        is True
    )


def test_heal_mark_does_not_reset_to_entry_when_live_mark_exists():
    engine = PortfolioEngine(principal=10_000.0, test_mode=True)
    engine._position_mark_prices["XRP/USDT"] = 1.463
    existing = MagicMock(current_price=1.3781356442088137)
    mark = engine._resolve_heal_mark("XRP/USDT", 1.3781356442088137, existing)
    assert mark == pytest.approx(1.463)


def test_heal_mark_keeps_nonzero_redis_mark_when_not_entry():
    engine = PortfolioEngine(principal=10_000.0, test_mode=True)
    existing = MagicMock(current_price=1.45)
    mark = engine._resolve_heal_mark("XRP/USDT", 1.378, existing)
    assert mark == pytest.approx(1.45)


def test_closed_lot_tombstone_is_lot_not_symbol_scoped():
    assert PortfolioEngine._closed_lot_tombstone_key("mystic_XRP/USDT_1") == "paper:closed_lot:mystic_XRP/USDT_1"
    assert PortfolioEngine._closed_lot_tombstone_key("mystic_XRP/USDT_2") != PortfolioEngine._closed_lot_tombstone_key(
        "mystic_XRP/USDT_1"
    )


@pytest.mark.asyncio
async def test_tp1_then_remainder_same_reason_and_qty_both_commit():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "tp1.db"
        symbol = "XRP/USDT"
        trade_id = "mystic_XRP/USDT_lot_partial"
        total = 1188.96626
        half = 594.48313
        entry = 1.3781356442088137
        _init_test_db(db_path, cash=10_000.0)
        _seed_position(
            db_path,
            symbol=symbol,
            trade_id=trade_id,
            qty=total,
            entry=entry,
            cash_after_buy=10_000.0 - total * entry,
            highest=1.6966,
        )

        engine = PortfolioEngine(db_path=str(db_path), principal=10_000.0, test_mode=True)
        await engine.initialize_from_db()
        engine.open_positions[symbol] = OpenPosition(
            symbol=symbol,
            quantity=total,
            entry_price=entry,
            entry_time=time.time() - 800,
            trade_id=trade_id,
            stop_price=entry * 0.97,
            take_profit_1_price=entry * 1.03,
            take_profit_2_price=0.0,
            highest_price=1.6966,
            lowest_price=1.3642,
            sleeve=Sleeve.ACTIVE.value,
            entry_strategy_id="day",
        )
        _wire_sell_engine(engine, symbol, half, 1.4111)

        first = await _sell(engine, symbol, half, 1.4111)
        assert first is not None
        with sqlite3.connect(str(db_path)) as conn:
            rem = conn.execute("SELECT quantity FROM portfolio_engine_positions WHERE symbol=?", (symbol,)).fetchone()
            sells = conn.execute("SELECT COUNT(*) FROM paper_trades WHERE side='SELL'").fetchone()[0]
        assert rem is not None
        assert rem[0] == pytest.approx(half, abs=1e-6)
        assert sells == 1
        assert symbol in engine.open_positions
        assert engine.open_positions[symbol].quantity == pytest.approx(half, abs=1e-6)

        engine.open_positions[symbol].tp1_hit = True
        engine._exit_in_progress.clear()
        second = await _sell(engine, symbol, half, 1.463)
        assert second is not None
        with sqlite3.connect(str(db_path)) as conn:
            rem = conn.execute("SELECT quantity FROM portfolio_engine_positions WHERE symbol=?", (symbol,)).fetchone()
            sells = conn.execute(
                "SELECT quantity, exit_reason FROM paper_trades WHERE side='SELL' ORDER BY id"
            ).fetchall()
        assert rem is None or float(rem[0] or 0) <= 0
        assert len(sells) == 2
        assert sells[0][0] == pytest.approx(half, abs=1e-6)
        assert sells[1][0] == pytest.approx(half, abs=1e-6)
        assert sells[0][1] == "NET_PROFIT_EXIT"
        assert sells[1][1] == "NET_PROFIT_EXIT"
        assert symbol not in engine.open_positions


@pytest.mark.asyncio
async def test_full_close_duplicate_sell_is_idempotent_single_pnl():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "full.db"
        symbol = "BTC/USDT"
        trade_id = "mystic_BTC/USDT_full"
        qty = 0.02
        entry = 77000.0
        _init_test_db(db_path, cash=25_000.0)
        _seed_position(db_path, symbol=symbol, trade_id=trade_id, qty=qty, entry=entry, cash_after_buy=25_000.0 - qty * entry)

        engine = PortfolioEngine(db_path=str(db_path), principal=25_000.0, test_mode=True)
        await engine.initialize_from_db()
        engine.open_positions[symbol] = OpenPosition(
            symbol=symbol,
            quantity=qty,
            entry_price=entry,
            entry_time=time.time() - 100,
            trade_id=trade_id,
            stop_price=entry * 0.97,
            take_profit_1_price=entry * 1.03,
            take_profit_2_price=0.0,
            highest_price=entry,
            lowest_price=entry,
            sleeve=Sleeve.ACTIVE.value,
            entry_strategy_id="day",
        )
        _wire_sell_engine(engine, symbol, qty, 78500.0)
        first = await _sell(engine, symbol, qty, 78500.0)
        engine._exit_in_progress.clear()
        second = await _sell(engine, symbol, qty, 78500.0)
        assert first is not None
        assert second is None
        with sqlite3.connect(str(db_path)) as conn:
            sell_n = conn.execute("SELECT COUNT(*) FROM paper_trades WHERE side='SELL'").fetchone()[0]
            open_n = conn.execute("SELECT COUNT(*) FROM portfolio_engine_positions").fetchone()[0]
        assert sell_n == 1
        assert open_n == 0


@pytest.mark.asyncio
async def test_failed_close_before_commit_leaves_position_and_no_sell_row():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "fail.db"
        symbol = "SOL/USDT"
        trade_id = "mystic_SOL/USDT_fail"
        qty = 10.0
        entry = 90.0
        _init_test_db(db_path, cash=10_000.0)
        _seed_position(db_path, symbol=symbol, trade_id=trade_id, qty=qty, entry=entry, cash_after_buy=10_000.0 - qty * entry)
        engine = PortfolioEngine(db_path=str(db_path), principal=10_000.0, test_mode=True)
        await engine.initialize_from_db()
        engine.open_positions[symbol] = OpenPosition(
            symbol=symbol,
            quantity=qty,
            entry_price=entry,
            entry_time=time.time() - 50,
            trade_id=trade_id,
            stop_price=entry * 0.97,
            take_profit_1_price=entry * 1.03,
            take_profit_2_price=0.0,
            highest_price=entry,
            lowest_price=entry,
            sleeve=Sleeve.ACTIVE.value,
            entry_strategy_id="day",
        )
        _wire_sell_engine(engine, symbol, qty, 91.0)
        engine._check_kill_switch_sell = MagicMock(return_value=(False, "TEST_BLOCK"))
        result = await _sell(engine, symbol, qty, 91.0)
        assert result is None
        with sqlite3.connect(str(db_path)) as conn:
            rem = conn.execute("SELECT quantity FROM portfolio_engine_positions WHERE symbol=?", (symbol,)).fetchone()
            sells = conn.execute("SELECT COUNT(*) FROM paper_trades WHERE side='SELL'").fetchone()[0]
        assert rem[0] == pytest.approx(qty)
        assert sells == 0


@pytest.mark.asyncio
async def test_heal_removes_redis_when_sqlite_closed_and_skips_tombstoned_lot():
    engine = PortfolioEngine(principal=10_000.0, test_mode=True)
    engine._live_execution_enabled = False
    engine.test_mode = False

    class _FakeRedis:
        def __init__(self):
            self.kv = {"paper:closed_lot:old_xrp": json.dumps({"kind": "FULL"})}
            self.hashes = {"paper:position:XRP/USDT": {"symbol": "XRP/USDT", "quantity": "594.48"}}
            self.active = {"XRP/USDT"}

        async def get(self, key):
            return self.kv.get(key)

        async def set(self, key, value, ex=None):
            self.kv[key] = value

        async def delete(self, key):
            self.kv.pop(key, None)
            self.hashes.pop(key, None)

        async def smembers(self, key):
            return set(self.active)

        async def srem(self, key, member):
            self.active.discard(member)

        async def scan_iter(self, match=None):
            if False:
                yield None

    fake = _FakeRedis()
    paper = MagicMock()
    paper.redis_client = fake
    paper.positions = {
        "XRP/USDT": MagicMock(
            quantity=594.48,
            average_price=1.378,
            current_price=1.378,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            repair_add_count=0,
            last_repair_add_ts=0,
            entry_strategy_id="day",
        )
    }
    paper._ensure_redis = AsyncMock()
    paper._persist_position_to_redis = AsyncMock()
    paper._delete_position_from_redis = AsyncMock(side_effect=lambda s: fake.active.discard(s) or fake.hashes.pop(f"paper:position:{s}", None))
    paper._persist_cash_balance_to_redis = AsyncMock()
    paper._persist_realized_pnl_total_to_redis = AsyncMock()
    paper.current_balance = 1000.0
    paper.realized_pnl_total = 0.0

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "heal.db"
        _init_test_db(db_path, cash=1000.0)
        engine.db_path = str(db_path)
        engine._ensure_db_schema()
        engine._paper_service = paper
        with patch("backend.services.paper_trading_service.get_paper_trading_service", return_value=paper):
            await engine._sync_paper_redis_from_sqlite_authoritative()
        paper._delete_position_from_redis.assert_awaited()
        assert "XRP/USDT" not in paper.positions


@pytest.mark.asyncio
async def test_heal_preserves_open_sqlite_lot_and_does_not_use_entry_when_mark_known():
    engine = PortfolioEngine(principal=10_000.0, test_mode=True)
    engine._live_execution_enabled = False
    engine.test_mode = False
    engine._position_mark_prices["XRP/USDT"] = 1.46315

    class _FakeRedis:
        def __init__(self):
            self.kv = {}

        async def get(self, key):
            return self.kv.get(key)

        async def delete(self, key):
            self.kv.pop(key, None)

        async def smembers(self, key):
            return set()

        async def scan_iter(self, match=None):
            if False:
                yield None

    paper = MagicMock()
    paper.redis_client = _FakeRedis()
    paper.positions = {}
    paper._ensure_redis = AsyncMock()
    paper._persist_position_to_redis = AsyncMock()
    paper._delete_position_from_redis = AsyncMock()
    paper._persist_cash_balance_to_redis = AsyncMock()
    paper._persist_realized_pnl_total_to_redis = AsyncMock()
    paper.current_balance = 7312.0
    paper.realized_pnl_total = 900.0

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "heal2.db"
        _init_test_db(db_path, cash=7312.0)
        _seed_position(
            db_path,
            symbol="XRP/USDT",
            trade_id="mystic_XRP/USDT_open",
            qty=594.48313,
            entry=1.3781356442088137,
            cash_after_buy=7312.0,
            highest=1.6966,
            tp1_hit=1,
        )
        engine.db_path = str(db_path)
        engine._paper_service = paper
        from backend.services.paper_trading_service import PaperPosition

        with patch("backend.services.paper_trading_service.get_paper_trading_service", return_value=paper):
            await engine._sync_paper_redis_from_sqlite_authoritative()
        paper._persist_position_to_redis.assert_awaited()
        pos = paper.positions["XRP/USDT"]
        assert float(pos.quantity) == pytest.approx(594.48313)
        assert float(pos.current_price) == pytest.approx(1.46315)
        assert float(pos.average_price) == pytest.approx(1.3781356442088137)


@pytest.mark.asyncio
async def test_new_symbol_lot_not_blocked_by_old_lot_idempotency(tmp_path: Path):
    db_path = tmp_path / "newlot.db"
    symbol = "XRP/USDT"
    old_id = "mystic_XRP/USDT_old"
    new_id = "mystic_XRP/USDT_new"
    qty = 100.0
    entry = 1.2
    _init_test_db(db_path, cash=5000.0)
    _seed_position(db_path, symbol=symbol, trade_id=old_id, qty=qty, entry=entry, cash_after_buy=5000.0 - qty * entry)
    engine = PortfolioEngine(db_path=str(db_path), principal=5000.0, test_mode=True)
    await engine.initialize_from_db()
    engine.open_positions[symbol] = OpenPosition(
        symbol=symbol,
        quantity=qty,
        entry_price=entry,
        entry_time=time.time() - 20,
        trade_id=old_id,
        stop_price=entry * 0.97,
        take_profit_1_price=entry * 1.03,
        take_profit_2_price=0.0,
        highest_price=entry,
        lowest_price=entry,
        sleeve=Sleeve.ACTIVE.value,
        entry_strategy_id="day",
    )
    _wire_sell_engine(engine, symbol, qty, 1.25)
    closed = await _sell(engine, symbol, qty, 1.25)
    assert closed is not None

    _seed_position(db_path, symbol=symbol, trade_id=new_id, qty=qty, entry=1.3, cash_after_buy=4000.0)
    engine.open_positions[symbol] = OpenPosition(
        symbol=symbol,
        quantity=qty,
        entry_price=1.3,
        entry_time=time.time(),
        trade_id=new_id,
        stop_price=1.25,
        take_profit_1_price=1.35,
        take_profit_2_price=0.0,
        highest_price=1.3,
        lowest_price=1.3,
        sleeve=Sleeve.ACTIVE.value,
        entry_strategy_id="day",
    )
    engine._exit_in_progress.clear()
    _wire_sell_engine(engine, symbol, qty, 1.35)
    again = await _sell(engine, symbol, qty, 1.35)
    assert again is not None
    with sqlite3.connect(str(db_path)) as conn:
        n = conn.execute("SELECT COUNT(*) FROM paper_trades WHERE side='SELL'").fetchone()[0]
    assert n == 2
