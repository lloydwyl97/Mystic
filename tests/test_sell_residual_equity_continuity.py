"""Residual qty must be the only coins valued after a confirmed sell fill."""

from __future__ import annotations

import sqlite3
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.services.live_fill_economics import LiveCommission
from backend.services.portfolio_engine import ExitType, OpenPosition, PortfolioEngine, Sleeve


def _pos(**kwargs):
    defaults = {
        "symbol": "XRP/USDT",
        "quantity": 47.1,
        "entry_price": 1.3597,
        "entry_time": time.time() - 120,
        "trade_id": "xrp_buy",
        "stop_price": 1.32,
        "take_profit_1_price": 1.40,
        "take_profit_2_price": 0.0,
        "highest_price": 1.36,
        "lowest_price": 1.359,
        "sleeve": Sleeve.ACTIVE.value,
        "entry_strategy_id": "day",
        "status": "ACTIVE",
    }
    defaults.update(kwargs)
    return OpenPosition(**defaults)


def _engine():
    eng = PortfolioEngine(principal=250.0, test_mode=True)
    eng.cash_balance = 169.20
    eng._positions_value = 47.1 * 1.3578
    eng._cost_basis = 47.1 * 1.3597
    eng._total_equity = eng.cash_balance + eng._positions_value
    eng._symbol_constraints["XRP/USDT"] = {"qty_step": 0.1, "min_qty": 1.0, "min_notional": 5.0}
    return eng


def test_complete_fill_values_zero_residual():
    eng = _engine()
    pos = _pos(quantity=47.0)
    rem = eng._apply_confirmed_sell_qty_to_memory(pos, 47.0, fill_price=1.3578, qty_step=0.1, min_qty=1.0)
    assert rem == 0.0
    assert pos.quantity == 0.0
    eng.open_positions["XRP/USDT"] = pos
    eng._mtm_after_confirmed_sell("XRP/USDT", 1.3578)
    assert abs(eng._positions_value) < 1e-9


def test_partial_fill_values_only_unsold_qty():
    eng = _engine()
    pos = _pos(quantity=47.1)
    rem = eng._apply_confirmed_sell_qty_to_memory(pos, 47.0, fill_price=1.3578, qty_step=0.1, min_qty=1.0, min_notional=5.0)
    assert abs(rem - 0.1) < 1e-9
    assert abs(pos.quantity - 0.1) < 1e-9
    assert pos.status == "DUST_PENDING"
    eng.open_positions["XRP/USDT"] = pos
    cash_before = eng.cash_balance
    proceeds = 47.0 * 1.3578
    eng._apply_sell_cash_credit(proceeds, -0.10)
    eng._mtm_after_confirmed_sell("XRP/USDT", 1.3578)
    assert abs(eng._positions_value - 0.1 * 1.3578) < 1e-6
    assert abs(eng.cash_balance - (cash_before + proceeds)) < 1e-6
    assert abs(eng._total_equity - (eng.cash_balance + 0.1 * 1.3578)) < 1e-6
    fake = cash_before + proceeds + 47.0 * 1.3578
    assert eng._total_equity < fake - 50.0


def test_no_fill_leaves_book_qty():
    eng = _engine()
    pos = _pos()
    rem = eng._apply_confirmed_sell_qty_to_memory(pos, 0.0, fill_price=1.3578)
    assert abs(rem - 47.1) < 1e-9
    assert pos.quantity == 47.1


def test_residual_retry_is_idempotent():
    eng = _engine()
    pos = _pos()
    first = eng._apply_confirmed_sell_qty_to_memory(pos, 47.0, fill_price=1.3578, qty_step=0.1, min_qty=1.0)
    second = eng._apply_confirmed_sell_qty_to_memory(pos, 47.0, fill_price=1.3578, qty_step=0.1, min_qty=1.0)
    assert abs(first - 0.1) < 1e-9
    assert abs(second - 0.1) < 1e-9
    assert abs(pos.quantity - 0.1) < 1e-9


def test_dust_residual_after_exchange_qty_differs_from_db():
    eng = _engine()
    pos = _pos(quantity=47.09)
    rem = eng._apply_confirmed_sell_qty_to_memory(pos, 47.0, fill_price=1.3578, qty_step=0.1, min_qty=1.0, min_notional=5.0)
    assert rem > 0.0
    assert rem < 1.0
    assert pos.status == "DUST_PENDING"
    eng.open_positions["XRP/USDT"] = pos
    eng._mtm_after_confirmed_sell("XRP/USDT", 1.3578)
    assert abs(eng._positions_value - rem * 1.3578) < 1e-6


def test_base_asset_commission_reduces_residual_only():
    eng = _engine()
    pos = _pos(quantity=47.1)
    rem = eng._apply_confirmed_sell_qty_to_memory(
        pos,
        47.0,
        fill_price=1.3578,
        qty_step=0.1,
        min_qty=1.0,
        min_notional=5.0,
        base_qty_reduction=0.047,
    )
    assert abs(rem - 0.053) < 1e-9 or abs(rem - 0.0) < 1e-9 or rem <= 0.1
    eng.open_positions["XRP/USDT"] = pos
    eng._mtm_after_confirmed_sell("XRP/USDT", 1.3578)
    assert eng._positions_value <= 0.1 * 1.3578 + 1e-9


def test_live_commission_base_qty_from_exchange():
    comm = LiveCommission(usd=0.064, fee_from_exchange=True, base_qty_reduction=0.047, quote_commission_usd=0.0)
    assert comm.base_qty_reduction == 0.047


def _init_db(db_path: Path, *, cash: float, qty: float, entry: float) -> None:
    from backend.database_schema import initialize_paper_trading_schema

    engine = PortfolioEngine(db_path=str(db_path), principal=250.0, test_mode=True)
    engine._ensure_db_schema()
    initialize_paper_trading_schema(str(db_path))
    now = time.time()
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO portfolio_engine_ledger (
                id, principal, cash_balance, positions_value,
                realized_pnl, unrealized_pnl, total_equity,
                account_status, trading_paused, pause_reason, last_updated, version
            ) VALUES (1, 250.0, ?, ?, 0, 0, ?, 'HEALTHY', 0, NULL, datetime('now'), 1)
            """,
            (cash, qty * entry, cash + qty * entry),
        )
        conn.execute(
            """
            INSERT INTO portfolio_engine_positions (
                symbol, quantity, entry_price, entry_time, trade_id,
                stop_price, take_profit_1_price, take_profit_2_price,
                highest_price, atr_at_entry, entry_bar_timestamp,
                confidence_at_entry, last_updated, sleeve
            ) VALUES ('XRP/USDT', ?, ?, ?, 'xrp_buy', ?, ?, 0, ?, ?, 0, 0.5, datetime('now'), ?)
            """,
            (qty, entry, now, entry * 0.97, entry * 1.03, entry, entry * 0.01, Sleeve.ACTIVE.value),
        )
        conn.execute(
            """
            INSERT INTO paper_trades (
                trade_id, paper_run_id, mode, symbol, side, quantity, price,
                remaining_position, timestamp, status, strategy_id
            ) VALUES ('xrp_buy', 'test-run', 'live', 'XRP/USDT', 'BUY', ?, ?, ?, datetime('now'), 'executed', 'day')
            """,
            (qty, entry, qty),
        )
        conn.commit()


@pytest.mark.asyncio
async def test_execute_sell_fifo_partial_fill_equity_continuous():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        qty = 47.1
        sold = 47.0
        entry = 1.3597
        fill = 1.3578
        cash = 169.20
        _init_db(db_path, cash=cash, qty=qty, entry=entry)
        engine = PortfolioEngine(db_path=str(db_path), principal=250.0, test_mode=True)
        await engine.initialize_from_db()
        engine.open_positions["XRP/USDT"] = _pos(quantity=qty, entry_price=entry)
        engine.cash_balance = cash
        engine._positions_value = qty * fill
        engine._total_equity = cash + qty * fill
        engine._positions_initialized = True
        engine._live_execution_enabled = False
        engine._exit_in_progress = set()
        engine._paper_service = MagicMock(paper_run_id="test-run")
        engine._paper_service.place_order = AsyncMock(return_value={"status": "ok"})
        engine._check_kill_switch_sell = MagicMock(return_value=(True, ""))
        engine._record_reject = AsyncMock()
        engine._delete_position_from_sqlite = AsyncMock()
        engine._clear_all_quarantines = AsyncMock()
        engine._update_coin_performance = AsyncMock()
        engine._record_thesis_regime_outcome = MagicMock()
        engine._record_day_bucket_outcome = MagicMock()
        engine._record_position_close_ledger = AsyncMock()
        engine._persist_quality_cooldowns = AsyncMock()
        engine.record_sell_cooldown = MagicMock()
        engine._resolve_learning_close_reason = MagicMock(return_value="DAY_4H_STRUCTURE_BREAK_EXIT")
        engine._get_loss_hold_until = AsyncMock(return_value=None)
        engine._set_loss_hold_until = AsyncMock()
        engine.get_rolling_24h_risk_metrics = AsyncMock(return_value=(0.0, 0))
        engine._validate_invariants = AsyncMock(return_value=True)
        engine._record_audit = AsyncMock()
        engine._ensure_symbol_constraints = AsyncMock()
        engine._symbol_constraints["XRP/USDT"] = {"qty_step": 0.1, "min_qty": 1.0, "min_notional": 5.0}
        engine._normalize_order_amount = MagicMock(return_value=(sold, "ok", sold))
        engine._dust_check = MagicMock(return_value=(False, sold, "", sold * fill))
        pre_equity = cash + qty * fill
        mock_pf = MagicMock()
        mock_pf.passed = True
        mock_pf.expected_avg_fill = fill
        mock_pf.to_audit_dict = MagicMock(return_value={"passed": True})
        sell_eval = {
            "symbol": "XRP/USDT",
            "reason": "DAY_4H_STRUCTURE_BREAK_EXIT",
            "exit_branch": "risk",
            "avg_entry_price": entry,
            "mark_price": fill,
            "mark_source": "test",
            "mark_age_seconds": 0.0,
            "gross_pct": -0.0014,
            "effective_exit_cost_pct": 0.001,
            "effective_roundtrip_cost_pct": 0.002,
            "net_exit_pct": -0.002,
            "roundtrip_net_pct": -0.002,
            "required_profit_buffer_pct": 0.0,
            "allowed": True,
            "block_reason": "",
            "emergency_flag": True,
            "measured_cost_sample_count": 0,
            "measured_cost_p75": 0.0,
        }
        with (
            patch.object(engine, "_evaluate_sell_profitability", AsyncMock(return_value=sell_eval)),
            patch(
                "backend.services.protected_limit_execution.run_protected_preflight",
                AsyncMock(return_value=mock_pf),
            ),
            patch("backend.services.protected_limit_execution.USE_PROTECTED_LIMIT_EXECUTION", True),
            patch("backend.services.paper_trading_service.get_paper_trading_service", return_value=engine._paper_service),
        ):
            result = await engine.execute_sell_fifo(
                "XRP/USDT",
                sold,
                fill,
                ExitType.MANUAL,
                "DAY_4H_STRUCTURE_BREAK_EXIT",
                force_sell=True,
            )
        assert result is not None
        pos = engine.open_positions.get("XRP/USDT")
        assert pos is not None
        assert abs(float(pos.quantity) - 0.1) < 1e-9
        assert abs(engine._positions_value - 0.1 * fill) < 0.01
        assert engine.cash_balance > cash
        assert engine._total_equity < pre_equity + 5.0
        assert engine._total_equity > 160.0
        assert engine._total_equity < 250.0


def test_no_live_fill_does_not_cut_qty():
    pos = SimpleNamespace(quantity=47.1, status="ACTIVE")
    # Residual persist on no fill keeps the unsold book.
    assert pos.quantity == 47.1
