"""Regression: execute_sell_fifo must credit sell proceeds to cash."""

from __future__ import annotations

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


def _seed_btc_position(db_path: Path, *, trade_id: str, qty: float, entry: float, cash_after_buy: float) -> None:
    now = time.time()
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            """
            INSERT INTO portfolio_engine_positions (
                symbol, quantity, entry_price, entry_time, trade_id,
                stop_price, take_profit_1_price, take_profit_2_price,
                highest_price, atr_at_entry, entry_bar_timestamp,
                confidence_at_entry, last_updated, sleeve
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), ?)
            """,
            (
                "BTC/USDT",
                qty,
                entry,
                now,
                trade_id,
                entry * 0.97,
                entry * 1.03,
                0.0,
                entry,
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
            ) VALUES (?, 'test-run', 'paper', 'BTC/USDT', 'BUY', ?, ?, ?, datetime('now'), 'executed', 'day')
            """,
            (trade_id, qty, entry, qty),
        )
        conn.execute(
            """
            UPDATE portfolio_engine_ledger SET cash_balance = ?, total_equity = ?, positions_value = ?
            WHERE id = 1
            """,
            (cash_after_buy, cash_after_buy + qty * entry, qty * entry),
        )
        conn.commit()


def _allowed_sell_eval(**overrides):
    base = {
        "symbol": "BTC/USDT",
        "reason": "NET_PROFIT_EXIT",
        "exit_branch": "profit",
        "avg_entry_price": 59542.393580501746,
        "mark_price": 59828.3,
        "mark_source": "test",
        "mark_age_seconds": 0.0,
        "gross_pct": 0.005,
        "effective_exit_cost_pct": 0.001,
        "effective_roundtrip_cost_pct": 0.002,
        "net_exit_pct": 0.004,
        "roundtrip_net_pct": 0.004,
        "required_profit_buffer_pct": 0.001,
        "allowed": True,
        "block_reason": "",
        "emergency_flag": False,
        "measured_cost_sample_count": 0,
        "measured_cost_p75": 0.0,
    }
    base.update(overrides)
    return base


@pytest.mark.asyncio
async def test_apply_sell_cash_credit_increases_cash():
    engine = PortfolioEngine(principal=25_000.0, test_mode=True)
    engine.cash_balance = 21_192.26
    engine._realized_pnl = -50.0
    delta = engine._apply_sell_cash_credit(3767.9863, 18.0064)
    assert abs(delta - 3767.9863) < 0.01
    assert abs(engine.cash_balance - (21_192.26 + 3767.9863)) < 0.01
    assert abs(engine._realized_pnl - (-50.0 + 18.0064)) < 0.01


@pytest.mark.asyncio
async def test_execute_sell_fifo_credits_cash_and_closes_position():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        principal = 25_000.0
        buy_cost = 3749.9799477
        cash_after_buy = 24_942.2364721
        qty = 0.06298
        entry = 59542.393580501746
        sell_price = 59828.3
        trade_id = "mystic_BTC/USDT_test_buy"

        _init_test_db(db_path, cash=principal)
        _seed_btc_position(db_path, trade_id=trade_id, qty=qty, entry=entry, cash_after_buy=cash_after_buy)

        engine = PortfolioEngine(db_path=str(db_path), principal=principal, test_mode=True)
        await engine.initialize_from_db()
        engine.open_positions["BTC/USDT"] = OpenPosition(
            symbol="BTC/USDT",
            quantity=qty,
            entry_price=entry,
            entry_time=time.time() - 700,
            trade_id=trade_id,
            stop_price=58000.0,
            take_profit_1_price=61000.0,
            take_profit_2_price=0.0,
            highest_price=sell_price,
            lowest_price=entry,
            sleeve=Sleeve.ACTIVE.value,
            entry_strategy_id="day",
        )
        engine._positions_value = qty * sell_price
        engine._cost_basis = buy_cost
        engine._total_equity = engine.cash_balance + engine._positions_value
        engine._positions_initialized = True
        engine._live_execution_enabled = False
        engine._exit_in_progress = set()
        engine._paper_service = MagicMock(paper_run_id="test-run")
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
        engine._resolve_learning_close_reason = MagicMock(return_value="NET_PROFIT_EXIT")
        engine._get_loss_hold_until = AsyncMock(return_value=None)
        engine._set_loss_hold_until = AsyncMock()
        engine.get_rolling_24h_risk_metrics = AsyncMock(return_value=(0.0, 0))
        engine._validate_invariants = AsyncMock(return_value=True)
        engine._record_audit = AsyncMock()
        engine._entry_ensure_constraints = AsyncMock()
        engine._ensure_symbol_constraints = AsyncMock(
            return_value=None,
        )
        engine._symbol_constraints["BTC/USDT"] = {
            "qty_step": 0.00001,
            "min_qty": 0.00001,
            "min_notional": 10.0,
        }
        engine._normalize_order_amount = MagicMock(return_value=(qty, "ok", qty))
        engine._dust_check = MagicMock(return_value=(False, qty, "", qty * sell_price))
        engine._floor_to_step = lambda q, _s: q

        pre_cash = float(engine.cash_balance)
        sell_eval = _allowed_sell_eval()

        mock_pf = MagicMock()
        mock_pf.passed = True
        mock_pf.expected_avg_fill = sell_price
        mock_pf.to_audit_dict = MagicMock(return_value={"passed": True})

        with patch.object(engine, "_evaluate_sell_profitability", AsyncMock(return_value=sell_eval)):
            with patch(
                "backend.services.protected_limit_execution.run_protected_preflight",
                AsyncMock(return_value=mock_pf),
            ):
                with patch(
                    "backend.services.protected_limit_execution.USE_PROTECTED_LIMIT_EXECUTION",
                    True,
                ):
                    with patch(
                        "backend.services.paper_trading_service.get_paper_trading_service",
                        return_value=engine._paper_service,
                    ):
                        result = await engine.execute_sell_fifo(
                            "BTC/USDT",
                            qty,
                            sell_price,
                            ExitType.TAKE_PROFIT_1,
                            "NET_PROFIT_EXIT",
                            force_sell=True,
                        )

        assert result is not None
        assert "BTC/USDT" not in engine.open_positions

        with sqlite3.connect(str(db_path)) as conn:
            open_n = conn.execute("SELECT COUNT(*) FROM portfolio_engine_positions").fetchone()[0]
            sell_row = conn.execute("SELECT pnl, price, quantity FROM paper_trades WHERE side='SELL' ORDER BY id DESC LIMIT 1").fetchone()
            ledger = conn.execute("SELECT cash_balance, realized_pnl, total_equity, positions_value FROM portfolio_engine_ledger WHERE id=1").fetchone()

        assert open_n == 0
        assert sell_row is not None
        paper_pnl = float(sell_row[0])
        proceeds = float(sell_row[1]) * float(sell_row[2])
        cash_delta = float(engine.cash_balance) - pre_cash

        assert abs(cash_delta - proceeds) < 0.05, f"cash_delta={cash_delta} proceeds={proceeds}"
        assert float(engine._positions_value) < 1.0
        assert abs(float(engine._realized_pnl) - paper_pnl) < 0.05
        assert abs(float(ledger[0]) - float(engine.cash_balance)) < 0.01
        assert float(ledger[3]) < 1.0
        assert engine._record_audit.await_count == 1
        audit_kwargs = engine._record_audit.await_args.kwargs
        post = audit_kwargs["post_ledger"]
        pre = audit_kwargs["pre_ledger"]
        assert float(post["cash_balance"]) - float(pre["cash_balance"]) > 0.0


@pytest.mark.asyncio
async def test_trade_3059_style_sell_post_cash_increases_not_flat():
    """Exact failure shape: post-cash must rise by proceeds, not stay equal to pre-cash."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "3059.db"
        qty = 0.06298
        entry = 59542.393580501746
        sell_price = 59828.3
        pre_cash = 21_192.256524399992
        trade_id = "mystic_BTC/USDT_1782403087662"

        _init_test_db(db_path, cash=25_000.0)
        _seed_btc_position(db_path, trade_id=trade_id, qty=qty, entry=entry, cash_after_buy=pre_cash)

        engine = PortfolioEngine(db_path=str(db_path), principal=25_000.0, test_mode=True)
        await engine.initialize_from_db()
        engine.cash_balance = pre_cash
        engine._total_equity = pre_cash + qty * sell_price
        engine._positions_value = qty * sell_price
        engine.open_positions["BTC/USDT"] = OpenPosition(
            symbol="BTC/USDT",
            quantity=qty,
            entry_price=entry,
            entry_time=time.time() - 702,
            trade_id=trade_id,
            stop_price=58022.33,
            take_profit_1_price=61318.86,
            take_profit_2_price=0.0,
            highest_price=59836.47,
            lowest_price=entry,
            sleeve=Sleeve.ACTIVE.value,
            entry_strategy_id="day",
        )
        engine._live_execution_enabled = False
        engine._exit_in_progress = set()
        engine._paper_service = MagicMock(paper_run_id="9b8bd253-af71-4a0c-83ad-558dc9f947a2")
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
        engine._resolve_learning_close_reason = MagicMock(return_value="NET_PROFIT_EXIT")
        engine._get_loss_hold_until = AsyncMock(return_value=None)
        engine._set_loss_hold_until = AsyncMock()
        engine.get_rolling_24h_risk_metrics = AsyncMock(return_value=(0.0, 0))
        engine._validate_invariants = AsyncMock(return_value=True)
        engine._entry_ensure_constraints = AsyncMock()
        engine._ensure_symbol_constraints = AsyncMock()
        engine._symbol_constraints["BTC/USDT"] = {"qty_step": 0.00001, "min_qty": 0.00001, "min_notional": 10.0}
        engine._normalize_order_amount = MagicMock(return_value=(qty, "ok", qty))
        engine._dust_check = MagicMock(return_value=(False, qty, "", qty * sell_price))
        engine._floor_to_step = lambda q, _s: q

        mock_pf = MagicMock()
        mock_pf.passed = True
        mock_pf.expected_avg_fill = sell_price
        mock_pf.to_audit_dict = MagicMock(return_value={})

        with patch.object(engine, "_evaluate_sell_profitability", AsyncMock(return_value=_allowed_sell_eval())):
            with patch("backend.services.protected_limit_execution.run_protected_preflight", AsyncMock(return_value=mock_pf)):
                with patch("backend.services.protected_limit_execution.USE_PROTECTED_LIMIT_EXECUTION", True):
                    with patch(
                        "backend.services.paper_trading_service.get_paper_trading_service",
                        return_value=engine._paper_service,
                    ):
                        await engine.execute_sell_fifo(
                            "BTC/USDT",
                            qty,
                            sell_price,
                            ExitType.TAKE_PROFIT_1,
                            "NET_PROFIT_EXIT",
                            force_sell=True,
                        )

        expected_proceeds = qty * sell_price * (1 - 0.001)  # maker fee approx
        assert float(engine.cash_balance) > pre_cash + 3700.0
        assert abs(float(engine.cash_balance) - (pre_cash + expected_proceeds)) < 5.0
        assert float(engine.cash_balance) != pre_cash


@pytest.mark.asyncio
async def test_duplicate_sell_blocked_after_close():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "dup.db"
        qty = 0.1
        entry = 100.0
        trade_id = "mystic_BTC/USDT_dup"
        _init_test_db(db_path, cash=9_000.0, principal=9_000.0)
        _seed_btc_position(db_path, trade_id=trade_id, qty=qty, entry=entry, cash_after_buy=9_000.0)

        engine = PortfolioEngine(db_path=str(db_path), principal=9_000.0, test_mode=True)
        await engine.initialize_from_db()
        engine.open_positions["BTC/USDT"] = OpenPosition(
            symbol="BTC/USDT",
            quantity=qty,
            entry_price=entry,
            entry_time=time.time(),
            trade_id=trade_id,
            stop_price=95.0,
            take_profit_1_price=110.0,
            take_profit_2_price=0.0,
            sleeve=Sleeve.ACTIVE.value,
        )
        engine._live_execution_enabled = False
        engine._exit_in_progress = set()
        engine._paper_service = MagicMock(paper_run_id="test-run")
        engine._check_kill_switch_sell = MagicMock(return_value=(True, ""))
        engine._record_reject = AsyncMock()
        engine._evaluate_sell_profitability = AsyncMock(return_value=_allowed_sell_eval(mark_price=105.0))

        mock_pf = MagicMock(passed=True, expected_avg_fill=105.0)
        mock_pf.to_audit_dict = MagicMock(return_value={})

        patches = [
            patch("backend.services.protected_limit_execution.run_protected_preflight", AsyncMock(return_value=mock_pf)),
            patch("backend.services.protected_limit_execution.USE_PROTECTED_LIMIT_EXECUTION", True),
            patch("backend.services.paper_trading_service.get_paper_trading_service", return_value=engine._paper_service),
        ]
        for p in patches:
            p.start()
        try:
            engine._delete_position_from_sqlite = AsyncMock()
            engine._clear_all_quarantines = AsyncMock()
            engine._update_coin_performance = AsyncMock()
            engine._validate_invariants = AsyncMock(return_value=True)
            engine._record_audit = AsyncMock()
            engine._entry_ensure_constraints = AsyncMock()
            engine._ensure_symbol_constraints = AsyncMock()
            engine._symbol_constraints["BTC/USDT"] = {"qty_step": 0.00001, "min_qty": 0.00001, "min_notional": 10.0}
            engine._normalize_order_amount = MagicMock(return_value=(qty, "ok", qty))
            engine._dust_check = MagicMock(return_value=(False, qty, "", qty * 105.0))
            engine._floor_to_step = lambda q, _s: q
            engine._record_position_close_ledger = AsyncMock()
            engine._persist_quality_cooldowns = AsyncMock()
            engine.record_sell_cooldown = MagicMock()
            engine._resolve_learning_close_reason = MagicMock(return_value="NET_PROFIT_EXIT")
            engine._get_loss_hold_until = AsyncMock(return_value=None)
            engine._set_loss_hold_until = AsyncMock()
            engine.get_rolling_24h_risk_metrics = AsyncMock(return_value=(0.0, 0))

            first = await engine.execute_sell_fifo("BTC/USDT", qty, 105.0, ExitType.TAKE_PROFIT_1, "NET_PROFIT_EXIT", force_sell=True)
            second = await engine.execute_sell_fifo("BTC/USDT", qty, 105.0, ExitType.TAKE_PROFIT_1, "NET_PROFIT_EXIT", force_sell=True)
        finally:
            for p in patches:
                p.stop()

        assert first is not None
        assert second is None
        with sqlite3.connect(str(db_path)) as conn:
            sell_count = conn.execute("SELECT COUNT(*) FROM paper_trades WHERE side='SELL'").fetchone()[0]
        assert sell_count == 1
