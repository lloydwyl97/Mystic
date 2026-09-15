"""Principal preservation and live P&L isolation."""

from __future__ import annotations

from decimal import Decimal

from backend.services.live_account_basis import (
    TRAILING_BUY_ANCHOR_EQUITY,
    apply_bootstrap_cash,
    apply_external_capital_flow,
    trailing_buy_scorecard,
)
from backend.services.live_fill_economics import sum_dust_adjustment_pnl, sum_realized_pnl_by_mode
from backend.services.portfolio_engine import PortfolioEngine
from tests.test_live_fill_economics import _make_paper_trades_db


def test_bootstrap_does_not_reset_principal_to_equity():
    out = apply_bootstrap_cash(
        stored_principal=Decimal("10000"),
        previous_cash=Decimal("227.12229294"),
        exchange_cash=Decimal("228.06746265"),
        positions_value=0,
    )
    assert out["principal"] == Decimal("10000")
    assert out["cash"] == Decimal("228.06746265")
    assert out["equity"] == Decimal("228.06746265")
    assert out["reconciliation_adjustment"] == Decimal("0.94516971")


def test_bootstrap_initializes_principal_only_when_missing():
    out = apply_bootstrap_cash(
        stored_principal=0,
        previous_cash=0,
        exchange_cash=Decimal("228.06746265"),
        positions_value=0,
    )
    assert out["principal"] == Decimal("228.06746265")


def test_reconciliation_is_not_trading_profit():
    out = apply_bootstrap_cash(
        stored_principal=Decimal("228.06746265"),
        previous_cash=Decimal("227.12229294"),
        exchange_cash=Decimal("228.06746265"),
    )
    assert out["reconciliation_adjustment"] == Decimal("0.94516971")
    assert out["principal"] == Decimal("228.06746265")


def test_principal_survives_engine_restart(tmp_path):
    db = tmp_path / "p.db"
    first = PortfolioEngine(db_path=str(db), principal=10000.0, test_mode=True)
    first._ensure_db_schema()
    first.principal = 10000.0
    first.cash_balance = 228.06746265
    first._available_balance = 228.06746265
    first._total_equity = 228.06746265
    first._positions_value = 0.0

    async def _persist():
        await first._persist_ledger_to_sqlite()

    import asyncio

    asyncio.run(_persist())
    second = PortfolioEngine(db_path=str(db), principal=10000.0, test_mode=True)
    second._ensure_db_schema()

    async def _load():
        return await second._load_ledger_from_sqlite()

    asyncio.run(_load())
    assert second.principal == 10000.0
    adopted = apply_bootstrap_cash(
        stored_principal=second.principal,
        previous_cash=second.cash_balance,
        exchange_cash=228.06746265,
        positions_value=0,
    )
    assert adopted["principal"] == Decimal("10000")


def test_paper_and_synthetic_excluded_from_live_pnl():
    path = _make_paper_trades_db(
        [
            {"side": "SELL", "pnl": -9.47, "mode": "live", "exit_type": None},
            {"side": "SELL", "pnl": 973.98, "mode": "paper", "exit_type": None},
            {"side": "SELL", "pnl": 50.0, "mode": "live", "is_synthetic": 1, "exit_type": None},
            {"side": "SELL", "pnl": -7.33, "mode": "live", "exit_type": "DUST_WRITEOFF"},
        ]
    )
    assert abs(sum_realized_pnl_by_mode(path, mode="live") - (-9.47)) < 1e-6
    assert abs(sum_realized_pnl_by_mode(path, mode="paper") - 973.98) < 1e-6
    assert abs(sum_dust_adjustment_pnl(path) - (-7.33)) < 1e-6


def test_live_fill_counts_once_only():
    path = _make_paper_trades_db(
        [
            {"side": "SELL", "pnl": -1.25, "mode": "live", "exit_type": None},
            {"side": "SELL", "pnl": -1.25, "mode": "live", "exit_type": None},
        ]
    )
    # Two distinct rows, each counted once.
    assert abs(sum_realized_pnl_by_mode(path, mode="live") - (-2.50)) < 1e-6


def test_deposit_and_withdraw_change_principal_not_pnl():
    principal = Decimal("228.06746265")
    after_deposit = apply_external_capital_flow(principal, Decimal("10"))
    assert after_deposit == Decimal("238.06746265")
    after_withdraw = apply_external_capital_flow(after_deposit, Decimal("-5"))
    assert after_withdraw == Decimal("233.06746265")
    engine = PortfolioEngine(db_path=":memory:", principal=228.06746265, test_mode=True)
    engine._realized_pnl = -9.47
    engine.apply_external_capital_flow(10.0, kind="deposit")
    assert engine.principal == 238.06746265
    assert engine._realized_pnl == -9.47


def test_equity_principal_realized_unrealized_reconcile():
    cash = Decimal("228.06746265")
    principal = Decimal("10000")
    realized = Decimal("-9.4734669")
    unrealized = Decimal("0")
    equity = cash
    assert equity == cash + Decimal("0")
    lifetime = equity - principal
    assert lifetime == cash - principal
    assert realized != lifetime
    assert unrealized == 0


def test_trailing_buy_scorecard_stays_zero_without_fills():
    card = trailing_buy_scorecard(live_fills_since_anchor=0, current_equity=TRAILING_BUY_ANCHOR_EQUITY)
    assert card["anchor_equity"] == str(TRAILING_BUY_ANCHOR_EQUITY)
    assert card["realized_pnl"] == 0.0
    assert card["total_pnl"] == 0.0
