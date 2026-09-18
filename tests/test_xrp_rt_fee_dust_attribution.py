"""First natural XRP round-trip fee, dust, and net-P&L attribution."""

from __future__ import annotations

import sqlite3
from decimal import Decimal

from backend.services.live_account_basis import persist_operational_json
from backend.services.live_exchange_equity import (
    clear_protected_preexisting_dust,
    load_protected_preexisting_dust,
    stamp_protected_preexisting_dust,
)
from backend.services.live_fill_economics import (
    FIRST_XRP_RT_CORRECTION_ID,
    apply_first_xrp_rt_correction,
    apply_live_buy_economics,
    extract_live_commission,
    first_xrp_rt_venue_facts,
    merge_venue_trades_into_order,
    plan_sell_quantity,
    quantity_identity,
)
from backend.services.live_order_identity import extract_identity


def test_base_asset_buy_commission_reduces_net_credited_qty():
    order = {
        "filled": 28.4,
        "average": 1.3863,
        "info": {"fills": [{"commission": "0.00568000", "commissionAsset": "XRP"}]},
        "trades": [{"commission": "0.00568000", "commissionAsset": "XRP"}],
        "fee": {"cost": 0.00568, "currency": "XRP"},
    }
    comm = extract_live_commission(order, symbol="XRP/USDT", fill_price=1.3863)
    assert comm.fee_from_exchange is True
    assert comm.base_qty_reduction == 0.00568
    qty, fee, cash = apply_live_buy_economics(
        filled_qty=28.4,
        fill_price=1.3863,
        modeled_fee=0.023622552,
        commission=comm,
    )
    assert abs(qty - 28.39432) < 1e-12
    assert abs(fee - 0.00568 * 1.3863) < 1e-9
    assert abs(cash - 39.37092) < 1e-8


def test_estimated_cost_cannot_become_actual_commission():
    comm = extract_live_commission(
        {
            "info": {"fills": [{"commission": "0.00568000", "commissionAsset": "XRP"}]},
        },
        symbol="XRP/USDT",
        fill_price=1.3863,
    )
    ident = extract_identity(
        {"id": "488230379", "filled": 28.4, "average": 1.3863, "timestamp": 1789757821812},
        symbol="XRP/USDT",
        side="BUY",
        fee_amount=0.023622552,
        fee_items=list(comm.items),
        fee_from_exchange=True,
    )
    assert ident.fee_amount == 0.00568
    assert ident.fee_asset == "XRP"
    assert ident.fee_amount != 0.023622552


def test_commission_amount_keeps_correct_asset():
    facts = first_xrp_rt_venue_facts()
    assert facts["buy"]["commission_asset"] == "XRP"
    assert facts["buy"]["commission_amount"] == "0.00568"
    assert facts["sell"]["commission_asset"] == "USDT"
    assert facts["sell"]["commission_amount"] == "0.00792814"
    assert "0.023622552" not in facts["actual_exchange_commission"]["buy_amount"]


def test_preexisting_dust_is_protected_from_normal_sell():
    planned = plan_sell_quantity(
        net_active_qty="28.39432",
        exchange_free_qty="28.48812",
        protected_dust_qty="0.0938",
        qty_step="0.1",
    )
    assert planned.sellable == Decimal("28.3")
    assert planned.borrowed_from_dust == Decimal("0")
    assert planned.protected_dust == Decimal("0.0938")
    assert planned.residual == Decimal("0.09432")


def test_sell_qty_floored_from_net_active_not_gross():
    planned = plan_sell_quantity(
        net_active_qty=28.39432,
        exchange_free_qty=28.48812,
        protected_dust_qty=0.0938,
        qty_step=0.1,
    )
    assert planned.sellable < Decimal("28.4")
    assert planned.sellable == Decimal("28.3")


def test_active_trade_residual_becomes_dust_not_loss():
    planned = plan_sell_quantity(
        net_active_qty="28.39432",
        exchange_free_qty="28.48812",
        protected_dust_qty="0.0938",
        qty_step="0.1",
    )
    assert planned.residual > 0
    facts = first_xrp_rt_venue_facts()
    assert facts["remaining_new_residual"] == "0"
    assert float(facts["total_economic_change"]) > 0


def test_cash_plus_asset_changes_equal_economic_pnl():
    facts = first_xrp_rt_venue_facts()
    cash = Decimal(facts["cash_increase"])
    dust_value = Decimal(facts["consumed_dust_value_at_sell"])
    economic = Decimal(facts["total_economic_change"])
    assert cash - dust_value == economic
    assert abs(economic - Decimal("0.253943716")) < Decimal("0.000000001")
    ending = quantity_identity(
        starting_dust="0.0938",
        gross_bought="28.4",
        buy_base_commission="0.00568",
        sold="28.4",
    )
    assert ending == Decimal("0.08812")
    assert ending == Decimal(facts["quantity_identity"]["ending"])


def test_restart_and_reconciliation_do_not_duplicate_dust_or_corrections(tmp_path):
    db = str(tmp_path / "corr.db")
    persist_operational_json(db, "seed", {"ok": True})
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            CREATE TABLE paper_trades (
                trade_id TEXT, side TEXT, pnl REAL, pnl_usd_net REAL,
                fees_paid REAL, is_synthetic INTEGER DEFAULT 0, mode TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO paper_trades VALUES (?,?,?,?,?,?,?)",
            ("mystic_sell_XRP/USDT_1789758633629", "SELL", 0.226370751999994, 0.226370751999994, 0.02378442, 0, "live"),
        )
        conn.commit()
    first = apply_first_xrp_rt_correction(db)
    second = apply_first_xrp_rt_correction(db)
    assert first["applied"] is True
    assert second["applied"] is True
    with sqlite3.connect(db) as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM live_accounting_corrections WHERE correction_id=?",
            (FIRST_XRP_RT_CORRECTION_ID,),
        ).fetchone()[0]
        row = conn.execute(
            "SELECT pnl, pnl_usd_net, fees_paid FROM paper_trades WHERE trade_id=?",
            ("mystic_sell_XRP/USDT_1789758633629",),
        ).fetchone()
        trades = conn.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0]
    assert n == 1
    assert trades == 1
    assert abs(row[0] - 0.253943716) < 1e-9
    assert abs(row[1] - 0.253943716) < 1e-9
    assert abs(row[2] - 0.00792814) < 1e-9
    stamp_protected_preexisting_dust(db, "XRP/USDT", "0.0938")
    stamp_protected_preexisting_dust(db, "XRP/USDT", "0.0938")
    assert load_protected_preexisting_dust(db, "XRP/USDT") == Decimal("0.0938")
    clear_protected_preexisting_dust(db, "XRP/USDT")
    assert load_protected_preexisting_dust(db, "XRP/USDT") == Decimal("0")


def test_duplicate_fee_aliases_are_not_summed():
    order = {
        "fee": {"cost": 0.00568, "currency": "XRP"},
        "trades": [{"commission": "0.00568", "commissionAsset": "XRP", "fee": {"cost": 0.00568, "currency": "XRP"}}],
        "info": {
            "commission": "0.00568",
            "commissionAsset": "XRP",
            "fills": [{"commission": "0.00568", "commissionAsset": "XRP"}],
        },
    }
    comm = extract_live_commission(order, symbol="XRP/USDT", fill_price=1.3863)
    assert comm.base_qty_reduction == 0.00568


def test_merge_venue_trades_does_not_invent_estimated_fee():
    order = {"id": "488230379", "status": "closed", "filled": 28.4, "average": 1.3863, "info": {}}
    merged = merge_venue_trades_into_order(
        order,
        [
            {
                "trade_id": "2627001",
                "order_id": "488230379",
                "qty": 28.4,
                "price": 1.3863,
                "quote_qty": 39.37092,
                "commission": 0.00568,
                "commission_asset": "XRP",
                "taker_or_maker": "taker",
                "timestamp": 1789757821812,
            }
        ],
    )
    comm = extract_live_commission(merged, symbol="XRP/USDT", fill_price=1.3863)
    assert comm.base_qty_reduction == 0.00568
    assert comm.fee_from_exchange is True
    modeled = extract_live_commission({"filled": 28.4}, symbol="XRP/USDT", fill_price=1.3863)
    assert modeled.fee_from_exchange is False
    assert modeled.usd == 0.0
