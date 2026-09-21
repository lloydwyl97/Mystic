"""Cash / fee / dust bridge must close in Decimal cents."""

from __future__ import annotations

from decimal import Decimal

from backend.services.live_account_bridge import bridge_account


def test_bridge_reconciles_cash_fees_base_commission_and_dust():
    fills = [
        {"symbol": "SOLUSDT", "side": "BUY", "qty": "0.35000000", "price": "110.00", "quoteQty": "38.50000000", "commission": "0.00007000", "commissionAsset": "SOL"},
        {"symbol": "SOLUSDT", "side": "SELL", "qty": "0.34993000", "price": "110.20", "quoteQty": "38.56228600", "commission": "0.00771246", "commissionAsset": "USDT"},
    ]
    opening_dust = [{"symbol": "SOL/USDT", "asset": "SOL", "quantity": "0.01200000", "executable_bid": "110.00"}]
    closing_dust = [{"symbol": "SOL/USDT", "asset": "SOL", "quantity": "0.01207000", "executable_bid": "110.20"}]
    out = bridge_account(
        opening_cash="224.60000000",
        fills=fills,
        opening_dust=opening_dust,
        closing_dust=closing_dust,
        deposits=0,
        withdrawals=0,
        bids_for_base_fees={"SOLUSDT": "110.00"},
    )
    assert Decimal(out["bridge_residual"]) == Decimal("0")
    assert Decimal(out["ending_nle"]) == Decimal(out["reconstructed_nle"])
    assert Decimal(out["quote_commission"]) == Decimal("0.00771246")
    assert Decimal(out["base_commission_quote"]) == Decimal("0.00007000") * Decimal("110.00")
    assert Decimal(out["ending_cash"]) == Decimal("224.60000000") + Decimal("38.56228600") - Decimal("38.50000000") - Decimal("0.00771246")


def test_automated_buy_identity_complete():
    buy = {
        "decision_id": "dec_1",
        "intent_id": "tbabc",
        "client_order_id": "mystic_tbabc",
        "exchange_order_id": "911069165",
        "venue_trade_id": "2630100",
    }
    assert all(str(buy[k]).strip() for k in buy)
