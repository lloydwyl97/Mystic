"""Live Binance commission accounting — quote, base, mixed; no double-count."""

from backend.services.live_fill_economics import (
    apply_live_buy_economics,
    apply_live_sell_economics,
    extract_live_commission,
    live_round_trip_net,
)


def test_quote_commission_usdt():
    order = {
        "info": {"fills": [{"commission": "0.01307088", "commissionAsset": "USDT"}]},
    }
    comm = extract_live_commission(order, symbol="BTCUSDT", fill_price=77802.87)
    assert comm.fee_from_exchange is True
    assert comm.base_qty_reduction == 0.0
    assert abs(comm.usd - 0.01307088) < 1e-9
    assert abs(comm.quote_commission_usd - 0.01307088) < 1e-9


def test_base_commission_btc():
    order = {
        "trades": [{"commission": "0.00000017", "commissionAsset": "BTC"}],
    }
    px = 77137.14
    comm = extract_live_commission(order, symbol="BTC/USDT", fill_price=px)
    assert comm.fee_from_exchange is True
    assert abs(comm.base_qty_reduction - 0.00000017) < 1e-12
    assert abs(comm.usd - 0.00000017 * px) < 1e-9
    assert comm.quote_commission_usd == 0.0


def test_mixed_commission_assets():
    order = {
        "info": {
            "fills": [
                {"commission": "0.00000017", "commissionAsset": "BTC"},
                {"commission": "0.01307088", "commissionAsset": "USDT"},
            ]
        }
    }
    px = 77137.14
    comm = extract_live_commission(order, symbol="BTCUSDT", fill_price=px)
    assert comm.fee_from_exchange is True
    assert abs(comm.base_qty_reduction - 0.00000017) < 1e-12
    assert abs(comm.usd - (0.00000017 * px + 0.01307088)) < 1e-8


def test_buy_base_fee_reduces_inventory_not_double_cash():
    px = 77137.14
    filled = 0.00084
    comm = extract_live_commission(
        {"fee": {"cost": 0.00000017, "currency": "BTC"}},
        symbol="BTCUSDT",
        fill_price=px,
    )
    modeled = filled * px * 0.001
    qty, fee, cash = apply_live_buy_economics(
        filled_qty=filled,
        fill_price=px,
        modeled_fee=modeled,
        commission=comm,
    )
    assert abs(qty - (filled - 0.00000017)) < 1e-12
    assert abs(fee - 0.00000017 * px) < 1e-9
    assert abs(cash - filled * px) < 1e-8
    assert fee < modeled


def test_gross_vs_net_no_double_count():
    qty = 0.00084
    entry = 77137.14
    exit_px = 77802.87
    buy_fee = 0.00000017 * entry
    sell_fee = 0.01307088
    out = live_round_trip_net(
        quantity=qty,
        entry_price=entry,
        exit_price=exit_px,
        entry_commission_usd=buy_fee,
        exit_commission_usd=sell_fee,
    )
    gross = qty * (exit_px - entry)
    assert abs(out["gross_price_pnl"] - gross) < 1e-8
    assert abs(out["net_realized_pnl"] - (gross - buy_fee - sell_fee)) < 1e-8
    # Fill prices already include bid/ask; do not subtract extra slippage.
    assert out["exchange_commission_usd"] == round(buy_fee + sell_fee, 8)


def test_sell_uses_exchange_fee_not_model():
    comm = extract_live_commission(
        {"fee": {"cost": 0.01307088, "currency": "USDT"}},
        symbol="BTCUSDT",
        fill_price=77802.87,
    )
    modeled = 0.00084 * 77802.87 * 0.001
    fee, proceeds = apply_live_sell_economics(
        quantity=0.00084,
        fill_price=77802.87,
        modeled_fee=modeled,
        commission=comm,
    )
    assert abs(fee - 0.01307088) < 1e-9
    assert abs(proceeds - (0.00084 * 77802.87 - 0.01307088)) < 1e-8


def test_paper_model_used_when_no_exchange_fee():
    comm = extract_live_commission({}, symbol="BTCUSDT", fill_price=100.0)
    assert comm.fee_from_exchange is False
    qty, fee, cash = apply_live_buy_economics(
        filled_qty=1.0,
        fill_price=100.0,
        modeled_fee=0.1,
        commission=comm,
    )
    assert qty == 1.0
    assert fee == 0.1
    assert cash == 100.1
