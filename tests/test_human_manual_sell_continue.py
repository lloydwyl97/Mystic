"""Human exchange flatten must book a sell and not pause entries."""

from backend.services.portfolio_engine import vanished_lot_is_human_sell


def test_vanished_active_lot_is_human_sell():
    assert vanished_lot_is_human_sell(status="ACTIVE", quantity=0.0248, entry_price=2516.15, fill_found=False) is True


def test_vanished_dust_qty_zero_is_human_sell():
    assert vanished_lot_is_human_sell(status="DUST_PENDING", quantity=0.0, entry_price=2516.15, fill_found=False) is True


def test_recovered_fill_is_always_human_sell():
    assert vanished_lot_is_human_sell(status="DUST_PENDING", quantity=0.0001, entry_price=2516.15, fill_found=True) is True


def test_true_dust_leftover_without_fill_stays_dust():
    assert vanished_lot_is_human_sell(status="DUST_PENDING", quantity=0.0001, entry_price=2516.15, fill_found=False) is False
