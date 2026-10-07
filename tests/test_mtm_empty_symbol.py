"""An empty open-lot symbol must not abort mark-to-market for the rest of the book."""

from backend.services.portfolio_engine import position_mtm_symbol


class _Lot:
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol


def test_empty_lot_symbol_uses_the_position_key():
    assert position_mtm_symbol(_Lot(""), "DAY_V2::XRP/USDT") == "XRP/USDT"


def test_empty_lot_and_empty_key_are_skipped():
    assert position_mtm_symbol(_Lot(""), "") == ""
    assert position_mtm_symbol(_Lot("   "), "not-a-symbol") == ""


def test_lot_symbol_still_prices_when_present():
    assert position_mtm_symbol(_Lot("xrpusdt"), "") == "XRP/USDT"
