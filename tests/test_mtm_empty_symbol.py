"""An empty open-lot symbol must not abort mark-to-market for the rest of the book."""

import asyncio
import sqlite3

from backend.services import adaptive_learning as al
from backend.services.portfolio_engine import PortfolioEngine, position_mtm_symbol


class _Lot:
    def __init__(self, symbol: str, quantity: float = 0.0, entry_price: float = 0.0) -> None:
        self.symbol = symbol
        self.quantity = quantity
        self.entry_price = entry_price


def test_empty_lot_symbol_uses_the_position_key():
    assert position_mtm_symbol(_Lot(""), "DAY_V2::XRP/USDT") == "XRP/USDT"


def test_empty_lot_and_empty_key_are_skipped():
    assert position_mtm_symbol(_Lot(""), "") == ""
    assert position_mtm_symbol(_Lot("   "), "not-a-symbol") == ""


def test_lot_symbol_still_prices_when_present():
    assert position_mtm_symbol(_Lot("xrpusdt"), "") == "XRP/USDT"


class _Cache:
    def invalidate(self, _name: str) -> None:
        return None


def _book() -> PortfolioEngine:
    engine = PortfolioEngine.__new__(PortfolioEngine)
    engine.open_positions = {
        "DAY_V2::ETH/USDT": _Lot("ETH/USDT", 0.1, 2000.0),
        "DAY_V2::BTC/USDT": _Lot("", 0.001, 70000.0),
        "dust": _Lot("", 5.0, 1.0),
    }
    engine.cash_balance = 731.77
    engine._buy_seq = 0
    engine.test_mode = True
    engine._portfolio_cache = _Cache()
    engine.db_path = ""
    return engine


def test_one_empty_lot_does_not_stop_valuation_or_learning(tmp_path):
    learning = str(tmp_path / "learn.db")
    version = al.current_strategy_version("DAY_V2")
    assert al.observe(
        learning,
        engine="DAY_V2",
        symbol="ETHUSDT",
        setup="EXHAUSTION_MR__MARKET",
        regime="btcdown_vollo",
        metric="policy_calibration",
        value=-0.002,
        strategy_version=version,
        now=1_700_000_000.0,
    )
    before = tmp_path.joinpath("learn.db").read_bytes()
    engine = _book()
    engine.db_path = str(tmp_path / "ledger.db")

    async def _marks(symbol: str) -> float:
        return {"ETH/USDT": 2100.0, "BTC/USDT": 80000.0}[symbol]

    engine._fetch_live_mark_for_open_position = _marks
    prices = asyncio.run(PortfolioEngine._fetch_mtm_prices_for_open_positions(engine))
    assert prices["ETH/USDT"] == 2100.0
    assert prices["BTC/USDT"] == 80000.0
    assert "dust" not in prices
    asyncio.run(PortfolioEngine._recompute_positions_values(engine, prices, allow_network_mtm=False))
    assert engine.cash_balance == 731.77
    assert abs(engine._positions_value - 290.0) < 1e-9
    assert abs(engine._total_equity - (731.77 + 290.0)) < 1e-9
    with sqlite3.connect(learning) as conn:
        ewma = conn.execute("SELECT ewma FROM adaptive_metric_state WHERE metric='policy_calibration'").fetchone()[0]
    assert abs(ewma - (-0.002)) < 1e-12
    assert tmp_path.joinpath("learn.db").read_bytes() == before
