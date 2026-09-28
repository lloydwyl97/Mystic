"""A SQLite reload must never detach the lot a sell is closing.

Ocean 2026-09-27 22:49 and 2026-09-28 02:13: an exit-monitor / MTM reload ran
while a SCALP XRP sell was in flight. The reload swapped open_positions, the
post-fill qty cut landed on the orphaned object, the fresh object kept the
pre-sell 22.6 XRP and the next exit sold 22.5 XRP of protected inventory.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock

import pytest

import backend.services.portfolio_engine as pe
from backend.services.portfolio_engine import (
    ExitType,
    OpenPosition,
    PortfolioEngine,
    make_position_key,
    retained_protected_after_sell,
)

KEY = make_position_key("SCALP_V2", "XRP/USDT")


def _engine(tmp_path) -> PortfolioEngine:
    eng = PortfolioEngine(db_path=str(tmp_path / "race.db"), principal=300.0, test_mode=True)
    eng._ensure_db_schema()
    return eng


def _xrp_lot(qty: float = 22.59548) -> OpenPosition:
    return OpenPosition(
        symbol="XRP/USDT",
        quantity=qty,
        entry_price=1.5076,
        entry_time=time.time(),
        trade_id="scalp_v2_XRPUSDT_1790560281058",
        stop_price=1.49,
        take_profit_1_price=1.52,
        take_profit_2_price=1.53,
        engine_id="SCALP_V2",
        original_position_cost=qty * 1.5076,
    )


async def _seed(eng: PortfolioEngine) -> OpenPosition:
    lot = _xrp_lot()
    await eng._persist_position_to_sqlite(lot)
    eng.open_positions = {KEY: lot}
    return lot


async def test_reload_while_sell_in_flight_keeps_live_lot(tmp_path):
    eng = _engine(tmp_path)
    lot = await _seed(eng)
    eng._sells_inflight = 1
    await eng._load_positions_from_sqlite(allow_mutations=False)
    assert eng.open_positions[KEY] is lot


async def test_reload_overlapping_a_finished_sell_is_discarded(tmp_path, monkeypatch):
    eng = _engine(tmp_path)
    lot = await _seed(eng)
    real_ro = pe.connect_ro

    def _ro_with_sell_finishing(*a, **k):
        eng._sell_seq = int(getattr(eng, "_sell_seq", 0) or 0) + 1
        return real_ro(*a, **k)

    monkeypatch.setattr(pe, "connect_ro", _ro_with_sell_finishing)
    await eng._load_positions_from_sqlite(allow_mutations=False)
    assert eng.open_positions[KEY] is lot


async def test_sell_may_rehydrate_itself_and_idle_reload_rebuilds(tmp_path):
    eng = _engine(tmp_path)
    lot = await _seed(eng)
    eng._sells_inflight = 1
    await eng._load_positions_from_sqlite(allow_mutations=False, during_sell=True)
    assert eng.open_positions[KEY] is not lot
    eng._sells_inflight = 0
    before = eng.open_positions[KEY]
    await eng._load_positions_from_sqlite(allow_mutations=False)
    assert eng.open_positions[KEY] is not before
    assert eng.open_positions[KEY].quantity == pytest.approx(22.59548)


async def test_execute_sell_fifo_marks_in_flight_and_bumps_generation(tmp_path):
    eng = _engine(tmp_path)
    seen: list[int] = []

    async def _locked(*a, **k):
        seen.append(eng._sells_inflight)
        raise RuntimeError("venue down")

    eng._execute_sell_fifo_locked = AsyncMock(side_effect=_locked)
    with pytest.raises(RuntimeError):
        await eng.execute_sell_fifo("XRP/USDT", 22.5, 1.515, ExitType.MANUAL, "SCALP_V2_NET_PROFIT", engine_id="SCALP_V2")
    assert seen == [1]
    assert eng._sells_inflight == 0
    assert eng._sell_seq == 1


def test_protected_stamp_survives_a_sell_of_the_strategy_lot():
    keep = retained_protected_after_sell(protected_before=27.05558, free_before=49.65106, sold_qty=22.5, lot_residual_qty=0.09548)
    assert keep == pytest.approx(27.05558)


def test_protected_stamp_shrinks_to_what_the_venue_still_holds():
    keep = retained_protected_after_sell(protected_before=27.05558, free_before=30.0, sold_qty=22.5, lot_residual_qty=0.09548)
    assert keep == pytest.approx(30.0 - 22.5 - 0.09548)
    assert retained_protected_after_sell(protected_before=5.0, free_before=22.6, sold_qty=22.5, lot_residual_qty=0.1) == 0.0
    assert retained_protected_after_sell(protected_before=5.0, free_before=None, sold_qty=1.0, lot_residual_qty=0.0) == 0.0
