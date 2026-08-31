"""DAY DUST_PENDING must not block same-symbol entry or consume slots.

ACTIVE / EXIT_RESIDUAL_PENDING remain held. Repair does not flatten dust.
"""

from __future__ import annotations

import asyncio
import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.services.day_mandatory_exit_execution import STATUS_EXIT_RESIDUAL_PENDING
from backend.services.portfolio_engine import OpenPosition, PortfolioEngine


def _pos(symbol: str, *, qty: float, status: str, price: float = 100.0) -> OpenPosition:
    return OpenPosition(
        symbol=symbol,
        quantity=qty,
        entry_price=price,
        entry_time=time.time(),
        trade_id=f"t-{symbol}-{status}",
        stop_price=price * 0.97,
        take_profit_1_price=price * 1.03,
        take_profit_2_price=0.0,
        status=status,
    )


def _engine() -> PortfolioEngine:
    engine = PortfolioEngine(principal=25_000.0, test_mode=True)
    engine.cash_balance = 25_000.0
    engine._available_balance = 25_000.0
    engine._total_open_risk = 0.0
    engine._last_bar_timestamp = int(time.time())
    engine._bar_interval_seconds = 60
    engine._lookup_position_close_cooldown = MagicMock(return_value=0.0)
    engine._get_loss_hold_until = AsyncMock(return_value=None)
    engine._ensure_symbol_constraints = AsyncMock()
    engine._remove_dust_position_canonical_cleanup = AsyncMock()
    return engine


def test_active_same_symbol_is_held_for_day_buy():
    engine = _engine()
    engine.open_positions["SOL/USDT"] = _pos("SOL/USDT", qty=0.6, status="ACTIVE", price=103.0)
    assert engine._day_position_blocks_new_entry(engine.open_positions["SOL/USDT"]) is True
    assert "SOL/USDT" in engine._day_entry_held_symbols()
    assert engine._day_path_ev_entry_block_reason("SOL/USDT", 4) == "DUPLICATE_SAME_SYMBOL"


def test_dust_pending_same_symbol_is_not_held_for_day_buy():
    engine = _engine()
    engine.open_positions["SOL/USDT"] = _pos("SOL/USDT", qty=0.001, status="DUST_PENDING", price=103.12)
    assert engine._day_position_blocks_new_entry(engine.open_positions["SOL/USDT"]) is False
    assert engine._day_entry_held_symbols() == set()
    assert engine._day_dust_symbols() == {"SOL/USDT"}
    assert engine._day_path_ev_entry_block_reason("SOL/USDT", 4) is None


def test_four_dust_rows_do_not_consume_day_entry_slots():
    engine = _engine()
    for sym, px, qty in (
        ("BTC/USDT", 78641.02, 1e-05),
        ("ETH/USDT", 2455.36, 0.0001),
        ("SOL/USDT", 103.12, 0.001),
        ("XRP/USDT", 1.4251, 0.1),
    ):
        engine.open_positions[sym] = _pos(sym, qty=qty, status="DUST_PENDING", price=px)
    assert engine._day_entry_held_count() == 0
    assert engine._day_entry_held_symbols() == set()
    assert engine._day_dust_symbols() == {"BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"}
    assert engine._day_path_ev_entry_block_reason("XRP/USDT", 4) is None
    assert engine._day_path_ev_entry_block_reason("SOL/USDT", 4) is None


def test_exit_residual_pending_still_blocks_same_symbol_buy():
    engine = _engine()
    engine.open_positions["ETH/USDT"] = _pos("ETH/USDT", qty=0.02, status=STATUS_EXIT_RESIDUAL_PENDING, price=2470.0)
    assert engine._day_position_blocks_new_entry(engine.open_positions["ETH/USDT"]) is True
    assert engine._day_path_ev_entry_block_reason("ETH/USDT", 4) == "DUPLICATE_SAME_SYMBOL"
    assert engine._day_entry_held_count() == 1


def test_real_max_open_still_blocks_when_held_slots_full():
    engine = _engine()
    engine.open_positions["BTC/USDT"] = _pos("BTC/USDT", qty=0.01, status="ACTIVE", price=78000.0)
    engine.open_positions["ETH/USDT"] = _pos("ETH/USDT", qty=0.2, status="ACTIVE", price=2470.0)
    engine.open_positions["SOL/USDT"] = _pos("SOL/USDT", qty=0.5, status="ACTIVE", price=103.0)
    engine.open_positions["XRP/USDT"] = _pos("XRP/USDT", qty=30.0, status="ACTIVE", price=1.4)
    assert engine._day_entry_held_count() == 4
    assert engine._day_path_ev_entry_block_reason("BTC/USDT", 4) == "DUPLICATE_SAME_SYMBOL"
    assert engine._day_path_ev_entry_block_reason("DOGE/USDT", 4) == "MAX_OPEN_LIMIT"


def test_max_open_uses_held_count_not_dust_dict_len():
    engine = _engine()
    engine.open_positions["BTC/USDT"] = _pos("BTC/USDT", qty=1e-05, status="DUST_PENDING", price=78641.0)
    engine.open_positions["ETH/USDT"] = _pos("ETH/USDT", qty=0.0001, status="DUST_PENDING", price=2455.0)
    engine.open_positions["SOL/USDT"] = _pos("SOL/USDT", qty=0.001, status="DUST_PENDING", price=103.0)
    engine.open_positions["XRP/USDT"] = _pos("XRP/USDT", qty=0.1, status="DUST_PENDING", price=1.42)
    assert len(engine.open_positions) == 4
    assert engine._day_path_ev_entry_block_reason("BTC/USDT", 4) is None


@pytest.mark.asyncio
async def test_reservation_duplicate_still_blocks_and_dust_does_not_fill_slots():
    with tempfile.TemporaryDirectory() as tmp:
        engine = _engine()
        engine.db_path = str(Path(tmp) / "reserve.db")
        for sym in ("BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"):
            engine.open_positions[sym] = _pos(sym, qty=0.001, status="DUST_PENDING")
        ok, reason = engine._try_reserve_entry("SOL/USDT", 50.0, decision_id="d-sol-1")
        assert ok is True, reason
        ok2, reason2 = engine._try_reserve_entry("SOL/USDT", 50.0, decision_id="d-sol-2")
        assert ok2 is False
        assert reason2 == "ENTRY_ALREADY_RESERVED"
        assert all(p.status == "DUST_PENDING" for p in engine.open_positions.values())


@pytest.mark.asyncio
async def test_can_open_allows_dust_symbol_without_flattening():
    engine = _engine()
    dust = _pos("XRP/USDT", qty=0.1, status="DUST_PENDING", price=1.4251)
    engine.open_positions["XRP/USDT"] = dust
    allowed, reason = await engine._can_open_position("XRP/USDT", 100.0)
    assert allowed is True, reason
    assert engine.open_positions["XRP/USDT"] is dust
    assert engine.open_positions["XRP/USDT"].status == "DUST_PENDING"
    engine._remove_dust_position_canonical_cleanup.assert_not_called()


@pytest.mark.asyncio
async def test_can_open_blocks_active_same_symbol():
    engine = _engine()
    engine.open_positions["SOL/USDT"] = _pos("SOL/USDT", qty=0.628, status="ACTIVE", price=102.87)
    allowed, reason = await engine._can_open_position("SOL/USDT", 100.0)
    assert allowed is False
    assert reason == "POSITION_ALREADY_OPEN"
    assert engine.open_positions["SOL/USDT"].status == "ACTIVE"


@pytest.mark.asyncio
async def test_can_open_blocks_exit_residual_same_symbol():
    engine = _engine()
    engine.open_positions["ETH/USDT"] = _pos("ETH/USDT", qty=0.02, status=STATUS_EXIT_RESIDUAL_PENDING, price=2470.0)
    allowed, reason = await engine._can_open_position("ETH/USDT", 100.0)
    assert allowed is False
    assert reason == "POSITION_ALREADY_OPEN"


@pytest.mark.asyncio
async def test_fetch_authoritative_ignores_dust_keeps_active():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "dust.db"
        engine = PortfolioEngine(db_path=str(db_path), principal=25_000.0, test_mode=True)
        engine._ensure_db_schema()
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(
                """
                INSERT INTO portfolio_engine_positions (
                    symbol, quantity, entry_price, entry_time, trade_id,
                    stop_price, take_profit_1_price, take_profit_2_price,
                    highest_price, atr_at_entry, entry_bar_timestamp,
                    confidence_at_entry, last_updated, sleeve, status
                ) VALUES ('SOL/USDT', 0.001, 103.12, ?, 'dust-sol',
                          97.0, 105.0, 108.0, 103.12, 1.5, 0, 0.5, datetime('now'),
                          'ACTIVE', 'DUST_PENDING')
                """,
                (time.time(),),
            )
            conn.execute(
                """
                INSERT INTO portfolio_engine_positions (
                    symbol, quantity, entry_price, entry_time, trade_id,
                    stop_price, take_profit_1_price, take_profit_2_price,
                    highest_price, atr_at_entry, entry_bar_timestamp,
                    confidence_at_entry, last_updated, sleeve, status
                ) VALUES ('BTC/USDT', 0.01, 78000.0, ?, 'active-btc',
                          76000.0, 80000.0, 81000.0, 78000.0, 700.0, 0, 0.5,
                          datetime('now'), 'ACTIVE', 'ACTIVE')
                """,
                (time.time(),),
            )
            conn.commit()
        dust_row = await asyncio.to_thread(engine._fetch_authoritative_open_position_sync, "SOL/USDT")
        active_row = await asyncio.to_thread(engine._fetch_authoritative_open_position_sync, "BTC/USDT")
        assert dust_row is None
        assert active_row is not None
        assert active_row[1] == "active-btc"


def test_repair_does_not_reclassify_dust_rows():
    engine = _engine()
    engine.open_positions["BTC/USDT"] = _pos("BTC/USDT", qty=1e-05, status="DUST_PENDING", price=78641.0)
    engine._day_entry_held_symbols()
    engine._day_path_ev_entry_block_reason("BTC/USDT", 4)
    assert engine.open_positions["BTC/USDT"].status == "DUST_PENDING"
    assert engine.open_positions["BTC/USDT"].quantity == 1e-05


def test_paper_and_live_use_same_held_predicate():
    paper = _engine()
    live = _engine()
    dust = _pos("ETH/USDT", qty=0.0001, status="DUST_PENDING", price=2455.36)
    active = _pos("ETH/USDT", qty=0.02, status="ACTIVE", price=2455.36)
    paper.open_positions["ETH/USDT"] = dust
    live.open_positions["ETH/USDT"] = dust
    assert paper._day_path_ev_entry_block_reason("ETH/USDT", 4) is None
    assert live._day_path_ev_entry_block_reason("ETH/USDT", 4) is None
    paper.open_positions["ETH/USDT"] = active
    live.open_positions["ETH/USDT"] = active
    assert paper._day_path_ev_entry_block_reason("ETH/USDT", 4) == "DUPLICATE_SAME_SYMBOL"
    assert live._day_path_ev_entry_block_reason("ETH/USDT", 4) == "DUPLICATE_SAME_SYMBOL"


def test_scalp_ranking_source_untouched():
    src = Path("backend/services/binance_scalp/scalp_candidate_ranking.py").read_text(encoding="utf-8")
    assert "DAY_ENTRY_HELD_SET" not in src
    assert "_day_path_ev_entry_block_reason" not in src


def test_path_ev_uses_canonical_held_predicate():
    src = Path("backend/services/portfolio_engine.py").read_text(encoding="utf-8")
    assert "DAY_ENTRY_HELD_SET" in src
    assert "_day_path_ev_entry_block_reason" in src
    assert "DAY_PATH_EV_SAFETY reject=DUPLICATE_SAME_SYMBOL" in src
    assert "if normalize_symbol(top_candidate.symbol) in self.open_positions:" not in src
