"""
Regression: canonical position reconciliation must never silently delete real
open inventory just because engine memory lost track of it. Only a completed
SELL after entry is deterministic evidence a position is actually closed;
absent that, the engine must reload/adopt the canonical row into memory.
"""

from __future__ import annotations

import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.services.portfolio_engine import PortfolioEngine, Sleeve


def _init_db(db_path: Path, *, principal: float = 25_000.0) -> None:
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
            (principal, principal, principal),
        )
        conn.commit()


def _seed_position(db_path: Path, *, symbol: str, qty: float, entry: float, entry_time: float, trade_id: str) -> None:
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
            (symbol, qty, entry, entry_time, trade_id, entry * 0.97, entry * 1.03, 0.0, entry, entry * 0.01, 0, 0.5, Sleeve.ACTIVE.value),
        )
        conn.commit()


async def _run_invariants_with_mocked_side_effects(engine: PortfolioEngine) -> None:
    engine._entry_ensure_constraints = AsyncMock()
    engine._ensure_symbol_constraints = AsyncMock()
    engine._symbol_constraints = {}
    engine._clear_all_quarantines = AsyncMock()
    with (
        patch("backend.services.paper_trading_service.get_paper_trading_service", return_value=None),
        patch("backend.config.redis_config.SharedRedisState.get_async_client", return_value=None),
    ):
        await engine._validate_invariants("test")


@pytest.mark.asyncio
async def test_open_position_without_sell_evidence_is_preserved_not_deleted():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "preserve.db"
        _init_db(db_path)
        # Position exists in canonical SQLite but engine memory never loaded it
        # (simulates a restart-timing bug / dropped in-memory entry) — and there
        # is NO sell trade at all for this symbol.
        _seed_position(db_path, symbol="BTC/USDT", qty=0.05, entry=60_000.0, entry_time=time.time() - 600, trade_id="preserve-test")

        engine = PortfolioEngine(db_path=str(db_path), principal=25_000.0, test_mode=False)
        engine.open_positions = {}  # engine memory empty -> "missing_in_memory" mismatch

        await _run_invariants_with_mocked_side_effects(engine)

        with sqlite3.connect(str(db_path)) as conn:
            row = conn.execute("SELECT COUNT(*) FROM portfolio_engine_positions WHERE symbol='BTC/USDT'").fetchone()
        assert row[0] == 1, "position must NOT be deleted absent completed-sell evidence"
        assert "BTC/USDT" in engine.open_positions, "position must be reloaded/adopted into engine memory"


@pytest.mark.asyncio
async def test_position_with_completed_sell_evidence_is_purged():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "purge.db"
        _init_db(db_path)
        entry_time = time.time() - 3600
        _seed_position(db_path, symbol="ETH/USDT", qty=1.0, entry=1800.0, entry_time=entry_time, trade_id="purge-test")
        # A completed SELL after entry_time exists -> deterministic evidence the position is closed.
        with sqlite3.connect(str(db_path)) as conn:
            from datetime import datetime, timezone

            sell_ts = datetime.fromtimestamp(entry_time + 1800, tz=timezone.utc).isoformat()
            conn.execute(
                """
                INSERT INTO paper_trades (
                    trade_id, paper_run_id, mode, symbol, side, quantity, price,
                    remaining_position, timestamp, status, strategy_id
                ) VALUES ('purge-test-sell', 'test-run', 'paper', 'ETH/USDT', 'SELL', 1.0, 1850.0, 0.0, ?, 'executed', 'day')
                """,
                (sell_ts,),
            )
            conn.commit()

        engine = PortfolioEngine(db_path=str(db_path), principal=25_000.0, test_mode=False)
        engine.open_positions = {}

        await _run_invariants_with_mocked_side_effects(engine)

        with sqlite3.connect(str(db_path)) as conn:
            row = conn.execute("SELECT COUNT(*) FROM portfolio_engine_positions WHERE symbol='ETH/USDT'").fetchone()
        assert row[0] == 0, "position with completed-sell evidence should be purged as genuinely stale"
