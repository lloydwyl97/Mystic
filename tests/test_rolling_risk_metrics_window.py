from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from backend.services.portfolio_engine import PortfolioEngine


@pytest.mark.asyncio
async def test_consecutive_losses_exclude_stale_sell_rows(tmp_path):
    db_path = tmp_path / "risk.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE portfolio_engine_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            action TEXT,
            pre_ledger_json TEXT,
            post_ledger_json TEXT
        )
        """
    )
    now = datetime.now(timezone.utc)
    loss_pre = json.dumps({"total_equity": 100.0})
    loss_post = json.dumps({"total_equity": 99.0})
    conn.executemany(
        """
        INSERT INTO portfolio_engine_audit
            (ts, action, pre_ledger_json, post_ledger_json)
        VALUES (?, 'SELL', ?, ?)
        """,
        [
            ((now - timedelta(hours=2)).isoformat(), loss_pre, loss_post),
            ((now - timedelta(hours=1)).isoformat(), loss_pre, loss_post),
            ((now - timedelta(days=3)).isoformat(), loss_pre, loss_post),
        ],
    )
    conn.commit()
    conn.close()

    engine = PortfolioEngine.__new__(PortfolioEngine)
    engine.db_path = str(db_path)
    engine._total_equity = 99.0

    _, consecutive_losses = await engine.get_rolling_24h_risk_metrics()

    assert consecutive_losses == 2
