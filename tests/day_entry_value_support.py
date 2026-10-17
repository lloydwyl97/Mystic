"""Install and teach the DAY continuation surface that prices DAY entry."""

from __future__ import annotations

import sqlite3
import time

from backend.services.adaptive_learning import DAY_ENGINE, current_economic_version
from backend.services.continuation_surface import ADVANTAGE_VERSION, DAY_ADVANTAGE_HORIZONS, record_advantage, state_features


def install_day_surface(db: str, how: str = "blended") -> None:
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS continuation_learning_meta (
                engine_id TEXT NOT NULL,
                economic_version TEXT NOT NULL,
                learning_version TEXT NOT NULL,
                source TEXT NOT NULL,
                anchor_commit TEXT NOT NULL,
                anchor_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                observations INTEGER NOT NULL,
                PRIMARY KEY (engine_id, economic_version)
            )
            """
        )
        conn.execute(
            "INSERT OR REPLACE INTO continuation_learning_meta VALUES (?,?,?,?,?,?,?,?)",
            (DAY_ENGINE, current_economic_version(DAY_ENGINE), ADVANTAGE_VERSION, f"horizon_hold_advantage:{how}", "test", "", "", 0),
        )
        conn.commit()
    finally:
        conn.close()


def fresh_state(cost: float = 0.00066) -> dict[str, float]:
    return state_features(entry=1.0, mark=1.0, net=-cost, mfe=0.0, mae=0.0, high_water=1.0, prev_net=None, age_sec=0.0)


def teach_entry(db: str, symbol: str, setup: str, regime: str, advantage: float, *, times: int = 1, now: float | None = None) -> None:
    """Fold ``times`` fresh-state advantages at every DAY horizon for one key."""
    install_day_surface(db)
    moment = float(now if now is not None else time.time())
    for _ in range(int(times)):
        for horizon in DAY_ADVANTAGE_HORIZONS:
            assert record_advantage(db, engine=DAY_ENGINE, symbol=symbol, setup=setup, regime=regime, horizon=horizon, features=fresh_state(), advantage=float(advantage), weight=1.0, now=moment)
