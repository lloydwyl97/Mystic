"""portfolio-engine /coins must reflect persisted coin_performance + open book."""

from __future__ import annotations

import sqlite3
import time

from backend.services.portfolio_engine import CoinPerformance, PortfolioEngine


def _seed_coin_table(path: str) -> None:
    con = sqlite3.connect(path)
    con.execute(
        """
        CREATE TABLE coin_performance (
            symbol TEXT PRIMARY KEY,
            trades_24h INTEGER,
            pnl_24h REAL,
            win_count_20 INTEGER,
            loss_count_20 INTEGER,
            total_trades_20 INTEGER,
            avg_win REAL,
            avg_loss REAL,
            expectancy REAL,
            stop_loss_hits_10 INTEGER,
            current_drawdown REAL,
            peak_value REAL,
            pause_until REAL,
            sizing_multiplier REAL,
            last_updated REAL,
            profit_factor REAL,
            trades_last_30d INTEGER,
            avg_pnl REAL,
            confidence_to_pnl_correlation REAL
        )
        """
    )
    con.execute(
        """
        INSERT INTO coin_performance VALUES
        ('BTC/USDT', 1, 10.0, 1, 0, 1, 10.0, 0.0, 10.0, 0, 0, 0, 0, 1.0, ?, NULL, 0, 0, 0)
        """,
        (time.time(),),
    )
    con.commit()
    con.close()


def test_load_coin_performance_and_status_not_no_data(tmp_path):
    db = str(tmp_path / "t.db")
    _seed_coin_table(db)
    eng = PortfolioEngine(db_path=db, principal=10000.0, test_mode=True)
    assert eng.coin_performance == {}
    n = eng._load_coin_performance_from_sqlite()
    assert n == 1
    st = eng.get_coin_status("BTCUSDT")
    assert st["status"] == "flat"
    assert st["trades_24h"] == 1
    assert st["pnl_24h"] == 10.0
    assert st["book"]["open"] is False


def test_coin_status_keys_cover_slash_and_bus():
    eng = PortfolioEngine(db_path=":memory:", principal=10000.0, test_mode=True)
    keys = eng._coin_status_keys("BTCUSDT")
    assert "BTC/USDT" in keys
    assert "BTCUSDT" in keys


def test_coin_status_book_falls_back_to_sqlite_positions(tmp_path):
    db = str(tmp_path / "t2.db")
    con = sqlite3.connect(db)
    con.execute(
        """
        CREATE TABLE portfolio_engine_positions (
            symbol TEXT, quantity REAL, entry_price REAL,
            highest_price REAL, lowest_price REAL
        )
        """
    )
    con.execute(
        "INSERT INTO portfolio_engine_positions VALUES ('XRP/USDT', 10.0, 1.30, 1.40, 1.28)"
    )
    con.commit()
    con.close()
    eng = PortfolioEngine(db_path=db, principal=10000.0, test_mode=True)
    st = eng.get_coin_status("XRPUSDT")
    assert st["status"] == "open"
    assert st["book"]["open"] is True
    assert st["book"]["quantity"] == 10.0
    assert st["book"]["entry_price"] == 1.30
