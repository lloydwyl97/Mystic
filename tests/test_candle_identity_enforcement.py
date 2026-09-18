"""The canonical candle identity is symbol + interval + open_time.

feature_ohlcv used to leave that identity unenforced and let a writer stamp the
import time into ts, which is how historical bars get rewritten as if they had
just opened. These tests hold both properties down.
"""

from __future__ import annotations

import sqlite3

import pytest

import backend.services.feature_store as fs


def _cols(db: str, table: str) -> list[str]:
    with sqlite3.connect(db) as c:
        return [r[1] for r in c.execute(f"PRAGMA table_info({table})")]


def test_ts_has_no_persist_now_default():
    """ts must not default to the write time.

    insert_ohlcv() hardcoded datetime.now() and insert_ohlcv_bulk() fell back to it,
    so a candle whose open time was missing or was epoch-ms silently recorded the
    moment of import as the bar's open.
    """
    ts_col = fs.FeatureOHLCV.__table__.c.ts
    assert ts_col.default is None, "ts must not fall back to now(); it is the bar's open time"
    assert ts_col.server_default is None
    assert ts_col.nullable is False, "a candle with no open time has no identity"


def test_persist_now_writers_are_gone():
    """The two uncalled writers that could only stamp write-time are removed."""
    assert not hasattr(fs, "insert_ohlcv")
    assert not hasattr(fs, "insert_ohlcv_bulk")
    assert "insert_ohlcv" not in fs.__all__
    assert "insert_ohlcv_bulk" not in fs.__all__
    # Tick writers are unaffected: a tick's ts really is its observation time.
    assert hasattr(fs, "insert_tick")


def test_identity_index_is_unique_in_the_model():
    idx = {i.name: i for i in fs.FeatureOHLCV.__table__.indexes}
    ident = idx.get("ix_feature_ohlcv_symbol_interval_ts")
    assert ident is not None, "the canonical identity index must exist"
    assert ident.unique is True, "symbol + interval + open_time must be unique"
    assert [c.name for c in ident.columns] == ["symbol", "interval", "ts"]


def test_migration_upgrades_a_legacy_nonunique_index(tmp_path, monkeypatch):
    """A database created before the constraint gets the unique index on startup."""
    db = tmp_path / "legacy.db"
    with sqlite3.connect(db) as c:
        c.execute("CREATE TABLE feature_ohlcv (id INTEGER PRIMARY KEY, symbol VARCHAR(32), interval VARCHAR(8), open FLOAT, high FLOAT, low FLOAT, close FLOAT, volume FLOAT, ts DATETIME)")
        c.execute("CREATE INDEX ix_feature_ohlcv_symbol_interval_ts ON feature_ohlcv (symbol, interval, ts)")
        c.execute("INSERT INTO feature_ohlcv (symbol, interval, open, high, low, close, volume, ts) VALUES ('BTC-USDT','15m',1,2,0.5,1.5,10,'2026-09-18 14:00:00.000000')")

    from sqlalchemy import create_engine

    engine = create_engine(f"sqlite:///{db}")
    monkeypatch.setattr(fs, "ENGINE", engine)
    fs.enforce_candle_identity_unique()

    with sqlite3.connect(db) as c:
        row = c.execute(
            "SELECT \"unique\" FROM pragma_index_list('feature_ohlcv') WHERE name=?",
            ("ix_feature_ohlcv_symbol_interval_ts",),
        ).fetchone()
        assert row is not None and row[0] == 1, "index should have been rebuilt as UNIQUE"

        # The identity is now actually enforced by the database.
        with pytest.raises(sqlite3.IntegrityError):
            c.execute("INSERT INTO feature_ohlcv (symbol, interval, open, high, low, close, volume, ts) VALUES ('BTC-USDT','15m',9,9,9,9,9,'2026-09-18 14:00:00.000000')")


def test_migration_refuses_to_run_when_duplicates_exist(tmp_path, monkeypatch, caplog):
    """Pre-existing duplicates must be reported, not crash startup."""
    db = tmp_path / "dupes.db"
    with sqlite3.connect(db) as c:
        c.execute("CREATE TABLE feature_ohlcv (id INTEGER PRIMARY KEY, symbol VARCHAR(32), interval VARCHAR(8), open FLOAT, high FLOAT, low FLOAT, close FLOAT, volume FLOAT, ts DATETIME)")
        c.execute("CREATE INDEX ix_feature_ohlcv_symbol_interval_ts ON feature_ohlcv (symbol, interval, ts)")
        for close in (1.5, 9.9):
            c.execute(
                "INSERT INTO feature_ohlcv (symbol, interval, open, high, low, close, volume, ts) VALUES ('BTC-USDT','15m',1,2,0.5,?,10,'2026-09-18 14:00:00.000000')",
                (close,),
            )

    from sqlalchemy import create_engine

    engine = create_engine(f"sqlite:///{db}")
    monkeypatch.setattr(fs, "ENGINE", engine)
    with caplog.at_level("ERROR"):
        fs.enforce_candle_identity_unique()

    assert "duplicate" in caplog.text.lower()
    with sqlite3.connect(db) as c:
        row = c.execute(
            "SELECT \"unique\" FROM pragma_index_list('feature_ohlcv') WHERE name=?",
            ("ix_feature_ohlcv_symbol_interval_ts",),
        ).fetchone()
        assert row[0] == 0, "must leave the old index alone rather than fail startup"


def test_migration_is_idempotent(tmp_path, monkeypatch):
    db = tmp_path / "idem.db"
    with sqlite3.connect(db) as c:
        c.execute("CREATE TABLE feature_ohlcv (id INTEGER PRIMARY KEY, symbol VARCHAR(32), interval VARCHAR(8), open FLOAT, high FLOAT, low FLOAT, close FLOAT, volume FLOAT, ts DATETIME)")
        c.execute("CREATE UNIQUE INDEX ix_feature_ohlcv_symbol_interval_ts ON feature_ohlcv (symbol, interval, ts)")

    from sqlalchemy import create_engine

    engine = create_engine(f"sqlite:///{db}")
    monkeypatch.setattr(fs, "ENGINE", engine)
    fs.enforce_candle_identity_unique()
    fs.enforce_candle_identity_unique()

    with sqlite3.connect(db) as c:
        rows = c.execute(
            "SELECT name, \"unique\" FROM pragma_index_list('feature_ohlcv') WHERE name=?",
            ("ix_feature_ohlcv_symbol_interval_ts",),
        ).fetchall()
    assert rows == [("ix_feature_ohlcv_symbol_interval_ts", 1)]
