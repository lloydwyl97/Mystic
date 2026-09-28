"""Learning provenance: engine/setup/dust tags, SCALP feed, deterministic backfill."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

from backend.services.learning_provenance import learning_provenance

ROOT = Path(__file__).resolve().parents[1]


def _backfill():
    spec = importlib.util.spec_from_file_location("lpb", ROOT / "scripts" / "learning_provenance_backfill.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _db(tmp_path) -> str:
    db = str(tmp_path / "l.db")
    c = sqlite3.connect(db)
    c.executescript(
        """
        CREATE TABLE paper_trades (id INTEGER PRIMARY KEY, symbol TEXT, side TEXT, timestamp TEXT, engine_id TEXT, trade_id TEXT, scalp_opportunity_id TEXT);
        CREATE TABLE scalp_v2_opportunities (id INTEGER PRIMARY KEY, opportunity_id TEXT, setup_family TEXT);
        CREATE TABLE day_v2_opportunity_state (id INTEGER PRIMARY KEY, trade_id TEXT, setup TEXT);
        CREATE TABLE trade_learning_outcomes (id INTEGER PRIMARY KEY, symbol TEXT, entry_timestamp REAL, exit_timestamp REAL, entry_price REAL, exit_price REAL,
            quantity REAL, fees_paid REAL, slippage_cost REAL, net_profit_usd REAL, net_profit_pct REAL, hold_seconds REAL, close_reason TEXT, extra_json TEXT);
        CREATE TABLE ai_outcome_training_rows (id INTEGER PRIMARY KEY, symbol TEXT, closed_at_utc TEXT, outcome_class TEXT, strategy_id TEXT, realized_pct REAL);
        INSERT INTO scalp_v2_opportunities (opportunity_id, setup_family) VALUES ('opp1', 'range_bounce_scalp');
        INSERT INTO day_v2_opportunity_state (trade_id, setup) VALUES ('day1', 'HTF_TREND_PULLBACK');
        INSERT INTO paper_trades (symbol, side, timestamp, engine_id, trade_id, scalp_opportunity_id) VALUES
            ('SOL/USDT', 'SELL', '2026-09-28T01:08:10+00:00', 'SCALP_V2', 'sc1', 'opp1'),
            ('BTC/USDT', 'SELL', '2026-09-28T02:00:00+00:00', 'DAY_V2', 'day1', NULL),
            ('XRP/USDT', 'SELL', '2026-09-28T03:00:00+00:00', 'SCALP_V2', 'x1', NULL),
            ('XRP/USDT', 'SELL', '2026-09-28T03:00:05+00:00', 'SCALP_V2', 'x2', NULL);
        """
    )
    t = 1790557690.0  # 2026-09-28T01:08:10Z
    rows = [
        (1, "SOL/USDT", t + 3, 0.16, "NET_PROFIT_EXIT", '{"source": "engine"}'),
        (2, "BTC/USDT", t + 3110, -0.2, "THESIS_INVALIDATION_EXIT", "{}"),
        (3, "XRP/USDT", t + 6712, 0.1, "NET_PROFIT_EXIT", "{}"),
        (4, "ETH/USDT", t, None, "DUST_WRITEOFF", "{}"),
        (5, "ETH/USDT", t, 1.0, "NET_PROFIT_EXIT", '{"engine_id": "DAY_V2"}'),
    ]
    for rid, sym, ts, pnl, reason, extra in rows:
        c.execute(
            "INSERT INTO trade_learning_outcomes (id, symbol, entry_timestamp, exit_timestamp, entry_price, exit_price, quantity, net_profit_usd, close_reason, extra_json) "
            "VALUES (?,?,?,?,1,2,3,?,?,?)",
            (rid, sym, ts - 60, ts, pnl, reason, extra),
        )
    c.executemany(
        "INSERT INTO ai_outcome_training_rows (id, symbol, closed_at_utc, outcome_class, strategy_id, realized_pct) VALUES (?,?,?,?,?,0.01)",
        [(1, "SOL/USDT", "2026-09-28T01:08:12+00:00", "WIN", "day"), (2, "BTC/USDT", "2026-09-28T02:00:01+00:00", "LOSS", "day"), (3, "ETH/USDT", "2026-09-28T01:00:00+00:00", "DUST", "day")],
    )
    c.commit()
    c.close()
    return db


def test_scalp_close_never_tagged_day(tmp_path):
    db = _db(tmp_path)
    pos = SimpleNamespace(engine_id="SCALP_V2", trade_id="sc1", scalp_opportunity_id="opp1", entry_thesis="", status="ACTIVE")
    prov = learning_provenance(db, pos, "NET_PROFIT_EXIT", {"setup_type_canonical": "HTF_TREND_PULLBACK"})
    assert prov == {"engine_id": "SCALP_V2", "trade_id": "sc1", "strategy": "scalp", "setup": "range_bounce_scalp", "is_dust": False, "label_strategy": "scalp"}


def test_day_setup_and_unknown_and_dust(tmp_path):
    db = _db(tmp_path)
    day = learning_provenance(db, SimpleNamespace(engine_id="DAY_V2", trade_id="day1", status="ACTIVE"), "THESIS_INVALIDATION_EXIT")
    assert (day["strategy"], day["setup"]) == ("day", "HTF_TREND_PULLBACK")
    legacy = learning_provenance(db, SimpleNamespace(engine_id="", trade_id="zz", status="DUST_PENDING"), "TRAILING_STOP_EXIT")
    assert legacy["engine_id"] == "LEGACY_DAY_LIVE" and legacy["setup"] == "UNKNOWN"
    assert legacy["is_dust"] and legacy["label_strategy"] == "dust"


def test_scalp_feed_reads_live_engine_rows_not_paper_tag(tmp_path):
    from backend.services.ai_learning_ingestion import ingest_scalp_outcomes

    db = _db(tmp_path)
    c = sqlite3.connect(db)
    c.execute("UPDATE trade_learning_outcomes SET extra_json=? WHERE id=1", (json.dumps({"engine_id": "SCALP_V2", "setup": "range_bounce_scalp", "is_dust": False}),))
    c.execute("UPDATE trade_learning_outcomes SET extra_json=? WHERE id=4", (json.dumps({"engine_id": "SCALP_V2", "is_dust": True}),))
    c.commit()
    c.close()
    ingest_scalp_outcomes(db)
    rows = sqlite3.connect(db).execute("SELECT source_id, setup_name FROM scalp_learning_outcomes").fetchall()
    assert rows == [(1, "range_bounce_scalp")]


def test_backfill_dry_run_then_apply_only_touches_provenance(tmp_path):
    lpb = _backfill()
    db = _db(tmp_path)
    before = sqlite3.connect(db).execute("SELECT id, symbol, exit_timestamp, net_profit_usd, entry_price, quantity FROM trade_learning_outcomes ORDER BY id").fetchall()
    res = lpb.plan(db, 20.0)
    c = res["counts"]
    assert (c["tlo_linked"], c["tlo_scalp"], c["tlo_day"], c["tlo_dust_only"], c["tlo_already_tagged"]) == (2, 1, 1, 1, 1)
    assert c["tlo_unmatched"] == 1  # XRP: two SELLs within the window -> ambiguous, untouched
    assert (c["aotr_to_scalp"], c["aotr_to_dust"]) == (1, 1)
    assert sqlite3.connect(db).execute("SELECT extra_json FROM trade_learning_outcomes WHERE id=1").fetchone()[0] == '{"source": "engine"}'

    backup = tmp_path / "bk.json"
    lpb.apply(db, res, backup)
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT id, symbol, exit_timestamp, net_profit_usd, entry_price, quantity FROM trade_learning_outcomes ORDER BY id").fetchall() == before
    sol = json.loads(conn.execute("SELECT extra_json FROM trade_learning_outcomes WHERE id=1").fetchone()[0])
    assert (sol["source"], sol["engine_id"], sol["strategy"], sol["setup"], sol["trade_id"]) == ("engine", "SCALP_V2", "scalp", "range_bounce_scalp", "sc1")
    assert json.loads(conn.execute("SELECT extra_json FROM trade_learning_outcomes WHERE id=3").fetchone()[0]) == {}
    assert conn.execute("SELECT id, strategy_id FROM ai_outcome_training_rows ORDER BY id").fetchall() == [(1, "scalp"), (2, "day"), (3, "dust")]
    saved = json.loads(backup.read_text())
    assert {u["id"] for u in saved["trade_learning_outcomes"]} == {1, 2, 4}
    assert saved["ai_outcome_training_rows"][0]["old_strategy_id"] == "day"


def test_live_close_writer_stamps_scalp_provenance(tmp_path, monkeypatch):
    from backend.services.portfolio_engine import OpenPosition, PortfolioEngine

    db = str(tmp_path / "live.db")
    c = sqlite3.connect(db)
    c.execute("CREATE TABLE scalp_v2_opportunities (id INTEGER PRIMARY KEY, opportunity_id TEXT, setup_family TEXT)")
    c.execute("INSERT INTO scalp_v2_opportunities (opportunity_id, setup_family) VALUES ('opp1', 'range_bounce_scalp')")
    c.commit()
    c.close()
    engine = PortfolioEngine.__new__(PortfolioEngine)
    engine.db_path = db
    engine.trade_explanations = {}
    engine._live_execution_enabled = False
    pos = OpenPosition(
        symbol="SOL/USDT",
        quantity=0.2,
        entry_price=120.0,
        entry_time=1790557600.0,
        trade_id="sc1",
        stop_price=0.0,
        take_profit_1_price=0.0,
        take_profit_2_price=0.0,
        highest_price=121.0,
        lowest_price=119.5,
        atr_at_entry=0.5,
        confidence_at_entry=0.5,
        engine_id="SCALP_V2",
        scalp_opportunity_id="opp1",
    )
    engine._record_learning_outcome(
        symbol="SOL/USDT",
        position=pos,
        close_reason="NET_PROFIT_EXIT",
        manual_sell=False,
        source="engine",
        exit_price=121.0,
        realized_profit=0.16,
        cooldown_until=0.0,
        fill_found=True,
    )
    conn = sqlite3.connect(db)
    extra = json.loads(conn.execute("SELECT extra_json FROM trade_learning_outcomes ORDER BY id DESC LIMIT 1").fetchone()[0])
    assert (extra["engine_id"], extra["strategy"], extra["setup"], extra["is_dust"]) == ("SCALP_V2", "scalp", "range_bounce_scalp", False)
    assert [r[0] for r in conn.execute("SELECT strategy_id FROM ai_outcome_training_rows")] == ["scalp"]


def test_promotion_holdout_rows_are_excluded_from_training(monkeypatch):
    import backend.ai_training_pipeline as tp
    import backend.services.ai_model_promotion_holdout as hold

    monkeypatch.setattr(hold, "holdout_window", lambda **k: {"ids": [3, 4], "n": 2, "min_id": 3, "max_id": 4} if k["symbol_bus"] == "BTCUSDT" else {"ids": [], "n": 0})
    rows = [{"id": i} for i in range(1, 6)]
    kept, windows = tp._exclude_promotion_holdout(rows, "day", 145, 5)
    assert [r["id"] for r in kept] == [1, 2, 5]
    assert windows["BTCUSDT"]["max_id"] == 4
