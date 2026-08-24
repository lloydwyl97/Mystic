"""Markout schedule/flush isolation and learning from executable nets."""

from __future__ import annotations

import sqlite3
import time

from backend.services.binance_scalp.scalp_markout import (
    flush_completed,
    observe_book,
    pending_count,
    reset_markouts,
    schedule_markout,
)
from backend.services.binance_scalp.scalp_micro_learning import micro_learning_adjustments, reset_learning_cache


def test_markout_extra_retains_feature_buckets(tmp_path):
    from backend.services.binance_scalp.scalp_micro_contract import feature_context_extra

    reset_markouts()
    db = str(tmp_path / "scalp_micro_extra.db")
    extra = feature_context_extra(
        {
            "symbol": "ETH",
            "ts": 1_000.0,
            "ofi_5s": 2.4,
            "obi_l5": 0.4,
            "microprice_pressure": 0.0002,
            "adverse_selection_score": 0.08,
            "agg_flow_imbalance_5s": 0.4,
            "bid_absorption_score": 0.6,
            "ask_absorption_score": 0.1,
        },
        {"rank_score": 1.7, "trade_id": "t1", "decision_epoch": 1_000.0},
    )
    assert extra["ofi_bucket"] == "strongly_positive"
    assert extra["obi_bucket"] == "positive"
    assert extra["microprice_bucket"] == "favorable"
    assert extra["adverse_bucket"] == "low"
    assert extra["model_version"] == "scalp_micro_ev_v1"
    assert extra["microstructure_version"] == "scalp_micro_v1"
    schedule_markout(
        kind="candidate",
        symbol="ETHUSDT",
        side="BUY",
        mid=100.0,
        entry_px=100.05,
        qty=0.0,
        notional=1.0,
        fee_pct=0.0004,
        slip_pct=0.0001,
        extra=extra,
        now=1_000.0,
    )
    observe_book("ETHUSDT", bid=100.2, ask=100.3, bids=[[100.2, 2.0]], asks=[[100.3, 2.0]], now=1_130.0)
    assert flush_completed(db, force=True, now=1_131.0) == 1
    import json

    with sqlite3.connect(db) as conn:
        row = conn.execute("SELECT extra_json, microstructure_version, model_version FROM scalp_micro_markouts").fetchone()
    stored = json.loads(row[0])
    assert stored["ofi_5s"] == 2.4
    assert stored["ofi_bucket"] == "strongly_positive"
    assert stored["trade_id"] == "t1"
    assert row[1] == "scalp_micro_v1"
    assert row[2] == "scalp_micro_ev_v1"


def test_learning_consumes_feature_context_extra(tmp_path):
    reset_learning_cache()
    db = str(tmp_path / "learn_ctx.db")
    import json

    from backend.services.binance_scalp.scalp_micro_contract import feature_context_extra

    t0 = time.time() - 80
    extra = json.dumps(
        feature_context_extra(
            {"ofi_5s": 2.0, "obi_l5": 0.4, "adverse_selection_score": 0.05},
            {"rank_score": 1.6},
        )
    )
    points = json.dumps({"10": {"executable_net_markout": 0.002}})
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scalp_micro_markouts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kind TEXT, symbol TEXT, side TEXT, t0 REAL, mid0 REAL, entry_px REAL,
                qty REAL, notional REAL, mfe REAL, mae REAL, points_json TEXT, extra_json TEXT,
                feature_version TEXT, microstructure_version TEXT, model_version TEXT, created_at REAL
            )
            """
        )
        for i in range(12):
            conn.execute(
                """
                INSERT INTO scalp_micro_markouts (
                    kind, symbol, side, t0, mid0, entry_px, qty, notional, mfe, mae,
                    points_json, extra_json, created_at
                ) VALUES ('entry','BTCUSDT','BUY',?,?,100,0.01,1,0.01,-0.01,?,?,?)
                """,
                (t0 + i, 100.0, points, extra, t0 + i),
            )
        conn.commit()
    adj = micro_learning_adjustments(db, symbol="BTCUSDT", ofi_5s=2.0, obi_l5=0.4, adverse_selection_score=0.05)
    assert adj["consumed"] is True
    assert adj["rank_delta"] > 0
    assert adj["size_mult"] > 1.0
    assert adj["eligibility"] is False


def test_buy_microstructure_stamp_survives_path_net_provenance():
    from backend.services.binance_scalp.scalp_micro_contract import (
        buy_microstructure_invariant_violations,
        version_stamps,
    )

    diag = {
        **version_stamps(),
        "microstructure_features": {"ofi_5s": 0.2, "obi_l1": 0.1},
        "EV_10s": 0.0001,
    }
    diag.update({"model_version": "scalp_path_net_v1", "forward_net_model_version": "scalp_path_net_v1"})
    diag.update(version_stamps())
    assert buy_microstructure_invariant_violations(diag) == []
    assert diag["model_version"] == "scalp_micro_ev_v1"
    assert diag["forward_net_model_version"] == "scalp_path_net_v1"


def test_buy_microstructure_stamp_invariant():
    from backend.services.binance_scalp.scalp_micro_contract import (
        buy_microstructure_invariant_violations,
        version_stamps,
    )

    good = {
        **version_stamps(),
        "microstructure_features": {"ofi_5s": 0.2, "obi_l1": 0.1},
        "EV_10s": 0.0001,
    }
    assert buy_microstructure_invariant_violations(good) == []
    assert buy_microstructure_invariant_violations({"model_version": "scalp_path_net_v1"}) != []
    assert buy_microstructure_invariant_violations({**version_stamps(), "microstructure_features": {}}) != []


def test_feature_context_extra_persists_tape_fields():
    from backend.services.binance_scalp.scalp_micro_contract import feature_context_extra

    extra = feature_context_extra(
        {
            "ofi_5s": 0.4,
            "agg_flow_imbalance_5s": 0.22,
            "trade_count_5s": 4,
            "flow_acceleration": 0.05,
            "bid_absorption_score": 0.3,
            "ask_absorption_score": 0.0,
        }
    )
    assert extra["agg_flow_imbalance_5s"] == 0.22
    assert extra["trade_count_5s"] == 4
    assert extra["flow_acceleration"] == 0.05
    assert extra["agg_flow_bucket"] == "buying"


def test_markout_completes_without_future_leak(tmp_path):
    reset_markouts()
    db = str(tmp_path / "scalp_micro.db")
    t0 = 1_000.0
    schedule_markout(
        kind="entry",
        symbol="BTCUSDT",
        side="BUY",
        mid=100.0,
        entry_px=100.05,
        qty=0.01,
        notional=1.0,
        fee_pct=0.0004,
        slip_pct=0.0001,
        extra={"ofi_5s": 1.0, "obi_l5": 0.2, "adverse_selection_score": 0.1},
        now=t0,
    )
    assert pending_count() == 1
    # Before +1s nothing completes.
    observe_book("BTCUSDT", bid=100.0, ask=100.1, bids=[[100.0, 2.0]], asks=[[100.1, 2.0]], now=t0 + 0.2)
    assert pending_count() == 1
    observe_book("BTCUSDT", bid=100.2, ask=100.3, bids=[[100.2, 2.0]], asks=[[100.3, 2.0]], now=t0 + 130.0)
    n = flush_completed(db, force=True, now=t0 + 131.0)
    assert n == 1
    assert pending_count() == 0
    with sqlite3.connect(db) as conn:
        row = conn.execute("SELECT kind, symbol FROM scalp_micro_markouts").fetchone()
        assert row[0] == "entry"
        assert row[1] == "BTCUSDT"


def test_learning_positive_markouts_raise_priority(tmp_path):
    reset_markouts()
    reset_learning_cache()
    db = str(tmp_path / "learn.db")
    t0 = time.time() - 100
    # Seed completed markouts directly.
    flush_completed(db, force=True)  # create table path
    import json

    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scalp_micro_markouts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kind TEXT, symbol TEXT, side TEXT, t0 REAL, mid0 REAL, entry_px REAL,
                qty REAL, notional REAL, mfe REAL, mae REAL, points_json TEXT, extra_json TEXT,
                feature_version TEXT, microstructure_version TEXT, model_version TEXT, created_at REAL
            )
            """
        )
        extra = json.dumps({"ofi_5s": 2.0, "obi_l5": 0.4, "adverse_selection_score": 0.05})
        points = json.dumps({"10": {"executable_net_markout": 0.002}})
        for i in range(12):
            conn.execute(
                """
                INSERT INTO scalp_micro_markouts (
                    kind, symbol, side, t0, mid0, entry_px, qty, notional, mfe, mae,
                    points_json, extra_json, created_at
                ) VALUES ('entry','BTCUSDT','BUY',?,?,100,0.01,1,0.01,-0.01,?,?,?)
                """,
                (t0 + i, 100.0, points, extra, t0 + i),
            )
        conn.commit()
    pos = micro_learning_adjustments(db, symbol="BTCUSDT", ofi_5s=2.0, obi_l5=0.4, adverse_selection_score=0.05)
    assert pos["consumed"] is True
    assert pos["rank_delta"] > 0
    assert pos["size_mult"] > 1.0
    assert pos["eligibility"] is False


def test_learning_negative_markouts_reduce_priority(tmp_path):
    reset_learning_cache()
    db = str(tmp_path / "learn_neg.db")
    import json
    import time as _t

    t0 = _t.time() - 50
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scalp_micro_markouts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kind TEXT, symbol TEXT, side TEXT, t0 REAL, mid0 REAL, entry_px REAL,
                qty REAL, notional REAL, mfe REAL, mae REAL, points_json TEXT, extra_json TEXT,
                feature_version TEXT, microstructure_version TEXT, model_version TEXT, created_at REAL
            )
            """
        )
        extra = json.dumps({"ofi_5s": -2.0, "obi_l5": -0.4, "adverse_selection_score": 0.6})
        points = json.dumps({"10": {"executable_net_markout": -0.003}})
        for i in range(12):
            conn.execute(
                """
                INSERT INTO scalp_micro_markouts (
                    kind, symbol, side, t0, mid0, entry_px, qty, notional, mfe, mae,
                    points_json, extra_json, created_at
                ) VALUES ('entry','XRPUSDT','BUY',?,?,2,1,2,-0.01,0.0,?,?,?)
                """,
                (t0 + i, 2.0, points, extra, t0 + i),
            )
        conn.commit()
    neg = micro_learning_adjustments(db, symbol="XRPUSDT", ofi_5s=-2.0, obi_l5=-0.4, adverse_selection_score=0.6)
    assert neg["rank_delta"] < 0
    assert neg["size_mult"] < 1.0
    assert neg["eligibility"] is False
