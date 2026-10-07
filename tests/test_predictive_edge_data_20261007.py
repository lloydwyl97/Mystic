"""Causal research data: 145 catalog and joins, label timing, queue capture,
counterfactual states, maker fills, research evaluation and rebuild linkage."""

from __future__ import annotations

import json
import math
import sqlite3
import time
import zlib
from pathlib import Path

import numpy as np
import pytest

import backend.services.adaptive_learning as al
import backend.services.economic_state_rebuild as esr
from backend.services import book_queue_capture as bq
from backend.services import counterfactual_continuation as cc
from backend.services import day_feature_catalog as cat
from backend.services import day_research_rows as drr
from backend.services import edge_research as er
from backend.services.day_v2.lifecycle_sim import LifecycleParams, simulate_lifecycle
from backend.services.economic_replay import DayCandidate
from backend.services.sqlite_large_table_retention import RETENTION_JUSTIFICATION, RETENTION_POLICIES

ROOT = Path(__file__).resolve().parents[1]

# --- 145 catalog -------------------------------------------------------------------


def test_catalog_covers_145_dims_and_classes():
    assert len(cat.DAY_FEATURE_NAMES) == cat.DAY_VECTOR_DIM == 145
    classes = [r["class"] for r in cat.catalog()]
    assert classes.count("D") == len(cat.DEAD_DIMS) == 7
    assert classes.count("E") == len(cat.DUPLICATE_DIMS) + len(cat.IDENTITY_DIMS)
    assert classes.count("C") == 0
    assert len(cat.CAUSAL_DIMS) == classes.count("A")
    for d, keep in cat.DUPLICATE_DIMS.items():
        assert cat.classify(keep) == "A", (d, keep)
    for d in (113, 114, 115):
        assert cat.classify(d) == "D"


def test_causal_features_use_only_the_same_row_and_drop_dead_and_duplicates():
    vec = [0.0] * 145
    vec[0] = 100.0
    vec[1] = 101.0  # high
    vec[38] = 2.0  # atr (duplicate of natr after transform)
    vec[4] = 1000.0  # volume
    vec[113] = 5.0  # dead proxy
    out = cat.causal_features(vec)
    assert math.isclose(out["high"], 0.01)
    assert math.isclose(out["price"], math.log(100.0))
    assert math.isclose(out["volume"], math.log1p(1000.0))
    assert "atr" not in out and "volume_imbalance" not in out and "rsi_14" not in out
    with pytest.raises(ValueError):
        cat.causal_features([0.0] * 124)


def test_profile_finds_constants_and_transformed_duplicates():
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(50):
        v = list(rng.random(145) + 1.0)
        v[83] = 0.0
        v[20] = v[19]
        rows.append(v)
    p = cat.profile({"BTCUSDT": rows, "ETHUSDT": rows})
    assert 83 in p["constant"]
    assert (19, 20) in p["duplicates"]


# --- research rows: feature_ts <= decision_ts, closed flow bar, policy split -----------


def _research_db(tmp_path):
    db = str(tmp_path / "r.db")
    version = "V_TEST"
    t = 1_791_000_000.0
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE ai_inference_log (id INTEGER PRIMARY KEY, decision_id TEXT, strategy_id TEXT, symbol TEXT, ts_utc TEXT, prob_buy REAL, prob_hold REAL, "
            "confidence REAL, buy_margin REAL, ctx_json TEXT, feature_version INTEGER, feature_dim INTEGER, features_json TEXT, model_artifact TEXT)"
        )
        for dt, price in ((-30.0, 100.0), (30.0, 999.0)):
            from datetime import UTC, datetime

            vec = [0.0] * 145
            vec[0] = price
            conn.execute(
                "INSERT INTO ai_inference_log (strategy_id, symbol, ts_utc, prob_buy, feature_version, feature_dim, features_json, model_artifact) VALUES ('day','BTCUSDT',?,0.6,5,145,?,'m.pkl')",
                (datetime.fromtimestamp(t + dt, tz=UTC).isoformat(), json.dumps(vec)),
            )
        conn.execute(
            "CREATE TABLE day_order_flow_bars (symbol TEXT, bar_open_epoch INTEGER, bar_sec INTEGER, trade_count INTEGER, imbalance REAL, "
            "notional_imbalance REAL, cvd_notional REAL, buy_notional REAL, sell_notional REAL, coverage_sec REAL)"
        )
        bar = int(t // 900 * 900)
        conn.execute("INSERT INTO day_order_flow_bars VALUES ('BTCUSDT', ?, 900, 7, 0.5, 0.4, 10, 20, 10, 800)", (bar - 900,))
        conn.execute("INSERT INTO day_order_flow_bars VALUES ('BTCUSDT', ?, 900, 99, -0.9, -0.9, -1, 1, 2, 10)", (bar,))
    for filled, realized in ((1, -0.001), (0, None)):
        rid = al.record_candidate(
            db,
            engine="DAY_V2",
            symbol="BTCUSDT",
            setup="RANGE_BOUNCE",
            regime="r",
            ref_price=100.0,
            roundtrip_cost=0.0006,
            signaled=True,
            evaluated_at=t,
            economic={"market_alpha": 0.001, "policy_gap": 0.002, "policy_value": 0.003, "rank_score": 0.003},
            features={"atr15_pct": 0.004},
        )
        with sqlite3.connect(db) as conn:
            conn.execute(
                "UPDATE adaptive_candidate_markouts SET economic_version=?, filled=?, realized_net=?, markouts_json=? WHERE id=?",
                (version, filled, realized, json.dumps({"lifecycle": {"net": 0.002, "reason": "HORIZON_MARK", "mfe": 0.01, "mae": -0.01}}), rid),
            )
    return db, version, t


def test_research_rows_are_causal_versioned_and_keep_policy_separate(tmp_path):
    db, version, _t = _research_db(tmp_path)
    rows = drr.build_day_research_rows(db, economic_version=version)
    assert len(rows) == 2
    for r in rows:
        assert r["feature_ts"] <= r["decision_ts"]
        assert math.isclose(r["features"]["price"], math.log(100.0))
        assert r["flow_bar_open"] + 900 <= r["decision_ts"]
        assert r["flow"]["flow_trades"] == 7.0
        assert r["versions"]["research"] == drr.DAY_RESEARCH_ROWS_VERSION and r["versions"]["economic_version"] == version
        assert r["state_features"] == {"atr15_pct": 0.004}
    filled = next(r for r in rows if r["selected"])
    unfilled = next(r for r in rows if not r["selected"])
    assert unfilled["realized_net"] is None and unfilled["policy_effect"] is None
    assert math.isclose(filled["market_label"] + filled["policy_effect"], filled["realized_net"])
    audit = drr.audit_rows(rows)
    assert audit["feature_after_decision"] == audit["flow_bar_unclosed"] == audit["policy_on_unfilled"] == audit["untagged"] == audit["policy_identity_broken"] == 0


def test_day_state_features_are_persisted_and_scalp_stays_micro_only(tmp_path):
    db = str(tmp_path / "f.db")
    day = al.record_candidate(
        db, engine="DAY_V2", symbol="ETHUSDT", setup="RANGE_BOUNCE", regime="r", ref_price=100.0, roundtrip_cost=0.0006, signaled=True, features={"atr15_pct": 0.003, "bad": float("nan")}
    )
    scalp = al.record_candidate(db, engine="SCALP_V2", symbol="ETHUSDT", setup="X", regime="r", ref_price=100.0, roundtrip_cost=0.0006, signaled=True, features={"atr15_pct": 0.003, "obi_l5": 0.4})
    with sqlite3.connect(db) as conn:
        got = dict(conn.execute("SELECT id, features_json FROM adaptive_candidate_markouts").fetchall())
    assert json.loads(got[day]) == {"atr15_pct": 0.003}
    assert "atr15_pct" not in json.loads(got[scalp]) and json.loads(got[scalp])["obi_l5"] == 0.4


# --- DAY lifecycle label starts when the ask was observable -------------------------


def test_lifecycle_label_ignores_the_pre_decision_part_of_the_first_minute():
    boundary = 1_791_338_400.0
    bars = [(boundary, 2657.0, 2659.66, 2645.0, 2645.5), (boundary + 60, 2645.0, 2646.0, 2620.0, 2628.0), (boundary + 120, 2628.0, 2630.0, 2627.0, 2629.0)]
    base = {"setup": "RANGE_BOUNCE", "entry_price": 2628.29, "atr_15m": 5.0, "structural_anchor": 2500.0, "target_price": 2700.0}
    at_boundary = simulate_lifecycle(LifecycleParams(entry_time=boundary, **base), bars, roundtrip_cost=0.0006, max_minutes=3, now=boundary + 10_000)
    at_ask = simulate_lifecycle(LifecycleParams(entry_time=boundary + 75.0, **base), bars, roundtrip_cost=0.0006, max_minutes=3, now=boundary + 10_000)
    assert at_boundary["mfe"] > 0.01  # the 2659.66 high printed before the ask existed
    assert at_ask["mfe"] < 0.001


def test_replay_candidate_entry_time_is_never_before_the_decision():
    c = DayCandidate("BTCUSDT", 1000.0, "RANGE_BOUNCE", "", 100.0, None, True)
    assert c.entry_time == 1000.0
    assert DayCandidate("BTCUSDT", 1000.0, "RANGE_BOUNCE", "", 100.0, None, True, ask_at=1042.0).entry_time == 1042.0
    assert DayCandidate("BTCUSDT", 1000.0, "RANGE_BOUNCE", "", 100.0, None, True, ask_at=900.0).entry_time == 1000.0


def test_live_day_candidate_stamps_the_ask_time_into_lifecycle_inputs():
    src = (ROOT / "backend/services/portfolio_engine_integration.py").read_text()
    assert "ask_at = time.time()" in src
    assert 'entry_time=float(cand.get("ask_at") or cand["as_of"])' in src


# --- book queue capture --------------------------------------------------------------


def _book(mid, n=20, qty=1.0):
    return [[mid - 0.5 - i, qty + i] for i in range(n)], [[mid + 0.5 + i, qty + i] for i in range(n)]


def test_queue_chunk_roundtrip_and_quality_counters(tmp_path):
    db = str(tmp_path / "q.db")
    cap = bq.BookQueueCapture(db)
    t0 = 1_791_000_000.0
    b, a = _book(100.0)
    cap.record("BTCUSDT", b, a, 10, ts=t0)
    for k in range(1, 500):
        b2 = [list(x) for x in b]
        b2[0][1] = 1.0 + k % 3
        cap.record("BTC", b2, a, 10 + k, ts=t0 + k * 0.1)
    cap.record("BTC", b, a, 10 + 499, ts=t0 + 50.0)  # duplicate id
    cap.record("BTC", b, a, 5, ts=t0 + 50.1)  # out of order
    cap.record("BTC", b, a, 2000, ts=t0 + 61.0)  # next minute flushes the first chunk
    cap.flush_all()
    cap.drain()
    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT n_updates, n_dup, n_out_of_order, usable, payload, first_update_id, last_update_id FROM book_queue_chunks ORDER BY chunk_start").fetchall()
    first = rows[0]
    assert first[0] == 500 and first[1] == 1 and first[2] == 1 and first[3] == 0
    snaps = bq.decode(first[4])
    assert len(snaps) == 500 and snaps[0]["update_id"] == 10 and snaps[-1]["update_id"] == 509
    assert snaps[7]["bids"][0][1] == 1.0 + 7 % 3 and snaps[-1]["asks"] == sorted(((float(p), float(q)) for p, q in a))
    assert len(zlib.decompress(first[4])) > len(first[4])


def test_usable_requires_continuity_and_uncrossed_books(tmp_path):
    cap = bq.BookQueueCapture(str(tmp_path / "u.db"))
    t0 = 1_791_000_000.0
    b, a = _book(100.0)
    for k in range(600):
        cap.record("ETH", b, a, k + 1, ts=t0 + k * 0.1)
    cap.record("ETH", b, a, 700, ts=t0 + 59.9)
    ok = cap._chunks["ETH"]
    assert ok.usable()
    cap2 = bq.BookQueueCapture(str(tmp_path / "u2.db"))
    cap2.record("ETH", b, a, 1, ts=t0)
    cap2.record("ETH", b, a, 2, ts=t0 + 1 + bq.STALE_GAP_MS / 1000.0)
    cap2.record("ETH", b, a, 3, ts=t0 + 59.0)
    assert not cap2._chunks["ETH"].usable()
    cap3 = bq.BookQueueCapture(str(tmp_path / "u3.db"))
    crossed_b = [[101.0, 1.0]]
    for k in range(600):
        cap3.record("ETH", crossed_b if k == 300 else b, a, k + 1, ts=t0 + k * 0.1)
    assert cap3._chunks["ETH"].n_crossed == 1 and not cap3._chunks["ETH"].usable()


def test_load_chunks_excludes_unusable_intervals(tmp_path):
    db = str(tmp_path / "l.db")
    cap = bq.BookQueueCapture(db)
    t0 = 1_791_000_000.0
    b, a = _book(100.0)
    for k in range(600):
        cap.record("SOL", b, a, k + 1, ts=t0 + k * 0.1)
    cap.record("SOL", b, a, 1000, ts=t0 + 60.0)  # partial chunk, unusable
    cap.record("SOL", b, a, 1001, ts=t0 + 125.0)
    cap.flush_all()
    cap.drain()
    snaps = bq.load_chunks(db, "SOLUSDT", t0, t0 + 200)
    assert snaps and all(t0 <= s["ts"] < t0 + 60 for s in snaps)
    assert len(bq.load_chunks(db, "SOL", t0, t0 + 200, usable_only=False)) > len(snaps)


def test_queue_features_measure_depletion_replenishment_and_shape():
    b, a = _book(100.0)
    s0 = {"ts": 0.0, "bids": [(p, q) for p, q in b], "asks": [(p, q) for p, q in a]}
    thin = [(b[0][0], 0.2)] + [(p, q) for p, q in b[1:]]
    s1 = {"ts": 0.1, "bids": thin, "asks": s0["asks"]}
    s2 = {"ts": 0.2, "bids": s0["bids"], "asks": s0["asks"][1:]}
    f = bq.queue_features([s0, s1, s2])
    assert f["bid_remove"] > 0 and f["bid_replenish"] == 1 and f["ask_depletions"] == 1
    assert 0 < f["bid_concentration"] <= 1 and f["bid_slope_bps"] > 0


def test_capture_hook_runs_in_existing_order_book_path_without_new_daemon():
    src = (ROOT / "backend/services/order_book_service.py").read_text()
    assert "record_depth(symbol, bids, asks, last_update_id)" in src
    assert not list((ROOT / "scripts").glob("*book_queue*"))


def test_retention_covers_new_capture_table():
    policies = {p.table: p for p in RETENTION_POLICIES}
    assert policies["book_queue_chunks"].ts_column == "chunk_start" and policies["book_queue_chunks"].keep_days <= 7
    assert "book_queue_chunks" in RETENTION_JUSTIFICATION


# --- counterfactual continuation and maker fills -----------------------------------------


def test_counterfactual_states_are_tagged_causal_and_kept_out_of_real_fills(tmp_path):
    ts = np.arange(0.0, 4000.0, 5.0)
    bid = 100.0 + 0.001 * ts
    book = {"ts": ts, "bid": bid, "ask": bid + 0.01}
    rows = cc.scalp_states(book, "BTC", [100.0], cost=0.0006)
    assert rows and all(r["kind"] == cc.CF_KIND for r in rows)
    first = rows[0]
    assert first["state_t"] == 160.0
    entry = first["entry_px"]
    assert math.isclose(first["features"]["net"], bid[ts <= 160.0][-1] / entry - 1 - 0.0006)
    assert math.isclose(first["advantage"]["30"], (bid[ts >= 190.0][0] - bid[ts <= 160.0][-1]) / entry)
    db = str(tmp_path / "cf.db")
    assert cc.write_states(db, rows) == len(rows)
    with pytest.raises(ValueError):
        cc.write_states(db, [dict(rows[0], kind=cc.REAL_KIND)])
    with sqlite3.connect(db) as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert tables == {cc.TABLE, "sqlite_sequence"}


def test_maker_fill_requires_a_print_through_the_bid():
    import importlib.util

    spec = importlib.util.spec_from_file_location("mcr", ROOT / "scripts/research/maker_counterfactual_research.py")
    mcr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mcr)
    sells = {"t": np.array([5.0, 10.0, 20.0]), "p": np.array([99.0, 100.0, 99.99])}
    assert mcr.first_trade_through(sells, 100.0, 6.0, 15.0) is None  # print at the bid is not a fill
    assert mcr.first_trade_through(sells, 100.0, 6.0, 25.0) == 20.0
    assert mcr.first_trade_through(sells, 100.0, 5.0, 25.0) == 20.0  # print at t is not after the order
    assert mcr.FILL_KIND == cc.CF_KIND


# --- research evaluation --------------------------------------------------------------


def _row(t, sym, label, group=None, exit_ts=None):
    return er.Row(t=t, group=group if group is not None else t, symbol=sym, x=np.array([label]), label=label, label_ts=exit_ts or t + 10, extra={"exit_ts": exit_ts or t + 10})


def test_fold_rows_embargo_train_labels_before_the_fold():
    rows = [_row(0, "A", 0.1, exit_ts=50), _row(10, "A", 0.1, exit_ts=200), _row(100, "A", 0.1)]
    train, test = er.fold_rows(rows, 100, 1000)
    assert [r.t for r in train] == [0] and [r.t for r in test] == [100]


def test_ranking_regret_and_closed_loop_slots():
    rows = [_row(0, "A", 0.01, group=0), _row(0, "B", -0.02, group=0), _row(1, "A", 0.03, group=1, exit_ts=5), _row(1, "B", 0.0, group=1)]
    m = er.ranking_metrics(rows, [1.0, 0.0, 0.0, 1.0])
    assert math.isclose(m["chosen"], 0.005) and math.isclose(m["best"], 0.02) and math.isclose(m["regret"], 0.015)
    book = er.closed_loop(rows, [1.0, -1.0, 1.0, 1.0], max_slots=4)
    assert book["trades"] == 2  # A is still held at t=1; B at t=1 enters
    assert math.isclose(book["net"], 0.01)


def test_online_sgd_never_learns_a_label_before_it_is_final():
    rows = [_row(float(t), "A", 1.0, exit_ts=float(t) + 100) for t in range(0, 50)]
    pred = er.online_sgd(rows)
    assert np.allclose(pred, 0.0)


# --- geometric claim is a feature, never a forecast -----------------------------------------


def test_cold_scalp_edge_prices_any_geometric_claim_at_minus_cost():
    from backend.services.scalp_v2.executable_edge import scalp_executable_edge

    for raw in (0.0, 0.0017, 0.05):
        e = scalp_executable_edge({}, raw_expected_move_pct=raw, spread_pct=0.0001, impact_pct=0.0, edge_source="STRATEGY_CLAIM")
        assert e.calibrated_move_pct == 0.0 and e.claim_capture == 0.0
        assert math.isclose(e.final_executable_edge_pct, -e.live_cost_pct)


def test_status_telemetry_names_the_claim_as_geometry():
    src = (ROOT / "backend/services/binance_scalp/status_snapshot.py").read_text()
    assert '"geometric_target_distance_pct"' in src and '"expected_move_pct": best_ranked_row' not in src


def test_research_modules_hold_no_live_authority():
    research = ("edge_research", "day_research_rows", "counterfactual_continuation", "day_feature_catalog", "research_extract")
    for live in ("portfolio_engine_integration.py", "adaptive_learning.py", "scalp_v2/executable_edge.py", "day_v2/ranking.py", "portfolio_engine.py"):
        src = (ROOT / "backend/services" / live).read_text()
        for name in research:
            assert name not in src, (live, name)


# --- rebuild linkage -----------------------------------------------------------------------


def test_link_closes_prefers_lineage_candidate_and_maps_repeated_fills_in_order():
    rows = [
        {"id": 1, "opportunity_id": "O", "filled": 1, "symbol": "X", "signaled": 1, "evaluated_at": 0.0},
        {"id": 2, "opportunity_id": "O", "filled": 1, "symbol": "X", "signaled": 1, "evaluated_at": 900.0},
        {"id": 3, "opportunity_id": "P", "filled": 1, "symbol": "X", "signaled": 1, "evaluated_at": 1800.0},
    ]
    closes = [
        {"opportunity_id": "O", "candidate_id": None, "symbol": "X", "entered_at": 10.0},
        {"opportunity_id": "O", "candidate_id": None, "symbol": "X", "entered_at": 910.0},
        {"opportunity_id": "", "candidate_id": 3, "symbol": "X", "entered_at": 1810.0},
    ]
    assert esr.link_closes("DAY_V2", rows, closes) == {0: 1, 1: 2, 2: 3}


def test_close_rows_link_by_identity_when_the_exit_stamp_trails_the_sell(tmp_path):
    db = str(tmp_path / "p.db")
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE paper_trades (trade_id TEXT, decision_id TEXT, scalp_opportunity_id TEXT, adaptive_decision_json TEXT, side TEXT, engine_id TEXT, symbol TEXT, timestamp TEXT)")
        lineage = json.dumps({"lineage": {"candidate_id": 32806}})
        conn.execute("INSERT INTO paper_trades VALUES ('B1','D1','', ?, 'BUY','DAY_V2','XRPUSDT','2026-10-07 01:32:12')", (lineage,))
        conn.execute("INSERT INTO paper_trades VALUES ('S1','D1','', ?, 'SELL','DAY_V2','XRPUSDT','2026-10-07 01:33:28')", (json.dumps({"close_lineage": {"entry": {"candidate_id": 32806}}}),))
    from datetime import UTC, datetime

    closed = datetime(2026, 10, 7, 1, 34, 0, tzinfo=UTC).timestamp()
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        buy, sell = esr._close_rows(conn, "DAY_V2", "XRPUSDT", "B1", closed)
    assert buy["trade_id"] == "B1" and sell["trade_id"] == "S1"
    assert esr._lineage_candidate(json.loads(sell["adaptive_decision_json"]), json.loads(buy["adaptive_decision_json"])) == 32806


def test_research_extract_spec_reads_capture_table_columns():
    from backend.services.research_extract import table_specs

    spec = next(s for s in table_specs() if s.table == "book_queue_chunks")
    assert "chunk_start" in spec.where and "start_ts" not in spec.where


def test_research_scripts_import_cleanly():
    import importlib.util

    for name in ("day_edge_research", "scalp_move_research", "maker_counterfactual_research", "continuation_counterfactual_research"):
        spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/research" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert callable(mod.main)
    assert time.time() > 0
