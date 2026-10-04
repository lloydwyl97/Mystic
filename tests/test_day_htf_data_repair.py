"""Frozen DAY 1d/1w repair: source depth, bounded reuse, stale protection, rebuild, holdout, logs."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from pathlib import Path

import pytest

from backend.config.canonical_candle_intervals import align_open_ms, interval_ms
from backend.config.day_active_timeframes import DAY_ACTIVE_TIMEFRAMES, min_bars_for_day_tf
from backend.services import canonical_candle_pipeline as ccp
from backend.services import day_htf_feature_rebuild as rb
from backend.services.day_active_market_bundle import (
    HTF_STATE_FRESH,
    HTF_STATE_REUSED,
    HTF_STATE_STALE,
    _fetch_day_active_ohlcv_bundle_raw,
    completed_rows_asof,
    htf_max_data_age_sec,
    validate_day_active_bundle,
)

REPO = Path(__file__).resolve().parents[1]
DAY_MS = 86_400_000


def _bars(n: int, tf: str, *, end_open_ms: int | None = None, base: float = 100.0, step: float = 0.3) -> list[list]:
    width = interval_ms(tf)
    if end_open_ms is None:
        end_open_ms = align_open_ms(int(time.time() * 1000), tf) - width
    out = []
    for i in range(n):
        c = base + step * i + (i % 7) * 0.11
        out.append([end_open_ms - (n - 1 - i) * width, c - 0.2, c + 0.5, c - 0.6, c, 10.0 + i])
    return out


# ---------------------------------------------------------------- source depth


class _MemStore:
    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], dict[int, dict]] = {}

    def load(self, symbol, interval, *, start_ms=None, end_ms=None, limit=None):
        data = self.rows.get((ccp.api_symbol(symbol), interval), {})
        out = [data[k] for k in sorted(data) if (start_ms is None or k >= start_ms) and (end_ms is None or k <= end_ms)]
        return out[-limit:] if limit else out

    def upsert(self, symbol, interval, completed):
        data = self.rows.setdefault((ccp.api_symbol(symbol), interval), {})
        for r in completed:
            data[int(r["open_ms"])] = r
        return {"inserted": len(completed), "updated": 0, "skipped_forming": 0}


def _exchange_fetch(history: dict[str, list[list]], page_cap: int, calls: list):
    async def fetch(*, symbol, interval, start_ms, end_ms, limit):
        calls.append((symbol, interval, start_ms, end_ms))
        rows = [k for k in history[interval] if start_ms <= k[0] <= end_ms]
        return [[k[0], str(k[1]), str(k[2]), str(k[3]), str(k[4]), str(k[5]), k[0] + interval_ms(interval) - 1] for k in rows[: min(limit, page_cap)]]

    return fetch


@pytest.mark.asyncio
async def test_htf_depth_backfill_paginates_to_bundle_depth(monkeypatch):
    store = _MemStore()
    monkeypatch.setattr(ccp, "load_aligned_candles", store.load)
    monkeypatch.setattr(ccp, "upsert_completed_candles", store.upsert)

    async def _no_redis(*a, **k):
        return None

    pipe = ccp.CanonicalCandlePipeline()
    monkeypatch.setattr(pipe, "publish_redis", _no_redis)
    monkeypatch.setattr(ccp.asyncio, "sleep", _no_redis)
    history = {"1d": _bars(500, "1d"), "1w": _bars(150, "1w")}
    # Store starts with only the canonical-window tail (59 daily, 7 weekly), as in production.
    store.upsert("BTCUSDT", "1d", [{"open_ms": k[0]} for k in history["1d"][-59:]])
    store.upsert("BTCUSDT", "1w", [{"open_ms": k[0]} for k in history["1w"][-7:]])
    calls: list = []
    pipe.set_fetch_fn(_exchange_fetch(history, page_cap=150, calls=calls))

    out = await pipe.ensure_htf_history_depth(["BTCUSDT"], force=True)

    by_tf = {r["interval"]: r for r in out}
    assert by_tf["1d"]["before"] == 59 and by_tf["1d"]["after"] >= ccp.htf_history_depth("1d")
    assert by_tf["1w"]["before"] == 7 and by_tf["1w"]["after"] >= ccp.htf_history_depth("1w")
    assert by_tf["1d"]["pages"] >= 3, "1d depth must page past the per-call cap"
    newest_closed_1d = align_open_ms(int(time.time() * 1000), "1d") - DAY_MS
    assert max(store.rows[("BTCUSDT", "1d")]) == newest_closed_1d

    calls.clear()
    again = await pipe.ensure_htf_history_depth(["BTCUSDT"])
    assert {r["action"] for r in again} == {"ok"} and not calls


@pytest.mark.asyncio
async def test_htf_depth_short_exchange_history_retries_later(monkeypatch):
    store = _MemStore()
    monkeypatch.setattr(ccp, "load_aligned_candles", store.load)
    monkeypatch.setattr(ccp, "upsert_completed_candles", store.upsert)

    async def _noop(*a, **k):
        return None

    pipe = ccp.CanonicalCandlePipeline()
    monkeypatch.setattr(pipe, "publish_redis", _noop)
    monkeypatch.setattr(ccp.asyncio, "sleep", _noop)
    calls: list = []
    pipe.set_fetch_fn(_exchange_fetch({"1d": _bars(30, "1d"), "1w": _bars(5, "1w")}, page_cap=1000, calls=calls))
    await pipe.ensure_htf_history_depth(["ETHUSDT"])
    n = len(calls)
    out = await pipe.ensure_htf_history_depth(["ETHUSDT"])
    assert {r["action"] for r in out} == {"retry_later"} and len(calls) == n


# ---------------------------------------------------------------- bounded reuse


class _Svc:
    def __init__(self, live: dict[str, list]):
        self.live = live

    async def get_ohlcv(self, symbol, tf, lim):
        return list(self.live.get(tf) or [])


def _full_prior() -> dict[str, list[list]]:
    return {tf: _bars(min_bars_for_day_tf(tf) + 5, tf) for tf in DAY_ACTIVE_TIMEFRAMES}


@pytest.mark.asyncio
async def test_recent_prior_htf_reused_with_original_age():
    prior = _full_prior()
    now = time.time()
    prior_ts = dict.fromkeys(DAY_ACTIVE_TIMEFRAMES, now - 4000.0)
    live = dict(prior)
    live["1d"] = prior["1d"][-59:]
    meta: dict = {}
    bundle, ts = await _fetch_day_active_ohlcv_bundle_raw(_Svc(live), "BTC/USDT", prior_bundle=prior, prior_tf_fetched_at=prior_ts, tf_meta_out=meta)
    assert validate_day_active_bundle(bundle)[0]
    assert meta["1d"]["state"] == HTF_STATE_REUSED
    assert ts["1d"] == pytest.approx(now - 4000.0), "reuse must not reset the copy's age to now"
    assert meta["1d"]["data_age_sec"] <= htf_max_data_age_sec("1d")
    assert meta["1d"]["last_bar_open_ms"] == prior["1d"][-1][0]


@pytest.mark.asyncio
async def test_frozen_sept18_prior_is_stale_and_never_fed():
    frozen_end = int(time.mktime((2026, 9, 17, 0, 0, 0, 0, 0, 0)) * 1000)
    if frozen_end > int(time.time() * 1000) - 3 * DAY_MS:
        frozen_end = align_open_ms(int(time.time() * 1000), "1d") - 16 * DAY_MS
    prior = _full_prior()
    prior["1d"] = _bars(400, "1d", end_open_ms=align_open_ms(frozen_end, "1d"))
    prior_ts = dict.fromkeys(DAY_ACTIVE_TIMEFRAMES, time.time() - 30.0)
    prior_ts["1d"] = frozen_end / 1000.0 + 3600
    live = dict(prior)
    live["1d"] = _bars(59, "1d")
    meta: dict = {}
    bundle, _ = await _fetch_day_active_ohlcv_bundle_raw(_Svc(live), "BTC/USDT", prior_bundle=prior, prior_tf_fetched_at=prior_ts, tf_meta_out=meta)
    assert meta["1d"]["state"] == HTF_STATE_STALE
    assert meta["1d"]["data_age_sec"] > htf_max_data_age_sec("1d")
    assert len(bundle["1d"]) < min_bars_for_day_tf("1d")
    ok, miss = validate_day_active_bundle(bundle)
    assert not ok and any("1d" in m for m in miss)


@pytest.mark.asyncio
async def test_fetched_rows_with_old_last_bar_are_stale():
    prior = {tf: [] for tf in DAY_ACTIVE_TIMEFRAMES}
    old_end = align_open_ms(int(time.time() * 1000), "1w") - 6 * interval_ms("1w")
    live = _full_prior()
    live["1w"] = _bars(120, "1w", end_open_ms=old_end)
    meta: dict = {}
    bundle, _ = await _fetch_day_active_ohlcv_bundle_raw(_Svc(live), "BTC/USDT", prior_bundle=prior, prior_tf_fetched_at={}, tf_meta_out=meta)
    assert meta["1w"]["state"] == HTF_STATE_STALE and not bundle["1w"]
    assert meta["1d"]["state"] == HTF_STATE_FRESH and meta["1d"]["source"]


def test_asof_bundle_keeps_completed_bars_only():
    rows = _bars(10, "1d", end_open_ms=10 * DAY_MS)
    t = 10 * DAY_MS + 5 * 3_600_000
    kept = completed_rows_asof("1d", rows, t)
    assert kept[-1][0] == 9 * DAY_MS and all(r[0] + DAY_MS <= t + 1 for r in kept)


def test_asof_reads_skip_live_cache_and_stale_fallback():
    src = (REPO / "backend/services/live_market_data.py").read_text()
    body = src[src.index("async def get_ohlcv_with_meta") :]
    body = body[: body.index("\n    async def ", 10)]
    assert "self._stale_ohlcv_fallback(cache_key) if end_time_ms is None else None" in body
    assert re.search(r"if end_time_ms is None:\s*\n\s+self\._store_ohlcv_cache", body)


# ---------------------------------------------------------------- live DAY unaffected


def test_day_v2_rule_entry_and_exit_do_not_read_ml_bundle():
    for name in ("live_signal.py", "live_exit_evaluator.py", "ranking.py", "structural_entry.py"):
        text = (REPO / "backend/services/day_v2" / name).read_text()
        assert "day_active_market_bundle" not in text, name
        assert "build_day_htf_feature_vector_145" not in text, name


# ---------------------------------------------------------------- rebuild


def _frozen_world():
    boundary = align_open_ms(int(time.time() * 1000), "1w") - 3 * interval_ms("1w") + 2 * DAY_MS + 35 * 60_000
    d1 = _bars(rb.DEPTH_1D + 40, "1d", end_open_ms=align_open_ms(int(time.time() * 1000), "1d") - DAY_MS, step=0.4)
    w1 = _bars(rb.DEPTH_1W + 10, "1w", end_open_ms=align_open_ms(int(time.time() * 1000), "1w") - interval_ms("1w"), step=2.0)
    forming_d = rb.day_open_ms(boundary)
    forming_w = rb.week_open_ms(boundary)
    done_d = rb.completed_asof(d1, "1d", boundary, rb.DEPTH_1D)
    done_w = rb.completed_asof(w1, "1w", boundary, rb.DEPTH_1W)
    frozen_d = [*done_d[1:], [forming_d, 0, 0, 0, done_d[-1][4] * 1.0123, 0]]
    frozen_w = [*done_w[1:], [forming_w, 0, 0, 0, done_w[-1][4] * 0.987, 0]]
    vals = rb.htf_dim_values(frozen_d, frozen_w)
    vec = [0.01 * (i % 13) for i in range(145)]
    vec[rb.DIM_SLOPE_4H] = 0.0
    for d, v in vals.items():
        if d != rb.DIM_MEAN_EMA:
            vec[d] = v
    others = [1.0, 0.5, 1.0, 1.0, 0.5, 1.0, 1.0]
    vec[rb.DIM_MEAN_EMA] = (sum(others) + rb.ema_term(frozen_d) + rb.ema_term(frozen_w)) / rb.EMA_TERM_COUNT
    return boundary, d1, w1, vec


def _db(boundary: int, vec: list[float]) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE ai_inference_log (id INTEGER PRIMARY KEY, decision_id TEXT, strategy_id TEXT, symbol TEXT, ts_utc TEXT,
            feature_version TEXT, feature_dim INTEGER, features_json TEXT, ctx_json TEXT, model_artifact TEXT);
        CREATE TABLE ai_outcome_training_rows (id INTEGER PRIMARY KEY, symbol TEXT, strategy_id TEXT, opened_at_utc TEXT,
            closed_at_utc TEXT, net_pnl REAL, features_json TEXT, context_json TEXT);
        CREATE TABLE paper_trades (id INTEGER PRIMARY KEY, symbol TEXT, side TEXT, qty REAL, price REAL);
        """
    )
    clean = list(vec)
    clean[rb.DIM_CHANGE_24H] += 0.5
    pre_freeze = list(vec)
    pre_freeze[rb.DIM_SLOPE_1D] += 0.003
    # First post-boundary row of a symbol that had not frozen yet.
    rows = [(1, f"day_BTCUSDT_{boundary - 600_000}", clean), (2, f"day_BTCUSDT_{boundary + 30_000}", pre_freeze)]
    for i in range(6):
        rows.append((3 + i, f"day_BTCUSDT_{boundary + 60_000 + i * 2 * DAY_MS}", vec))
    for rid, did, v in rows:
        t = int(did.rsplit("_", 1)[-1])
        conn.execute(
            "INSERT INTO ai_inference_log VALUES (?,?,?,?,?,?,?,?,?,?)",
            (rid, did, "day", "BTCUSDT", rb._ms_to_iso(t), "day_htf_145_v1", 145, json.dumps(v), json.dumps({"k": 1}), "rf_v1"),
        )
    conn.execute(
        "INSERT INTO ai_outcome_training_rows VALUES (1,'BTC/USDT','day',?,?,1.5,?,?)",
        (rb._ms_to_iso(boundary + 60_000 + 2 * DAY_MS), rb._ms_to_iso(boundary + 3 * DAY_MS), json.dumps(vec), "{}"),
    )
    conn.execute("INSERT INTO paper_trades VALUES (1,'BTCUSDT','BUY',0.1,60000)")
    conn.commit()
    return conn


def test_rebuild_reconstructs_frozen_copy_and_is_deterministic():
    boundary, d1, w1, vec = _frozen_world()
    args = {"bars_1d": {"BTCUSDT": d1}, "bars_1w": {"BTCUSDT": w1}, "boundary_ms": boundary}
    a = _db(boundary, vec)
    b = _db(boundary, vec)
    dry = rb.rebuild_contaminated_rows(a, **args)
    assert dry["symbols"]["BTCUSDT"]["daily_ok"] and dry["symbols"]["BTCUSDT"]["frozen_1w_copies"] == 1
    assert dry["inference"]["contaminated"] == 6 and dry["inference"]["rebuildable"] == 6
    assert dry["training"]["contaminated"] == 1 and dry["training"]["rebuildable"] == 1
    assert a.execute("SELECT features_json FROM ai_inference_log WHERE id=3").fetchone()[0] == json.dumps(vec), "dry run writes nothing"

    ra = rb.rebuild_contaminated_rows(a, apply=True, **args)
    rb.rebuild_contaminated_rows(b, apply=True, **args)
    assert ra["applied"] == {"inference": 6, "training": 1}
    fa = a.execute("SELECT features_json FROM ai_inference_log ORDER BY id").fetchall()
    assert fa == b.execute("SELECT features_json FROM ai_inference_log ORDER BY id").fetchall()

    out = [json.loads(r[0]) for r in fa]
    assert out[0][rb.DIM_CHANGE_24H] == pytest.approx(vec[rb.DIM_CHANGE_24H] + 0.5), "pre-boundary row untouched"
    changed = {i for i in range(145) if out[3][i] != vec[i]}
    assert changed and changed <= set(rb.HTF_DERIVED_DIMS)
    assert out[1][rb.DIM_SLOPE_1D] == pytest.approx(vec[rb.DIM_SLOPE_1D] + 0.003), "pre-freeze row untouched"
    assert out[2][rb.DIM_CHANGE_7D] != out[7][rb.DIM_CHANGE_7D], "rows now move with their own timestamps"
    t_ms = boundary + 60_000 + 2 * DAY_MS
    want = rb.htf_dim_values(rb.completed_asof(d1, "1d", t_ms, rb.DEPTH_1D), rb.completed_asof(w1, "1w", t_ms, rb.DEPTH_1W))
    for d in (rb.DIM_CHANGE_24H, rb.DIM_SLOPE_1D, rb.DIM_SLOPE_1W, rb.DIM_MONTH_LOG_RET):
        assert out[3][d] == pytest.approx(want[d])
    tr = a.execute("SELECT features_json, context_json, net_pnl, closed_at_utc FROM ai_outcome_training_rows").fetchone()
    assert json.loads(tr[0]) == out[3] and "_htf_rebuild" in tr[1] and tr[2] == 1.5
    assert a.execute("SELECT * FROM paper_trades").fetchall() == [(1, "BTCUSDT", "BUY", 0.1, 60000.0)]
    assert json.loads(a.execute("SELECT ctx_json FROM ai_inference_log WHERE id=3").fetchone()[0])["k"] == 1

    again = rb.rebuild_contaminated_rows(a, apply=True, **args)
    assert again.get("applied", {}).get("inference", 0) == 0, "marked rows are never rebuilt twice"


def test_legacy_4h_rows_move_to_current_contract_only_when_reproduced():
    boundary, d1, w1, vec = _frozen_world()
    h4 = _bars(rb.DEPTH_4H + 40, "4h", end_open_ms=align_open_ms(int(time.time() * 1000), "4h") - interval_ms("4h"), step=0.05)
    t_ms = boundary + 60_000
    rows4 = rb.completed_asof(h4, "4h", t_ms, rb.DEPTH_4H)
    e4h = rb.ema_term(rows4)
    legacy = list(vec)
    legacy[rb.DIM_SLOPE_4H] = rb._slope_norm(rb._summarize_tf(rows4)["slope"])
    assert legacy[rb.DIM_SLOPE_4H] != 0.0
    legacy[rb.DIM_MEAN_EMA] = (vec[rb.DIM_MEAN_EMA] * rb.EMA_TERM_COUNT + e4h) / (rb.EMA_TERM_COUNT + 1)
    conn = _db(boundary, vec)
    conn.execute("UPDATE ai_inference_log SET features_json=? WHERE id=3", (json.dumps(legacy),))
    bad = list(legacy)
    bad[rb.DIM_SLOPE_4H] += 0.001
    conn.execute("UPDATE ai_inference_log SET features_json=? WHERE id=4", (json.dumps(bad),))
    conn.commit()
    rep = rb.rebuild_contaminated_rows(conn, bars_1d={"BTCUSDT": d1}, bars_1w={"BTCUSDT": w1}, bars_4h={"BTCUSDT": h4}, boundary_ms=boundary, apply=True)
    assert rep["legacy_4h_converted"] == 1 and rep["legacy_4h_kept"] == 1
    assert rep["failure_reasons"] == {}
    got = json.loads(conn.execute("SELECT features_json FROM ai_inference_log WHERE id=3").fetchone()[0])
    plain = _db(boundary, vec)
    rb.rebuild_contaminated_rows(plain, bars_1d={"BTCUSDT": d1}, bars_1w={"BTCUSDT": w1}, boundary_ms=boundary, apply=True)
    ref = json.loads(plain.execute("SELECT features_json FROM ai_inference_log WHERE id=3").fetchone()[0])
    assert got[rb.DIM_SLOPE_4H] == 0.0
    assert got == pytest.approx(ref), "converted legacy row equals the current-contract rebuild"
    marker = json.loads(conn.execute("SELECT ctx_json FROM ai_inference_log WHERE id=3").fetchone()[0])["_htf_rebuild"]
    assert marker["legacy_4h_converted"] is True and rb.DIM_SLOPE_4H in marker["dims"]
    kept = json.loads(conn.execute("SELECT features_json FROM ai_inference_log WHERE id=4").fetchone()[0])
    assert kept[rb.DIM_SLOPE_4H] == bad[rb.DIM_SLOPE_4H], "unreproducible 4h input stays as built"
    t4 = boundary + 60_000 + 2 * DAY_MS
    want = rb.htf_dim_values(rb.completed_asof(d1, "1d", t4, rb.DEPTH_1D), rb.completed_asof(w1, "1w", t4, rb.DEPTH_1W))
    assert kept[rb.DIM_CHANGE_24H] == pytest.approx(want[rb.DIM_CHANGE_24H]), "frozen 1d values still replaced"
    plain4 = json.loads(plain.execute("SELECT features_json FROM ai_inference_log WHERE id=4").fetchone()[0])
    assert kept[rb.DIM_MEAN_EMA] == pytest.approx((plain4[rb.DIM_MEAN_EMA] * rb.EMA_TERM_COUNT + e4h) / (rb.EMA_TERM_COUNT + 1))
    kmark = json.loads(conn.execute("SELECT ctx_json FROM ai_inference_log WHERE id=4").fetchone()[0])["_htf_rebuild"]
    assert kmark["legacy_4h_kept"] is True and kmark["legacy_4h_converted"] is False


def test_rebuild_refuses_when_frozen_copy_cannot_be_validated():
    boundary, d1, w1, vec = _frozen_world()
    vec = list(vec)
    vec[rb.DIM_SLOPE_1D] += 0.01
    conn = _db(boundary, vec)
    rep = rb.rebuild_contaminated_rows(conn, bars_1d={"BTCUSDT": d1}, bars_1w={"BTCUSDT": w1}, boundary_ms=boundary, apply=True)
    assert rep["symbols"]["BTCUSDT"]["daily_ok"] is False
    assert rep["applied"]["inference"] == 0 and rep["inference"]["failed"] == 6


# ---------------------------------------------------------------- training cache


def _history(anchor_open: int) -> dict[str, list[list]]:
    from backend.config.day_active_timeframes import fetch_limit_for_day_tf

    end = anchor_open + 3 * interval_ms("1w")
    hist = {}
    for k, tf in enumerate(DAY_ACTIVE_TIMEFRAMES):
        width = interval_ms(tf)
        n = fetch_limit_for_day_tf(tf) + (end - anchor_open) // width + 40
        hist[tf] = _bars(n, tf, end_open_ms=align_open_ms(end, tf), base=50.0 + k, step=0.01 + 0.002 * k)
    return hist


def test_cache_asof_bundle_keeps_completed_bars_only():
    from backend.services import day_training_cache_rebuild as cr

    anchor = align_open_ms(1_790_000_000_000, "4h")
    hist = _history(anchor)
    end = cr.anchor_asof_ms(anchor)
    bundle = cr.asof_bundle_from_history(hist, end)
    for tf, rows in bundle.items():
        assert rows and all(int(r[0]) + interval_ms(tf) <= end + 1 for r in rows), tf
    assert bundle["4h"][-1][0] == anchor


def test_cache_rebuild_removes_forming_bar_lookahead_and_keeps_labels():
    from backend.services import day_training_cache_rebuild as cr
    from backend.services.ai_day_htf_features import build_day_htf_feature_vector_145

    anchor = align_open_ms(1_790_000_000_000, "4h") + 4 * 3_600_000
    hist = _history(anchor)
    end = cr.anchor_asof_ms(anchor)
    clean = cr.asof_bundle_from_history(hist, end)
    leaky = {tf: list(rows) for tf, rows in clean.items()}
    for tf in ("8h", "12h", "1d", "1w"):
        forming = next(r for r in hist[tf] if r[0] <= end < r[0] + interval_ms(tf))
        leaky[tf] = [*leaky[tf][1:], [forming[0], forming[1], forming[2] * 1.2, forming[3], forming[4] * 1.15, forming[5]]]
    assert validate_day_active_bundle(leaky)[0]
    leaky_feats = build_day_htf_feature_vector_145(symbol_ccxt="BTC/USDT", day_bundle=leaky, volume_profile=None, orderbook=None, sentiment=None, ai_context={})
    assert validate_day_active_bundle(clean)[0]
    want = build_day_htf_feature_vector_145(symbol_ccxt="BTC/USDT", day_bundle=clean, volume_profile=None, orderbook=None, sentiment=None, ai_context={})
    assert leaky_feats != want
    sample = {"features": leaky_feats, "label_anchor_4h_open_ms": anchor, "label_anchor_close": 123.0, "timestamp": "t", "feature_version": 5}

    out, rep = cr.rebuild_cache_samples([sample, dict(sample)], hist, "BTC/USDT")
    assert rep["rebuilt"] == 2 and rep["lower_tf_match"] == 2
    assert out[0]["features"] == want and out[1]["features"] == want
    assert out[0]["label_anchor_close"] == 123.0 and out[0]["label_anchor_4h_open_ms"] == anchor
    assert out[0]["htf_rebuild"]["asof_ms"] == end
    assert set(rep["changed_dims"]) & {5, 6, 7, 132, 133}
    again, rep2 = cr.rebuild_cache_samples(out, hist, "BTC/USDT")
    assert rep2["already"] == 2 and again == out


# ---------------------------------------------------------------- holdout


def test_holdout_excludes_rows_the_incumbent_trained_on(tmp_path):
    import pickle

    from backend.services import ai_model_promotion_holdout as h

    def _art(path: Path, **meta) -> Path:
        path.write_bytes(pickle.dumps({"model": object.__name__, "scaler": "s", **meta}))
        return path

    active = _art(tmp_path / "day_BTCUSDT.pkl", trained_at="2026-09-27T16:22:00+00:00", train_outcome_max_id=40)
    cut = h.incumbent_seen_cutoff(active)
    assert cut["seen_max_id"] == 40 and cut["trained_at"]
    seen_by_time = {"id": 90, "closed_at_utc": "2026-09-27T10:00:00+00:00"}
    seen_by_id = {"id": 39, "closed_at_utc": "2026-09-30T10:00:00+00:00"}
    unseen = {"id": 91, "closed_at_utc": "2026-09-28T10:00:00+00:00"}
    assert not h._row_unseen_by_incumbent(seen_by_time, cut)
    assert not h._row_unseen_by_incumbent(seen_by_id, cut)
    assert h._row_unseen_by_incumbent(unseen, cut)
    assert h._row_unseen_by_incumbent(seen_by_time, {})

    stamped = _art(tmp_path / "day_ETHUSDT.pkl", trained_at="2026-09-26T09:06:00+00:00", train_outcome_max_id=12, holdout_window={"max_id": 77})
    assert h.incumbent_seen_cutoff(stamped)["seen_max_id"] == 77
    assert h.incumbent_seen_cutoff(tmp_path / "missing.pkl") == {}
    assert h.incumbent_seen_cutoff(None) == {}


def test_training_exclusion_and_validation_use_same_incumbent_cutoff():
    src = (REPO / "backend/ai_training_pipeline.py").read_text()
    seg = src[src.index("def _exclude_promotion_holdout") :]
    seg = seg[: seg.index("\ndef ", 5)]
    assert "active_path=" in seg
    hsrc = (REPO / "backend/services/ai_model_promotion_holdout.py").read_text()
    seg2 = hsrc[hsrc.index("def build_holdout_validation_metrics") :]
    assert seg2.count("active_path=") >= 2


# ---------------------------------------------------------------- logs


LOG_NAMES = ("backend", "live_md", "signal", "portfolio", "learning", "ai_context")


def test_start_script_appends_to_all_six_logs():
    text = (REPO / "start_mystic.sh").read_text()
    assert not re.search(r"[^>]> /home/mystic/mystic/logs/", text)
    for name in LOG_NAMES:
        assert f">> /home/mystic/mystic/logs/mystic_{name}.log" in text
    assert text.count("9>&-") >= 6


def test_append_redirect_leaves_no_holes_after_copytruncate(tmp_path):
    def _cycle(flags: int) -> tuple[int, bytes]:
        p = tmp_path / f"log_{flags}"
        p.write_bytes(b"")
        fd = os.open(p, flags)
        os.write(fd, b"x" * 65536)
        os.truncate(p, 0)  # logrotate copytruncate
        os.write(fd, b"after\n")
        os.close(fd)
        return p.stat().st_size, p.read_bytes()

    size, data = _cycle(os.O_WRONLY | os.O_APPEND)
    assert size == 6 and b"\x00" not in data
    size_w, data_w = _cycle(os.O_WRONLY)
    assert size_w > 65536 and data_w.startswith(b"\x00")


def test_preserved_log_keeps_previous_content(tmp_path):
    p = tmp_path / "mystic_portfolio.log"
    p.write_text("previous run\n")
    with open(p, "a") as fh:
        fh.write("new run\n")
    assert p.read_text() == "previous run\nnew run\n"


def test_portfolio_and_learning_logs_stamp_full_utc_date():
    import logging

    for rel in ("start_portfolio_engine_integration.py", "start_ai_learning.py"):
        text = (REPO / rel).read_text()
        m = re.search(r'"(%Y-%m-%dT%H:%M:%SZ)"', text)
        assert m, rel
        assert "time.gmtime" in text, rel
    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ")
    fmt.converter = time.gmtime
    rec = logging.LogRecord("x", logging.INFO, __file__, 1, "m", None, None)
    rec.created = 1791130147.0
    assert fmt.format(rec).startswith("2026-10-04T16:09:07Z")


# ---------------------------------------------------------------- protected no-fill label


class _Eng:
    last_buy_reject_reason = ""
    last_buy_failure_reason = ""


def test_protected_limit_no_fill_label_is_specific():
    from backend.services.day_v2.live_entry import day_v2_submit_reject_reason

    eng = _Eng()
    assert day_v2_submit_reject_reason(eng) == "UNSPECIFIED"
    eng.last_buy_failure_reason = "PROTECTED_LIMIT_BUY_NOT_FILLED"
    assert day_v2_submit_reject_reason(eng) == "PROTECTED_LIMIT_BUY_NOT_FILLED"
    eng2 = _Eng()
    eng2.last_buy_reject_reason = "MIN_NOTIONAL"
    assert day_v2_submit_reject_reason(eng2) == "MIN_NOTIONAL"


def test_no_fill_label_does_not_touch_execution_reason():
    src = (REPO / "backend/services/portfolio_engine.py").read_text()
    sites = [m.start() for m in re.finditer(r'error_msg = "PROTECTED_LIMIT_BUY_NOT_FILLED"', src)]
    assert len(sites) == 2
    for s in sites:
        window = src[s : s + 400]
        assert "self.last_buy_failure_reason = error_msg" in window
        assert "last_buy_reject_reason" not in window, "trailing-buy retry classification must stay unchanged"
