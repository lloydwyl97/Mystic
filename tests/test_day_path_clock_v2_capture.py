"""Decision-time clock-v2 capture. Observability only. No live ranking."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from backend.services.day_decision_observability import (
    build_group_contract,
    record_day_ranking_group,
)
from backend.services.day_direct_path_ev_authority import select_action
from backend.services.day_forward_lock import TABLE as TABLE_LOCK
from backend.services.day_forward_lock import register_lock
from backend.services.day_path_clock_pipeline import FORBIDDEN_OUTCOME_KEYS
from backend.services.day_path_clock_v2_capture import (
    CANDIDATE_INELIGIBLE,
    FEATURE_OHLCV_SOURCE,
    KLINE_SOURCE,
    NO_QUOTE,
    NOT_COMPUTED_FOR_CANDIDATE,
    NOT_PERSISTED,
    SOURCE_DATA_GAP,
    SOURCE_DATA_STALE,
    TABLE_ARTIFACT,
    artifact_fingerprint,
    build_candidate_artifact,
    capture_clock_v2_group,
    classify_historical_clock,
    classify_historical_p_buy,
    classify_historical_spread,
    group_completeness,
    recompute_spread_bps,
)
from backend.services.day_path_input_validity import MAX_GAP_SEC, MAX_LAST_BAR_AGE_SEC
from backend.services.day_path_net import predict_decision_net, reset_day_artifact_cache, resolve_day_path_ev

_COINS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")


def _book(*_args, **_kwargs):
    return _quote("BTCUSDT")


class FakeRedis:
    def __init__(self, hashes=None, strings=None):
        self.hashes = hashes or {}
        self.strings = strings or {}

    def hgetall(self, key):
        return dict(self.hashes.get(key) or {})

    def get(self, key):
        return self.strings.get(key)


def _dense_klines(n: int = 90, start: datetime | None = None, close0: float = 100.0) -> list[list[float]]:
    t0 = start or datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc)
    rows = []
    px = close0
    for i in range(n):
        px *= 1.00015
        ts = t0 + timedelta(minutes=i)
        rows.append([ts.timestamp(), px, px * 1.0001, px * 0.9999, px, 12.0])
    return rows


def _gappy_klines() -> list[list[float]]:
    t0 = datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc)
    rows = []
    px = 100.0
    for i in (0, 1, 2, 20, 40, 80):
        ts = t0 + timedelta(minutes=i)
        rows.append([ts.timestamp(), px, px, px, px, 1.0])
    return rows


def _stale_klines() -> list[list[float]]:
    # Last bar 13:25 vs decision 13:29 => age 240s > 180s freshness contract.
    t0 = datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc)
    return _dense_klines(86, start=t0, close0=90.0)


def _fourh(_symbol: str = "") -> dict:
    return {
        "production_4h_break_true_at_decision": False,
        "distance_to_4h_break_bps": 12.5,
        "4h_range_position": 0.4,
    }


def _contract(*, as_of: datetime, eligible=None, p_buys=None, path_evs=None, ranks=None) -> dict:
    eligible = eligible or dict.fromkeys(_COINS, True)
    p_buys = p_buys or {}
    path_evs = path_evs or dict.fromkeys(eligible, 0.01)
    ranks = ranks or dict.fromkeys(eligible, 0.01)
    cands = []
    for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "HOLD"):
        if sym == "HOLD":
            cands.append(
                {
                    "symbol": "HOLD",
                    "eligible": True,
                    "exclusion_reason": None,
                    "p_buy": None,
                    "path_ev": 0.0,
                    "final_rank_score": 0.0,
                    "4h_entry_telemetry": None,
                }
            )
            continue
        cands.append(
            {
                "symbol": sym,
                "eligible": bool(eligible.get(sym)),
                "exclusion_reason": None if eligible.get(sym) else "NO_SCORED_CANDIDATE",
                "p_buy": p_buys.get(sym),
                "path_ev": path_evs.get(sym),
                "final_rank_score": ranks.get(sym),
                "4h_entry_telemetry": _fourh(sym),
            }
        )
    return {
        "decision_group_id": f"daygrp_{int(as_of.timestamp())}",
        "decision_timestamp": as_of.isoformat(),
        "created_at": as_of.isoformat(),
        "selected_action": "HOLD",
        "selected_symbol": "HOLD",
        "candidates": cands,
        "4h_entry_telemetry": {s: _fourh(s) for s in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")},
    }


def _redis_all(*, klines=None, p_buys=None) -> FakeRedis:
    klines = klines or {}
    default = _dense_klines()
    strings = {f"klines:{s}:1m": json.dumps(klines.get(s, default)) for s in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")}
    hashes = {f"ai_signal:day:{s}": {"prob_buy": str((p_buys or {}).get(s, 0.4 + i * 0.05))} for i, s in enumerate(("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"))}
    return FakeRedis(hashes=hashes, strings=strings)


def _quote(symbol: str) -> dict:
    px = {"BTCUSDT": 80000.0, "ETHUSDT": 4000.0, "SOLUSDT": 180.0, "XRPUSDT": 1.4}[symbol]
    return {
        "best_bid": px,
        "best_ask": px * 1.0002,
        "mid": px * 1.0001,
        "spread_pct": 0.0002,
        "book_source": "redis_depth_cache",
        "book_age_sec": 0.4,
        "ts_utc": datetime(2026, 9, 2, 13, 29, tzinfo=timezone.utc).isoformat(),
    }


def test_all_candidate_p_buy_shadow_and_production_unchanged(monkeypatch):
    as_of = datetime(2026, 9, 2, 13, 29, tzinfo=timezone.utc)
    contract = _contract(as_of=as_of, eligible={"BTCUSDT": True, "ETHUSDT": False, "SOLUSDT": False, "XRPUSDT": False}, p_buys={"BTCUSDT": 0.77})
    redis = _redis_all(p_buys={"BTCUSDT": 0.11, "ETHUSDT": 0.22, "SOLUSDT": 0.33, "XRPUSDT": 0.44})
    monkeypatch.setattr("backend.services.decision_book_tape.snapshot_book", _book)
    arts = [build_candidate_artifact(contract, s, redis_client=redis, as_of=as_of) for s in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")]
    btc = arts[0]
    assert btc["production_p_buy"] == 0.77
    assert btc["shadow_candidate_p_buy"] == 0.11
    assert btc["p_buy_provenance"] == "production_candidate"
    assert btc["features"]["p_buy"] == 0.77
    eth = arts[1]
    assert eth["production_p_buy"] is None
    assert eth["shadow_candidate_p_buy"] == 0.22
    assert eth["p_buy_provenance"] == "redis_signal"
    assert eth["eligible"] is False
    assert eth["eligibility_reason"] == "NO_SCORED_CANDIDATE"


def test_clock_feature_point_in_time_and_spread(monkeypatch):
    as_of = datetime(2026, 9, 2, 13, 29, tzinfo=timezone.utc)
    contract = _contract(as_of=as_of, p_buys=dict.fromkeys(_COINS, 0.5))
    redis = _redis_all()
    monkeypatch.setattr("backend.services.decision_book_tape.snapshot_book", _book)
    art = build_candidate_artifact(contract, "BTCUSDT", redis_client=redis, as_of=as_of)
    feats = art["features"]
    for name in ("ret_5m", "ret_15m", "ret_30m", "realized_vol_10m", "btc_rel_ret_5m", "rel_volume_15m"):
        assert feats[name] is not None
    assert art["quote"]["spread_bps"] is not None
    assert art["quote"]["quote_timestamp"]
    assert art["quote"]["quote_source"] == "redis_depth_cache"
    assert art["quote"]["quote_age_ms"] == 400.0
    assert feats["spread_bps"] == art["quote"]["spread_bps"]
    assert feats["estimated_all_in_cost_bps"] != feats["spread_bps"]
    assert art["provenance"]["feature_cutoff_ts"]
    assert art["provenance"]["source_latest_ts"]
    assert art["provenance"]["observation_count"] > 0
    assert art["provenance"]["feature_contract_version"]


def test_missing_feature_null_no_zero_impute(monkeypatch):
    as_of = datetime(2026, 9, 2, 13, 29, tzinfo=timezone.utc)
    contract = _contract(as_of=as_of, p_buys={"BTCUSDT": 0.5})
    redis = _redis_all(klines=dict.fromkeys(_COINS, _gappy_klines()))
    monkeypatch.setattr("backend.services.decision_book_tape.snapshot_book", _book)
    art = build_candidate_artifact(contract, "BTCUSDT", redis_client=redis, as_of=as_of)
    assert art["features"]["ret_30m"] is None
    assert art["features"]["ret_30m"] != 0
    assert art["missingness_reasons"]["ret_30m"] in {SOURCE_DATA_GAP, SOURCE_DATA_STALE}
    stale = _redis_all(klines=dict.fromkeys(_COINS, _stale_klines()))
    art_stale = build_candidate_artifact(contract, "BTCUSDT", redis_client=stale, as_of=as_of)
    assert art_stale["features"]["ret_5m"] is None
    assert art_stale["missingness_reasons"]["ret_5m"] == SOURCE_DATA_STALE


def test_no_quote_is_null_not_cost_model(monkeypatch):
    as_of = datetime(2026, 9, 2, 13, 29, tzinfo=timezone.utc)
    contract = _contract(as_of=as_of, p_buys={"BTCUSDT": 0.5})
    redis = _redis_all()
    monkeypatch.setattr("backend.services.decision_book_tape.snapshot_book", lambda *_a, **_k: {"best_bid": None, "best_ask": None})
    art = build_candidate_artifact(contract, "BTCUSDT", redis_client=redis, as_of=as_of)
    assert art["features"]["spread_bps"] is None
    assert art["quote"]["reason"] == "NO_VALID_DECISION_QUOTE"
    assert art["missingness_reasons"]["spread_bps"] == NO_QUOTE
    assert art["features"]["estimated_all_in_cost_bps"] is not None


def test_eligible_ineligible_and_complete_group(monkeypatch):
    as_of = datetime(2026, 9, 2, 13, 29, tzinfo=timezone.utc)
    contract = _contract(
        as_of=as_of,
        eligible={"BTCUSDT": True, "ETHUSDT": True, "SOLUSDT": False, "XRPUSDT": False},
        p_buys={"BTCUSDT": 0.6, "ETHUSDT": 0.55},
    )
    redis = _redis_all()
    monkeypatch.setattr("backend.services.decision_book_tape.snapshot_book", _book)
    arts = [build_candidate_artifact(contract, s, redis_client=redis, as_of=as_of) for s in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "HOLD")]
    comp = group_completeness(arts)
    assert comp["FEATURE_COMPLETE"] is True
    assert "SOLUSDT" in comp["ineligible_symbols"]
    assert "HOLD" in comp["eligible_symbols"]
    assert arts[-1]["features"]["legacy_path_ev"] == 0.0
    assert arts[-1]["features"]["spread_bps"] == 0.0
    sol = arts[2]
    assert sol["eligible"] is False
    assert sol["shadow_candidate_p_buy"] is not None


def test_complete_group_and_independence(tmp_path, monkeypatch):
    as_of = datetime(2026, 9, 2, 13, 29, tzinfo=timezone.utc)
    later = as_of + timedelta(minutes=15)
    redis = _redis_all()
    monkeypatch.setattr("backend.services.decision_book_tape.snapshot_book", _book)
    db = str(tmp_path / "cap.db")
    a = capture_clock_v2_group(db, _contract(as_of=as_of, p_buys=dict.fromkeys(_COINS, 0.5)), redis_client=redis)
    b = capture_clock_v2_group(db, _contract(as_of=later, p_buys=dict.fromkeys(_COINS, 0.5)), redis_client=redis)
    assert a["completeness"]["FEATURE_COMPLETE"] is True
    assert b["decision_group_id"] != a["decision_group_id"]
    n = sqlite3.connect(db).execute(f"SELECT COUNT(DISTINCT decision_group_id) FROM {TABLE_ARTIFACT}").fetchone()[0]
    assert n == 2


def test_lock_feature_only_never_inspected(tmp_path, monkeypatch):
    as_of = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)
    start = datetime(2026, 9, 4, 10, 30, tzinfo=timezone.utc)
    redis = _redis_all(klines=dict.fromkeys(_COINS, _dense_klines(90, start=start)))
    monkeypatch.setattr("backend.services.decision_book_tape.snapshot_book", _book)
    db = str(tmp_path / "lock.db")
    register_lock(db)
    out = capture_clock_v2_group(db, _contract(as_of=as_of, p_buys=dict.fromkeys(_COINS, 0.4)), redis_client=redis)
    assert out["inspected"] is False
    rows = sqlite3.connect(db).execute(f"SELECT inspected, feature_json FROM {TABLE_ARTIFACT}").fetchall()
    assert rows
    for inspected, raw in rows:
        assert inspected == 0
        payload = json.loads(raw)
        assert FORBIDDEN_OUTCOME_KEYS.isdisjoint(payload)
    lock_flag = sqlite3.connect(db).execute(f"SELECT inspected FROM {TABLE_LOCK}").fetchone()[0]
    assert lock_flag in (0, False)


def test_future_data_and_timezone_invariance(monkeypatch):
    as_of = datetime(2026, 9, 2, 13, 29, tzinfo=timezone.utc)
    future = _dense_klines(10, start=as_of + timedelta(hours=2))
    past = _dense_klines(90, start=as_of - timedelta(minutes=89))
    redis = _redis_all(klines=dict.fromkeys(_COINS, past + future))
    monkeypatch.setattr("backend.services.decision_book_tape.snapshot_book", _book)
    contract = _contract(as_of=as_of, p_buys=dict.fromkeys(_COINS, 0.5))
    art = build_candidate_artifact(contract, "BTCUSDT", redis_client=redis, as_of=as_of)
    latest = datetime.fromisoformat(art["provenance"]["source_latest_ts"])
    cutoff = datetime.fromisoformat(art["provenance"]["feature_cutoff_ts"])
    assert latest <= cutoff
    z_as_of = as_of.isoformat().replace("+00:00", "Z")
    art_z = build_candidate_artifact(contract, "BTCUSDT", redis_client=redis, as_of=z_as_of)
    assert art["features"]["ret_5m"] == art_z["features"]["ret_5m"]


def test_historical_missingness_categories_are_exclusive():
    assert classify_historical_p_buy(eligible=False, p_buy=None, exclusion_reason="NO_SCORED_CANDIDATE") == NOT_PERSISTED
    assert classify_historical_p_buy(eligible=True, p_buy=None, exclusion_reason=None) == NOT_COMPUTED_FOR_CANDIDATE
    assert classify_historical_p_buy(eligible=False, p_buy=None, exclusion_reason="ALREADY_OPEN") == CANDIDATE_INELIGIBLE
    assert classify_historical_clock(quality_reasons=["stale_last_bar"], persisted=True) == SOURCE_DATA_STALE
    assert classify_historical_clock(quality_reasons=["gap_exceeded"], persisted=True) == SOURCE_DATA_GAP
    assert classify_historical_clock(quality_reasons=[], persisted=False) == NOT_PERSISTED
    assert classify_historical_spread(contract_spread=4.2, clock_spread=None) == NOT_PERSISTED
    assert classify_historical_spread(contract_spread=None, clock_spread=None) == NO_QUOTE
    assert MAX_GAP_SEC == 180
    assert MAX_LAST_BAR_AGE_SEC == 180


def test_golden_telemetry_off_vs_on_identity(tmp_path, monkeypatch):
    reset_day_artifact_cache()
    bars = []
    t0 = datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc)
    px = 100.0
    for i in range(40):
        px *= 1.0001
        ts = t0 + timedelta(minutes=i)
        bars.append({"open": px, "high": px * 1.0002, "low": px * 0.9998, "close": px, "volume": 10.0, "ts": ts})
    dd = {"bars_1m": bars, "symbol": "ETHUSDT", "btc_ret_5": 0.0}
    ev_off, stamped_off = resolve_day_path_ev(dd, symbol="ETHUSDT")
    pred_off = predict_decision_net(dd)
    decision_inputs = {
        "btc_path_ev": 0.0,
        "eth_path_ev": ev_off,
        "sol_path_ev": 0.0,
        "xrp_path_ev": 0.0,
        "valid": {"btc": False, "eth": True, "sol": False, "xrp": False},
        "path_net_status": "predicted",
        "path_net_model_id": "day_path_net_v1",
    }
    sel_off = select_action(decision_inputs)
    monkeypatch.setenv("DAY_CLOCK_V2_CAPTURE", "false")
    contract_off = build_group_contract(decision=sel_off, bar_timestamp=int(t0.timestamp()))
    monkeypatch.setenv("DAY_CLOCK_V2_CAPTURE", "true")
    ev_on, stamped_on = resolve_day_path_ev(dd, symbol="ETHUSDT")
    pred_on = predict_decision_net(dd)
    sel_on = select_action(decision_inputs)
    contract_on = build_group_contract(decision=sel_on, bar_timestamp=int(t0.timestamp()))
    assert ev_off == ev_on == pred_off == pred_on
    assert stamped_off["path_input_valid"] is True
    assert stamped_on["path_input_valid"] is True
    assert sel_off["selected_action"] == sel_on["selected_action"]
    assert sel_off.get("selected_symbol") == sel_on.get("selected_symbol")
    assert contract_off["selected_action"] == contract_on["selected_action"]
    assert [c["p_buy"] for c in contract_off["candidates"]] == [c["p_buy"] for c in contract_on["candidates"]]
    assert [c["path_ev"] for c in contract_off["candidates"]] == [c["path_ev"] for c in contract_on["candidates"]]
    assert [c["final_rank_score"] for c in contract_off["candidates"]] == [c["final_rank_score"] for c in contract_on["candidates"]]
    db = str(tmp_path / "gold.db")
    monkeypatch.setenv("DAY_DECISION_OBSERVABILITY", "true")
    gid = record_day_ranking_group(db, decision=sel_on, bar_timestamp=int(t0.timestamp()))
    assert gid
    reset_day_artifact_cache()


def test_pit_mutation_future_klines_do_not_change_artifact(monkeypatch):
    as_of = datetime(2026, 9, 2, 13, 29, tzinfo=timezone.utc)
    past = _dense_klines(90, start=as_of - timedelta(minutes=89))
    redis = _redis_all(klines=dict.fromkeys(_COINS, past))
    monkeypatch.setattr("backend.services.decision_book_tape.snapshot_book", _book)
    contract = _contract(as_of=as_of, p_buys=dict.fromkeys(_COINS, 0.5))
    before = build_candidate_artifact(contract, "BTCUSDT", redis_client=redis, as_of=as_of)
    future = _dense_klines(20, start=as_of + timedelta(minutes=1), close0=50.0)
    redis.strings = {k: json.dumps(past + future) for k in redis.strings}
    after = build_candidate_artifact(contract, "BTCUSDT", redis_client=redis, as_of=as_of)
    assert artifact_fingerprint(before) == artifact_fingerprint(after)
    latest = datetime.fromisoformat(after["provenance"]["source_latest_ts"])
    assert latest <= as_of


def test_feature_ohlcv_fallback_same_clock_contract(tmp_path, monkeypatch):
    as_of = datetime(2026, 9, 2, 13, 29, tzinfo=timezone.utc)
    db = str(tmp_path / "ohlcv.db")
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE feature_ohlcv (symbol TEXT, interval TEXT, ts TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL)")
    t0 = as_of - timedelta(minutes=89)
    px = 100.0
    for i in range(90):
        ts = t0 + timedelta(minutes=i)
        px *= 1.00015
        for sym in _COINS:
            conn.execute(
                "INSERT INTO feature_ohlcv VALUES (?,?,?,?,?,?,?,?)",
                (sym, "1m", ts.isoformat(), px, px, px, px, 12.0),
            )
    conn.commit()
    conn.close()
    empty = FakeRedis(
        hashes={f"ai_signal:day:{s}": {"prob_buy": "0.4"} for s in _COINS},
        strings={f"klines:{s}:1m": None for s in _COINS},
    )
    monkeypatch.setattr("backend.services.decision_book_tape.snapshot_book", _book)
    art = build_candidate_artifact(
        _contract(as_of=as_of, p_buys={"BTCUSDT": 0.4}),
        "BTCUSDT",
        db_path=db,
        redis_client=empty,
        as_of=as_of,
    )
    assert art["provenance"]["kline_source"] == FEATURE_OHLCV_SOURCE
    assert art["features"]["ret_5m"] is not None
    latest = datetime.fromisoformat(art["provenance"]["source_latest_ts"])
    assert latest <= as_of


def test_redis_1m_preferred_when_present(monkeypatch):
    as_of = datetime(2026, 9, 2, 13, 29, tzinfo=timezone.utc)
    redis = _redis_all()
    monkeypatch.setattr("backend.services.decision_book_tape.snapshot_book", _book)
    art = build_candidate_artifact(
        _contract(as_of=as_of, p_buys={"BTCUSDT": 0.5}),
        "BTCUSDT",
        redis_client=redis,
        as_of=as_of,
    )
    assert art["provenance"]["kline_source"] == KLINE_SOURCE
    assert art["provenance"]["observation_count"] >= 80


def test_spread_recompute_matches_stored(monkeypatch):
    as_of = datetime(2026, 9, 2, 13, 29, tzinfo=timezone.utc)
    redis = _redis_all()
    monkeypatch.setattr("backend.services.decision_book_tape.snapshot_book", _book)
    art = build_candidate_artifact(
        _contract(as_of=as_of, p_buys={"BTCUSDT": 0.5}),
        "BTCUSDT",
        redis_client=redis,
        as_of=as_of,
    )
    q = art["quote"]
    recomputed = recompute_spread_bps(q["best_bid"], q["best_ask"], q["mid"])
    assert recomputed == q["spread_bps"]
    assert art["features"]["estimated_all_in_cost_bps"] != q["spread_bps"]
    assert q["quote_source"]
    assert q["quote_timestamp"]


def test_ineligible_null_clock_does_not_fail_group(monkeypatch):
    as_of = datetime(2026, 9, 2, 13, 29, tzinfo=timezone.utc)
    redis = _redis_all(klines={"BTCUSDT": _dense_klines(), "ETHUSDT": _gappy_klines(), "SOLUSDT": _gappy_klines(), "XRPUSDT": _gappy_klines()})
    monkeypatch.setattr("backend.services.decision_book_tape.snapshot_book", _book)
    contract = _contract(
        as_of=as_of,
        eligible={"BTCUSDT": True, "ETHUSDT": False, "SOLUSDT": False, "XRPUSDT": False},
        p_buys={"BTCUSDT": 0.55},
    )
    arts = [build_candidate_artifact(contract, s, redis_client=redis, as_of=as_of) for s in (*_COINS, "HOLD")]
    comp = group_completeness(arts)
    assert comp["status"] == "FEATURE_COMPLETE"
    assert comp["FEATURE_PARTIAL"] is False
    assert arts[1]["eligible"] is False
    assert arts[1]["features"]["ret_30m"] is None


def test_eligible_stale_clock_is_partial(monkeypatch):
    as_of = datetime(2026, 9, 2, 13, 29, tzinfo=timezone.utc)
    redis = _redis_all(klines=dict.fromkeys(_COINS, _stale_klines()))
    monkeypatch.setattr("backend.services.decision_book_tape.snapshot_book", _book)
    arts = [build_candidate_artifact(_contract(as_of=as_of, p_buys=dict.fromkeys(_COINS, 0.4)), s, redis_client=redis, as_of=as_of) for s in (*_COINS, "HOLD")]
    comp = group_completeness(arts)
    assert comp["status"] == "FEATURE_PARTIAL"
    assert comp["UNUSABLE"] is False


def test_clock_modules_still_absent_from_live_authority():
    for rel in (
        "backend/services/day_path_net.py",
        "backend/services/day_direct_path_ev_authority.py",
        "backend/services/day_path_input_validity.py",
        "backend/services/portfolio_engine.py",
    ):
        text = Path(rel).read_text()
        assert "day_path_clock" not in text
        assert "day_path_clock_v2_capture" not in text
