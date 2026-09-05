"""Clock-consistent path research dataset integrity. Offline only."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from backend.services.day_direct_path_ev_authority import select_action
from backend.services.day_forward_lock import FORWARD_LOCK_START
from backend.services.day_model_readiness import MIN_CHRONOLOGICAL_BLOCKS, MIN_EVENTS_PER_FEATURE, evaluate_readiness
from backend.services.day_path_clock_compare import (
    compare_legacy_vs_clock,
    refuse_legacy_coefficients_on_clock_features,
)
from backend.services.day_path_clock_dataset import (
    assert_group_integrity,
    build_group_record,
    dataset_counts,
    in_sealed_lock,
    lock_window_status,
)
from backend.services.day_path_clock_features import build_clock_features, clock_return, normalize_bars
from backend.services.day_path_clock_labels import bars_have_future_leak, build_clock_labels, clock_markout_net
from backend.services.day_path_clock_v2 import (
    CLOCK_LABEL_HORIZONS_SEC,
    SCHEMA_VERSION,
    clock_challenger_export_schema,
    feature_contract,
    future_acceptance_bar,
    future_decision_contract,
)
from backend.services.day_path_input_validity import five_bar_return
from backend.services.day_path_net import predict_decision_net, reset_day_artifact_cache, resolve_day_path_ev


def _ts(iso: str) -> datetime:
    return datetime.fromisoformat(iso.replace("Z", "+00:00"))


def _dense(n: int = 80, start: datetime | None = None, close0: float = 100.0, step: float = 0.0002) -> list[dict]:
    t0 = start or datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)
    bars = []
    px = close0
    for i in range(n):
        px *= 1.0 + step
        ts = t0 + timedelta(seconds=60 * i)
        bars.append({"open": px, "high": px * 1.0001, "low": px * 0.9999, "close": px, "volume": 10.0 + i, "ts": ts})
    return bars


def test_clock_lookback_exactness():
    t0 = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)
    bars = []
    for i in range(20):
        px = 100.0 + i
        bars.append({"open": px, "high": px, "low": px, "close": px, "volume": 1.0, "ts": t0 + timedelta(minutes=i)})
    as_of = t0 + timedelta(minutes=19)
    ret = clock_return(normalize_bars(bars), as_of, 5 * 60)
    assert ret is not None
    assert abs(ret - (119.0 - 114.0) / 114.0) < 1e-12


def test_no_future_data_in_features_or_labels():
    t0 = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)
    bars = _dense(40, start=t0, close0=100.0)
    as_of = bars[20]["ts"]
    future = {
        "open": 1.0,
        "high": 1.0,
        "low": 1.0,
        "close": 1.0,
        "volume": 1.0,
        "ts": as_of + timedelta(minutes=30),
    }
    mixed = [*bars, future]
    feats = build_clock_features(mixed, as_of=as_of, symbol="ETHUSDT")
    assert feats["ret_5m"] is not None
    labels = build_clock_labels(mixed, decision_ts=as_of, symbol="ETHUSDT", cost_bps=4.0)
    cutoff = as_of + timedelta(seconds=CLOCK_LABEL_HORIZONS_SEC["15m"])
    used = normalize_bars(mixed)
    assert not bars_have_future_leak([b for b in used if b.ts <= cutoff], cutoff)
    assert labels["clock_net_bps"]["15m"] is not None


def test_timezone_invariance():
    t0 = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)
    bars = _dense(40, start=t0)
    as_of = bars[-1]["ts"]
    a = build_clock_features(bars, as_of=as_of, symbol="BTCUSDT")
    z_bars = [{**b, "ts": b["ts"].isoformat().replace("+00:00", "Z")} for b in bars]
    offset_bars = [{**b, "ts": b["ts"].isoformat()} for b in bars]
    naive_bars = [{**b, "ts": b["ts"].replace(tzinfo=None)} for b in bars]
    b = build_clock_features(z_bars, as_of=as_of.isoformat().replace("+00:00", "Z"), symbol="BTCUSDT")
    c = build_clock_features(offset_bars, as_of=as_of.isoformat(), symbol="BTCUSDT")
    d = build_clock_features(naive_bars, as_of=as_of.replace(tzinfo=None), symbol="BTCUSDT")
    assert a["ret_5m"] == b["ret_5m"] == c["ret_5m"] == d["ret_5m"]
    assert a["ret_15m"] == d["ret_15m"]


def test_btc_relative_return_exactness():
    t0 = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)
    btc = []
    eth = []
    for i in range(20):
        ts = t0 + timedelta(minutes=i)
        btc.append({"open": 100 + i * 0.2, "high": 100 + i * 0.2, "low": 100 + i * 0.2, "close": 100 + i * 0.2, "volume": 1.0, "ts": ts})
        eth.append({"open": 50 + i * 0.5, "high": 50 + i * 0.5, "low": 50 + i * 0.5, "close": 50 + i * 0.5, "volume": 1.0, "ts": ts})
    as_of = t0 + timedelta(minutes=19)
    feats = build_clock_features(eth, as_of=as_of, symbol="ETHUSDT", btc_bars=btc)
    btc_ret = (103.8 - 102.8) / 102.8
    eth_ret = (59.5 - 57.0) / 57.0
    assert feats["btc_ret_5m"] is not None
    assert abs(feats["btc_ret_5m"] - btc_ret) < 1e-12
    assert abs(feats["ret_5m"] - eth_ret) < 1e-12
    assert abs(feats["btc_rel_ret_5m"] - (eth_ret - btc_ret)) < 1e-12


def test_missing_and_stale_bars_are_unavailable():
    t0 = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)
    as_of = t0 + timedelta(minutes=30)
    early = _dense(10, start=t0, close0=100.0)
    late = _dense(5, start=as_of - timedelta(seconds=60 * 4), close0=110.0)
    hole = [*early, *late]
    missing = build_clock_features(hole, as_of=as_of, symbol="SOLUSDT")
    assert missing["ret_5m"] is None or missing["feature_available"] is False
    stale_bars = _dense(40, start=t0)
    stale = build_clock_features(stale_bars, as_of=stale_bars[-1]["ts"] + timedelta(seconds=181), symbol="SOLUSDT")
    assert stale["ret_5m"] is None
    assert stale["feature_available"] is False


def test_candidate_group_integrity_and_hold_zero():
    t0 = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)
    bars = {sym: _dense(80, start=t0, close0=px) for sym, px in (("BTCUSDT", 80000.0), ("ETHUSDT", 2500.0), ("SOLUSDT", 100.0), ("XRPUSDT", 1.4))}
    group = build_group_record(
        decision_group_id="daygrp_research_1",
        decision_ts=t0 + timedelta(minutes=79),
        bars_by_symbol=bars,
        lock_cutoff=FORWARD_LOCK_START,
    )
    assert_group_integrity(group)
    assert [c["symbol"] for c in group["candidates"]] == ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "HOLD"]
    hold = group["candidates"][-1]
    assert hold["label"]["clock_net_bps"]["15m"] == 0.0
    assert hold["label"]["clock_net_bps"]["4h"] == 0.0
    assert hold["label"]["hold_value_bps"] == 0.0
    counts = dataset_counts([group])
    assert counts["independent_decisions"] == 1
    assert counts["candidate_rows_including_hold"] == 5
    assert counts["primary_unit"] == "decision_group"


def test_same_horizon_label_comparability_and_costs():
    t0 = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)
    bars = _dense(200, start=t0, close0=100.0, step=0.0003)
    as_of = t0 + timedelta(minutes=40)
    cost = 4.4
    a = clock_markout_net(bars, decision_ts=as_of, horizon_sec=3600, cost_bps=cost)
    b = clock_markout_net(bars, decision_ts=as_of, horizon_sec=3600, cost_bps=cost)
    assert a["gross_bps"] == b["gross_bps"]
    assert a["net_bps"] == a["gross_bps"] - cost
    labels = build_clock_labels(bars, decision_ts=as_of, symbol="BTCUSDT", cost_bps=cost, commission_bps=4.0, spread_bps=0.2, slippage_bps=0.2)
    assert labels["same_target_as_production_exit"] is False
    assert labels["counterfactual"] is True
    assert labels["all_in_cost_bps"] == cost
    assert set(labels["label_horizon_seconds"]) == set(CLOCK_LABEL_HORIZONS_SEC)


def test_sealed_lock_strips_outcomes():
    locked_ts = _ts("2026-09-04T23:45:00+00:00")
    assert in_sealed_lock(locked_ts)
    bars = {sym: _dense(80, start=locked_ts - timedelta(minutes=80), close0=10.0) for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")}
    group = build_group_record(
        decision_group_id="daygrp_1788565500",
        decision_ts=locked_ts,
        bars_by_symbol=bars,
        production_exits={"XRPUSDT": {"production_exit_net_bps": 999.0, "exit_reason": "do_not_inspect"}},
        lock_cutoff=FORWARD_LOCK_START,
    )
    assert group["lock_excluded"] is True
    assert_group_integrity(group)
    xrp = next(c for c in group["candidates"] if c["symbol"] == "XRPUSDT")
    assert xrp["label"] is None
    assert xrp["provenance"]["production_exit_attached"] is False


def test_readiness_gate_not_weakened():
    schema = clock_challenger_export_schema()
    assert schema["train"] is False
    assert schema["live_gate"] is False
    assert schema["readiness_required_mature_trade_labels"] == 14 * MIN_EVENTS_PER_FEATURE
    assert schema["readiness_required_chronological_blocks"] == MIN_CHRONOLOGICAL_BLOCKS
    assert future_decision_contract()["activated"] is False
    bar = future_acceptance_bar()
    assert bar["hold_value_bps"] == 0.0
    assert any("profit factor" in c for c in bar["criteria"])
    contract = feature_contract()
    assert "ret_5" not in contract["features"]
    assert "ret_5m" in contract["features"]
    assert contract["features"]["ret_5m"]["clock_lookback_seconds"] == 300


def test_legacy_vs_clock_and_no_coefficient_reuse():
    t0 = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)
    samples = []
    for i in range(8):
        bars = _dense(80, start=t0 + timedelta(minutes=i), close0=100.0 + i)
        samples.append({"as_of": bars[-1]["ts"], "symbol": "ETHUSDT", "bars": bars, "btc_bars": bars, "legacy_btc_ret_5": 0.0})
    out = compare_legacy_vs_clock(samples)
    assert out["do_not_reuse_legacy_coefficients_on_clock_inputs"] is True
    assert out["ret_5_vs_ret_5m"]["n"] >= 3
    assert out["ret_5_vs_ret_5m"]["correlation"] is not None
    assert out["ret_5_vs_ret_5m"]["correlation"] > 0.99
    refuse = refuse_legacy_coefficients_on_clock_features()
    assert refuse["allowed"] is False


def test_sparse_row_count_differs_from_clock():
    t0 = datetime(2026, 9, 4, 8, 25, 15, tzinfo=timezone.utc)
    bars = []
    px = 1.45
    for i in range(20):
        bars.append({"open": px, "high": px, "low": px, "close": px, "volume": 1.0, "ts": t0 + timedelta(seconds=30 * i)})
    t1 = datetime(2026, 9, 4, 23, 25, 15, tzinfo=timezone.utc)
    px = 1.3995
    for i in range(5):
        bars.append({"open": px, "high": px, "low": px, "close": px, "volume": 1.0, "ts": t1 + timedelta(seconds=30 * i)})
    legacy = five_bar_return(bars)
    clock = build_clock_features(bars, as_of=bars[-1]["ts"], symbol="XRPUSDT")
    assert legacy is not None
    assert abs(float(legacy)) > 0.01
    assert clock["ret_5m"] is None


def test_legacy_production_golden_unchanged():
    reset_day_artifact_cache()
    bars = _dense(40, start=datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc), close0=100.0)
    dd = {"bars_1m": bars, "symbol": "ETHUSDT", "btc_ret_5": 0.0}
    legacy = predict_decision_net(dd)
    ev, stamped = resolve_day_path_ev(dd, symbol="ETHUSDT")
    assert ev == legacy
    assert stamped["path_input_valid"] is True
    out = select_action({"btc_path_ev": ev, "eth_path_ev": 0.0, "sol_path_ev": 0.0, "xrp_path_ev": 0.0, "valid": {"btc": True, "eth": False, "sol": False, "xrp": False}})
    if float(ev or 0.0) > 0.0:
        assert out["selected_action"] == "BUY_BTCUSDT"
    else:
        assert out["selected_action"] == "HOLD"
    reset_day_artifact_cache()


def test_clock_modules_not_imported_by_live_path():
    for rel in (
        "backend/services/day_path_net.py",
        "backend/services/day_direct_path_ev_authority.py",
        "backend/services/day_path_input_validity.py",
        "backend/services/portfolio_engine.py",
    ):
        text = Path(rel).read_text()
        assert "day_path_clock" not in text


def test_lock_status_defaults_protect_historical_66():
    status = lock_window_status()
    assert status["historical_66_excluded"] is True
    assert status["inspected"] is False
    assert status["dataset_cutoff"] == FORWARD_LOCK_START


def test_readiness_import_still_blocks_training_without_span(tmp_path):
    out = evaluate_readiness(str(tmp_path / "missing.db"))
    assert out["ready"] is False
    assert SCHEMA_VERSION == "day_path_clock_v2"
