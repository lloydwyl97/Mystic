import json
from datetime import datetime, timezone

from backend.services.day_4h_entry_features import (
    HOLD_SYMBOL,
    SCHEMA_VERSION,
    asof_bundle_from_1m,
    build_4h_entry_features,
    drop_bars_after,
    hold_4h_entry_features,
    shadow_4h_structure_score,
)
from backend.services.day_4h_entry_telemetry import collect_4h_entry_telemetry, merge_4h_entry_extras
from backend.services.day_controlled_exits import evaluate_engine_managed_exit
from backend.services.day_direct_path_ev_authority import select_action
from backend.services.day_ocean_live_book import BRIEFING_N, OCEAN_BOOK_COUNT
from backend.services.day_trade_thesis import (
    day_4h_structure_snapshot,
    htf_4h_rise_broken,
    htf_4h_rise_intact,
    resolve_day_4h_structure_bundle,
)
from backend.services.portfolio_engine import get_coin_profile


def _ts(iso: str) -> float:
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


DAY_4H_MS = 4 * 3600 * 1000


def _bar(open_iso: str, o: float, h: float, low: float, c: float) -> list:
    return [int(_ts(open_iso) * 1000), o, h, low, c, 100.0]


def _rising_prefix(n: int, end_open_ms: int, start: float = 100.0) -> list[list]:
    rows = []
    px = start
    ot = end_open_ms - n * DAY_4H_MS
    for _ in range(n):
        o = px
        c = px * 1.004
        rows.append([ot, o, c * 1.001, o * 0.999, c, 100.0])
        px = c
        ot += DAY_4H_MS
    return rows


def _bundle(*, forming_close: float, prior_low: float = 100.0, now_iso: str = "2026-08-28T06:00:00Z") -> dict:
    now_open = int(_ts(now_iso[:11] + "04:00:00Z") * 1000)
    prior_open = now_open - DAY_4H_MS
    prefix = _rising_prefix(8, prior_open, start=prior_low * 0.95)
    prior = [prior_open, prior_low * 1.01, prior_low * 1.03, prior_low, prior_low * 1.005, 100.0]
    forming = [now_open, prior_low * 1.002, prior_low * 1.02, prior_low * 0.995, forming_close, 100.0]
    return {"4h": [*prefix, prior, forming]}


def test_no_future_data_in_drop_and_builder():
    now = _ts("2026-08-28T02:00:00Z")
    past = _bar("2026-08-28T00:00:00Z", 100, 101, 99, 100.5)
    future = _bar("2026-08-28T04:00:00Z", 90, 91, 89, 88)
    kept = drop_bars_after([past, future], now)
    assert kept == [past]
    bundle = {"4h": [past, future]}
    feats = build_4h_entry_features(bundle=bundle, now_epoch=now, current_price=100.5, symbol="BTCUSDT")
    resolved = resolve_day_4h_structure_bundle(bundle, current_price=100.5, now_epoch=now)
    last_ot = resolved["4h"][-1][0]
    assert last_ot == past[0]
    assert feats["current_4h_close_or_mark"] == 100.5


def test_asof_tracker_cannot_see_future_1m():
    t0 = int(_ts("2026-08-28T04:00:00Z"))
    seed = [[float(t0 - 14400), 100.0, 101.0, 99.0, 100.5, 0.0]]
    bars = [
        (t0, 100.5, 101.0, 100.0, 100.8, 1.0),
        (t0 + 60, 100.8, 102.0, 100.7, 101.5, 1.0),
        (t0 + 120, 101.5, 110.0, 101.0, 109.0, 1.0),
    ]
    bundle = asof_bundle_from_1m(bars, float(t0 + 90), seed_4h=seed)
    forming = bundle["4h"][-1]
    assert forming[2] == 102.0
    assert forming[4] == 101.5
    assert forming[2] < 110.0


def test_future_data_mutation_is_byte_identical():
    t0 = int(_ts("2026-08-28T04:00:00Z"))
    now = float(t0 + 90)
    seed = [[float(t0 - 14400), 100.0, 101.0, 99.0, 100.5, 0.0]]
    bars = [
        (t0, 100.5, 101.0, 100.0, 100.8, 1.0),
        (t0 + 60, 100.8, 102.0, 100.7, 101.5, 1.0),
        (t0 + 120, 101.5, 110.0, 101.0, 109.0, 1.0),
        (t0 + 180, 109.0, 80.0, 70.0, 75.0, 1.0),
    ]
    first = build_4h_entry_features(
        bundle=asof_bundle_from_1m(bars, now, seed_4h=seed),
        now_epoch=now,
        current_price=101.5,
        symbol="ETHUSDT",
    )
    mutated = list(bars)
    mutated[2] = (t0 + 120, 1.0, 999.0, 0.1, 0.2, 99.0)
    mutated[3] = (t0 + 180, 0.2, 0.3, 0.1, 0.1, 99.0)
    second = build_4h_entry_features(
        bundle=asof_bundle_from_1m(mutated, now, seed_4h=seed),
        now_epoch=now,
        current_price=101.5,
        symbol="ETHUSDT",
    )
    keys = (
        "prior_completed_4h_low",
        "current_4h_high_so_far",
        "current_4h_close_or_mark",
        "distance_to_4h_break_bps",
        "production_4h_break_true_now",
        "minutes_into_current_4h_bar",
    )
    assert json.dumps({k: first[k] for k in keys}, sort_keys=True) == json.dumps({k: second[k] for k in keys}, sort_keys=True)


def test_future_mark_does_not_leak_when_clipped():
    now = _ts("2026-08-28T05:00:00Z")
    bundle = _bundle(forming_close=100.4, prior_low=100.0)
    a = build_4h_entry_features(bundle=bundle, now_epoch=now, current_price=100.4, symbol="BTCUSDT")
    future_bundle = dict(bundle)
    future_bundle["4h"] = [*bundle["4h"], _bar("2026-08-28T08:00:00Z", 50, 51, 49, 48)]
    b = build_4h_entry_features(bundle=future_bundle, now_epoch=now, current_price=100.4, symbol="BTCUSDT")
    assert a["distance_to_4h_break_bps"] == b["distance_to_4h_break_bps"]
    assert a["production_4h_break_true_now"] == b["production_4h_break_true_now"]


def test_forming_vs_completed_and_predicate_equality():
    now = _ts("2026-08-28T06:15:00Z")
    bundle = _bundle(forming_close=100.8, prior_low=100.0)
    feats = build_4h_entry_features(bundle=bundle, now_epoch=now, current_price=100.8, symbol="ETHUSDT")
    assert feats["prior_completed_4h_low"] == 100.0
    assert feats["production_4h_break_true_now"] is False
    assert feats["production_4h_intact_at_decision"] is True
    assert htf_4h_rise_broken(bundle, current_price=100.8, now_epoch=now) is False
    assert htf_4h_rise_intact(bundle, current_price=100.8, now_epoch=now) is True
    broken = build_4h_entry_features(bundle=bundle, now_epoch=now, current_price=99.5, symbol="ETHUSDT")
    assert broken["production_4h_break_true_now"] is True
    assert htf_4h_rise_broken(bundle, current_price=99.5, now_epoch=now) is True
    assert day_4h_structure_snapshot(bundle, current_price=99.5, now_epoch=now)["htf_4h_rise_broken"] is True


def test_distance_to_break_and_hold_nulls():
    now = _ts("2026-08-28T05:00:00Z")
    bundle = _bundle(forming_close=100.4, prior_low=100.0)
    far = build_4h_entry_features(bundle=bundle, now_epoch=now, current_price=101.0, symbol="SOLUSDT")
    near = build_4h_entry_features(bundle=bundle, now_epoch=now, current_price=100.25, symbol="SOLUSDT")
    assert far["distance_to_4h_break_bps"] > near["distance_to_4h_break_bps"]
    hold = hold_4h_entry_features()
    assert hold["symbol"] == HOLD_SYMBOL
    assert hold["production_4h_break_true_now"] is None
    assert hold["path_ev"] == 0.0
    assert hold["4h_structure_schema_version"] == SCHEMA_VERSION
    score = shadow_4h_structure_score(near)
    assert score["live_gate"] is False


def test_golden_behavior_equality():
    scores = {
        "btc_path_ev": 0.0001,
        "eth_path_ev": 0.0008,
        "sol_path_ev": 0.0002,
        "xrp_path_ev": 0.0001,
        "path_net_status": "predicted",
        "path_net_model_id": "day_path_net_v1",
    }
    a = select_action(scores, old_rank_nominee="BTCUSDT", old_rank_score=9.0)
    b = select_action(scores, old_rank_nominee="BTCUSDT", old_rank_score=9.0)
    extras_a = {"selected_action": a["selected_action"], "path_ev_winner": a["path_ev_winner"], "btc_path_ev": a["btc_path_ev"]}
    extras_c = merge_4h_entry_extras(dict(extras_a), a)
    for key in ("selected_action", "selected_symbol", "path_ev_winner", "selected_ev", "why_selected"):
        assert a[key] == b[key]
    assert extras_c["selected_action"] == a["selected_action"]
    assert extras_c["path_ev_winner"] == a["path_ev_winner"]
    assert extras_c.get("4h_telemetry_live_gate") is False
    now = _ts("2026-08-28T06:15:08Z")
    bundle = _bundle(forming_close=100.2, prior_low=100.0)

    class _Pos:
        symbol = "BTC/USDT"
        entry_price = 100.2
        highest_price = 100.2
        lowest_price = 100.2
        stop_price = 0.0
        trailing_stop_price = 0.0
        trail_pct = 0.004
        take_profit_1_price = 0.0
        entry_thesis = "HTF_TREND_PULLBACK"
        entry_vwap = 100.2
        thesis_invalid_level = 0.0
        thesis_target_level = 0.0
        thesis_score = 0.7
        max_hold_min = 360
        day_route_regime_at_entry = "bull"

    out_a = evaluate_engine_managed_exit(
        position=_Pos(),
        current_price=100.2,
        net_pnl_pct=0.0,
        hold_minutes=1.0,
        coin_profile=get_coin_profile("BTCUSDT"),
        bundle=bundle,
        now_epoch=now,
    )
    out_b = evaluate_engine_managed_exit(
        position=_Pos(),
        current_price=100.2,
        net_pnl_pct=0.0,
        hold_minutes=1.0,
        coin_profile=get_coin_profile("BTCUSDT"),
        bundle=bundle,
        now_epoch=now,
    )
    assert out_a["action"] == out_b["action"]
    assert out_a.get("reason") == out_b.get("reason")


def test_4h_break_semantics_regression():
    now = _ts("2026-08-28T06:15:08Z")
    bundle = _bundle(forming_close=100.2, prior_low=100.0)
    assert htf_4h_rise_broken(bundle, current_price=100.2, now_epoch=now) is False
    assert htf_4h_rise_broken(bundle, current_price=99.8, now_epoch=now) is True


def test_candidate_btc_eth_sol_xrp_hold_and_66_constants():
    tel = collect_4h_entry_telemetry(
        {
            "selected_action": "BUY_ETHUSDT",
            "selected_symbol": "ETHUSDT",
            "prediction_timestamp": "2026-08-28T06:15:00+00:00",
            "eth_path_ev": 0.001,
        }
    )
    for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "HOLD"):
        assert sym in tel["4h_entry_telemetry"]
    assert tel["4h_entry_telemetry"]["HOLD"]["distance_to_4h_break_bps"] is None
    assert tel["4h_telemetry_live_gate"] is False
    assert "selected_already_broken_at_ranking" in tel
    assert BRIEFING_N == 53
    assert OCEAN_BOOK_COUNT == 66


def test_scalp_and_ranker_unaffected():
    scalp = open("backend/services/binance_scalp/scalp_candidate_ranking.py", encoding="utf-8").read()
    authority = open("backend/services/day_direct_path_ev_authority.py", encoding="utf-8").read()
    assert "day_4h_entry_features" not in scalp
    assert "day_4h_entry_telemetry" not in scalp
    assert "day_4h_entry_features" not in authority
    assert "htf_4h_rise_broken" not in authority
    scores = {"btc_path_ev": 0.01, "eth_path_ev": 0.0, "sol_path_ev": 0.0, "xrp_path_ev": 0.0, "path_net_status": "predicted"}
    out = select_action(scores)
    assert out["selected_action"] == "BUY_BTCUSDT"
