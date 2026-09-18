import sqlite3
import time
from datetime import datetime, timezone
from unittest.mock import patch

from backend.services.day_production_lifecycle_replay import (
    ReplayPos,
    apply_production_exit_env,
    in_unlocked_band,
    rank_key,
    veto_current,
    veto_honest_plus_quality_floor,
    veto_honest_pure_net,
)


def test_quality_floor_matches_current_veto_on_p_buy_band():
    low = {"symbol": "ETHUSDT", "p_buy": 0.11, "p_sell": 0.0, "p_hold": 0.89}
    mid = {"symbol": "ETHUSDT", "p_buy": 0.20, "p_sell": 0.0, "p_hold": 0.80}
    assert veto_current(low)[0] is False
    assert veto_honest_plus_quality_floor(low)[0] is False
    assert veto_honest_pure_net(low)[0] is True
    assert veto_current(mid)[0] is True
    assert veto_honest_plus_quality_floor(mid)[0] is True


def test_unlocked_band_bounds():
    assert in_unlocked_band(0.047) is True
    assert in_unlocked_band(0.18333) is False
    assert in_unlocked_band(0.20) is False


def test_replay_exit_calls_production_manager():
    apply_production_exit_env()
    pos = ReplayPos(
        symbol="BTCUSDT",
        entry_price=100.0,
        entry_time=time.time() - 3600,
        quantity=1.0,
        notional=100.0,
        stop_price=90.0,
        trail_pct=0.004,
        highest_price=100.0,
        lowest_price=100.0,
    )
    bars = [(int(pos.entry_time) + 60, 100.0, 100.0, 99.0, 99.5)]
    with patch(
        "backend.services.day_production_lifecycle_replay.evaluate_engine_managed_exit",
        return_value={"action": "sell", "reason": "DAY_4H_STRUCTURE_BREAK_EXIT"},
    ) as mocked:
        from backend.services.day_production_lifecycle_replay import _advance_position

        closed = _advance_position(pos, bars, [], int(pos.entry_time) + 1, int(pos.entry_time) + 120, 0.0006)
        assert mocked.called
        assert closed is not None
        assert closed.exit_reason == "DAY_4H_STRUCTURE_BREAK_EXIT"


def test_fourh_bundle_excludes_unclosed_and_future_bars():
    from backend.services.day_production_lifecycle_replay import FOURH_SEC, fourh_bundle

    # Three 4H opens: 00:00, 04:00, 08:00 UTC on a synthetic day.
    t0 = 1_700_000_000
    t0 = (t0 // FOURH_SEC) * FOURH_SEC
    rows = [
        [float(t0), 100.0, 110.0, 90.0, 105.0, 0.0],
        [float(t0 + FOURH_SEC), 105.0, 120.0, 100.0, 118.0, 0.0],
        [float(t0 + 2 * FOURH_SEC), 118.0, 130.0, 117.0, 125.0, 0.0],
    ]
    # Evaluate 1h into the third bar. Completed = first two only unless 1m forming is supplied.
    now = t0 + 2 * FOURH_SEC + 3600
    asof = fourh_bundle(rows, float(now))
    opens = [r[0] for r in asof["4h"]]
    assert t0 in opens
    assert t0 + FOURH_SEC in opens
    assert t0 + 2 * FOURH_SEC not in opens
    for r in asof["4h"]:
        assert r[0] + FOURH_SEC <= now + 1e-9

    # Forming bar rebuilt from 1m <= now must use the 1m close, not the leaked 4H close 125.
    bars_1m = [
        (t0 + 2 * FOURH_SEC, 118.0, 119.0, 117.5, 118.5),
        (now - 60, 118.5, 119.2, 118.0, 119.0),
        (t0 + 3 * FOURH_SEC - 60, 124.0, 130.0, 123.0, 125.0),  # after now — must not leak
    ]
    formed = fourh_bundle(rows, float(now), bars_1m)
    last = formed["4h"][-1]
    assert last[0] == float(t0 + 2 * FOURH_SEC)
    assert last[4] == 119.0
    assert last[2] == 119.2
    assert last[3] == 117.5


def test_prebuy_no_longer_blocks_on_asof_4h_broken():
    """4H removed from trading authority (2026-09-17). Pre-buy no longer
    blocks on a broken 4H structure."""
    from backend.services.day_controlled_exits import evaluate_pre_buy_exit_consistency
    from backend.services.day_production_lifecycle_replay import FOURH_SEC, fourh_bundle
    from backend.services.portfolio_engine import get_coin_profile

    t0 = (1_700_000_000 // FOURH_SEC) * FOURH_SEC
    rows = []
    px = 90.0
    for i in range(60):
        o = px
        c = px + 1.0
        rows.append([float(t0 + i * FOURH_SEC), o, c + 0.2, o - 0.5, c, 0.0])
        px = c
    prior_open = t0 + 59 * FOURH_SEC
    now = prior_open + FOURH_SEC + 600
    bars_1m = [
        (prior_open + FOURH_SEC, 149.0, 149.2, 94.0, 95.0),
        (now, 95.0, 95.1, 94.5, 95.0),
    ]
    bundle = fourh_bundle(rows, float(now), bars_1m)
    profile = get_coin_profile("BTCUSDT")
    pre = evaluate_pre_buy_exit_consistency(
        setup="HTF_TREND_PULLBACK",
        entry_price=95.0,
        stop_price=90.0,
        thesis_invalid_level=0.0,
        thesis_target_level=100.0,
        entry_vwap=95.0,
        entry_ts=float(now),
        coin_profile=profile,
        bundle=bundle,
        bar_ts=float(now),
    )
    assert pre.get("allowed") is True
    assert "STRUCTURE" not in str(pre.get("immediate_exit_reason") or pre.get("block_reason") or "")


def test_fourh_bundle_last_completed_close_not_after_eval():
    from backend.services.day_production_lifecycle_replay import FOURH_SEC, fourh_bundle

    t0 = (1_700_000_000 // FOURH_SEC) * FOURH_SEC
    rows = [[float(t0 + i * FOURH_SEC), 100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i, 0.0] for i in range(6)]
    now = t0 + 3 * FOURH_SEC + 10
    bundle = fourh_bundle(rows, float(now))
    for r in bundle["4h"]:
        close_ts = r[0] + FOURH_SEC
        is_forming = r[0] == (now // FOURH_SEC) * FOURH_SEC
        if not is_forming:
            assert close_ts <= now + 1e-9


def test_run_arm_respects_one_position_per_symbol():
    apply_production_exit_env()
    now = int(time.time()) // 900 * 900
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE feature_ohlcv (symbol TEXT, interval TEXT, ts INTEGER, open REAL, high REAL, low REAL, close REAL, volume REAL)")
    conn.execute("CREATE TABLE ai_inference_log (symbol TEXT, ts_utc TEXT, prob_buy REAL, prob_hold REAL, prob_sell REAL, strategy_id TEXT)")
    for i in range(40):
        ts = now + i * 60
        conn.execute(
            "INSERT INTO feature_ohlcv VALUES (?,?,?,?,?,?,?,?)",
            ("BTC-USDT", "1m", ts, 100.0, 100.2, 99.8, 100.0, 1.0),
        )
    conn.execute(
        "INSERT INTO ai_inference_log VALUES (?,?,?,?,?,?)",
        ("BTCUSDT", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)), 0.7, 0.3, 0.0, "day"),
    )
    conn.commit()
    from backend.services.day_production_lifecycle_replay import (
        decision_bars,
        load_1m_bars,
        load_inferences,
        resample_4h,
        run_arm,
        veto_current,
    )

    bars = load_1m_bars(conn)
    events = decision_bars(load_inferences(conn))
    closed, acc, _rej = run_arm(
        name="t",
        events=events,
        bars=bars,
        fourh={"BTCUSDT": resample_4h(bars["BTCUSDT"])},
        admit=veto_current,
        authority_mode="legacy",
    )
    assert acc >= 1
    assert len(closed) <= 1


def test_rank_key_prefers_final_selection_score():
    assert rank_key({"p_buy": 0.9, "final_selection_score": 0.1}) == 0.1
    assert rank_key({"p_buy": 0.42}) == 0.42


def test_pick_path_ev_hold_when_bars_missing():
    from backend.services.day_production_lifecycle_replay import pick_direct_path_ev_winner

    winner, evs = pick_direct_path_ev_winner([], bars={s: [] for s in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")}, epoch=1_788_000_000)
    assert winner is None
    assert evs["BTCUSDT"] == 0.0


def test_parse_epoch_treats_naive_ohlcv_as_utc():
    from backend.services.day_production_lifecycle_replay import parse_epoch

    naive = parse_epoch("2026-08-25 22:27:22.210415")
    aware = parse_epoch("2026-08-25T22:27:22.210415+00:00")
    assert naive == aware
    assert naive == int(datetime(2026, 8, 25, 22, 27, 22, tzinfo=timezone.utc).timestamp())
