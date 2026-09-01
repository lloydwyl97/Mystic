import sqlite3
import time
from unittest.mock import patch

from backend.services.day_production_lifecycle_replay import (
    ReplayPos,
    apply_production_exit_env,
    in_unlocked_band,
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
    )
    assert acc >= 1
    assert len(closed) <= 1
