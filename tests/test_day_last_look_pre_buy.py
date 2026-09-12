"""Last-look BUY uses the live executable price, not the earlier decision mid."""

from backend.services.day_controlled_exits import last_look_buy_mark


def test_last_look_uses_lowest_live_print():
    """2026-09-11T12:30Z ETH: decision mid 2466.85, live fill 2447.11."""
    mark = last_look_buy_mark(
        decision_price=2466.85,
        expected_fill=2447.11,
        best_bid=2447.00,
        best_ask=2447.20,
        limit_price=2467.17,
    )
    assert mark == 2447.00


def test_last_look_ignores_missing_book():
    mark = last_look_buy_mark(decision_price=2614.41)
    assert mark == 2614.41


def test_last_look_14_15_unchanged_when_book_matches():
    mark = last_look_buy_mark(
        decision_price=2614.41,
        expected_fill=2613.60,
        best_bid=2614.29,
        best_ask=2614.52,
        limit_price=2613.60,
    )
    assert mark == 2613.60


def test_portfolio_engine_calls_last_look():
    src = open("backend/services/portfolio_engine.py", encoding="utf-8").read()
    assert "last_look_buy_mark" in src
    assert "BUY_BLOCKED_LAST_LOOK" in src
    assert "LAST_LOOK_ENTRY_EXIT" in src


def test_incident_1415_giveback_cuts_after_18bps_mfe(monkeypatch):
    from backend.services.day_controlled_exits import EXIT_GIVEBACK, evaluate_giveback_exit

    monkeypatch.setenv("DAY_GIVEBACK_EXIT_ENABLED", "true")
    monkeypatch.setenv("DAY_GIVEBACK_MIN_HOLD_MIN", "20")
    monkeypatch.setenv("DAY_GIVEBACK_MIN_MFE_PCT", "0.0015")
    monkeypatch.setenv("DAY_GIVEBACK_TRIGGER_PNL_PCT", "-0.0015")
    out = evaluate_giveback_exit(
        entry_price=2613.6,
        highest_price=2618.33,
        net_pnl_pct=-0.00224,
        hold_minutes=25.0,
    )
    assert out is not None
    assert out["action"] == "sell"
    assert out["reason"] == EXIT_GIVEBACK


def test_incident_1415_peak_turn_sells_while_green(monkeypatch):
    from backend.services.day_controlled_exits import EXIT_PEAK_TURN, evaluate_peak_turn_exit

    monkeypatch.setenv("DAY_PEAK_TURN_EXIT_ENABLED", "true")
    monkeypatch.setenv("DAY_PEAK_TURN_MIN_HOLD_MIN", "2")
    monkeypatch.setenv("DAY_PEAK_TURN_MIN_MFE_PCT", "0.0015")
    monkeypatch.setenv("DAY_PEAK_TURN_PULLBACK_PCT", "0.0008")
    out = evaluate_peak_turn_exit(
        entry_price=2613.6,
        highest_price=2618.33,
        current_price=2615.8,
        net_pnl_pct=0.00024,
        hold_minutes=8.0,
    )
    assert out is not None
    assert out["action"] == "sell"
    assert out["reason"] == EXIT_PEAK_TURN


def test_incident_1415_spike_fade_blocks_buy_after_top():
    from backend.services.day_controlled_exits import BUY_BLOCKED_SPIKE_FADE, evaluate_spike_fade_entry

    bundle = {"4h": [[1, 2452.92, 2664.29, 2452.92, 2613.60, 1.0], [2, 2452.92, 2664.29, 2452.92, 2613.60, 1.0]]}
    out = evaluate_spike_fade_entry(mark=2613.60, bundle=bundle)
    assert out is not None
    assert out["block_reason"] == BUY_BLOCKED_SPIKE_FADE
    assert out["fade_pct"] > 0.004


def test_incident_1415_stall_cuts_dead_hold(monkeypatch):
    from backend.services.day_controlled_exits import EXIT_STALL_DEAD, evaluate_stall_exit

    monkeypatch.setenv("DAY_STALL_EXIT_ENABLED", "true")
    monkeypatch.setenv("DAY_STALL_MIN_HOLD_MIN", "120")
    monkeypatch.setenv("DAY_STALL_MAX_MFE_PCT", "0.0050")
    monkeypatch.setenv("DAY_STALL_MIN_ADVERSE_PCT", "0.003")
    out = evaluate_stall_exit(
        entry_price=2613.6,
        highest_price=2618.33,
        lowest_price=2531.6,
        current_price=2531.6,
        net_pnl_pct=-0.0319,
        hold_minutes=120.0,
        max_hold_min=360,
    )
    assert out is not None
    assert out["action"] == "sell"
    assert out["reason"] == EXIT_STALL_DEAD
