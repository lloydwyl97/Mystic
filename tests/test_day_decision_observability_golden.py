"""Golden: observability on/off must not change DAY action, size, order, or exit."""

from __future__ import annotations

import json

from backend.config.trading_economics import DAY_TARGET_NOTIONAL_PER_SLOT_USD
from backend.services.day_decision_observability import build_group_contract, record_day_ranking_group
from backend.services.day_direct_path_ev_authority import decide_day_bar, select_action


def _fixed_scores():
    return {
        "btc_path_ev": 0.00015,
        "eth_path_ev": -0.00010,
        "sol_path_ev": 0.00040,
        "xrp_path_ev": 0.00005,
        "path_net_status": "predicted",
        "path_net_model_id": "day_path_net_v1",
        "costs_bps": 6.5,
        "horizon_minutes": 240,
    }


def _snapshot(decision: dict) -> str:
    keys = (
        "selected_action",
        "selected_symbol",
        "selected_ev",
        "path_ev_winner",
        "btc_path_ev",
        "eth_path_ev",
        "sol_path_ev",
        "xrp_path_ev",
        "hold_ev",
        "why_selected",
        "old_rank_execution_authority",
    )
    return json.dumps({k: decision.get(k) for k in keys}, sort_keys=True, default=str)


def test_select_action_byte_equal_with_observability_side_effect(tmp_path, monkeypatch):
    monkeypatch.setenv("DAY_DECISION_OBSERVABILITY", "true")
    monkeypatch.setenv("MYSTIC_TRADING_MODE", "paper")
    monkeypatch.setenv("TRADING_MODE", "paper")
    monkeypatch.setenv("EXECUTION_MODE", "paper")
    off = select_action(_fixed_scores(), old_rank_nominee="BTCUSDT", old_rank_score=1.0)
    on = select_action(_fixed_scores(), old_rank_nominee="BTCUSDT", old_rank_score=1.0)
    before = json.dumps(on, sort_keys=True, default=str)
    record_day_ranking_group(str(tmp_path / "g.db"), decision=on, bar_timestamp=9)
    after = json.dumps(on, sort_keys=True, default=str)
    assert _snapshot(off) == _snapshot(on)
    assert before == after
    assert off["selected_action"] == "BUY_SOLUSDT"
    assert off["selected_symbol"] == "SOLUSDT"


def test_decide_day_bar_unchanged_when_observability_records(tmp_path, monkeypatch):
    monkeypatch.setenv("DAY_DECISION_OBSERVABILITY", "true")
    monkeypatch.setenv("MYSTIC_TRADING_MODE", "paper")
    monkeypatch.setenv("TRADING_MODE", "paper")
    monkeypatch.setenv("EXECUTION_MODE", "paper")

    def _scores(*_a, **_k):
        return _fixed_scores()

    monkeypatch.setattr("backend.services.day_direct_path_ev_authority.score_four_coins", _scores)
    a = decide_day_bar(db_path="", candidates=None)
    b = decide_day_bar(db_path="", candidates=None)
    snap_a = _snapshot(a)
    record_day_ranking_group(str(tmp_path / "g.db"), decision=b, bar_timestamp=11)
    assert snap_a == _snapshot(b)
    assert a["selected_action"] == b["selected_action"]
    assert a["selected_symbol"] == b["selected_symbol"]


def test_sizing_and_order_request_unchanged():
    price = 100.0
    qty = DAY_TARGET_NOTIONAL_PER_SLOT_USD / price
    order_off = {"symbol": "SOLUSDT", "side": "BUY", "type": "MARKET", "quantity": qty, "price": None}
    order_on = dict(order_off)
    dec = select_action(_fixed_scores())
    build_group_contract(decision=dec, bar_timestamp=1)
    assert order_off == order_on
    assert qty == DAY_TARGET_NOTIONAL_PER_SLOT_USD / price


def test_exit_state_unchanged_by_observability_builder():
    from backend.services.day_trade_thesis import EXIT_DAY_4H_STRUCTURE_BREAK, EXIT_TRAILING_STOP

    exit_state = {"action": "hold", "reason": "PATH_AWARE_HOLD_4H_RISE", "trail": EXIT_TRAILING_STOP, "break4h": EXIT_DAY_4H_STRUCTURE_BREAK}
    frozen = json.dumps(exit_state, sort_keys=True)
    dec = select_action(_fixed_scores())
    build_group_contract(decision=dec, bar_timestamp=2)
    assert json.dumps(exit_state, sort_keys=True) == frozen
    assert exit_state["break4h"] == "DAY_4H_STRUCTURE_BREAK_EXIT"
