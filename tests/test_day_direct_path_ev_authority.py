"""Direct four-coin path-EV is DAY paper authority. Old rank is telemetry only."""

from __future__ import annotations

from backend.services.day_direct_path_ev_authority import (
    DAY_AUTHORITY_MODE,
    HOLD_EV,
    OLD_RANK_EXECUTION_AUTHORITY,
    decide_day_bar,
    old_rank_telemetry,
    post_cost_economics_ev,
    select_action,
)
from backend.services.entry_decision_authority import build_day_entry_provenance, is_model_controlled


class _Cand:
    def __init__(self, symbol: str, score: float):
        self.symbol = symbol
        self.decision_data = {"final_selection_score": score}

    def rank_score(self) -> float:
        return float(self.decision_data["final_selection_score"])


def test_old_rank_btc_path_ev_sol_selects_sol():
    scores = {
        "btc_path_ev": 0.0001,
        "eth_path_ev": 0.0002,
        "sol_path_ev": 0.0009,
        "xrp_path_ev": 0.0003,
        "path_net_status": "predicted",
        "path_net_model_id": "day_path_net_v1",
    }
    out = select_action(scores, old_rank_nominee="BTCUSDT", old_rank_score=9.9)
    assert out["old_rank_execution_authority"] is False
    assert out["old_rank_nominee"] == "BTCUSDT"
    assert out["path_ev_winner"] == "SOLUSDT"
    assert out["selected_action"] == "BUY_SOLUSDT"
    assert out["selected_symbol"] == "SOLUSDT"
    assert out["day_authority_mode"] == DAY_AUTHORITY_MODE
    assert out["hold_ev"] == HOLD_EV


def test_old_rank_nominee_below_hold_selects_hold():
    scores = {
        "btc_path_ev": -0.0004,
        "eth_path_ev": -0.0002,
        "sol_path_ev": -0.0001,
        "xrp_path_ev": -0.0003,
        "path_net_status": "predicted",
        "path_net_model_id": "day_path_net_v1",
    }
    out = select_action(scores, old_rank_nominee="ETHUSDT", old_rank_score=8.0)
    assert out["selected_action"] == "HOLD"
    assert out["path_ev_winner"] == "HOLD"
    assert out["selected_ev"] == 0.0
    assert out["old_rank_nominee"] == "ETHUSDT"
    assert out["old_rank_execution_authority"] is False


def test_missing_old_rank_still_selects_from_four_coins():
    scores = {
        "btc_path_ev": -0.0001,
        "eth_path_ev": 0.0005,
        "sol_path_ev": 0.0002,
        "xrp_path_ev": 0.0001,
        "path_net_status": "predicted",
        "path_net_model_id": "day_path_net_v1",
    }
    out = select_action(scores, old_rank_nominee="", old_rank_score=None)
    assert out["old_rank_nominee"] == ""
    assert out["selected_symbol"] == "ETHUSDT"
    assert out["selected_action"] == "BUY_ETHUSDT"


def test_hold_ev_is_exactly_zero():
    out = select_action(
        {"btc_path_ev": 0.0, "eth_path_ev": 0.0, "sol_path_ev": 0.0, "xrp_path_ev": 0.0, "path_net_status": "predicted"},
        old_rank_nominee="BTCUSDT",
        old_rank_score=1.0,
    )
    assert out["hold_ev"] == 0.0
    assert out["selected_action"] == "HOLD"


def test_old_rank_telemetry_only():
    nominee, score = old_rank_telemetry([_Cand("BTC/USDT", 1.0), _Cand("SOL/USDT", 3.0)])
    assert nominee == "SOLUSDT"
    assert score == 3.0
    assert OLD_RANK_EXECUTION_AUTHORITY is False


def test_all_four_plus_hold_keys_present():
    out = select_action({"btc_path_ev": 0.01, "eth_path_ev": 0.0, "sol_path_ev": 0.0, "xrp_path_ev": 0.0, "path_net_status": "predicted"})
    for key in ("btc_path_ev", "eth_path_ev", "sol_path_ev", "xrp_path_ev", "hold_ev"):
        assert key in out
    assert out["hold_ev"] == 0.0


def test_no_setup_regime_or_fbr_in_selector():
    src = open("backend/services/day_direct_path_ev_authority.py", encoding="utf-8").read()
    banned = ("SETUP_REGIME", "FBR", "HTF_ANCHOR", "LOW_MFE", "entry_confirmation", "hard_block")
    for token in banned:
        assert token not in src


def test_provenance_stamps_direct_mode():
    dec = select_action(
        {
            "btc_path_ev": 0.0001,
            "eth_path_ev": 0.0008,
            "sol_path_ev": 0.0002,
            "xrp_path_ev": 0.0001,
            "path_net_status": "predicted",
            "path_net_model_id": "day_path_net_v1",
        },
        old_rank_nominee="BTCUSDT",
        old_rank_score=9.0,
    )
    prov = build_day_entry_provenance(decision_data=dec, symbol="ETH/USDT", decision_id="t1")
    assert prov["day_authority_mode"] == DAY_AUTHORITY_MODE
    assert prov["old_rank_execution_authority"] is False
    assert prov["old_rank_nominee"] == "BTCUSDT"
    assert prov["selected_action"].startswith("BUY")
    assert prov["path_ev_winner"] == "ETHUSDT"
    assert is_model_controlled(prov, engine="day") is True


def test_decide_does_not_require_candidates():
    out = decide_day_bar(db_path="", candidates=None)
    assert out["day_authority_mode"] == DAY_AUTHORITY_MODE
    assert out["hold_ev"] == 0.0
    assert "btc_path_ev" in out
    assert out["old_rank_execution_authority"] is False


def test_tape_persists_all_four_coins_and_hold(monkeypatch):
    captured: list[dict] = []

    def _fake_record(rows, force=False):
        captured.extend(rows)
        assert force is True
        return len(rows)

    monkeypatch.setattr("backend.services.decision_book_tape.record_rows", _fake_record)
    monkeypatch.setattr("backend.services.decision_book_tape.snapshot_book", lambda *_a, **_k: {})
    from backend.services.decision_book_tape import record_day_bar_authority

    n = record_day_bar_authority(
        {
            "day_authority_mode": DAY_AUTHORITY_MODE,
            "old_rank_nominee": "BTCUSDT",
            "old_rank_score": 9.9,
            "btc_path_ev": 0.0001,
            "eth_path_ev": 0.0002,
            "sol_path_ev": 0.0009,
            "xrp_path_ev": 0.0003,
            "hold_ev": 0.0,
            "path_ev_winner": "SOLUSDT",
            "selected_action": "BUY_SOLUSDT",
            "selected_symbol": "SOLUSDT",
            "selected_ev": 0.0009,
            "path_net_model_id": "day_path_net_v1",
            "path_aware_policy_id": "day_path_aware_v1",
            "why_selected": "PATH_NET_BEATS_HOLD",
            "prediction_timestamp": "2026-08-16T22:00:00+00:00",
        }
    )
    assert n == 5
    symbols = [r["symbol"] for r in captured]
    assert symbols == ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "HOLD"]
    hold = next(r for r in captured if r["symbol"] == "HOLD")
    assert hold["hold_ev"] == 0.0
    assert hold["buy_ev"] == 0.0
    extras = captured[0]["extras_json"]
    assert "old_rank_execution_authority" in extras
    assert "false" in extras.lower()
    assert "BTCUSDT" in extras


def test_true_safety_gates_still_in_execute_buy():
    src = open("backend/services/portfolio_engine.py", encoding="utf-8").read()
    for token in (
        "KILL SWITCH CHECK",
        "BUY_BLOCKED_ALREADY_OPEN",
        "BUY_BLOCKED_MAX_POSITIONS",
        "BUY_BLOCKED_NO_CASH",
        "BUY_BLOCKED_INSUFFICIENT_CASH",
        "EXIT_MARK_STALE",
        "LIVE_TEST_BUY_BLOCKED",
    ):
        assert token in src
    assert "DAY_PATH_EV_SAFETY reject=DUPLICATE_SAME_SYMBOL" in src
    assert "DAY_PATH_EV_SAFETY reject=MAX_OPEN_LIMIT" in src


def test_scalp_authority_unchanged():
    src = open("backend/services/binance_scalp/scalp_candidate_ranking.py", encoding="utf-8").read()
    assert "def pick_best_global_candidate" in src
    assert "HOLD_ACTION_EV" in src
    assert "day_direct_path_ev_authority" not in src
    from backend.services.binance_scalp.scalp_candidate_ranking import HOLD_ACTION_EV

    assert HOLD_ACTION_EV == 0.0


def test_day_exits_unchanged():
    src = open("backend/services/day_controlled_exits.py", encoding="utf-8").read()
    assert "EXIT_STALL" in src
    assert "EXIT_GIVEBACK" in src
    assert "day_direct_path_ev_authority" not in src
    auth = open("backend/services/day_direct_path_ev_authority.py", encoding="utf-8").read()
    assert "day_controlled_exits" not in auth
    assert "STALL" not in auth
    assert "GIVEBACK" not in auth


def test_live_flags_remain_false_in_authority_files():
    files = (
        "backend/services/day_direct_path_ev_authority.py",
        "backend/services/entry_decision_authority.py",
        "backend/services/decision_book_tape.py",
    )
    banned = (
        "LIVE_TRADES_ALLOWED = True",
        "SCALP_LIVE = True",
        'TRADING_MODE = "live"',
        'EXECUTION_MODE = "live"',
    )
    for path in files:
        src = open(path, encoding="utf-8").read()
        for token in banned:
            assert token not in src


def test_post_cost_economics_ev_negative_cannot_beat_hold():
    ev = post_cost_economics_ev(
        {
            "expected_favorable_excursion": 0.002,
            "expected_adverse_excursion": 0.008,
            "prob_buy": 0.40,
            "prob_sell": 0.40,
            "prob_hold": 0.20,
            "estimated_fees_pct": 0.001,
            "estimated_slippage_pct": 0.0005,
            "spread_pct": 0.0004,
        }
    )
    assert ev is not None
    assert ev < 0.0
    scores = {
        "btc_path_ev": 0.01,
        "eth_path_ev": 0.0,
        "sol_path_ev": 0.0,
        "xrp_path_ev": 0.0,
        "path_net_status": "predicted",
    }
    out = select_action(scores)
    assert out["selected_action"].startswith("BUY")
    # Ranking EV itself must be the economic number when identifiable and non-positive.
    scores["btc_path_ev"] = min(0.01, ev)
    out2 = select_action(scores)
    assert out2["selected_action"] == "HOLD"


def test_post_cost_economics_ev_missing_fields_is_none():
    assert post_cost_economics_ev({"buy_margin": -0.2, "confidence": 0.9}) is None


def test_process_bar_uses_direct_selector_not_old_queue():
    src = open("backend/services/portfolio_engine.py", encoding="utf-8").read()
    assert "_select_direct_path_ev_candidate" in src
    assert "decide_day_bar" in src
    assert "record_day_bar_authority" in src
    assert "top_candidate = _buy_queue[0]" not in src
    assert "MULTI_BUY_SKIP_OLD_RANK" in src
