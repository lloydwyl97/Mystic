"""Golden test: corrected observability must not change any trading decision.

Old observability vs corrected observability over the same inputs. Path-EV, the
p_buy used live, the rank used live, the selected symbol, BUY/HOLD, execution
authorization, size inputs and the order request must be identical. Only the
recorded research semantics may differ.
"""

from __future__ import annotations

import copy
from typing import Any

from backend.services.day_decision_observability import build_group_contract
from backend.services.day_direct_path_ev_authority import (
    OLD_RANK_EXECUTION_AUTHORITY,
    old_rank_telemetry,
    select_action,
)

SCORES = {
    "btc_path_ev": 0.0005243229492899691,
    "eth_path_ev": 0.000787350605847948,
    "sol_path_ev": 0.0004884052241086762,
    "xrp_path_ev": 9.922790316291389e-05,
    "valid": {"btc": True, "eth": True, "sol": True, "xrp": True},
    "path_input_by_symbol": {
        "BTCUSDT": {"path_input_valid": True, "path_invalid_reason": None},
        "ETHUSDT": {"path_input_valid": True, "path_invalid_reason": None},
        "SOLUSDT": {"path_input_valid": True, "path_invalid_reason": None},
        "XRPUSDT": {"path_input_valid": True, "path_invalid_reason": None},
    },
    "path_net_model_id": "day_path_net_v1",
}

TRADING_KEYS = (
    "btc_path_ev",
    "eth_path_ev",
    "sol_path_ev",
    "xrp_path_ev",
    "hold_ev",
    "path_ev_winner",
    "selected_action",
    "selected_symbol",
    "selected_ev",
    "selected_net_expected_value",
    "predicted_net_return",
    "why_selected",
    "old_rank_nominee",
    "old_rank_score",
    "old_rank_execution_authority",
    "path_net_model_id",
    "path_aware_policy_id",
    "true_safety_reject_reason",
    "valid",
)


class _Cand:
    """Minimal stand-in for BuyCandidate with the fields ranking/sizing read."""

    def __init__(self, symbol: str, score: float, p_buy: float, conf: float, price: float, atr: float):
        self.symbol = symbol
        self.confidence = conf
        self.current_price = price
        self.atr = atr
        self.decision_id = f"day_{symbol.replace('/', '')}_1"
        self.price_structure_regime = "range"
        self.chop_score = 0.5
        self.decision_data: dict[str, Any] = {
            "final_selection_score": score,
            "prob_buy": p_buy,
            "ml_score": p_buy,
            "estimated_fees_pct": 0.001,
        }

    def rank_score(self) -> float:
        return float(self.decision_data["final_selection_score"])


def _candidates() -> list[_Cand]:
    # Reproduces the real 06:15 bar: only SOL and XRP were buy-intent candidates.
    return [
        _Cand("XRP/USDT", 0.480773, 0.7190491939510026, 0.268, 1.4189, 0.02),
        _Cand("SOL/USDT", 0.465366, 0.6609622029562143, 0.287, 101.91, 1.4),
    ]


class _Engine:
    def __init__(self, cands):
        self.current_bar_candidates = list(cands)
        self.open_positions: dict[str, Any] = {}
        self.db_path = ""


def _decide(cands) -> dict[str, Any]:
    nominee, score = old_rank_telemetry(cands)
    return select_action(copy.deepcopy(SCORES), old_rank_nominee=nominee, old_rank_score=score)


def _account():
    return {"open_symbols": [], "slots_used": 0, "slot_count": 4, "cash_available": 168.09644128999997, "cash_balance": 168.09644128999997}


def test_selection_identical_with_and_without_corrected_capture():
    """Old path: decide only. New path: decide, then build the corrected contract."""
    old_cands = _candidates()
    old_decision = _decide(old_cands)

    new_cands = _candidates()
    new_decision = _decide(new_cands)
    build_group_contract(
        decision=new_decision,
        candidates=new_cands,
        bar_timestamp=1788588900,
        engine=_Engine(new_cands),
        account_state=_account(),
    )

    for key in TRADING_KEYS:
        assert old_decision.get(key) == new_decision.get(key), f"trading field changed: {key}"
    assert new_decision["selected_action"] == "BUY_ETHUSDT"
    assert new_decision["selected_symbol"] == "ETHUSDT"
    assert OLD_RANK_EXECUTION_AUTHORITY is False


def test_contract_build_never_mutates_the_decision():
    cands = _candidates()
    decision = _decide(cands)
    snapshot = copy.deepcopy(decision)
    build_group_contract(
        decision=decision,
        candidates=cands,
        bar_timestamp=1788588900,
        engine=_Engine(cands),
        account_state=_account(),
    )
    assert decision == snapshot


def test_contract_build_never_mutates_candidate_decision_data():
    cands = _candidates()
    decision = _decide(cands)
    before = [copy.deepcopy(c.decision_data) for c in cands]
    build_group_contract(
        decision=decision,
        candidates=cands,
        bar_timestamp=1788588900,
        engine=_Engine(cands),
        account_state=_account(),
    )
    after = [c.decision_data for c in cands]
    assert before == after


def test_live_p_buy_and_live_rank_are_recorded_unchanged():
    cands = _candidates()
    decision = _decide(cands)
    contract = build_group_contract(
        decision=decision,
        candidates=cands,
        bar_timestamp=1788588900,
        engine=_Engine(cands),
        account_state=_account(),
    )
    rows = {r["symbol"]: r for r in contract["candidates"]}
    assert rows["XRPUSDT"]["p_buy"] == 0.7190491939510026
    assert rows["SOLUSDT"]["p_buy"] == 0.6609622029562143
    # Historical field keeps its v1 value and meaning.
    assert rows["XRPUSDT"]["final_rank_score"] == 0.480773
    assert rows["SOLUSDT"]["final_rank_score"] == 0.465366
    # Old-rank telemetry is unchanged and still non-authoritative.
    assert decision["old_rank_nominee"] == "XRPUSDT"
    assert decision["old_rank_score"] == 0.480773


def test_only_research_semantics_change_for_the_unscored_selected_symbol():
    cands = _candidates()
    decision = _decide(cands)
    contract = build_group_contract(
        decision=decision,
        candidates=cands,
        bar_timestamp=1788588900,
        engine=_Engine(cands),
        account_state=_account(),
    )
    eth = next(r for r in contract["candidates"] if r["symbol"] == "ETHUSDT")
    # v1 semantics preserved exactly as before the correction.
    assert eth["eligible"] is False
    assert eth["exclusion_reason"] == "NO_SCORED_CANDIDATE"
    # Corrected semantics added alongside, not in place of.
    assert eth["action_available"] is True
    assert eth["legacy_rank_candidate_present"] is False
    assert eth["legacy_final_rank_score"] is None
    assert eth["legacy_final_rank_score_valid"] is False
    assert eth["production_selected"] is True
    assert contract["selected_action_invariant"]["pass"] is True


def test_hold_decision_is_unchanged_by_capture():
    scores = copy.deepcopy(SCORES)
    for key in ("btc_path_ev", "eth_path_ev", "sol_path_ev", "xrp_path_ev"):
        scores[key] = -1e-6
    old = select_action(copy.deepcopy(scores))
    new = select_action(copy.deepcopy(scores))
    cands = _candidates()
    build_group_contract(
        decision=new,
        candidates=cands,
        bar_timestamp=1788588900,
        engine=_Engine(cands),
        account_state=_account(),
    )
    for key in TRADING_KEYS:
        assert old.get(key) == new.get(key)
    assert new["selected_action"] == "HOLD"
    assert new["hold_ev"] == 0.0


def test_path_input_invalid_coin_is_excluded_from_selection_exactly_as_before():
    scores = copy.deepcopy(SCORES)
    scores["valid"]["eth"] = False
    scores["path_input_by_symbol"]["ETHUSDT"] = {
        "path_input_valid": False,
        "path_invalid_reason": "PATH_INPUT_INVALID_GAP",
    }
    decision = select_action(copy.deepcopy(scores))
    assert decision["selected_symbol"] == "BTCUSDT"
    cands = _candidates()
    contract = build_group_contract(
        decision=decision,
        candidates=cands,
        bar_timestamp=1788588900,
        engine=_Engine(cands),
        account_state=_account(),
    )
    eth = next(r for r in contract["candidates"] if r["symbol"] == "ETHUSDT")
    assert eth["action_available"] is False
    assert eth["action_unavailable_reason"] == "PATH_INPUT_INVALID_GAP"


def test_execution_resolvable_set_is_recorded_separately_from_ranked_set():
    """The executor resolves against ranked + current_bar_candidates."""
    ranked = _candidates()
    decision = _decide(ranked)
    engine = _Engine(ranked)
    # A stale ETH candidate from a prior bar, as observed in production at 06:15.
    engine.current_bar_candidates.append(_Cand("ETH/USDT", 0.0, 0.05998694989838594, 0.814, 2451.96, 33.3))
    contract = build_group_contract(
        decision=decision,
        candidates=ranked,
        bar_timestamp=1788588900,
        engine=engine,
        account_state=_account(),
    )
    eth = next(r for r in contract["candidates"] if r["symbol"] == "ETHUSDT")
    assert eth["legacy_rank_candidate_present"] is False
    assert eth["execution_resolvable_candidate_present"] is True
