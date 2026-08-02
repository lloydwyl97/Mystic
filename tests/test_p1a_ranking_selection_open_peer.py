"""P1A: final_selection_score is primary; open-position demotion must not fake a score win."""

from __future__ import annotations

from types import SimpleNamespace

from backend.services.symbol_setup_outcome_penalty import (
    assign_v3_selection_ranks,
    build_truthful_selection_reason,
)


def _cand(symbol: str, final_sel: float, *, nev: float = 0.0, rank: float = 0.0, margin: float = 0.0, conf: float = 0.0):
    return SimpleNamespace(
        symbol=symbol,
        confidence=conf,
        decision_data={
            "final_selection_score": final_sel,
            "selection_score": final_sel,
            "selected_net_expected_value": nev,
            "adjusted_ev": nev,
            "rank_score": rank,
            "outcome_adjusted_rank_score": rank,
            "buy_margin": margin,
            "ai_confidence": conf,
        },
        rank_score=lambda r=rank: r,
    )


def _score_primary_key(c, open_symbols: set[str]):
    """Mirror portfolio_engine._ml_rank_key contract for unit tests."""
    d = c.decision_data or {}
    bm = float(d.get("buy_margin") or 0.0)
    nev = float(d.get("selected_net_expected_value") or d.get("adjusted_ev") or 0.0)
    conf = float(getattr(c, "confidence", None) or d.get("ai_confidence") or 0.0)
    ts = float(d.get("signal_timestamp") or d.get("bar_timestamp") or 0.0)
    return (
        -float(d.get("final_selection_score") or d.get("selection_score") or 0.0),
        -nev,
        -float(d.get("outcome_adjusted_rank_score") or c.rank_score()),
        -bm,
        -conf,
        -ts,
        c.symbol in open_symbols,
        str(c.symbol or ""),
    )


def test_higher_score_open_peer_not_demoted_before_score():
    """Snap-8109 shape: SOL 0.221 open, ETH 0.134 unheld — SOL ranks above ETH by score."""
    sol = _cand("SOL/USDT", 0.22113819, nev=0.00570947, rank=0.462, margin=0.45, conf=0.28)
    eth = _cand("ETH/USDT", 0.13351102, nev=0.03349088, rank=0.255, margin=0.10, conf=0.15)
    open_syms = {"SOL/USDT"}
    ordered = sorted([eth, sol], key=lambda c: _score_primary_key(c, open_syms))
    assert [c.symbol for c in ordered] == ["SOL/USDT", "ETH/USDT"]


def test_lower_score_candidate_cannot_claim_score_victory():
    sol = _cand("SOL/USDT", 0.22113819, nev=0.0057)
    eth = _cand("ETH/USDT", 0.13351102, nev=0.0335)
    ordered = [sol, eth]
    reason = build_truthful_selection_reason(eth, ordered, open_symbols={"SOL/USDT"})
    why = reason["why_selected"]
    assert "0.133511 >" not in why
    assert "0.13351102 > 0.22113819" not in why
    assert reason["winner_score"] == 0.13351102
    assert reason["runner_up_score"] == 0.22113819
    assert reason["winner_score"] < reason["runner_up_score"]
    assert reason["selection_key_used"] == "open_symbol_skipped_capacity"
    assert reason["skipped_reason"] == "same_symbol_already_open"
    assert "SOL/USDT" in why
    assert "skipped" in why


def test_higher_score_non_open_beats_lower_score_non_open():
    sol = _cand("SOL/USDT", 0.40, nev=0.30)
    eth = _cand("ETH/USDT", 0.23, nev=0.10)
    ordered = sorted([eth, sol], key=lambda c: _score_primary_key(c, set()))
    assert ordered[0].symbol == "SOL/USDT"
    assign_v3_selection_ranks(ordered, open_symbols=set())
    why = ordered[0].decision_data["why_selected"]
    assert ordered[0].decision_data["selection_key_used"] == "highest_final_selection_score"
    assert "0.400000 > ETH/USDT 0.230000" in why
    assert ordered[0].decision_data["winner_score"] > ordered[0].decision_data["runner_up_score"]


def test_open_status_only_tiebreaks_equal_scores():
    a = _cand("ETH/USDT", 0.25, nev=0.10, rank=0.3, margin=0.2, conf=0.2)
    b = _cand("SOL/USDT", 0.25, nev=0.10, rank=0.3, margin=0.2, conf=0.2)
    open_syms = {"SOL/USDT"}
    ordered = sorted([a, b], key=lambda c: _score_primary_key(c, open_syms))
    # Equal scores → non-open preferred as final tie-break.
    assert ordered[0].symbol == "ETH/USDT"
    # Material gap → score still primary even if winner is open.
    high_open = _cand("SOL/USDT", 0.40, nev=0.10)
    low_free = _cand("ETH/USDT", 0.20, nev=0.10)
    ordered2 = sorted([low_free, high_open], key=lambda c: _score_primary_key(c, {"SOL/USDT"}))
    assert ordered2[0].symbol == "SOL/USDT"


def test_snap_8109_style_executable_selection_reason():
    sol = _cand("SOL/USDT", 0.22113819, nev=0.00570947, rank=0.462)
    eth = _cand("ETH/USDT", 0.13351102, nev=0.03349088, rank=0.255)
    ordered = sorted([eth, sol], key=lambda c: _score_primary_key(c, {"SOL/USDT"}))
    assert ordered[0].symbol == "SOL/USDT"
    # Execution filter skips open SOL → ETH is executable winner.
    assign_v3_selection_ranks(ordered, open_symbols={"SOL/USDT"}, selected=eth)
    dd = eth.decision_data
    assert dd["selection_key_used"] == "open_symbol_skipped_capacity"
    assert dd["winner_symbol"] == "ETH/USDT"
    assert dd["runner_up_symbol"] == "SOL/USDT"
    assert dd["winner_score"] < dd["runner_up_score"]
    assert "same_symbol_already_open" in dd["skipped_reason"]
    assert ">" not in dd["why_selected"] or "skipped" in dd["why_selected"]
    # Must not claim ETH score > SOL score.
    assert "0.133511 > SOL/USDT 0.221138" not in dd["why_selected"]


def test_assign_v3_solo_still_truthful():
    solo = [_cand("BTC/USDT", 0.33, nev=0.12)]
    assign_v3_selection_ranks(solo)
    assert solo[0].decision_data["why_selected"] == "solo_candidate_no_peer"
    assert solo[0].decision_data["selection_key_used"] == "solo_candidate_no_peer"


def test_nev_tiebreak_when_scores_equal():
    a = _cand("ETH/USDT", 0.30, nev=0.20)
    b = _cand("SOL/USDT", 0.30, nev=0.05)
    ordered = sorted([b, a], key=lambda c: _score_primary_key(c, set()))
    assert ordered[0].symbol == "ETH/USDT"
    reason = build_truthful_selection_reason(ordered[0], ordered, open_symbols=set())
    assert reason["selection_key_used"] == "higher_nev_tiebreak"
