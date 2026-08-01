"""P1B/P1C: BUY explainability + SELL learning rank_data measurement stamps."""

from __future__ import annotations

from backend.services.portfolio_engine import TradeExplainability


def _sample_explainability(**overrides) -> TradeExplainability:
    ex = TradeExplainability(
        trade_id="t1",
        symbol="BTCUSDT",
        side="BUY",
        timestamp="2026-08-01T00:00:00+00:00",
    )
    ex.prob_buy = 0.55
    ex.prob_hold = 0.30
    ex.prob_sell = 0.15
    ex.quality_opinion_penalty = 0.02
    ex.signal_side_penalty = 0.01
    ex.rank_score = 1.25
    ex.final_selection_score = 1.22
    ex.entry_buy_margin = 0.08
    ex.selected_net_expected_value = 0.0045
    ex.selected_score = 1.22
    ex.argmax_action = "BUY"
    ex.prediction = "BUY"
    ex.side_signal = "buy"
    for k, v in overrides.items():
        setattr(ex, k, v)
    return ex


def test_8_buy_explainability_includes_prob_penalty_rank_fields():
    payload = _sample_explainability().to_dict()
    for key in (
        "prob_buy",
        "prob_hold",
        "prob_sell",
        "quality_opinion_penalty",
        "signal_side_penalty",
        "rank_score",
        "final_selection_score",
        "buy_margin",
        "selected_net_expected_value",
    ):
        assert key in payload
        assert payload[key] is not None
    assert payload["side"] == "BUY"
    assert payload["side_signal"] == "buy"
    assert payload["prob_buy"] == 0.55
    assert payload["rank_score"] == 1.25
    assert payload["buy_margin"] == 0.08


def test_9_sell_learning_rank_data_maps_scores_from_explainability():
    """Mirror portfolio_engine learning enrichment mapping from BUY explainability."""
    ex_payload = _sample_explainability().to_dict()
    _sel = (
        ex_payload.get("selected_score")
        or ex_payload.get("final_selection_score")
        or ex_payload.get("rank_score")
    )
    rank_data = {
        "selected_score": _sel,
        "entry_score": _sel,
        "rank_score": ex_payload.get("rank_score") or _sel,
        "final_selection_score": ex_payload.get("final_selection_score") or _sel,
        "buy_margin": ex_payload.get("buy_margin")
        if ex_payload.get("buy_margin") is not None
        else ex_payload.get("entry_buy_margin"),
        "selected_net_expected_value": ex_payload.get("selected_net_expected_value"),
        "prob_buy": ex_payload.get("prob_buy"),
        "prob_hold": ex_payload.get("prob_hold"),
        "prob_sell": ex_payload.get("prob_sell"),
        "quality_opinion_penalty": ex_payload.get("quality_opinion_penalty"),
        "signal_side_penalty": ex_payload.get("signal_side_penalty"),
    }
    assert float(rank_data["entry_score"] or 0) > 0
    assert float(rank_data["selected_score"] or 0) > 0
    assert float(rank_data["rank_score"] or 0) > 0
    assert float(rank_data["buy_margin"] or 0) > 0
    assert float(rank_data["final_selection_score"] or 0) > 0


def test_10_missing_optional_telemetry_does_not_crash():
    ex = TradeExplainability(
        trade_id="t2",
        symbol="ETHUSDT",
        side="BUY",
        timestamp="2026-08-01T00:00:00+00:00",
    )
    # Defaults: probs None, penalties/rank zero — still serializable.
    payload = ex.to_dict()
    assert payload["prob_buy"] is None
    assert payload["prob_hold"] is None
    assert payload["prob_sell"] is None
    assert payload["quality_opinion_penalty"] == 0.0
    assert payload["signal_side_penalty"] == 0.0
    assert payload["rank_score"] == 0.0
    assert payload["side"] == "BUY"

    # Learning mapping tolerates empties.
    _sel = (
        payload.get("selected_score")
        or payload.get("final_selection_score")
        or payload.get("rank_score")
    )
    rank_data = {
        "selected_score": _sel,
        "entry_score": _sel,
        "rank_score": payload.get("rank_score") or _sel,
        "buy_margin": payload.get("buy_margin"),
    }
    assert rank_data["entry_score"] in (0, 0.0, None) or float(rank_data["entry_score"] or 0) == 0.0
