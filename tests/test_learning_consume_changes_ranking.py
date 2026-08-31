"""Prove a completed outcome changes later ranking / size (not a gate)."""

from __future__ import annotations

import random
from pathlib import Path

from backend.services.day_outcome_bandit import apply_bandit_to_decision_data
from backend.services.trade_learning_writer import (
    TradeLearningRecord,
    consume_setup_outcomes_for_ranking,
    record_trade_outcome,
)

SETUP = "VWAP_EMA_RECLAIM"


def _write_loss(db: str, *, symbol: str, vol: float, n: int = 10) -> None:
    for i in range(n):
        rec = TradeLearningRecord(
            symbol=symbol,
            entry_timestamp=1_700_000_000 + i,
            exit_timestamp=1_700_000_180 + i,
            entry_price=100.0,
            exit_price=99.7,
            quantity=1.0,
            fees_paid=0.04,
            slippage_cost=0.02,
            net_profit_usd=-0.35,
            net_profit_pct=-0.0035,
            hold_seconds=180,
            close_reason="EARLY_SCRATCH_EXIT",
            extra={
                "setup": SETUP,
                "volatility": vol,
                "momentum": 0.0008,
                "regime": "chop",
                "mfe_pct": 0.0004,
            },
            indicators_while_holding={"max_favorable_pct": 0.0004},
        )
        assert record_trade_outcome(rec, db_path=db) is True


def test_consume_rank_delta_moves_after_losses(tmp_path: Path):
    db = str(tmp_path / "learn.db")
    before = consume_setup_outcomes_for_ranking(db, SETUP)
    assert before["consumed"] is False
    assert before["rank_delta"] == 0.0

    _write_loss(db, symbol="BTCUSDT", vol=0.008, n=10)

    after = consume_setup_outcomes_for_ranking(
        db,
        SETUP,
        features={"volatility": 0.008, "momentum": 0.0008, "regime": "chop", "mfe_pct": 0.0004},
    )
    assert after["consumed"] is True
    assert after["n"] >= 8
    assert after["rank_delta"] < 0
    assert after["size_factor"] < 1.0
    assert after["feature_matched_n"] >= 1


def test_apply_bandit_score_changes_after_outcomes(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DAY_OUTCOME_BANDIT_ENABLED", "true")
    db = str(tmp_path / "bandit_learn.db")
    dd0 = {
        "setup_type": SETUP,
        "day_route_regime": "chop",
        "final_selection_score": 0.62,
        "selection_score": 0.62,
        "realized_volatility_pct": 0.008,
        "mid_change_60s": 0.0008,
        "confidence": 0.7,
    }
    first = apply_bandit_to_decision_data(dd0, "BTC/USDT", db_path=db, rng=random.Random(7))
    score_before = float(first["final_selection_score"])
    assert first.get("learning_outcomes_consumed") is False

    _write_loss(db, symbol="ETHUSDT", vol=0.008, n=10)

    second = apply_bandit_to_decision_data(dd0, "BTC/USDT", db_path=db, rng=random.Random(7))
    assert second.get("learning_outcomes_consumed") is True
    assert int(second.get("learning_outcomes_n") or 0) >= 8
    assert float(second["learning_outcomes_rank_delta"]) < 0
    assert float(second["final_selection_score"]) != score_before
    assert float(second["final_selection_score"]) < score_before
    assert float(second["thesis_size_factor"]) < float(first.get("thesis_size_factor") or 1.0)
