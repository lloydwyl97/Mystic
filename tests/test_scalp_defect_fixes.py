from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from backend.services.binance_scalp.paper_engine import _round_trip_execution_costs
from backend.services.binance_scalp.scalp_position_lifecycle import _stale_review_due
from backend.services.binance_scalp import status_snapshot


class _Signal:
    def __init__(self, symbol: str, score: float) -> None:
        self.symbol = symbol
        self.setup_name = "test_setup"
        self.score = score
        self.passed = True

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "setup_name": self.setup_name,
            "score": self.score,
            "passed": self.passed,
        }


class _Tracker:
    def record(self, *_args) -> None:
        pass

    def diagnostics(self, *_args):
        return SimpleNamespace(as_dict=lambda: {"mid_change_15s": 0.001})


def test_status_uses_enriched_global_pick_and_only_marks_selected_symbol(monkeypatch):
    rows = []
    for symbol, score in (("BTCUSDT", 1.8), ("SOLUSDT", 1.7)):
        signal = _Signal(symbol, score)
        rows.append(
            {
                "symbol": symbol,
                "snap": SimpleNamespace(best_bid=100.0, mid=100.0, spread_pct=0.0002),
                "signal": signal,
                "all_signals": [signal.as_dict()],
                "rank_meta": {
                    "ranked": [{"setup_name": "test_setup", "rank_score": score}],
                    "regime": "range",
                    "reachability_surplus": 0.01,
                },
                "rank_score": score,
                "entry_eligible": True,
                "best_setup": "test_setup",
            }
        )

    class _Router:
        def __init__(self, **_kwargs) -> None:
            pass

        def evaluate_all(self, **_kwargs):
            return rows

        def _current_regime(self, *_args):
            return "range"

        def strategy_inventory(self):
            return {"enabled": ["test_setup"]}

    class _Klines:
        def get(self, _symbol):
            return []

    def _enrich(candidates, **_kwargs):
        enriched = [dict(row) for row in candidates]
        enriched[1]["rank_score_raw"] = enriched[1]["rank_score"]
        enriched[1]["rank_score"] = 2.0
        enriched[1]["intelligence"] = {"boost": 0.3, "unsafe": float("nan")}
        return enriched

    monkeypatch.setattr(status_snapshot, "ScalpStrategyRouter", _Router)
    monkeypatch.setattr(status_snapshot, "KlineCache", _Klines)
    monkeypatch.setattr(
        "backend.services.scalp_ai_rank_enrichment.enrich_scalp_ranked_candidates",
        _enrich,
    )

    result = status_snapshot._evaluate_strategy_router(
        SimpleNamespace(products=("BTCUSDT", "SOLUSDT"), max_notional_paper=150.0),
        SimpleNamespace(),
        SimpleNamespace(),
        _Tracker(),
        warm_rounds=0,
    )

    assert result["overall_entry_ready"] is True
    assert result["best_global_candidate"]["symbol"] == "SOLUSDT"
    assert result["symbols"]["SOLUSDT"]["router_entry_ready"] is True
    assert result["symbols"]["BTCUSDT"]["router_entry_ready"] is False
    assert result["symbols"]["BTCUSDT"]["per_symbol_entry_eligible"] is True
    assert result["symbols"]["SOLUSDT"]["intelligence"]["unsafe"] is None
    json.dumps(result, allow_nan=False)


def test_stale_exit_preview_waits_for_same_review_interval_as_engine():
    now = datetime.now(timezone.utc).timestamp()
    recent = datetime.fromtimestamp(now - 10, timezone.utc).isoformat()
    old = datetime.fromtimestamp(now - 31, timezone.utc).isoformat()

    assert not _stale_review_due(
        hold_sec=600,
        stale_timeout_sec=300,
        stale_review_count=1,
        last_review_ts=recent,
        now_epoch=now,
        review_interval_sec=30,
    )
    assert _stale_review_due(
        hold_sec=600,
        stale_timeout_sec=300,
        stale_review_count=1,
        last_review_ts=old,
        now_epoch=now,
        review_interval_sec=30,
    )
    assert _stale_review_due(
        hold_sec=300,
        stale_timeout_sec=300,
        stale_review_count=0,
        last_review_ts=recent,
        now_epoch=now,
        review_interval_sec=30,
    )


def test_learning_costs_use_persisted_entry_and_exit_economics():
    econ = SimpleNamespace(taker_fee_pct=0.001, slippage_buffer_pct=0.0005)
    fees, slippage = _round_trip_execution_costs(
        entry_notional=150.0,
        exit_notional=153.0,
        econ=econ,
        persisted_entry_fee=0.17,
        persisted_entry_slippage=0.08,
    )

    assert fees == pytest.approx(0.17 + 153.0 * 0.001)
    assert slippage == pytest.approx(0.08 + 153.0 * 0.0005)
